# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import os
import random
import tarfile
import tempfile
import time
from io import BytesIO
from typing import Optional
import pdb
import dataclasses
import json

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.transforms import Resize
from tqdm import tqdm

import fourm.utils as utils
import fourm.utils.clip as clip
from fourm.data import CenterCropImageAugmenter, RandomCropImageAugmenter
from fourm.data.modality_info import MODALITY_TRANSFORMS_DIVAE
from fourm.vq.vqvae import DiVAE

from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates


FEATURE_TASKS = ["CLIP-B16"]
VIDEO_EXTENSIONS = (".mp4",)

QUERY_ACTION_BASE = (
    "You are to assist me in accomplishing a task about the input video."
    "# Task\n"
    "Describe the main characters and actions in the provided video, without mentioning the background.\n"
)


def find_image_extension(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file:
                return os.path.splitext(file)[1]
    return None


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array(
        [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return offsets


@dataclasses.dataclass
class MetaDataRGBClass:
    # string, but default is None
    shard: str
    metadata_dir: Optional[str] = None
    video_rgb_dir: Optional[str] = None


class SaveVQDataset(Dataset):
    def __init__(
        self,
        root: str,
        metadata_dir: str,
        video_rgb_dir: str,
        captions_dir: str,
        task: str,
        input_size: int = 336,
        task_transforms: dict = MODALITY_TRANSFORMS_DIVAE,
        resample_mode: str = "bilinear",
        corrupt_samples_log: Optional[str] = None,
        dryrun: bool = False,
        n_total_frames: int = 16,
    ):
        super().__init__()

        self.data_root = root
        self.metadata_root = os.path.join(
            root, metadata_dir
        )  # XXX: in general, paths are a bit hacky now. Clean up later.
        self.caption_root = os.path.join(root, captions_dir)
        print(f"Using:\nData: {self.data_root}")
        self.input_size = input_size
        self.task = task
        self.task_transforms = task_transforms
        self.resample_mode = resample_mode
        self.n_total_frames = n_total_frames

        self.dryrun = dryrun

        # FIXME
        self.classes = ["train"]
        self.class_to_idx = {"train": 0}

        self.all_samples = make_dataset(
            root, self.class_to_idx, ("tar",), None, allow_empty=True
        )
        # find samples with video_rgb_dir in them
        # video_rgb_idx = self.class_to_idx[video_rgb_dir]
        video_rgb_idx = 0
        # metadata_idx = self.class_to_idx[metadata_dir]
        self.samples = []
        # ensure both idcs have the same tarfiles
        # for sample in self.all_samples:
        #     print(sample)
        #     if sample[1] == metadata_idx:
        #         # TODO: should we call shards "shard-000.tar" or just "000.tar" (CURRENTLY INCONSISTENT)
        #         self.samples.append(
        #             MetaDataRGBClass(
        #                 metadata_dir=sample[0],
        #                 shard=sample[0].split("/")[-1].split("-")[-1],
        #             )
        #         )
        # for sample in self.all_samples:
        #     if sample[1] == video_rgb_idx:
        #         # check self.samples and see if there is a matching .tar
        #         shard = sample[0].split("/")[-1]
        #         for self_sample in self.samples:
        #             if self_sample.shard == shard:
        #                 self_sample.video_rgb_dir = sample[0]
        #                 break
        # for sample in self.samples:
        #     if sample.metadata_dir is None or sample.video_rgb_dir is None:
        #         print(f"Could not find matching metadata and video_rgb for {sample}")

        # only use samples from video_rgb_idx
        for sample in self.all_samples:
            print(sample)
            if sample[1] == video_rgb_idx:
                # TODO: should we call shards "shard-000.tar" or just "000.tar" (CURRENTLY INCONSISTENT)
                self.samples.append(
                    MetaDataRGBClass(
                        video_rgb_dir=sample[0],
                        shard=sample[0].split("/")[-1].split("-")[-1],
                    )
                )
        self.samples = [dataclasses.asdict(sample) for sample in self.samples]

    def loader(self, path, extensions):
        with tarfile.open(path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(
                    member.name.endswith(ext) for ext in extensions
                ):
                    video_base_name = member.name.rsplit(".", 1)[0]  # Extract base name

                    # Find the associated JSON file
                    for json_member in tar.getmembers():
                        if (
                            json_member.isfile()
                            and json_member.name == f"{video_base_name}.json"
                        ):
                            with tar.extractfile(json_member) as json_file_obj:
                                json_dict = json.load(json_file_obj)
                                break  # Found the match, stop looking
                    else:  # No matching JSON found
                        json_dict = None

                    with tar.extractfile(member) as file_obj:
                        yield BytesIO(file_obj.read()), json_dict

    def load_video(self, video_bytes, cuts):
        # TODO: check resolution
        transforms = Resize(size=self.input_size)

        # TODO: wok around decord?!
        video_bytes.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes.read())
            temp_file_name = temp_file.name

        cap = cv2.VideoCapture(temp_file_name)
        if not cap.isOpened():
            os.remove(temp_file_name)
            raise ValueError("Failed to open video stream from bytes")

        num_frames = cuts[-1][1]
        print(num_frames)
        # cuts is a list of lists, each containing 2 ints: start/end of video.
        clips = []
        for cut in cuts:
            # sample self.n_total_frames frames uniformly from the cut
            clip_indices = get_index(cut[1] - cut[0], self.n_total_frames) + cut[0]
            images_group = []
            for cli_idx in clip_indices:
                # TODO: get rid of h264 error, but have to iterate over frames...
                cap.set(cv2.CAP_PROP_POS_FRAMES, cli_idx)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                images_group.append(transforms(frame))
            clips.append(images_group)
        cap.release()
        return clips

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """Gets a single *shard* of videos and the corresponding shard path.

        Args:
            index (int): Index of the shard from the dataset to get.

        Raises:
            NotImplementedError: If a config value used previously for images is set.
            FileNotFoundError: If the crop settings file does not exist.

        Returns:
            list: List of videos with len(videos) == len(shard), each of shape:
                (n_frames, n_crops, c, h, w), where c=3, h=w=224 (=input_size; due to tokenizer)
            str: Path to the current shard of videos (tokenized/output).
        """

        paths = self.samples[index]
        # TODO: clean up this
        shard, metadata_dir, video_rgb_dir = (
            paths["shard"],
            paths["metadata_dir"],
            paths["video_rgb_dir"],
        )
        path = video_rgb_dir
        videos = self.loader(video_rgb_dir, VIDEO_EXTENSIONS)

        start = time.time()

        class_id, file_id = path.split("/")[-2:]
        file_id = file_id.split(".")[0]

        caption_path = os.path.join(self.caption_root, f"{file_id}.tar")
        print(caption_path)
        if not self.dryrun:
            os.makedirs(os.path.dirname(caption_path), exist_ok=True)

        all_video_frames = []
        # Perform augmentations and optionally mask images
        for video, cut_json in videos:
            if cut_json["status"] != "success":
                print("FAIL: ", cut_json)
                continue
            cuts = cut_json["cuts"]["cuts_original_fps"]
            full_video = self.load_video(video_bytes=video, cuts=cuts)
            all_video_frames.append(full_video)

        return all_video_frames, caption_path


def video_collate_fn(batch):
    videos, caption_path = zip(*batch)

    return videos, caption_path


def load_model_and_dataset(
    pretrained_model_name_or_path,
    num_frames,
    use_lora,
    lora_alpha,
    weight_dir,
):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(
        pretrained_model_name_or_path,
        num_frames=num_frames,
        use_lora=use_lora,
        lora_alpha=lora_alpha,
        weight_dir=weight_dir,
    )
    print("Done loading pllava!")
    model = model.eval()
    return model, processor


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Loading model")
    model, processor = load_model_and_dataset(
        pretrained_model_name_or_path=args.pllava_dir,
        num_frames=args.n_total_frames,
        use_lora=True,
        lora_alpha=4,  # TODO: double-check
        weight_dir=args.pllava_dir,
    )

    conv = conv_templates[args.conv_mode].copy()
    conv.user_query(QUERY_ACTION_BASE, None, None, is_mm=True)

    # TODO: model prompt - default, configurable, test

    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    print("start loading ds")
    dataset = SaveVQDataset(
        root=args.data_root,
        metadata_dir=args.metadata_dir,
        video_rgb_dir=args.video_rgb_dir,
        captions_dir=args.caption_dir,
        task="caption",  # TODO: check
        input_size=args.input_size,  # TODO: check default, 336 fine?
        resample_mode=args.resample_mode,
        corrupt_samples_log=args.corrupt_samples_log,
        n_total_frames=args.n_total_frames,
    )
    print("loaded dataset!")

    print(num_tasks, sampler_rank)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_dataloader,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=video_collate_fn,
    )

    model.to(device)

    time.sleep(1)

    print("Starting captioning")
    start_time = time.time()

    if global_rank == 0 and args.verbose:
        pbar = tqdm(total=len(data_loader))
    else:
        pbar = None

    for videos_batch, tar_paths in data_loader:
        # NOTE: videos_batch is a batch of video shards (tar files), each containing a list of videos (of multiple/variable-length frames)
        # processing files in this way ensures that shard structure is preserved.

        # Filter out already saved video shards
        videos_batch_filtered, tar_paths_filtered = [], []
        for imgs, tar_path in zip(videos_batch, tar_paths):
            if not os.path.exists(tar_path) or args.corrupt_samples_log is not None:
                videos_batch_filtered.append(imgs)
                tar_paths_filtered.append(tar_path)
        if len(videos_batch_filtered) == 0:
            if pbar is not None:
                pbar.update(1)
            continue
        videos_batch = videos_batch_filtered
        tar_paths = tar_paths_filtered
        print(f"Processing {len(videos_batch)} video shards.")
        print("Processing video shards: ", tar_paths)

        all_captions = []

        # TODO: batch-ify (but super cumbersome, PLLaVA does not provide this natively)
        for shard in videos_batch:
            shard_caption = []
            for video in tqdm(shard):
                video_caption = []
                for clip in video:
                    llm_response, _ = pllava_answer(
                        conv=conv,
                        model=model,
                        processor=processor,
                        do_sample=False,
                        img_list=clip,
                        max_new_tokens=256,
                        print_res=args.show_res,
                    )
                    conv = conv_templates[args.conv_mode].copy()
                    conv.user_query(query_action_base, None, None, is_mm=True)
                    video_caption.append(llm_response)
                shard_caption.append(video_caption)
            all_captions.append(shard_caption)
            # TODO: "In the image" --> too short scenes?
            # TODO: add len of clip/num_clip_frames to prompt?

        print(f"Tokenized video shards.")

        print("Saving tokenized video shards to disk.")
        # TODO: save, to jsonl? How? Add other info like clips?
        for shard_captions, tar_path in zip(all_captions, tar_paths):
            with tarfile.open(tar_path, "w") as tar:
                for i, video_captions in enumerate(shard_captions):
                    # TODO: prefix with shard-... or not?
                    # TODO: if we save like this here, does this even retain intra-tar order?
                    save_name = f"{i:05d}.json"
                    if args.dryrun:
                        print(
                            f"Dryrun: rank {global_rank} -> {video_captions}/{save_name}"
                        )
                    else:
                        # save to json
                        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                            f.write(json.dumps(video_captions))
                        tar.add(f.name, save_name)
                        os.remove(f.name)

        if pbar is not None:
            pbar.update(1)

    # torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Caption time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VQ token saver")

    parser.add_argument(
        "--pllava_dir",
        type=str,
        default="/store/swissai/a08/models/pllava-7b",
        help="Dir to PLLaVa model. Must be alreaded downloadd via python_scripts/hf.py (PLLaVa repo).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/store/swissai/a08/data/4m/splits/video_rgb/",  # FIXME
        help="Path to dataset root",
    )
    # FIXME: replace (temporarily) with whisperX
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="video_metadata",
        help="Relative path from metadata. From data_root/train.",
    )
    parser.add_argument(
        "--video_rgb_dir",
        type=str,
        default="video_rgb",
        help="Relative path from data_root. From data_root/train.",
    )
    parser.add_argument("--input_size", type=int, default=336, help="Image size")
    parser.add_argument("--task", type=str, default="caption", help="Task name")
    parser.add_argument(
        "--resample_mode",
        type=str,
        default=None,
        help="PIL resample mode for resizing loaded images. One out of ['bilinear', 'bicubic', 'nearest', None]. (default: %(default)s)",
    )
    parser.add_argument(
        "--corrupt_samples_log",
        type=str,
        default=None,
        help="Path to log file with corrupted samples from find_corrupted_pseudolabels.py. \
              If provided, only corrupted samples will be re-tokenized.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Set to enable progress bar",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=False,
        help="Set to do a dry run that creates the tokens and prints the paths without saving them to disk.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use for tokenization",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--caption_dir",
        type=str,
        default="video_caption",
        help="Suffix to add to the folder under which the tokens are saved.",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        default=False,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--batch_size_dataloader",
        default=1,
        type=int,
        help="Dataloader batch size (default: %(default)s) (how many video shards to load at once in the dataloader)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (default: %(default)s) (how many frames to tokenize at once)",
    )

    # Distributed parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--n_total_frames",
        default=16,
        type=int,
        help="How many frames to use in total, per clip.",
    )

    parser.add_argument(
        "--conv_mode",
        type=str,
        required=False,
        default="plain",
    )

    parser.add_argument(
        "--store_res",
        action="store_true",
        default=False,
        help="Print LLM prompt + response for every clip or not.",
    )

    args = parser.parse_args()
    main(args)