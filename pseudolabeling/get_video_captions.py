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
import json
import os
import random
import tarfile
import tempfile
import time
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_pllava, pllava_answer
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import Resize
from tqdm import tqdm

import fourm.utils as utils
from fourm.data.modality_info import MODALITY_TRANSFORMS_DIVAE

VIDEO_EXTENSIONS = (".mp4",)

# TODO: "In the image" --> too short scenes?
# TODO: add len of clip/num_clip_frames to prompt?
QUERY_ACTION_BASE = (
    "You are to assist me in accomplishing a task about the input video."
    "# Task\n"
    "Describe the main characters and actions in the provided video, without mentioning the background.\n"
)


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array(
        [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return offsets


class SaveCaptionDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        captions_dir: str,
        input_size: int = 336,
        dryrun: bool = False,
        n_total_frames: int = 16,
    ):
        super().__init__()

        self.data_root = root

        self.caption_root = os.path.join(root, captions_dir).replace(
            "filtered_raw", "4m"
        )
        print(f"Using:\nData: {self.data_root}\nCaptions: {self.caption_root}")
        self.input_size = input_size
        self.n_total_frames = n_total_frames

        self.dryrun = dryrun

        self.classes = [dataset_name]
        self.class_to_idx = {dataset_name: 0}

        self.samples = make_dataset(
            root, self.class_to_idx, ("tar",), None, allow_empty=True
        )

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
                        yield BytesIO(file_obj.read()), json_dict, json_member.name

    def load_video(self, video_bytes, cuts):
        # TODO: check resolution
        transforms = Resize(size=self.input_size)

        video_bytes.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes.read())
            temp_file_name = temp_file.name

        cap = cv2.VideoCapture(temp_file_name)
        if not cap.isOpened():
            os.remove(temp_file_name)
            raise ValueError("Failed to open video stream from bytes")

        # num_frames = cuts[-1][1]

        # cuts is a list of lists, each containing 2 ints: start/end of video.
        clips = []
        for cut in cuts:
            # sample self.n_total_frames frames uniformly from the cut
            clip_indices = get_index(cut[1] - cut[0], self.n_total_frames) + cut[0]
            images_group = []

            # start only from the beginning of the cut
            frame_count = cut[0]

            while True:
                # Check if frame should be extracted
                if frame_count in clip_indices:
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = transforms(frame)
                    images_group.append(frame)
                else:
                    # Skip frames to achieve the desired FPS
                    cap.grab()  # Move to the next frame without decoding
                frame_count += 1
                # Check if we have reached the end of the cut
                if frame_count > clip_indices[-1]:
                    break

            clips.append(images_group)
        cap.release()
        os.remove(temp_file_name)
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

        path, _ = self.samples[index]
        videos = self.loader(path, VIDEO_EXTENSIONS)

        _, file_id = path.split("/")[-2:]
        file_id = file_id.split(".")[0]

        caption_path = os.path.join(self.caption_root, f"{file_id}.tar")

        if not self.dryrun:
            os.makedirs(os.path.dirname(caption_path), exist_ok=True)

        all_video_frames = []
        all_json_names = []
        # Perform augmentations and optionally mask images
        for video, cut_json, json_name in videos:
            if cut_json["status"] != "success":
                print("FAIL: ", cut_json)
                continue
            cuts = cut_json["cuts"]["cuts_original_fps"]
            full_video = self.load_video(video_bytes=video, cuts=cuts)
            all_video_frames.append(full_video)
            all_json_names.append(json_name)

        out_dict = {"video_frames": all_video_frames, "json_names": all_json_names}
        return out_dict, caption_path


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
    dataset = SaveCaptionDataset(
        root=args.data_root,
        dataset_name=args.dataset_name,
        captions_dir=args.caption_dir,
        input_size=args.input_size,  # TODO: check default, 336 fine?
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
            if not os.path.exists(tar_path):
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
            for video_content, video_path in zip(
                shard["video_frames"], shard["json_names"]
            ):
                # video_content = video["video_frames"]
                # video_path = video["json_names"]
                video_caption = []
                for clip in video_content:
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
                    conv.user_query(QUERY_ACTION_BASE, None, None, is_mm=True)
                    video_caption.append(llm_response)
                shard_caption.append(
                    {"video_path": video_path, "captions": video_caption}
                )
            all_captions.append(shard_caption)

        print(f"Tokenized video shards.")

        print("Saving tokenized video shards to disk.")

        for shard_captions, tar_path in zip(all_captions, tar_paths):
            with tarfile.open(tar_path, "w") as tar:
                for video_captions in shard_captions:
                    save_name = video_captions["video_path"]
                    if args.dryrun:
                        print(
                            f"Dryrun: rank {global_rank} -> {video_captions}/{save_name}"
                        )
                    else:
                        # save to json
                        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                            f.write(json.dumps(video_captions["captions"], indent=4))
                        tar.add(f.name, save_name)
                        os.remove(f.name)
                    # print(f"Saved {save_name} to {tar_path}")

        if pbar is not None:
            pbar.update(1)

    # torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Caption time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Video clip caption saver")

    parser.add_argument(
        "--pllava_dir",
        type=str,
        default="/store/swissai/a08/models/pllava-7b",
        help="Dir to PLLaVa model. Must be alreaded downloadd via python_scripts/hf.py (PLLaVa repo).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/store/swissai/a08/data/filtered_raw/howto100m",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="v2d_5000",
        help="Upmost directory of dataset. So dataset is in data_root/dataset_name.",
    )
    parser.add_argument("--input_size", type=int, default=336, help="Image size")

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
        "--show_res",
        action="store_true",
        default=False,
        help="Print LLM prompt + response for every clip or not.",
    )

    args = parser.parse_args()
    main(args)
