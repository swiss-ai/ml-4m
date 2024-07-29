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
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

import fourm.utils as utils
import fourm.utils.clip as clip
from fourm.data import CenterCropImageAugmenter, RandomCropImageAugmenter
from fourm.data.modality_info import MODALITY_TRANSFORMS_DIVAE
from fourm.vq.vqvae import DiVAE

from tasks.eval.model_utils import load_pllava

FEATURE_TASKS = ["CLIP-B16"]
VIDEO_EXTENSIONS = (".mp4",)


def find_image_extension(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file:
                return os.path.splitext(file)[1]
    return None


@dataclass
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
        print(f"Using:\nData: {self.data_root}")
        self.input_size = input_size
        self.task = task
        self.task_transforms = task_transforms
        self.resample_mode = resample_mode
        self.n_total_frames = n_total_frames

        self.dryrun = dryrun

        # XXX: not sure how 4m originally did this; may be unnecessary
        # pdb.set_trace()
        self.classes, self.class_to_idx = find_classes(root)

        self.all_samples = make_dataset(
            root, self.class_to_idx, ("tar",), None, allow_empty=True
        )
        # find samples with video_rgb_dir in them
        video_rgb_idx = self.class_to_idx[video_rgb_dir]
        metadata_idx = self.class_to_idx[metadata_dir]
        self.samples = []
        # ensure both idcs have the same tarfiles
        for sample in self.all_samples:
            if sample[1] == metadata_idx:
                # TODO: should we call shards "shard-000.tar" or just "000.tar" (CURRENTLY INCONSISTENT)
                self.samples.append(
                    MetaDataRGBClass(
                        metadata_dir=sample[0],
                        shard=sample[0].split("/")[-1].split("-")[-1],
                    )
                )
        for sample in self.all_samples:
            if sample[1] == video_rgb_idx:
                # check self.samples and see if there is a matching .tar
                shard = sample[0].split("/")[-1]
                for self_sample in self.samples:
                    if self_sample.shard == shard:
                        self_sample.video_rgb_dir = sample[0]
                        break
        for sample in self.samples:
            if sample.metadata_dir is None or sample.video_rgb_dir is None:
                print(f"Could not find matching metadata and video_rgb for {sample}")
    def loader(self, path, extensions):
        with tarfile.open(path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and any(
                    member.name.endswith(ext) for ext in extensions
                ):
                    file_obj = tar.extractfile(member)
                    yield BytesIO(file_obj.read())

    def _extract_frames(self, video_bytes):
        # ensure bytes stream is at beginning
        video_bytes.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes.read())
            temp_file_name = temp_file.name

        cap = cv2.VideoCapture(temp_file_name)
        # print(cap.get(5))  # gets fps if needed
        if not cap.isOpened():
            os.remove(temp_file_name)
            raise ValueError("Failed to open video stream from bytes")

        frame_count = 0

        try:
            while True:
                if frame_count % self.every_nth_frame == 0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    yield frame
                else:
                    ret = cap.grab()
                    if not ret:
                        break
                frame_count += 1
        finally:
            cap.release()
            os.remove(temp_file_name)

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
        # FIXME: remove
        print(index)

        rank = torch.distributed.get_rank()

        path, _ = self.samples[index]
        pdb.set_trace()
        # TODO: integrate clips from metadata, split video depending on this
        # TODO: adapt this to behave like load_video in pllava_caption.py
        # clips = split_on_clips()
        # should be a generator? returns 1 video of N clips per call?
        # selects 16 frames per clip, uniformly distributed (like in get_index in pllava_caption)
        videos = self.loader(path, VIDEO_EXTENSIONS)

        start = time.time()

        class_id, file_id = path.split("/")[-2:]
        file_id = file_id.split(".")[0]

        # FIXME
        tokens_path = os.path.join(self.tokens_root, f"{file_id}.tar")
        print(tokens_path)
        if not self.dryrun:
            os.makedirs(os.path.dirname(tokens_path), exist_ok=True)


        all_video_frames = []
        # Perform augmentations and optionally mask images
        v_idx = 0
        for video in videos:
            # TODO: do this also here or previously, as part of generator? --> check memory!
            video_frames = self._extract_frames(video)

            # Create or load crop settings
            # TODO: remove cropping/augmentation
            if os.path.exists(crop_settings_path) and not self.force_new_crop:
                try:
                    settings = np.load(crop_settings_path)
                except:
                    raise FileNotFoundError
            else:
                # since we do this here, a video is cropped the same way for all its frames
                settings = []

                # First crop is always non-flipped center crop
                crop_coords, _, _, _, _ = self.center_crop_augmenter(
                    {self.task: next(video_frames)}, None
                )
                settings.append((*crop_coords, 0))

                # Subsequent crops are random
                for _ in range(1, self.n_crops):
                    # TODO: replace next(...)
                    crop_coords, h_flip, _, _, _ = self.random_crop_augmenter(
                        {self.task: next(video_frames)}, None
                    )
                    settings.append((*crop_coords, 1 if h_flip else 0))

                settings = np.array(settings)
                if not self.dryrun:
                    os.makedirs(os.path.dirname(crop_settings_path), exist_ok=True)
                    np.save(crop_settings_path, settings)
            current_video = []
            for frame in video_frames:
                current_frame = []
                for i, j, h, w, h_flip in settings:
                    img_mod = self.task_transforms[self.task].preprocess(frame.copy())
                    img_mod = self.task_transforms[self.task].image_augment(
                        img_mod,
                        (i, j, h, w),
                        h_flip,
                        None,
                        (self.input_size, self.input_size),
                        None,
                        self.resample_mode,
                    )
                    img_mod = self.task_transforms[self.task].postprocess(img_mod)
                    # FIXME: can we do to fp16?
                    img_mod = img_mod.half()
                    current_frame.append(img_mod)

                    if self.mask_value is not None:
                        raise NotImplementedError

                current_video.append(torch.stack(current_frame))
            all_video_frames.append(torch.stack(current_video))
            v_idx += 1

        end = time.time()
        print(f"IDX {index}: Extracting/Augmenting took {end - start} seconds.")
        return all_video_frames, tokens_path


def video_collate_fn(batch):
    videos, tokens_paths = zip(*batch)

    return videos, tokens_paths


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
    print("done loading llava")
    #  position embedding
    model = model.eval()
    return model, processor


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("loading model")
    model, processor = load_model_and_dataset(
        pretrained_model_name_or_path=args.pllava_dir,
        num_frames=args.n_total_frames,
        use_lora=True,
        lora_alpha=4,  # TODO: double-check
        weight_dir=args.pllava_dir,
    )
    
    # TODO: model prompt - default, configurable, test

    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    print("start loading ds")
    dataset = SaveVQDataset(
        root=os.path.join(args.data_root, args.split),
        metadata_dir=args.metadata_dir,
        video_rgb_dir=args.video_rgb_dir,
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

    for videos_batch, tokens_paths in data_loader:
        # NOTE: videos_batch is a batch of video shards (tar files), each containing a list of videos (of multiple/variable-length frames)
        # processing files in this way ensures that shard structure is preserved.

        # Filter out already saved video shards
        videos_batch_filtered, tokens_paths_filtered = [], []
        for imgs, tokens_path in zip(videos_batch, tokens_paths):
            if not os.path.exists(tokens_path) or args.corrupt_samples_log is not None:
                videos_batch_filtered.append(imgs)
                tokens_paths_filtered.append(tokens_path)
        if len(videos_batch_filtered) == 0:
            if pbar is not None:
                pbar.update(1)
            continue
        videos_batch = videos_batch_filtered
        tokens_paths = tokens_paths_filtered
        print(f"Processing {len(videos_batch)} video shards.")
        print("Processing video shards: ", tokens_paths)

        # TODO: probably can remove this
        num_frames = []
        for shard in videos_batch:
            num_frames.append([len(video) for video in shard])

        # TODO: probably also remove this
        if "semseg" in args.task:
            raise NotImplementedError
            # Merge batch and number of augmentation dimensions
            videos_batch = rearrange(videos_batch, "b n h w -> (b n) h w")
        else:
            video_batch = []
            # unroll videos
            for shard in videos_batch:
                for video in shard:
                    video_batch.extend(video)
            videos_batch_merged = torch.stack(video_batch)
        # merge batch and augmentation dimensions
        # thus, dim 0 contains: all frames from all videos from all shards in the batch
        videos_batch_merged = rearrange(videos_batch_merged, "b n c h w -> (b n) c h w")
        # For efficiency, process images with batch size that might be different from loader batch size or num augmentations
        sub_batches = videos_batch_merged.split(args.batch_size, dim=0)

        all_tokens = []

        # TODO: change; batch should be shards x videos x clips
        for sub_batch in tqdm(
            sub_batches, "Tokenizing batch", disable=not args.verbose
        ):
            # back to fp32
            # TODO: can we also store imgs in float16 here?
            sub_batch = sub_batch.float()
            sub_batch = sub_batch.to(device)

            with torch.no_grad():
                if "CLIP" in args.task:
                    raise NotImplementedError

                # TODO: proper model call
                tokens = model.tokenize(sub_batch)
                tokens = rearrange(tokens, "b h w -> b (h w)")

            tokens = tokens.detach().cpu().numpy().astype(np.int16)
            all_tokens.append(tokens)

        all_tokens = np.concatenate(all_tokens)
        all_tokens = rearrange(all_tokens, "(b n) d -> b n d", n=args.n_crops)
        print(f"Tokenized {all_tokens.shape[0]} images.")

        # Split tokens back to num_frames
        # TODO
        tokens = []
        for i, shard_num_frames in enumerate(num_frames):
            shard_tokens = []
            for j, num_frame in enumerate(shard_num_frames):
                shard_tokens.append(all_tokens[:num_frame])
                all_tokens = all_tokens[num_frame:]
                if num_frame != len(videos_batch[i][j]):
                    raise ValueError(
                        f"Expected num_frame to be {len(videos_batch[i][j])}, but got {num_frame}."
                    )
            if len(shard_tokens) != len(videos_batch[i]):
                raise ValueError(
                    f"Expected shard_tokens length to be {len(videos_batch[i])}, but got {len(shard_tokens)}."
                )
            tokens.append(shard_tokens)

        if len(tokens) != len(videos_batch):
            raise ValueError(
                f"Expected tokens length to be {len(videos_batch)}, but got {len(tokens)}."
            )

        if [len(video) for video in tokens] != [len(video) for video in videos_batch]:
            raise ValueError(
                f"Expected tokens structure {[len(video) for video in videos_batch]}, but got {[len(video) for video in tokens]}."
            )
        # TODO
        for shard_tokens, tokens_path in zip(tokens, tokens_paths):
            with tarfile.open(tokens_path, "w") as tar:
                for i, video_tokens in enumerate(shard_tokens):
                    save_name = f"{i:05d}.npy"
                    if args.dryrun:
                        print(
                            f"Dryrun: rank {global_rank} -> {tokens_path}/{save_name}"
                        )
                    else:
                        with BytesIO() as bio:
                            np.save(bio, video_tokens)
                            bio.seek(0)
                            tarinfo = tarfile.TarInfo(save_name)
                            tarinfo.size = len(bio.getvalue())
                            tar.addfile(tarinfo, bio)

        if pbar is not None:
            pbar.update(1)

    # torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Tokenization time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VQ token saver")

    parser.add_argument(
        "--pllava_dir",
        type=str,
        default="/cluster/work/cotterell/mfrohmann/PLLaVA/MODELS/pllava-7b",
        help="Dir to PLLaVa model. Must be alreaded downloadd via python_scripts/hf.py (PLLaVa repo).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/cluster/work/cotterell/mm_swissai/raw/v2d_500/howto100m",
        help="Path to dataset root",
    )
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
    parser.add_argument("--split", type=str, default="train", help="train or val")
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
    parser.add_argument("--num_workers", default=6, type=int)
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
        default=10,
        type=int,
        help="How many frames to use in total, per clip.",
    )

    args = parser.parse_args()
    main(args)
