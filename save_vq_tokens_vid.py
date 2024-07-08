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

FEATURE_TASKS = ["CLIP-B16"]
VIDEO_EXTENSIONS = (".mp4",)


def find_image_extension(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file:
                return os.path.splitext(file)[1]
    return None


class SaveVQDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokens_dir: str,
        crop_settings_dir: str,
        task: str,
        n_crops: int = 10,
        min_crop_scale: float = 0.2,
        input_size: int = 224,
        mask_value: Optional[float] = None,
        task_transforms: dict = MODALITY_TRANSFORMS_DIVAE,
        resample_mode: str = "bilinear",
        corrupt_samples_log: Optional[str] = None,
        dryrun: bool = False,
        force_load_crop: bool = False,
    ):
        super().__init__()

        self.data_root = root
        self.tokens_root = os.path.join(
            root, tokens_dir
        )  # XXX: in general, paths are a bit hacky now. Clean up later.
        self.crop_settings_root = os.path.join(
            os.path.abspath(__file__ + "/../"), crop_settings_dir
        )
        self.n_crops = n_crops
        self.input_size = input_size
        self.task = task
        self.mask_value = mask_value
        self.task_transforms = task_transforms
        self.resample_mode = resample_mode

        self.force_load_crop = force_load_crop

        self.dryrun = dryrun
        self.force_load_crop = force_load_crop

        # XXX: not sure how 4m originally did this; may be unnecessary
        self.classes, self.class_to_idx = find_classes(root)

        self.samples = make_dataset(
            root, self.class_to_idx, ("tar",), None, allow_empty=True
        )

        # TODO: check augmentations (implemented but unchecked)
        self.center_crop_augmenter = CenterCropImageAugmenter(
            target_size=self.input_size, hflip=0.0, main_domain=task
        )
        self.random_crop_augmenter = RandomCropImageAugmenter(
            target_size=self.input_size,
            hflip=0.5,
            crop_scale=(min_crop_scale, 1.0),
            crop_ratio=(0.75, 1.3333),
            main_domain=task,
        )

    def loader(self, path, extensions):
        video_files = []
        with tarfile.open(path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(extensions):
                    file_obj = tar.extractfile(member)
                    video_files.append(BytesIO(file_obj.read()))
        return video_files

    def _extract_frames(self, video_bytes) -> list:
        video_bytes.seek(0)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(video_bytes.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise ValueError("Failed to open video stream from bytes")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # FIXME: do we need to convert to RGB here?
            frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()

        os.remove(temp_file.name)

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        videos = self.loader(path, VIDEO_EXTENSIONS)

        video_frames = [self._extract_frames(video) for video in videos]

        class_id, file_id = path.split("/")[-2:]
        file_id = file_id.split(".")[0]

        if self.mask_value is not None:
            raise NotImplementedError
            # FIXME: what to do with the masking stuff?
            mask_path = os.path.join(
                self.data_root, "mask_valid", class_id, f"{file_id}.png"
            )
            mask = Image.open(mask_path)

        tokens_path = os.path.join(self.tokens_root, f"{file_id}.tar")
        if not self.dryrun:
            os.makedirs(os.path.dirname(tokens_path), exist_ok=True)

        crop_settings_path = os.path.join(
            self.crop_settings_root, class_id, f"{file_id}.npy"
        )

        # Perform augmentations and optionally mask images
        videos = []
        for video in video_frames:
            # Create or load crop settings
            if os.path.exists(crop_settings_path) or self.force_load_crop:
                try:
                    settings = np.load(crop_settings_path)
                except:
                    raise FileNotFoundError
            else:
                # since we do this here, a video is cropped the same way for all its frames
                settings = []

                # First crop is always non-flipped center crop
                crop_coords, _, _, _, _ = self.center_crop_augmenter(
                    {self.task: video[0]}, None
                )
                settings.append((*crop_coords, 0))

                # Subsequent crops are random
                for _ in range(1, self.n_crops):
                    crop_coords, h_flip, _, _, _ = self.random_crop_augmenter(
                        {self.task: video[0]}, None
                    )
                    settings.append((*crop_coords, 1 if h_flip else 0))

                settings = np.array(settings)
                if not self.dryrun:
                    os.makedirs(os.path.dirname(crop_settings_path), exist_ok=True)
                    np.save(crop_settings_path, settings)
            current_video = []
            for frame in video:
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
                    current_frame.append(img_mod)

                    if self.mask_value is not None:
                        raise NotImplementedError
                    #     mask_valid = self.task_transforms["mask_valid"].preprocess(mask.copy())
                    #     mask_valid = self.task_transforms["mask_valid"].image_augment(
                    #         mask_valid,
                    #         (i, j, h, w),
                    #         h_flip,
                    #         None,
                    #         (self.input_size, self.input_size),
                    #         None,
                    #         None,
                    #     )
                    #     mask_valid = self.task_transforms["mask_valid"].postprocess(mask_valid)
                    #     img_mod[~repeat(mask_valid, "1 h w -> c h w", c=img_mod.shape[0])] = (
                    #         self.mask_value
                    #     )
                    #     mask_valid = (
                    #         mask_valid.float() * 2 - 1
                    #     )  # Valid regions -> 1, Masked-out regions -> -1
                    #     img_mod = torch.cat(
                    #         [img_mod, mask_valid], dim=0
                    #     )  # Concat image with mask

                current_video.append(torch.stack(current_frame))
            videos.append(torch.stack(current_video))

        return videos, tokens_path


def get_feature_extractor(args):
    if args.task == "CLIP-B16":
        raise NotImplementedError
        teacher_model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        teacher_model = teacher_model.visual
        return teacher_model.eval()
    else:
        return None


def video_collate_fn(batch):
    videos, tokens_paths = zip(*batch)

    return videos, tokens_paths


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = DiVAE.from_pretrained(args.hf_model_id)

    feature_extractor = get_feature_extractor(args)

    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    loader_task = "rgb" if args.task in FEATURE_TASKS else args.task
    dataset = SaveVQDataset(
        root=os.path.join(args.data_root, args.split),
        crop_settings_dir="crop_settings",
        tokens_dir=args.tokens_dir,
        task=loader_task,
        min_crop_scale=args.min_crop_scale,
        n_crops=args.n_crops,
        input_size=args.input_size,
        mask_value=args.mask_value,
        resample_mode=args.resample_mode,
        corrupt_samples_log=args.corrupt_samples_log,
        force_load_crop=args.force_load_crop,
    )

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
    if feature_extractor is not None:
        feature_extractor.to(device)

    print("Starting tokenization")
    start_time = time.time()

    if global_rank == 0 and args.verbose and not args.dryrun:
        pbar = tqdm(total=len(data_loader))
    else:
        pbar = None

    for imgs_batch, tokens_paths in data_loader:
        # Filter out already saved images
        imgs_batch_filtered, tokens_paths_filtered = [], []
        for imgs, tokens_path in zip(imgs_batch, tokens_paths):
            # FIXME: this excludes already saved shards; do we want this?
            if not os.path.exists(tokens_path) or args.corrupt_samples_log is not None:
                imgs_batch_filtered.append(imgs)
                tokens_paths_filtered.append(tokens_path)
        if len(imgs_batch_filtered) == 0:
            if pbar is not None:
                pbar.update(1)
            continue
        # imgs_batch = torch.stack(imgs_batch_filtered)
        imgs_batch = imgs_batch_filtered
        tokens_paths = tokens_paths_filtered

        num_frames = []
        for shard in imgs_batch:
            num_frames.append([len(video) for video in shard])

        # Merge batch and number of augmentation dimensions
        if "semseg" in args.task:
            raise NotImplementedError
            imgs_batch = rearrange(imgs_batch, "b n h w -> (b n) h w")
        else:
            video_batch = []
            # unroll videos
            for shard in imgs_batch:
                for video in shard:
                    video_batch.extend(video)
            imgs_batch_merged = torch.stack(video_batch)
        # merge batch and augmentation dimensions
        imgs_batch_merged = rearrange(imgs_batch_merged, "b n c h w -> (b n) c h w")
        # For efficiency, process images with batch size that might be different from loader batch size or num augmentations
        sub_batches = imgs_batch_merged.split(args.batch_size, dim=0)

        all_tokens = []

        for sub_batch in tqdm(
            sub_batches, "Tokenizing batch", disable=not args.verbose
        ):
            sub_batch = sub_batch.to(device)

            with torch.no_grad():
                if "CLIP" in args.task:
                    raise NotImplementedError
                    B, C, H, W = sub_batch.shape
                    P_H, P_W = feature_extractor.conv1.kernel_size
                    N_H, N_W = H // P_H, W // P_W
                    sub_batch = feature_extractor(
                        sub_batch, return_final_tokens_no_cls=True
                    )
                    sub_batch = rearrange(
                        sub_batch, "b (nh nw) d -> b d nh nw", nh=N_H, nw=N_W
                    )

                tokens = model.tokenize(sub_batch)
                tokens = rearrange(tokens, "b h w -> b (h w)")

            tokens = tokens.detach().cpu().numpy().astype(np.int16)
            all_tokens.append(tokens)

        all_tokens = np.concatenate(all_tokens)
        all_tokens = rearrange(all_tokens, "(b n) d -> b n d", n=args.n_crops)
        # TODO: why is loading (__getiem__) so slow?

        # Split tokens back to num_frames
        tokens = []
        all_tokens_cp = (
            all_tokens.copy()
        )  # FIXME: remove this copy when we have seen this works.
        for i, shard_num_frames in enumerate(num_frames):
            shard_tokens = []
            for j, num_frame in enumerate(shard_num_frames):
                shard_tokens.append(all_tokens_cp[:num_frame])
                all_tokens_cp = all_tokens_cp[num_frame:]
                assert num_frame == len(imgs_batch[i][j])
            assert len(shard_tokens) == len(imgs_batch[i])
            tokens.append(shard_tokens)

        # FIXME: asserts can be removed after testing
        assert len(tokens) == len(imgs_batch)
        assert [len(video) for video in tokens] == [len(video) for video in imgs_batch]

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
                            # TODO: is this efficient enough?
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
        "--hf_model_id",
        type=str,
        default="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        help="ID of tokenizer to load.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/store/swissai/a08/data/4m-data",
        help="Path to video_rgb dataset root",
    )
    parser.add_argument("--split", type=str, default="train", help="train or val")
    parser.add_argument(
        "--n_crops",
        type=int,
        default="1",
        help="Number of crops to save. If 1, only a center crop will be saved. \
             If > 1, first image will be center cropped, the subsequent ones will be randomly cropped.",
    )
    parser.add_argument(
        "--min_crop_scale",
        type=float,
        default=0.8,
        help="Minimum crop scale (Only for n_crops > 1)",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Image size")
    parser.add_argument("--task", type=str, default="rgb", help="Task name")
    parser.add_argument(
        "--mask_value",
        type=float,
        default=None,
        help="Optionally set masked-out regions to this value after data augs (default: %(default)s)",
    )
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
        default="cpu",
        help="Device to use for tokenization",  # FIXME: use cuda by default (disabled due to debugging @todi)
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--tokens_dir",
        type=str,
        default="video_rgb_tok",
        help="Suffix to add to the folder under which the tokens are saved.",
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--batch_size_dataloader",
        default=64,
        type=int,
        help="Dataloader batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (default: %(default)s)",
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
        "--force_load_crop",
        action="store_true",
        help="Make sure to load crops locally, otherwise break the code.",
    )

    args = parser.parse_args()
    print("Force loading existing crop settings: {}".format(args.force_load_crop))
    main(args)
