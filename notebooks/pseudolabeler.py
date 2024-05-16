import os
import logging
import torch
from webdataset import WebLoader
from video2dataset.dataloader import get_video_dataset

# Enable detailed error logging
os.environ["TORCHELASTIC_ERROR_FILE"] = "/tmp/torch_distributed_error.log"

# Set up logging
logging.basicConfig(filename='/tmp/script_error.log', level=logging.DEBUG)

# import os
# import sys
# import tempfile
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp

# from torch.nn.parallel import DistributedDataParallel as DDP

# # On Windows platform, the torch.distributed package only
# # supports Gloo backend, FileStore and TcpStore.
# # For FileStore, set init_method parameter in init_process_group
# # to a local file. Example as follow:
# # init_method="file:///f:/libtmp/some_file"
# # dist.init_process_group(
# #    "gloo",
# #    rank=rank,
# #    init_method=init_method,
# #    world_size=world_size)
# # For TcpStore, same way as on Linux.

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# WebVid validation split
SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/howto100m/0000000000.tar"

if __name__ == "__main__":
    try:
        # Disable distributed features
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        print("Distributed features disabled")
        decoder_kwargs = {
            "n_frames": 8,  # get 8 frames from each video
            "fps": 10,  # downsample to 10 FPS
            "num_threads": 12,  # use 12 threads to decode the video
        }
        resize_size = crop_size = 256
        batch_size = 32

        dset = get_video_dataset(
            urls=SHARDS,
            batch_size=batch_size,
            decoder_kwargs=decoder_kwargs,
            resize_size=resize_size,
            crop_size=crop_size,
        )
        print("Got dataset")

        num_workers = 6 # 6 dataloader workers

        dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
        print("Ran Webloader")

        for sample in dl:
            video_batch = sample["mp4"]
            print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])

            # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
            text_batch = sample["txt"]
            print(text_batch[0])
            metadata_batch = sample["json"]
    except Exception as e:
        print("oopsies FAILED")
        print(e)