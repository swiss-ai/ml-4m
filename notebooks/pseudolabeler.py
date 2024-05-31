import os
import logging
import torch
from webdataset import WebLoader
from video2dataset.dataloader import get_video_dataset

# https://stackoverflow.com/questions/78537817/importerror-cannot-import-name-dill-available if running with pytorch 2.3.0 need to use the fix here
# might also get `undefined symbol when importing torchaudio with pytorch` when running with torch 2.3.0
# WebVid validation split
# SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/howto100m/0000000000.tar"
# SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/hdvila/1000_hd_vila_shuffled/0000000000.tar"
SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/hdvila/1000_hd_vila_shuffled/00000{00000..00032}.tar"

if __name__ == "__main__":
    decoder_kwargs = {
        "n_frames": 8,  # get 8 frames from each video
        "fps": 10,  # downsample to 10 FPS
        "num_threads": 12,  # use 12 threads to decode the video
    }
    resize_size = crop_size = 256
    batch_size = 10

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
        enforce_additional_keys=[],
    )

    num_workers = 6  # 6 dataloader workers

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
    # TODO: figure out how to make the dataloader have batch size >1 (current error: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>
    # dl = WebLoader(dset, batch_size=None, num_workers=num_workers) )

    for sample in dl:
        print("in loop")
        video_batch = sample["mp4"]
        print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])

        # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
        text_batch = sample["json"]
        # print(text_batch[0])
    print("done")
