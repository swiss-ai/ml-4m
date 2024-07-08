import os
import logging
import torch
from webdataset import WebLoader
from video2dataset.dataloader import get_video_dataset
from torch.utils.data.dataloader import default_collate
from ultralytics import YOLO

SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/hdvila/1000_hd_vila_shuffled/00000{00000..00032}.tar"

def generate_pseudolabels(video_batch, model):
    pseudolabels = []
    for video in video_batch:
        # Perform inference using the YOLO model on each frame of the video
        results = model(video)
        
        # Process the inference results and generate pseudolabels
        video_pseudolabels = []
        for result in results:
            boxes = result.boxes
            # ... process the bounding boxes and save pseudolabels ...
            video_pseudolabels.append(boxes)
        
        pseudolabels.append(video_pseudolabels)
    
    return pseudolabels

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
    
    num_workers = 2  # 6 dataloader workers
    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)
    
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    
    for sample in dl:
        print("in loop")
        video_batch = sample["mp4"]
        print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])
        
        # Generate pseudolabels using the YOLO model
        pseudolabels = generate_pseudolabels(video_batch, model)
        
        # TODO: Save the pseudolabels or perform further processing
        
        text_batch = sample["json"]
        print("done")