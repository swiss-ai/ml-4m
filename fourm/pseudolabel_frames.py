import tarfile
import os
import cv2
from ultralytics import YOLO
import argparse
import json

# Set up argument parser
parser = argparse.ArgumentParser(description='Process video frames with YOLO')
parser.add_argument('--shards', type=str, required=True, help='Path to the tar file containing video shards')
parser.add_argument('--nth_frame', type=int, default=30, help='Select every nth frame (default: 30)')
parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (default: None, process all)')
args = parser.parse_args()

SHARDS = args.shards
NTH_FRAME = args.nth_frame
MAX_FRAMES = args.max_frames
OUTPUT_DIR = "extracted_frames"
LABELED_OUTPUT_DIR = "labeled_frames"
JSON_OUTPUT_DIR = "json_output"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELED_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# Load the YOLO model
model = YOLO('/cluster/work/cotterell/yemara/ml-4m/bbox-yolo/yolov8n.pt') # pretrained YOLOv8n model

# Extract the tar file
with tarfile.open(SHARDS, "r") as tar:
    tar.extractall(path="extracted_files")

# Iterate through extracted files
for root, dirs, files in os.walk("extracted_files"):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            video = cv2.VideoCapture(video_path)
            frame_paths = []
            frame_count = 0
            processed_frames = 0

            while True:
                success, frame = video.read()
                if not success:
                    break
                
                if frame_count % NTH_FRAME == 0:
                    # Save the frame as an image
                    frame_path = os.path.join(OUTPUT_DIR, f"{file[:-4]}_frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    processed_frames += 1
                
                frame_count += 1

                if MAX_FRAMES and processed_frames >= MAX_FRAMES:
                    break

            video.release()
            
            # Apply pseudolabeling to the extracted frames
            results = model(frame_paths, project=LABELED_OUTPUT_DIR, name=file[:-4])
            
            for i, result in enumerate(results):
                # Save labeled image
                result.save(filename=f'{file[:-4]}_labeled_frame_{i}.jpg')
                
                # Extract bounding box information
                boxes = result.boxes
                json_data = []
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()  # get box coordinates
                    conf = box.conf.item()  # get confidence score
                    cls = int(box.cls.item())  # get class id
                    json_data.append({
                        "bbox": xyxy,
                        "confidence": conf,
                        "class": cls,
                        "class_name": result.names[cls]
                    })
                
                # Save JSON file
                json_filename = os.path.join(JSON_OUTPUT_DIR, f"{file[:-4]}_frame_{i}_boxes.json")
                with open(json_filename, 'w') as f:
                    json.dump(json_data, f, indent=4)

print("Frame extraction, pseudolabeling, and JSON export complete.")
