import tarfile
import os
import cv2
from ultralytics import YOLO

SHARDS = "/cluster/work/cotterell/mm_swissai/datasets/hdvila/1000_hd_vila_shuffled/0000000000.tar"
OUTPUT_DIR = "bbox-yolo/extracted_frames"
LABELED_OUTPUT_DIR = "bbox-yolo/labeled_frames"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELED_OUTPUT_DIR, exist_ok=True)

# Load the YOLO model
model = YOLO('bbox-yolo/yolov8n.pt')  # pretrained YOLOv8n model

# Extract the tar file
with tarfile.open(SHARDS, "r") as tar:
    tar.extractall(path="bbox-yolo/extracted_files")

# Iterate through extracted files
for root, dirs, files in os.walk("bbox-yolo/extracted_files"):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            video = cv2.VideoCapture(video_path)
            
            frame_paths = []
            for frame_count in range(3):  # Only process the first 3 frames to test on euler
                success, frame = video.read()
                if not success:
                    break
                
                # Save the frame as an image
                frame_path = os.path.join(OUTPUT_DIR, f"{file[:-4]}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            video.release()
            
            # Apply pseudolabeling to the extracted frames
            results = model(frame_paths, project=LABELED_OUTPUT_DIR, name=file[:-4])
            
            for i, result in enumerate(results):
                result.save(filename=f'{file[:-4]}_labeled_frame_{i}.jpg')

print("Frame extraction and pseudolabeling complete.")
