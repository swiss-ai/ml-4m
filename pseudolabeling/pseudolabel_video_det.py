import tarfile
import os
import cv2
from ultralytics import YOLO
import argparse
import json
import jsonlines
from pathlib import Path
import tempfile


# Set up argument parser
parser = argparse.ArgumentParser(description='Process video frames with YOLO')
parser.add_argument('--source_dir', type=str, required=True, help='Path to the source dir containing tar files of video shards')
parser.add_argument('--yolo_path', type=str, default="/store/swissai/a08/pseudolabelers/yolov8n.pt", help='Path to the YOLO model')
parser.add_argument('--nth_frame', type=int, default=30, help='Select every nth frame (default: 30)')
parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (default: None, process all)')
parser.add_argument('--save_frames', type=bool, default=False, help='Whether to save frames')
args = parser.parse_args()

SOURCE_DIR = args.source_dir
NTH_FRAME = args.nth_frame
MAX_FRAMES = args.max_frames
SAVE_FRAMES = args.save_frames
JSON_OUTPUT_DIR = Path(SOURCE_DIR).parent.absolute() / "video_det/"

# Ensure output directories exist
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# Load the YOLO model
model = YOLO(args.yolo_path) # pretrained YOLOv8n model

for tfile in sorted(os.listdir(SOURCE_DIR)):
    print(tfile)
    if tfile.endswith(".tar"):
        # Get the shard number from the input file name and create the output tar file
        shard_number = os.path.splitext(os.path.basename(tfile))[0]
        output_tar_path = os.path.join(JSON_OUTPUT_DIR, f"{shard_number}.tar")

        # Extract the tar file
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            output_dir = os.path.join(tmpdirname, "extracted_frames")
            labeled_output_dir = os.path.join(tmpdirname, "labeled_frames")

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(labeled_output_dir, exist_ok=True)

            with tarfile.open(os.path.join(SOURCE_DIR, tfile), "r") as tar:
                tar.extractall(path=tmpdirname, numeric_owner=True)



            with tarfile.open(output_tar_path, "w") as output_tar:
                # Iterate through extracted files
                for root, dirs, files in sorted(os.walk(tmpdirname)):
                    for file in files:
                        if file.endswith(".mp4"):
                            video_path = os.path.join(root, file)
                            video = cv2.VideoCapture(video_path)
                            frame_paths = []
                            frame_count = 0
                            processed_frames = 0
                            json_data_list = []

                            while True:
                                success, frame = video.read()
                                if not success:
                                    break
                                if frame_count % NTH_FRAME == 0:
                                    # Save the frame as an image
                                    frame_path = os.path.join(output_dir, f"{file[:-4]}_frame_{frame_count}.jpg")
                                    cv2.imwrite(frame_path, frame)
                                    frame_paths.append(frame_path)
                                    processed_frames += 1
                                frame_count += 1
                                if MAX_FRAMES and processed_frames >= MAX_FRAMES:
                                    break
                            video.release()

                            # Apply pseudolabeling to the extracted frames
                            import pdb; pdb.set_trace()
                            results = model(frame_paths, project=labeled_output_dir, name=file[:-4])
                            
                            for i, result in enumerate(results):
                                # Save labeled image
                                if SAVE_FRAMES:
                                    result.save(filename=f'{file[:-4]}_labeled_frame_{i}.jpg')
                                
                                # Extract bounding box information
                                boxes = result.boxes
                                frame_data = []
                                for box in boxes:
                                    xyxy = box.xyxy[0].tolist()  # get box coordinates
                                    conf = box.conf.item()  # get confidence score
                                    cls = int(box.cls.item())  # get class id
                                    frame_data.append({
                                        "bbox": xyxy,
                                        "confidence": conf,
                                        "class": cls,
                                        "class_name": result.names[cls]
                                    })
                                json_data_list.append(frame_data)

                            # Save JSONL file
                            jsonl_filename = f"{file[:-4]}.jsonl"
                            jsonl_path = os.path.join(JSON_OUTPUT_DIR, jsonl_filename)
                            with jsonlines.open(jsonl_path, mode='w') as writer:
                                writer.write_all(json_data_list)

                            # Add JSONL file to the tar archive
                            output_tar.add(jsonl_path, arcname=jsonl_filename)

                            # Remove the temporary JSONL file
                            os.remove(jsonl_path)

print("Frame extraction, pseudolabeling, and JSONL export complete.")
