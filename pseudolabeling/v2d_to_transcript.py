import argparse
import json
import os
import shutil
import tarfile
import tempfile
from datetime import timedelta


def timestamp_to_frames(timestamp, fps):
    """Converts a timestamp in the format 'min.ms' into a frame count."""
    total_seconds = float(timestamp)
    print(total_seconds)
    # TODO: right-exlusive, left-inclusive.
    return round(total_seconds * fps)


def process_tar_files(source_directory, target_directory, skip_existing=True):
    """Extract, process, and re-package JSON files in TAR archives."""
    # TODO: this path
    # source_directory = os.path.join(source_directory, "video_rgb")
    target_directory = os.path.join(target_directory, "video_transcript")

    os.makedirs(target_directory, exist_ok=True)

    for tar_path in os.listdir(source_directory):
        if tar_path.endswith(".tar"):
            shard_name = "shard-" + os.path.splitext(tar_path)[0] + ".tar"
            target_tar_path = os.path.join(target_directory, shard_name)
            print(target_tar_path)

            if skip_existing and os.path.exists(target_tar_path):
                print(f"Skipping already processed file: {target_tar_path}")
                continue

            source_tar_path = os.path.join(source_directory, tar_path)
            with tarfile.open(source_tar_path, "r") as tar:
                temp_dir = tempfile.mkdtemp()
                try:
                    tar.extractall(path=temp_dir)

                    # process json files
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith(".json"):
                                process_json_file(os.path.join(root, file), temp_dir)

                    with tarfile.open(target_tar_path, "w") as out_tar:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith(".jsonl"):
                                    out_tar.add(os.path.join(root, file), arcname=file)
                finally:
                    shutil.rmtree(temp_dir)


def process_json_file(json_file_path, output_dir):
    """Reads and processes a single JSON file to convert it to the required format."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        video_key = os.path.splitext(os.path.basename(json_file_path))[0]

        if data["status"] != "success":
            # errored while downloading
            return
        elif "subtitles" not in data["yt_meta_dict"]:
            print(data)
            # TODO: what to do with videos that have no subtitles? When can this occur?
            return
        if data["yt_meta_dict"]["subtitles"].keys() != {"en"}:
            # XXX: for now, we decided to only exclude non-English videos
            return
        subtitles = data["whisper_alignment"]["segments"]
        fps = data["yt_meta_dict"]["info"]["fps"]

        json_content = []
        for subtitle in subtitles:
            start_frame = timestamp_to_frames(subtitle["start"], fps)
            end_frame = timestamp_to_frames(subtitle["end"], fps)
            sentence = subtitle["text"]
            word_timestamps = []
            for word in subtitle["words"]:
                word_timestamps.append(
                    {
                        "word": word["word"],
                        "start": timestamp_to_frames(word["start"], fps)
                        if "start" in word.keys() else None,
                        "end": timestamp_to_frames(word["end"], fps)
                        if "end" in word.keys() else None,
                    }
                )

            json_content.append(
                {
                    "sentence": sentence,
                    "start": start_frame,
                    "end": end_frame,
                    "words": word_timestamps,
                }
            )

        jsonl_filename = f"{video_key}.jsonl"
        with open(os.path.join(output_dir, jsonl_filename), "w") as outfile:
            json.dump(json_content, outfile, indent=4)


def main(args):
    for folder in os.listdir(args.data_root):
        if folder in ["train", "val", "test"]:
            current_folder = os.path.join(args.data_root, folder, args.whisper_dir)
            print(f"Processing {current_folder}.")
            process_tar_files(
                source_directory=current_folder,
                target_directory=os.path.join(args.data_root, folder),
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tarfiles containing JSONs and convert to structured JSONL format."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        # FIXME: default dir
        # default="/store/swissai/a08/data/4m-data/train/DEBUG/v2d_40k",
        default="/cluster/work/cotterell/mm_swissai/raw/v2d_500/howto100m",
        help="Dir containing the JSON files to process.",
    )
    parser.add_argument(
        "--whisper_dir",
        type=str,
        default="whisperx",
        help="Dir containing the WhisperX transcripts.",
    )
    parser.add_argument(
        "--skip_existing",
        default=False,  # FIXME
        help="Skip tarfiles already processed (exist in the target directory).",
    )

    args = parser.parse_args()
    main(args)
