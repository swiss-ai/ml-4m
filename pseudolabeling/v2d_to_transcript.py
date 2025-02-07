import argparse
import json
import os
import shutil
import tarfile
import tempfile
from tqdm import tqdm
from datetime import timedelta


def timestamp_to_frames(timestamp, fps):
    """Converts a timestamp in the format 'min.ms' into a frame count."""
    total_seconds = float(timestamp)
    print(total_seconds)
    # TODO: right-exlusive, left-inclusive.
    return round(total_seconds * fps)


def process_tar_files(source_directory, target_directory, skip_existing=True):
    """Extract, process, and re-package JSON files in TAR archives."""
    os.makedirs(target_directory, exist_ok=True)

    for filename in tqdm(os.listdir(source_directory)):
        if filename.endswith(".tar"):
            target_tar_path = os.path.join(target_directory, filename)
            print(target_tar_path)

            if skip_existing and os.path.exists(target_tar_path):
                print(f"Skipping already processed file: {target_tar_path}")
                continue

            source_tar_path = os.path.join(source_directory, filename)
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
                        if "start" in word.keys()
                        else None,
                        "end": timestamp_to_frames(word["end"], fps)
                        if "end" in word.keys()
                        else None,
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
    if "filtered_raw" not in args.input_dir:
        raise ValueError(f"Expected input dir to be a subdir of `filtered_raw/`, instead received {args.input_dir}.")

    current_folder = os.path.join(args.input_dir, args.whisper_dir)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(args.input_dir.replace("filtered_raw", "4m"), "video_transcript")
    )
    print(f"Processing {current_folder}.")
    process_tar_files(
        source_directory=current_folder,
        target_directory=output_dir,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tarfiles containing JSONs and convert to structured JSONL format."
    )
    parser.add_argument(
        "-I",
        "--input_dir",
        type=str,
        default="/store/swissai/a08/data/filtered_raw/howto100m/v2d_5000/",
        # default="/cluster/work/cotterell/mm_swissai/raw/v2d_500/howto100m",
        help="A `filtered_raw` dir containing the JSON files to process.",
    )
    parser.add_argument(
        "-O",
        "--output_dir",
        type=str,
        default=None,
        help="Output dir to save the pseudolabeled transcripts.",
    )
    parser.add_argument(
        "-W",
        "--whisper_dir",
        type=str,
        default="whisperx",
        help="Dir containing the WhisperX transcripts.",
    )
    parser.add_argument(
        "-S",
        "--skip_existing",
        default=False,  # FIXME
        help="Skip tarfiles already processed (exist in the target directory).",
    )

    args = parser.parse_args()
    main(args)
