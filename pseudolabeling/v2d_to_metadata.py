import argparse
import json
import os
import shutil
import tarfile
import tempfile
from datetime import timedelta

# FIXME: may need adaptation
METADATA_MAPPING = {
    "webpage_url": "url",
    "title": "title",
    "duration": "duration",
    "channel": "channel",
    "fps": "fps",
    "tags": "tags",
    "resolution": "resolution",
    "aspect_ratio": "aspect_ratio",
}


def process_tar_files(source_directory, target_directory, dataset, skip_existing=True):
    """Extract, process, and re-package JSON files in TAR archives."""
    # TODO: this path
    # source_directory = os.path.join(source_directory, "video_rgb")
    target_directory = os.path.join(target_directory, "video_metadata")

    os.makedirs(target_directory, exist_ok=True)

    for tar_path in os.listdir(source_directory):
        print(source_directory)
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
                                process_json_file(
                                    os.path.join(root, file), temp_dir, dataset
                                )

                    with tarfile.open(target_tar_path, "w") as out_tar:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith(".json"):
                                    out_tar.add(os.path.join(root, file), arcname=file)
                finally:
                    shutil.rmtree(temp_dir)


def process_json_file(json_file_path, output_dir, dataset):
    """Reads and processes a single JSON file to convert it to the required format."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        # remove filepath of json
        os.remove(json_file_path)
        video_key = os.path.splitext(os.path.basename(json_file_path))[0]

        json_content = {}

        if data["status"] != "success":
            # errored while downloading
            return
        elif "subtitles" not in data["yt_meta_dict"]:
            print(data)
            # XXX: always ensure to only write metadata where we have everything we need
            # (transcript, video, ...)
            return
        if data["yt_meta_dict"]["subtitles"].keys() != {"en"}:
            # XXX: for now, we decided to only exclude non-English videos
            return
        for key, value in METADATA_MAPPING.items():
            if value in data["yt_meta_dict"]["info"]:
                json_content[key] = data["yt_meta_dict"]["info"][value]

        json_content["dataset"] = dataset
        json_filename = f"{video_key}.json"
        with open(os.path.join(output_dir, json_filename), "w") as outfile:
            json.dump(json_content, outfile, indent=4)


def main(args):
    for folder in os.listdir(args.data_root):
        if folder in ["train", "val", "test"]:
            print(f"Processing {folder}.")
            process_tar_files(
                source_directory=os.path.join(
                    args.data_root,
                    folder,
                ),
                target_directory=os.path.join(args.data_root, folder),
                dataset=args.dataset,
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
        "--skip_existing",
        default=False,  # FIXME
        help="Skip tarfiles already processed (exist in the target directory).",
    )
    # TODO: is this also in filestructure or do we have to provide it like this?
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Which dataset tar is coming from (HDVILA/HowTo100M)",
    )

    args = parser.parse_args()
    main(args)
