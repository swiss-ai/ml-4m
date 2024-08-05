import argparse
import json
import os
import shutil
import tarfile
import tempfile
from tqdm import tqdm
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
            print(data["status"])
            return
        elif "subtitles" not in data["yt_meta_dict"]:
            print("NO SUBTITLES: ", data)
            # indeed, there are some videos without subtitles (np speech)
            return
        if (
            data["yt_meta_dict"]["subtitles"].keys() != {"en"}
            and len(data["yt_meta_dict"]["subtitles"].keys()) > 0
        ):
            # XXX: for now, we decided to only exclude non-English videos.
            raise ValueError(
                f"Non-English subtitles found: {data['yt_meta_dict']['subtitles'].keys()}"
            )
        for key, value in METADATA_MAPPING.items():
            if value in data["yt_meta_dict"]["info"]:
                json_content[key] = data["yt_meta_dict"]["info"][value]

        json_content["dataset"] = dataset
        json_filename = f"{video_key}.json"
        with open(os.path.join(output_dir, json_filename), "w") as outfile:
            json.dump(json_content, outfile, indent=4)


def main(args):
    if "filtered_raw" not in args.input_dir:
        raise ValueError(f"Expected input dir to be a subdir of `filtered_raw/`, instead received {args.input_dir}.")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(args.input_dir.replace("filtered_raw", "4m"), "video_metadata")
    )
    process_tar_files(
        source_directory=args.input_dir,
        target_directory=output_dir,
        dataset=args.dataset,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tarfiles from `filtered_raw` format containing JSONs and extract relevant metadata into the `video_metadata` modality."
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
        help="Output dir to save the pseudolabeled metadata.",
    )
    parser.add_argument(
        "-S",
        "--skip_existing",
        default=False,  # FIXME
        help="Skip tarfiles already processed (exist in the target directory).",
    )
    # TODO: is this also in filestructure or do we have to provide it like this?
    parser.add_argument(
        "-D",
        "--dataset",
        type=str,
        required=True,
        help="Which dataset tar is coming from (HDVILA/HowTo100M)",
    )

    args = parser.parse_args()
    main(args)
