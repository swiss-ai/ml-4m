import os
import tarfile
import argparse
from tqdm import tqdm
from collections import defaultdict
import logging

from utils import setup_logging


def filter_tar_files(input_dir: str, output_dir: str):
    """
    Given a path to a raw dataset of v2d tar files, filter out all files within the tar that are "bad", i.e., don't have an mp4.
    E.g., given the path raw/howto100m containing:

    raw/howto100m
    |    | - 00000.tar
    |    |   | - 00000.mp4
    |    |   | - 00000.m4a
    |    |   | - 00000.json
    |    |   | - 00002.mp4
    |    |   | - 00002.m4a
    |    |   | - 00002.json
    |    |   | - 00010.json
    |    |   | - 00011.json
    |    |   | - 00012.json
    we should end up with filtered_raw/howto100m containing:

    filtered_raw/howto100m
    |    | - shard-00000.tar
    |    |   | - 00000.mp4
    |    |   | - 00000.m4a
    |    |   | - 00000.json
    |    |   | - 00002.mp4
    |    |   | - 00002.m4a
    |    |   | - 00002.json
    The video_rgb and other modalities should then all be drawn/pseudolabeled from filtered_raw/howto100m.
    """
    logging.info(f"Creating {output_dir} by filtering out files without corresponding .mp4s from {input_dir}.")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".tar"):
            input_path = os.path.join(input_dir, filename)
            shard_name = "shard-" + os.path.splitext(filename)[0] + ".tar"
            output_path = os.path.join(output_dir, shard_name)

            # Open the input tar file
            with tarfile.open(input_path, "r") as tar_in:
                # Group files by their base name (without extension)
                file_groups = defaultdict(list)
                for member in tar_in.getmembers():
                    base_name = os.path.splitext(member.name)[0]
                    file_groups[base_name].append(member)

                # Create a new tar file for filtered contents
                with tarfile.open(output_path, "w") as tar_out:
                    for base_name, files in file_groups.items():
                        # Check if there's an .mp4 file in this group
                        if any(f.name.endswith(".mp4") for f in files):
                            # If so, add all files in this group to the new tar
                            for file in files:
                                file_content = tar_in.extractfile(file)
                                tar_out.addfile(file, file_content)
                        else:
                            logging.info(f"Filtering out {base_name} from tar {filename}.")

    logging.info(f"Filtered tar files have been saved to {output_dir}.")


def validate_output(output_dir: str):
    """Validate that all tar files within the output_dir pass our filter."""
    logging.info(f"Validating output directory: {output_dir}")
    all_valid = True

    failed_tars = []
    for filename in tqdm(os.listdir(output_dir)):
        if filename.endswith(".tar"):
            file_path = os.path.join(output_dir, filename)
            if not validate_tar_file(file_path):
                all_valid = False
                logging.error(f"Validation failed for {filename}")
                failed_tars.append(filename)
            else:
                logging.info(f"Validation passed for {filename}")

    if all_valid:
        logging.info("All tar files in the output directory are valid.")
    else:
        logging.error(f"{len(failed_tars)} tar files in the output directory are invalid:\n\t{failed_tars}")

    return all_valid


def validate_tar_file(file_path):
    with tarfile.open(file_path, "r") as tar:
        file_groups = defaultdict(list)
        for member in tar.getmembers():
            base_name = os.path.splitext(member.name)[0]
            file_groups[base_name].append(member.name)

        for group in file_groups.values():
            if not any(file.endswith(".mp4") for file in group):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Filter tar files to keep only those with corresponding .mp4 files.")
    parser.add_argument(
        "-I",
        "--input_dir",
        default="/store/swissai/a08/data/raw/howto100m/v2d_5000",
        help="Path to the input directory containing tar files.",
    )
    parser.add_argument(
        "-O",
        "--output_dir",
        default=None,
        help="Path to the output directory for filtered tar files. Default is None (will be inferred from the input dir).",
    )
    parser.add_argument(
        "-L", "--log_file", default="filter_raw.log", help="Path to the log file (default: filter_mp4s.log)"
    )
    parser.add_argument(
        "-V", "--validate-only", action="store_true", help="Validate the output directory after filtering"
    )
    args = parser.parse_args()
    setup_logging(args.log_file)

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir.replace("raw", "filtered_raw")

    if not args.validate_only:
        logging.info("Beginning filtering raw process.")
        filter_tar_files(args.input_dir, output_dir)
        logging.info("Completed filtering raw process.")

    if not validate_output(output_dir):
        logging.error("Validation failed. Please check the output directory.")


if __name__ == "__main__":
    main()
