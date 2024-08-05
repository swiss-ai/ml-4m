import os
import tarfile
import argparse
from tqdm import tqdm
import logging

from utils import setup_logging


def filter_mp4_from_tars(input_dir, output_dir):
    """
    Create tars at output_dir which mirror those of input_dir but filtering to be only the mp4s.

    E.g., given input_dir

    filtered_raw/howto100m
    |    | - shard-00000.tar
    |    |   | - 00000.mp4
    |    |   | - 00000.m4a
    |    |   | - 00000.json
    |    |   | - 00002.mp4
    |    |   | - 00002.m4a
    |    |   | - 00002.json

    we want to end up with output_dir

    4m/howto100m/video_rgb
    |    | - shard-00000.tar
    |    |   | - 00000.mp4
    |    |   | - 00002.mp4
    """
    logging.info(f"Creating {output_dir} by filtering in only mp4s from {input_dir}.")
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".tar"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            logging.info(f"Processing {input_path}")

            with tarfile.open(input_path, "r") as tar_in, tarfile.open(output_path, "w") as tar_out:
                for member in tar_in.getmembers():
                    if member.name.endswith(".mp4"):
                        file_content = tar_in.extractfile(member)
                        tar_out.addfile(member, file_content)
                        logging.info(f"Added {member.name} to {output_path}")
        else:
            logging.warning(f"Expected only tarfiles in input dir {input_dir}, instead found {filename}.")


def validate_output(output_dir):
    """Validate that all tar files within the output_dir contain only mp4s."""
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
        for member in tar.getmembers():
            if not member.name.endswith(".mp4"):
                logging.error(f"Non-mp4 file found in {file_path}: {member.name}")
                return False
    return True


def main():
    """
    Extract the video_rgb/ modality for a given filtered_raw dataset.
    """
    parser = argparse.ArgumentParser(description="Filter tar files to keep only .mp4 files.")
    parser.add_argument(
        "-I",
        "--input_dir",
        default="/store/swissai/a08/data/filtered_raw/howto100m/v2d_5000",
        help="Path to the input directory containing tar files (default: raw/howto100m)",
    )
    parser.add_argument(
        "-O",
        "--output_dir",
        default=None,
        help="Path to the output directory for filtered tar files. If None is passed, then will be inferred/constructed from the input path.",
    )
    parser.add_argument(
        "-L", "--log_file", default="make_video_rgb.log", help="Path to the log file (default: filter_mp4s.log)"
    )
    parser.add_argument(
        "-V", "--validate-only", action="store_true", help="Validate the output directory after filtering"
    )

    args = parser.parse_args()
    setup_logging(args.log_file)

    if "filtered_raw" not in args.input_dir:
        raise ValueError(f"Expected input dir to be a subdir of `filtered_raw/`, instead received {args.input_dir}.")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(args.input_dir.replace("filtered_raw", "4m"), "video_rgb")
    )

    if not args.validate_only:
        logging.info("Starting .mp4 filtering process")
        filter_mp4_from_tars(args.input_dir, output_dir)
        logging.info("Filtering process completed")

    if not validate_output(output_dir):
        logging.error("Validation failed. Please check the output directory.")


if __name__ == "__main__":
    main()
