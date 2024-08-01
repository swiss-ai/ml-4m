import argparse
import os
import shutil
from sklearn.model_selection import train_test_split


def main(args):
    """Main function to partition datasets into train, val, and test splits."""

    # Get all class directories from the source directory
    dset_dirs = [
        d
        for d in os.listdir(args.source_dir)
        if os.path.isdir(os.path.join(args.source_dir, d))
    ]
    print(f"Subdirectories of source dir {args.source_dir}: {dset_dirs}")

    for dset_dir in dset_dirs:
        dset_path = os.path.join(args.source_dir, dset_dir)
        print(dset_path)
        all_files = os.listdir(dset_path)
        if len(all_files) == 0:
            print(f"Skipping dataset {dset_dir} as it has no files.")
            continue

        # filter out files not ending with .tar
        all_files = sorted([f for f in all_files if f.endswith(".tar")])
        # Split shards into train/temp
        train_files, temp_files = train_test_split(
            all_files,
            train_size=args.train_ratio,
            random_state=42,
            shuffle=args.shuffle,
        )

        # Split temp into val/test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=args.test_ratio / (1 - args.train_ratio),
            random_state=42,
            shuffle=args.shuffle,
        )

        # move files to respective splits
        for dataset, files in zip(
            ["train", "val", "test"], [train_files, val_files, test_files]
        ):
            split_path = os.path.join(args.output_dir, dataset, dset_dir)
            print(f"Move {dset_path} -----------> {split_path}")
            os.makedirs(split_path, exist_ok=True)  # Create class directory in split
            for file in files:
                if args.copy:
                    shutil.copy(os.path.join(dset_path, file), os.path.join(split_path, file))
                else:
                    shutil.move(
                        os.path.join(dset_path, file), os.path.join(split_path, file)
                    )


if __name__ == "__main__":
    """
    Given a source directory containing the data for multiple modalities, e.g.,
    
    ```
    |--source_dir/
    |  |--modality_a/
    |  |--modality_b/
    |  |--modality_c/
    ```

    move the files into a specified output_dir/ with the structure:
    ```
    |--source_dir/
    |  |--train/
    |  |  |--modality_a/
    |  |  |--modality_b/
    |  |  |--modality_c/
    |  |--val/
    |  |  |--modality_a/
    |  |  |--modality_b/
    |  |  |--modality_c/
    |  |--test/
    |  |  |--modality_a/
    |  |  |--modality_b/
    |  |  |--modality_c/
    ```
    """
    parser = argparse.ArgumentParser(
        description="Partition datasets into train, val, and test splits."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source directory containing dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to store the splits. (--output_dir/dataset/split)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data for the training set.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio of data for the test set (remaining will be validation).",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="Whether to shuffle shards befores splitting. Otherwise, train is 0, 1, 2, etc.",
    )
    parser.add_argument(
        "--copy",
        type=bool,
        default=False,
        help="Whether to copy the files instead of move. Defaults to False.",
    )
    args = parser.parse_args()

    main(args)
