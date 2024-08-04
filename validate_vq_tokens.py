import tarfile
import numpy as np
from io import BytesIO
import os


def read_tarfile(file_path):
    with tarfile.open(file_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                file_obj = tar.extractfile(member)
                if file_obj:
                    bio = BytesIO(file_obj.read())
                    bio.seek(0)
                    array = np.load(bio)
                    print(
                        f"File: {member.name}, Shape: {array.shape}, Dtype: {array.dtype}"
                    )


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tar"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                read_tarfile(file_path)


# tar dir
directory_path = "/store/swissai/a08/data/4m-data/train/video_rgb_tok"

process_directory(directory_path)
