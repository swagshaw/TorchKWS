import argparse
import os
import shutil
import urllib.request
import tarfile


def main(args):
    data_dir = args.data_dir
    # Download the dataset
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    filename = "speech_commands_v0.02.tar.gz"
    urllib.request.urlretrieve(url, filename)

    # Extract the dataset
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()

    # Create train and valid directories
    os.makedirs(os.path.join(data_dir, "train"))
    os.makedirs(os.path.join(data_dir, "valid"))

    # Move files to train and valid directories
    for name in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, name)):
            for i, file_name in enumerate(os.listdir(os.path.join(data_dir, name))):
                src = os.path.join(data_dir, name, file_name)
                if i < 8000:
                    dst = os.path.join(os.path.join(data_dir, "train"), name, file_name)
                else:
                    dst = os.path.join(os.path.join(data_dir, "valid"), name, file_name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

    # Remove unnecessary directories
    for name in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, name)):
            shutil.rmtree(os.path.join(data_dir, name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="dataset folder")
    args = parser.parse_args()

    main(args)
