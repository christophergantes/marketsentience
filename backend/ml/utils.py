import torch
import os
import zipfile
import requests
import random
from pathlib import Path


def download_data(source: str, destination: str, remove_source: bool):
    data_path = Path("data/")
    download_path = data_path / destination

    if download_path.is_dir():
        print(f"{download_path} directory already exists. Skipping download.")
    else:
        print(f"directory {download_path} not found. creating new directory.")
        download_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(source).name

        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"Downloading data into {target_file} from {source}...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, "r") as zip:
            print(f"Extracting {target_file} into {download_path}")
            zip.extractall(download_path)

        if remove_source:
            os.remove(data_path / target_file)


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    random.seed(seed)


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
