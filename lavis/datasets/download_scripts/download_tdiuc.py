import os
from pathlib import Path

from omegaconf import OmegaConf
import json
import requests
import zipfile
from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


import gdown


def download_and_extract_zip(output_path, extract_to):
    if os.path.exists(output_path) == False:

        url = 'https://drive.google.com/uc?export=download&id=1Hevf7eQNzg-qlXbfz9nPbATmQciexkDp'
        print(f"Downloading {url} to {storage_dir}")
        gdown.download(url, output_path, quiet=False)

    # Extract the ZIP file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted files to {extract_to}")

    os.remove(output_path)


def download_json(json_path):
    url = 'https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TDIUC_sampling.json'

    print(f"downloading {url} to {json_path}....")
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        data = response.json()

        # Save the JSON content to a file
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)
        print('File downloaded and saved as downloaded_file.json')
    else:
        print(f'Failed to download file: {response.status_code}')

def download_datasets(root, url):
    download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)


if __name__ == "__main__":
    config_path = get_abs_path("configs/datasets/TDIUC/default_TDIUC.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.TDIUC.build_info.images.storage

    json_path = OmegaConf.load(
        config_path
    ).datasets.TDIUC.build_info.annotations.test.storage
    json_path = get_cache_path(json_path)

    download_json(json_path)

    storage_dir = Path(get_cache_path(storage_dir))
    print(f"length: {len(os.listdir(storage_dir))}, {storage_dir}")
    if os.path.exists(storage_dir) == False:
        os.makedirs(storage_dir)
        download_and_extract_zip('val2014.zip', storage_dir)
    elif os.path.exists(os.path.join(storage_dir,'val2014_TDIUC')) == False:
        download_and_extract_zip('val2014.zip', storage_dir)
    else:
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)
