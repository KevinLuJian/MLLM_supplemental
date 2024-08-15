import os
from pathlib import Path

from omegaconf import OmegaConf
import json
import requests
from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)


DATA_URL = {
    "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
}

def download_json(json_path):
    url = 'https://github.com/KevinLuJian/MLLM-evaluation/raw/main/VQDv1_sampling.json'

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

    config_path = get_abs_path("configs/datasets/VQDv1/defaults_VQDv1.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.VQDv1.build_info.images.storage


    json_path = OmegaConf.load(
        config_path
    ).datasets.VQDv1.build_info.annotations.val.storage
    json_path = get_cache_path(json_path)

    directory = os.path.dirname(json_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directories: {directory}")

    print(f"cache: {json_path}")
    download_json(json_path)


    download_dir = Path(get_cache_path(storage_dir)).parent / "download"
    storage_dir = Path(get_cache_path(storage_dir))

    print(f" downloading {DATA_URL['val']} to {storage_dir}")

    if storage_dir.exists():
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)

    try:
        for k, v in DATA_URL.items():
            print("Downloading {} to {}".format(v, k))
            download_datasets(download_dir, v)
    except Exception as e:
        print(f"Exception {e}")
        # remove download dir if failed
        cleanup_dir(download_dir)
        print("Failed to download or extracting datasets. Aborting.")

    cleanup_dir(download_dir)
