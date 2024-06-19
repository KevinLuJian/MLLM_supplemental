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


val = {
    "val1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "val2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}

DATA_URL = {
    "val": val,
}


def download_datasets(root, url):
    download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)

def download_json(json_path):
    url = 'https://github.com/KevinLuJian/MLLM-evaluation/raw/main/TallyQA_test.json'

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


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/TallyQA/default_tallyqa.yaml")

    storage_dir = OmegaConf.load(
        config_path
    ).datasets.TallyQA.build_info.images.storage

    json_path = OmegaConf.load(
        config_path
    ).datasets.TallyQA.build_info.annotations.test.storage
    json_path = get_cache_path(json_path)

    download_json(json_path)

    download_dir = Path(get_cache_path(storage_dir)).parent / "download"

    storage_dir = Path(get_cache_path(storage_dir))

    if storage_dir.exists():
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)

    try:
        for dataset, url_list in DATA_URL.items():
            for k, v in url_list.items():
                print("Downloading {} to {}".format(v, k))
                download_datasets(download_dir, v)

    except Exception as e:
        print(f"Exception {e}")
        # remove download dir if failed
        cleanup_dir(download_dir)
        print("Failed to download or extracting datasets. Aborting.")

    cleanup_dir(download_dir)
