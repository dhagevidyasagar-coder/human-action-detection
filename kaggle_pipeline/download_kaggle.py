import os
import sys
import json
import argparse
from pathlib import Path

# Common Kaggle datasets for Action & Emotion Recognition
DATASETS = {
    "fer2013": {
        "slug": "msambare/fer2013",
        "description": "Facial Expression Recognition 2013 (FER-2013) - 35k labeled face images (Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust)",
        "type": "emotion",
        "target_dir": "data/fer2013"
    },
    "har": {
        "slug": "meetnagadia/human-action-recognition-har-dataset",
        "description": "Human Action Recognition (HAR) - 12.6k labeled action images (sitting, standing, clapping, cycling, dancing, eating, etc.)",
        "type": "action",
        "target_dir": "data/har"
    }
}

def check_kaggle_credentials():
    """Checks if Kaggle API credentials exist in environment or ~/.kaggle/kaggle.json"""
    home = Path.home()
    default_creds = home / ".kaggle" / "kaggle.json"
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if env_user and env_key:
        return True, "Found KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    if default_creds.exists():
        return True, f"Found credentials at {default_creds}"
    if env_dir and (Path(env_dir) / "kaggle.json").exists():
        return True, f"Found credentials at {Path(env_dir) / 'kaggle.json'}"

    return False, "Kaggle credentials not found. Please place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME & KAGGLE_KEY."

def download_dataset(dataset_key):
    if dataset_key not in DATASETS:
        print(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASETS.keys())}")
        return False

    info = DATASETS[dataset_key]
    has_creds, msg = check_kaggle_credentials()
    print(f"[*] Checking Kaggle credentials: {msg}")

    target_path = Path(info["target_dir"])
    target_path.mkdir(parents=True, exist_ok=True)

    if not has_creds:
        print("\n" + "="*70)
        print("⚠️  KAGGLE API TOKEN REQUIRED FOR DIRECT KAGGLE DOWNLOAD")
        print("="*70)
        print("To download directly from Kaggle:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ or set environment variables:")
        print("   export KAGGLE_USERNAME='your_username'")
        print("   export KAGGLE_KEY='your_api_key'")
        print(f"\nAlternatively, manually download from:\nhttps://www.kaggle.com/datasets/{info['slug']}")
        print(f"and extract files into: {target_path.resolve()}\n")
        print("="*70 + "\n")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[*] Downloading {info['slug']} to {target_path}...")
        api.dataset_download_files(info["slug"], path=str(target_path), unzip=True)
        print(f"[✓] Successfully downloaded and extracted {dataset_key}!")
        return True
    except Exception as e:
        print(f"[!] Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Action and Emotion datasets from Kaggle.")
    parser.add_argument("--dataset", choices=["fer2013", "har", "all"], default="all",
                        help="Which Kaggle dataset to download")
    args = parser.parse_args()

    if args.dataset == "all":
        for k in DATASETS:
            download_dataset(k)
    else:
        download_dataset(args.dataset)
