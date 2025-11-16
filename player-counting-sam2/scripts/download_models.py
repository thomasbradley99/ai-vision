#!/usr/bin/env python3
"""
Download required model checkpoints
"""

import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    # Create checkpoints directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading model checkpoints...")
    print()
    
    # YOLO models (optional - ultralytics will auto-download)
    yolo_models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    }
    
    for model_name, url in yolo_models.items():
        model_path = checkpoint_dir / model_name
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            download_url(url, model_path)
        else:
            print(f"✓ {model_name} already downloaded")
    
    # SAM2 checkpoints
    print()
    print("For SAM2 checkpoints, please:")
    print("1. Install SAM2:")
    print("   git clone https://github.com/facebookresearch/segment-anything-2.git")
    print("   cd segment-anything-2 && pip install -e .")
    print()
    print("2. Download checkpoints:")
    print("   cd checkpoints && ./download_ckpts.sh")
    print()
    print("3. Copy checkpoint to this project:")
    print(f"   cp segment-anything-2/checkpoints/sam2_hiera_large.pt {checkpoint_dir}/")
    print()
    
    print("✓ Setup complete!")


if __name__ == "__main__":
    main()

