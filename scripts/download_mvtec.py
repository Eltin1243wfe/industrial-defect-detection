"""
Grabs a single-category subset of MVTec-AD from the official mirror and
unpacks it into data/mvtec/<category>. The full dataset is ~5GB across all
15 categories so by default this only pulls the one category you ask for.

    python scripts/download_mvtec.py --category bottle

Categories available: bottle, cable, capsule, carpet, grid, hazelnut,
leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

Note: MVTec's servers have historically been flaky about direct scripted
downloads / occasionally require accepting a license click-through on their
site. If this script fails, just grab it manually from
https://www.mvtec.com/company/research/datasets/mvtec-ad and drop the
extracted folder into data/mvtec/<category>.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/{category}.tar.xz"

VALID_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def download(category: str, dest_root: str = "data/mvtec"):
    if category not in VALID_CATEGORIES:
        raise ValueError(f"unknown category '{category}', pick from {VALID_CATEGORIES}")

    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    archive_path = dest_root / f"{category}.tar.xz"
    target_dir = dest_root / category

    if target_dir.exists():
        print(f"{target_dir} already exists, skipping download")
        return

    url = BASE_URL.format(category=category)
    print(f"downloading {category} from mvtec mirror...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        print(f"download failed ({e}). grab it manually from mvtec's site instead, see "
              f"docstring at the top of this file for the link.")
        return

    print("extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_root)

    archive_path.unlink()
    print(f"done, dataset is at {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle", choices=VALID_CATEGORIES)
    parser.add_argument("--dest", default="data/mvtec")
    args = parser.parse_args()
    download(args.category, args.dest)
