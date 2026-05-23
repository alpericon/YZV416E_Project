"""
Download and extract DAVIS 2017 (480p) to data/DAVIS/.

Usage:
    python scripts/download_davis.py
    python scripts/download_davis.py --root data/DAVIS --resolution 1080p
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DAVIS_URLS = {
    "480p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
    "1080p": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip",
}

# Also download the test-dev split (optional)
DAVIS_TEST_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip"

SAM2_CHECKPOINTS = {
    "hiera_t": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_tiny.pt",
    ),
    "hiera_s": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_small.pt",
    ),
    "hiera_b+": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_base_plus.pt",
    ),
    "hiera_l": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "sam2_hiera_large.pt",
    ),
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dest), reporthook=_progress)
    print()


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    print("  Done.")


def download_davis(root: str = "data", resolution: str = "480p",
                   keep_zip: bool = False) -> None:
    root = Path(root)
    url = DAVIS_URLS.get(resolution)
    if url is None:
        sys.exit(f"Unknown resolution '{resolution}'. Choose from: {list(DAVIS_URLS)}")

    zip_name = url.split("/")[-1]
    zip_path = root / zip_name

    if (root / "DAVIS" / "ImageSets").exists():
        print(f"DAVIS already present at {root / 'DAVIS'}. Skipping download.")
    else:
        if not zip_path.exists():
            download_file(url, zip_path)
        extract_zip(zip_path, root)
        if not keep_zip:
            zip_path.unlink()
            print(f"Removed {zip_path}")

    print(f"\nDAVIS 2017 ({resolution}) available at: {root / 'DAVIS'}")
    _verify_davis(root / "DAVIS")


def download_sam2_checkpoint(variant: str = "hiera_b+",
                              dest_dir: str = "checkpoints") -> None:
    url, filename = SAM2_CHECKPOINTS.get(variant, (None, None))
    if url is None:
        sys.exit(f"Unknown SAM 2 variant '{variant}'. Choose from: {list(SAM2_CHECKPOINTS)}")

    dest = Path(dest_dir) / filename
    if dest.exists():
        print(f"Checkpoint already exists: {dest}")
        return
    download_file(url, dest)
    print(f"SAM 2 checkpoint saved: {dest}")


def _verify_davis(davis_root: Path) -> None:
    required = [
        davis_root / "JPEGImages",
        davis_root / "Annotations",
        davis_root / "ImageSets" / "2017" / "train.txt",
        davis_root / "ImageSets" / "2017" / "val.txt",
    ]
    ok = True
    for p in required:
        if p.exists():
            print(f"  [OK] {p.relative_to(davis_root.parent)}")
        else:
            print(f"  [MISSING] {p}")
            ok = False
    if not ok:
        print("\n[WARN] Some DAVIS files are missing. Re-run or download manually.")
    else:
        print("\nDAVIS 2017 integrity check passed.")


def main():
    parser = argparse.ArgumentParser(description="Download DAVIS 2017 and SAM 2 checkpoint")
    parser.add_argument("--root", default="data", help="Directory to store DAVIS (default: data/)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "1080p"])
    parser.add_argument("--checkpoint", default="hiera_b+",
                        choices=list(SAM2_CHECKPOINTS),
                        help="SAM 2 variant to download (default: hiera_b+)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for SAM 2 checkpoint (default: checkpoints/)")
    parser.add_argument("--skip-davis", action="store_true",
                        help="Skip DAVIS download (only download checkpoint)")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Skip checkpoint download (only download DAVIS)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the zip file after extraction")
    args = parser.parse_args()

    if not args.skip_davis:
        download_davis(args.root, args.resolution, args.keep_zip)
    if not args.skip_checkpoint:
        download_sam2_checkpoint(args.checkpoint, args.checkpoint_dir)


if __name__ == "__main__":
    main()
