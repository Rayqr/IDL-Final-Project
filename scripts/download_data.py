from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.out_dir / "CMAPSSData.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} -> {zip_path}")
        urllib.request.urlretrieve(URL, zip_path)
    else:
        print(f"Found existing {zip_path}")

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
