"""Download Tier-1/2 open datasets into data/raw (real data, no synthetic fallbacks)."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "heart_cleveland.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "parkinsons.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    "pima_diabetes.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
}


def main() -> None:
    for fname, url in URLS.items():
        dest = RAW / fname
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"fetch {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
