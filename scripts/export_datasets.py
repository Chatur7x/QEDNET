#!/usr/bin/env python3
"""Export the sklearn breast-cancer dataset as CSV so the dataset package
contains all 4 Tier-1/2 datasets, plus a README manifest."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer

RAW = Path("/home/z/my-project/data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# 1. remove the junk 9-byte placeholder
junk = RAW / "pima-indians-diabetes.data"
if junk.exists() and junk.stat().st_size < 32:
    junk.unlink()
    print("removed junk pima-indians-diabetes.data (9 bytes)")

# 2. export breast cancer
ds = load_breast_cancer()
header = ["diagnosis"] + list(ds.feature_names)
out = RAW / "breast_cancer.csv"
with out.open("w") as f:
    f.write(",".join(header) + "\n")
    for x, y in zip(ds.data, ds.target):
        f.write(str(int(y)) + "," + ",".join(f"{v:.10g}" for v in x) + "\n")
print(f"wrote {out.name}: {ds.data.shape[0]} rows x {ds.data.shape[1]+1} cols")

# 3. manifest
readme = RAW / "README.md"
readme.write_text("""# QED-Net 2.0 — bundled open datasets

Four public, de-identified benchmark datasets (no restricted data included).
Place these files in `data/raw/` of the project (this IS that directory).

| file | dataset | size | target | source |
|------|---------|------|--------|--------|
| `breast_cancer.csv` | Wisconsin Diagnostic Breast Cancer | 569 x 31 | `diagnosis` (0=benign, 1=malignant) | sklearn `load_breast_cancer` export |
| `processed.cleveland.data` | UCI Heart Disease (Cleveland) | 303 x 14 | `num` (>0 = disease) | UCI ML Repository |
| `parkinsons.data` | UCI Parkinson's (voice features) | 195 x 23 | `status` (1=PD) | UCI ML Repository |
| `pima-indians-diabetes.csv` | Pima Indians Diabetes | 768 x 9 | `Outcome` (last column) | UCI / jbrownlee/Datasets |

Sources:
- UCI ML Repository: https://archive.ics.uci.edu/
  - heart-disease/processed.cleveland.data
  - parkinsons/parkinsons.data
- Pima: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
- Breast cancer: bundled inside scikit-learn (public domain copy of the UCI
  original); exported here for completeness.

All files are standard public research benchmarks. `?` marks in the Cleveland
file are treated as missing values by the ingestion layer (F1).
""")
print(f"wrote {readme.name}")
print(json.dumps({"ok": True}))
