# QED-Net 2.0 — bundled open datasets

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
