# Research Report — exp_breast_cancer

**Dataset:** breast_cancer (5121 samples, 30 features)  
**Protocol:** 5-fold outer CV x 3-fold inner CV, seeds [42, 7, 2026]  
**Compression:** pca -> 8 dims  
**Configuration hash:** `d540d564dc27`

> RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.

## Benchmark results (mean +/- std over folds x seeds)

| Model | Family | AUROC | AUPRC | MCC | Brier | Params |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | classical | 0.9943 +/- 0.0063 | 0.9935 | 0.9408 | 0.0249 | 9 |
| qsvm | quantum | 0.9933 +/- 0.0055 | 0.9905 | 0.9211 | 0.0269 | 126 |
| rbf_svm | classical | 0.9931 +/- 0.0059 | 0.9900 | 0.9149 | 0.0292 | 146 |
| matched_mlp | classical | 0.9928 +/- 0.0079 | 0.9917 | 0.9378 | 0.0503 | 41 |
| xgboost | classical | 0.9922 +/- 0.0075 | 0.9903 | 0.9303 | 0.0253 | 2938 |
| random_forest | classical | 0.9883 +/- 0.0093 | 0.9847 | 0.9020 | 0.0429 | 12016 |
| hybrid_qnn | quantum | 0.9874 +/- 0.0136 | 0.9884 | 0.9396 | 0.0239 | 98 |
| vqc | quantum | 0.9826 +/- 0.0109 | 0.9798 | 0.8755 | 0.0438 | 25 |
| reupload | quantum | 0.9762 +/- 0.0153 | 0.9721 | 0.8445 | 0.0553 | 73 |

## Second-opinion cascade (F12)

- Routed to quantum: 72/1707 (4.2%)  
- AUROC — screening: 0.9926, quantum: 0.9934, cascade: 0.9923

## Bottleneck honesty meter (F10)

- Compressed-space baseline AUROC: 0.9971  
- Quantum AUROC: 0.9974  
- Gap: 0.000292 — most predictive performance comes from the classical compression; quantum layer adds no measurable value

## Quantum Advantage Certificate (F9)

**Verdict: No Demonstrated Advantage**  
best quantum AUROC 0.9933 <= best classical 0.9943; classical models remain competitive or superior on this dataset

Full certificate: `experiments/runs/exp_breast_cancer/certificate.json`

## Reproducibility

```json
{
  "python_version": "3.12.14",
  "platform": "Linux-5.10.134-013.15.kangaroo.al8.x86_64-x86_64-with-glibc2.41",
  "package_versions": {
    "numpy": "2.1.3",
    "scikit-learn": "1.5.2",
    "pennylane": "0.45.1",
    "xgboost": "2.1.3",
    "shap": "0.52.0"
  },
  "qednet_version": "2.0.0",
  "config_hash": "d540d564dc27",
  "created_unix": 1789674252
}
```

Reproduce with: `python scripts/reproduce.py --config configs/experiments/breast_cancer.yaml`