# Research Report — exp_heart_disease

**Dataset:** heart_disease (2727 samples, 13 features)  
**Protocol:** 5-fold outer CV x 3-fold inner CV, seeds [42, 7, 2026]  
**Compression:** pca -> 8 dims  
**Configuration hash:** `0f8f65c30665`

> RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.

## Benchmark results (mean +/- std over folds x seeds)

| Model | Family | AUROC | AUPRC | MCC | Brier | Params |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | classical | 0.9041 +/- 0.0371 | 0.8985 | 0.6802 | 0.1222 | 9 |
| qsvm | quantum | 0.8935 +/- 0.0340 | 0.8848 | 0.6387 | 0.1310 | 134 |
| rbf_svm | classical | 0.8935 +/- 0.0376 | 0.8856 | 0.6252 | 0.1323 | 165 |
| random_forest | classical | 0.8922 +/- 0.0355 | 0.8752 | 0.6306 | 0.1343 | 11870 |
| matched_mlp | classical | 0.8916 +/- 0.0432 | 0.8904 | 0.6613 | 0.1438 | 41 |
| vqc | quantum | 0.8843 +/- 0.0402 | 0.8880 | 0.6221 | 0.1351 | 25 |
| xgboost | classical | 0.8743 +/- 0.0297 | 0.8564 | 0.6022 | 0.1553 | 4172 |
| reupload | quantum | 0.8548 +/- 0.0350 | 0.8430 | 0.5454 | 0.1567 | 73 |
| hybrid_qnn | quantum | 0.8313 +/- 0.0510 | 0.7906 | 0.5261 | 0.1890 | 98 |

## Second-opinion cascade (F12)

- Routed to quantum: 132/909 (14.5%)  
- AUROC — screening: 0.9052, quantum: 0.8904, cascade: 0.8994

## Bottleneck honesty meter (F10)

- Compressed-space baseline AUROC: 0.9232  
- Quantum AUROC: 0.9305  
- Gap: 0.007289 — most predictive performance comes from the classical compression; quantum layer adds no measurable value

## Quantum Advantage Certificate (F9)

**Verdict: No Demonstrated Advantage**  
best quantum AUROC 0.8935 <= best classical 0.9041; classical models remain competitive or superior on this dataset

Full certificate: `experiments/runs/exp_heart_disease/certificate.json`

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
  "config_hash": "0f8f65c30665",
  "created_unix": 1789674143
}
```

Reproduce with: `python scripts/reproduce.py --config configs/experiments/heart_disease.yaml`