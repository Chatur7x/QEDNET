# Research Report — exp_parkinsons

**Dataset:** parkinsons (1755 samples, 22 features)  
**Protocol:** 5-fold outer CV x 3-fold inner CV, seeds [42, 7, 2026]  
**Compression:** pca -> 8 dims  
**Configuration hash:** `f252624203ee`

> RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.

## Benchmark results (mean +/- std over folds x seeds)

| Model | Family | AUROC | AUPRC | MCC | Brier | Params |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | classical | 0.9499 +/- 0.0325 | 0.9805 | 0.7304 | 0.0752 | 1968 |
| random_forest | classical | 0.9477 +/- 0.0291 | 0.9828 | 0.6862 | 0.0851 | 5830 |
| rbf_svm | classical | 0.9445 +/- 0.0315 | 0.9811 | 0.6977 | 0.0801 | 70 |
| reupload | quantum | 0.9373 +/- 0.0454 | 0.9793 | 0.6759 | 0.0910 | 73 |
| qsvm | quantum | 0.9264 +/- 0.0385 | 0.9741 | 0.6593 | 0.0915 | 82 |
| hybrid_qnn | quantum | 0.9186 +/- 0.0563 | 0.9674 | 0.6860 | 0.1009 | 98 |
| vqc | quantum | 0.9129 +/- 0.0478 | 0.9710 | 0.5675 | 0.1093 | 25 |
| logistic_regression | classical | 0.8946 +/- 0.0348 | 0.9645 | 0.5750 | 0.1095 | 9 |
| matched_mlp | classical | 0.8834 +/- 0.0874 | 0.9590 | 0.4904 | 0.1398 | 41 |

## Second-opinion cascade (F12)

- Routed to quantum: 50/585 (8.5%)  
- AUROC — screening: 0.8840, quantum: 0.9144, cascade: 0.8905

## Bottleneck honesty meter (F10)

- Compressed-space baseline AUROC: 0.8727  
- Quantum AUROC: 0.9258  
- Gap: 0.05303 — quantum layer adds measurable value beyond the classical compressed representation

## Quantum Advantage Certificate (F9)

**Verdict: No Demonstrated Advantage**  
best quantum AUROC 0.9373 <= best classical 0.9499; classical models remain competitive or superior on this dataset

Full certificate: `experiments/runs/exp_parkinsons/certificate.json`

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
  "config_hash": "f252624203ee",
  "created_unix": 1789673802
}
```

Reproduce with: `python scripts/reproduce.py --config configs/experiments/parkinsons.yaml`