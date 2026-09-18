# Research Report — exp_pima_diabetes

**Dataset:** pima_diabetes (6912 samples, 8 features)  
**Protocol:** 5-fold outer CV x 3-fold inner CV, seeds [42, 7, 2026]  
**Compression:** pca -> 8 dims  
**Configuration hash:** `15ed6bf597f3`

> RESEARCH AND EDUCATIONAL USE ONLY — not a medical device, not clinically validated, not for diagnosis.

## Benchmark results (mean +/- std over folds x seeds)

| Model | Family | AUROC | AUPRC | MCC | Brier | Params |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | classical | 0.8370 +/- 0.0254 | 0.7198 | 0.4788 | 0.1565 | 9 |
| matched_mlp | classical | 0.8281 +/- 0.0244 | 0.7171 | 0.4758 | 0.1659 | 41 |
| qsvm | quantum | 0.8248 +/- 0.0244 | 0.7022 | 0.4528 | 0.1643 | 347 |
| rbf_svm | classical | 0.8247 +/- 0.0300 | 0.7078 | 0.4408 | 0.1621 | 376 |
| vqc | quantum | 0.8215 +/- 0.0298 | 0.6892 | 0.4539 | 0.1665 | 25 |
| random_forest | classical | 0.8179 +/- 0.0293 | 0.6921 | 0.4432 | 0.1643 | 63384 |
| xgboost | classical | 0.7950 +/- 0.0218 | 0.6478 | 0.3983 | 0.1904 | 3824 |
| reupload | quantum | 0.7941 +/- 0.0322 | 0.6419 | 0.3927 | 0.1817 | 73 |
| hybrid_qnn | quantum | 0.7793 +/- 0.0493 | 0.6291 | 0.4227 | 0.1895 | 98 |

## Second-opinion cascade (F12)

- Routed to quantum: 506/2304 (22.0%)  
- AUROC — screening: 0.8357, quantum: 0.8209, cascade: 0.8314

## Bottleneck honesty meter (F10)

- Compressed-space baseline AUROC: 0.8323  
- Quantum AUROC: 0.8203  
- Gap: -0.012016 — simple classical model on the same compressed representation outperforms the quantum layer

## Quantum Advantage Certificate (F9)

**Verdict: No Demonstrated Advantage**  
best quantum AUROC 0.8248 <= best classical 0.8370; classical models remain competitive or superior on this dataset

Full certificate: `experiments/runs/exp_pima_diabetes/certificate.json`

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
  "config_hash": "15ed6bf597f3",
  "created_unix": 1789676572
}
```

Reproduce with: `python scripts/reproduce.py --config configs/experiments/pima_diabetes.yaml`