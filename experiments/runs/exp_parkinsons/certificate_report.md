# QED-Net 2.0 Quantum Advantage Certificate

**Experiment:** `exp_parkinsons`  
**Dataset:** parkinsons  
**Quantum model:** reupload  
**Configuration hash:** `f252624203ee`  
**Seeds:** [42, 7, 2026]

## Verdict

**No Demonstrated Advantage** — best quantum AUROC 0.9373 <= best classical 0.9499; classical models remain competitive or superior on this dataset

## Performance (mean AUROC over folds/seeds)

| Model | Family | AUROC | 95% CI |
| --- | --- | --- | --- |
| xgboost | classical | 0.9499 | [0.9217, 0.9646] |
| random_forest | classical | 0.9477 | [0.9232, 0.9601] |
| rbf_svm | classical | 0.9445 | [0.9165, 0.9570] |
| reupload | quantum | 0.9373 | [0.9075, 0.9499] |
| qsvm | quantum | 0.9264 | [0.8899, 0.9371] |
| hybrid_qnn | quantum | 0.9186 | [0.8845, 0.9389] |
| vqc | quantum | 0.9129 | [0.8732, 0.9230] |
| logistic_regression | classical | 0.8946 | [0.8548, 0.9111] |
| matched_mlp | classical | 0.8834 | [0.8053, 0.8735] |

## Statistical evidence

```json
{
  "delong_test": {
    "delong_available": true,
    "auc1": 0.930036,
    "auc2": 0.944082,
    "auc_diff": -0.014046,
    "z": -1.210137,
    "p_value": 0.22622649090148883,
    "significant_005": false
  },
  "corrected_cv_test": {
    "available": true,
    "mean_diff": -0.012516,
    "t": -0.327306,
    "p_value": 0.7482790732631017,
    "significant_005": false,
    "correction": "Nadeau-Bengio corrected resampled t-test",
    "k_folds": 5,
    "n_train": 156,
    "n_test": 39
  },
  "bootstrap": {
    "logistic_regression": {
      "metric": "auroc",
      "point": 0.8839600655076846,
      "ci_low": 0.8547616376316303,
      "ci_high": 0.9110568785643687,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "random_forest": {
      "metric": "auroc",
      "point": 0.9420351473922902,
      "ci_low": 0.9231945867971435,
      "ci_high": 0.960138482506668,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "xgboost": {
      "metric": "auroc",
      "point": 0.9440822625346434,
      "ci_low": 0.9216765736278684,
      "ci_high": 0.9646437377178042,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "rbf_svm": {
      "metric": "auroc",
      "point": 0.9378464348702443,
      "ci_low": 0.916506524041958,
      "ci_high": 0.9570012913587906,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "matched_mlp": {
      "metric": "auroc",
      "point": 0.8404509952129,
      "ci_low": 0.8053007194768073,
      "ci_high": 0.8735024530409048,
     
```

## Entanglement ablation

```json
{
  "available": true,
  "model": "vqc",
  "entangled": {
    "auroc": 0.9257575757575758,
    "n_parameters": 25,
    "circuit_depth": 6,
    "entangling_gates": 21
  },
  "non_entangled_parameter_matched": {
    "auroc": 0.7863636363636365,
    "n_parameters": 25,
    "circuit_depth": 4,
    "entangling_gates": 0
  },
  "auroc_delta_entangled_minus_nonentangled": 0.139394,
  "capacity_preserved": true,
  "entanglement_contribution": "positive contribution",
  "interpretation": "Same-parameter-count comparison; differences within noise are reported as no measurable contribution. The result is whatever the experiment actually shows."
}
```

## Dequantization

```json
{
  "quantum_kernel_auroc": 0.859091,
  "rff": {
    "method": "Random Fourier Features (surrogate of quantum kernel distances)",
    "n_components": 200,
    "gamma_median_heuristic": 0.553489,
    "auroc": 0.9515151515151515,
    "note": "classical kernel approximation evaluated on identical test folds"
  },
  "nystrom": {
    "method": "Nystrom low-rank approximation of quantum kernel",
    "n_components": 100,
    "auroc": 0.8560606060606061,
    "kernel_relative_reconstruction_error": 0.044515,
    "note": "classical low-rank surrogate evaluated on identical test folds"
  }
}
```

## Bottleneck honesty meter

```json
{
  "compressed_space_baseline": {
    "model": "logistic regression on squashed compressed features",
    "auroc": 0.8727272727272728,
    "n_dims": 8
  },
  "quantum_score": {
    "model": "vqc",
    "auroc": 0.9257575757575758
  },
  "performance_gap_quantum_minus_baseline": 0.05303,
  "reading": "quantum layer adds measurable value beyond the classical compressed representation",
  "purpose": "measures how much predictive performance comes from classical feature compression versus the quantum layer"
}
```

## Resources

- Qubits: 8  
- Circuit depth: 10  
- Quantum parameters: 73  
- Matched MLP parameters: 41

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

---

Research and educational platform — not a medical device, not clinically validated, not for diagnosis.
