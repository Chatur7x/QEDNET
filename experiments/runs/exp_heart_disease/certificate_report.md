# QED-Net 2.0 Quantum Advantage Certificate

**Experiment:** `exp_heart_disease`  
**Dataset:** heart_disease  
**Quantum model:** qsvm  
**Configuration hash:** `0f8f65c30665`  
**Seeds:** [42, 7, 2026]

## Verdict

**No Demonstrated Advantage** — best quantum AUROC 0.8935 <= best classical 0.9041; classical models remain competitive or superior on this dataset

## Performance (mean AUROC over folds/seeds)

| Model | Family | AUROC | 95% CI |
| --- | --- | --- | --- |
| logistic_regression | classical | 0.9041 | [0.8850, 0.9243] |
| qsvm | quantum | 0.8935 | [0.8673, 0.9115] |
| rbf_svm | classical | 0.8935 | [0.8673, 0.9102] |
| random_forest | classical | 0.8922 | [0.8662, 0.9095] |
| matched_mlp | classical | 0.8916 | [0.8551, 0.9014] |
| vqc | quantum | 0.8843 | [0.8552, 0.9021] |
| xgboost | classical | 0.8743 | [0.8472, 0.8941] |
| reupload | quantum | 0.8548 | [0.8270, 0.8764] |
| hybrid_qnn | quantum | 0.8313 | [0.7994, 0.8530] |

## Statistical evidence

```json
{
  "delong_test": {
    "delong_available": true,
    "auc1": 0.890356,
    "auc2": 0.905173,
    "auc_diff": -0.014817,
    "z": -2.963023,
    "p_value": 0.003046334766194869,
    "significant_005": true
  },
  "corrected_cv_test": {
    "available": true,
    "mean_diff": -0.010552,
    "t": -0.766044,
    "p_value": 0.45637268790546903,
    "significant_005": false,
    "correction": "Nadeau-Bengio corrected resampled t-test",
    "k_folds": 5,
    "n_train": 243,
    "n_test": 60
  },
  "bootstrap": {
    "logistic_regression": {
      "metric": "auroc",
      "point": 0.9051734222378194,
      "ci_low": 0.8849559846584241,
      "ci_high": 0.9242704546273222,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "random_forest": {
      "metric": "auroc",
      "point": 0.8893007545183366,
      "ci_low": 0.8662177351794662,
      "ci_high": 0.909475684211237,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "xgboost": {
      "metric": "auroc",
      "point": 0.87171238618861,
      "ci_low": 0.8471948929134919,
      "ci_high": 0.894084090298668,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "rbf_svm": {
      "metric": "auroc",
      "point": 0.8892885691446843,
      "ci_low": 0.8673313490675119,
      "ci_high": 0.9102185908319186,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "matched_mlp": {
      "metric": "auroc",
      "point": 0.8791113450702852,
      "ci_low": 0.8551022169563776,
      "ci_high": 0.901427834568863,
    
```

## Entanglement ablation

```json
{
  "available": true,
  "model": "vqc",
  "entangled": {
    "auroc": 0.9305150631681245,
    "n_parameters": 25,
    "circuit_depth": 6,
    "entangling_gates": 21
  },
  "non_entangled_parameter_matched": {
    "auroc": 0.9378036929057337,
    "n_parameters": 25,
    "circuit_depth": 4,
    "entangling_gates": 0
  },
  "auroc_delta_entangled_minus_nonentangled": -0.007289,
  "capacity_preserved": true,
  "entanglement_contribution": "no measurable contribution",
  "interpretation": "Same-parameter-count comparison; differences within noise are reported as no measurable contribution. The result is whatever the experiment actually shows."
}
```

## Dequantization

```json
{
  "quantum_kernel_auroc": 0.897473,
  "rff": {
    "method": "Random Fourier Features (surrogate of quantum kernel distances)",
    "n_components": 200,
    "gamma_median_heuristic": 0.561431,
    "auroc": 0.8979591836734694,
    "note": "classical kernel approximation evaluated on identical test folds"
  },
  "nystrom": {
    "method": "Nystrom low-rank approximation of quantum kernel",
    "n_components": 100,
    "auroc": 0.9062196307094266,
    "kernel_relative_reconstruction_error": 0.086626,
    "note": "classical low-rank surrogate evaluated on identical test folds"
  }
}
```

## Bottleneck honesty meter

```json
{
  "compressed_space_baseline": {
    "model": "logistic regression on squashed compressed features",
    "auroc": 0.9232264334305151,
    "n_dims": 8
  },
  "quantum_score": {
    "model": "vqc",
    "auroc": 0.9305150631681245
  },
  "performance_gap_quantum_minus_baseline": 0.007289,
  "reading": "most predictive performance comes from the classical compression; quantum layer adds no measurable value",
  "purpose": "measures how much predictive performance comes from classical feature compression versus the quantum layer"
}
```

## Resources

- Qubits: 8  
- Circuit depth: 8  
- Quantum parameters: 134  
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
  "config_hash": "0f8f65c30665",
  "created_unix": 1789674143
}
```

---

Research and educational platform — not a medical device, not clinically validated, not for diagnosis.
