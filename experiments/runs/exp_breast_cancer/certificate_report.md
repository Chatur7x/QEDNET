# QED-Net 2.0 Quantum Advantage Certificate

**Experiment:** `exp_breast_cancer`  
**Dataset:** breast_cancer  
**Quantum model:** qsvm  
**Configuration hash:** `d540d564dc27`  
**Seeds:** [42, 7, 2026]

## Verdict

**No Demonstrated Advantage** — best quantum AUROC 0.9933 <= best classical 0.9943; classical models remain competitive or superior on this dataset

## Performance (mean AUROC over folds/seeds)

| Model | Family | AUROC | 95% CI |
| --- | --- | --- | --- |
| logistic_regression | classical | 0.9943 | [0.9877, 0.9964] |
| qsvm | quantum | 0.9933 | [0.9903, 0.9959] |
| rbf_svm | classical | 0.9931 | [0.9902, 0.9957] |
| matched_mlp | classical | 0.9928 | [0.9823, 0.9944] |
| xgboost | classical | 0.9922 | [0.9877, 0.9953] |
| random_forest | classical | 0.9883 | [0.9832, 0.9920] |
| hybrid_qnn | quantum | 0.9874 | [0.9779, 0.9926] |
| vqc | quantum | 0.9826 | [0.9733, 0.9874] |
| reupload | quantum | 0.9762 | [0.9629, 0.9794] |

## Statistical evidence

```json
{
  "delong_test": {
    "delong_available": true,
    "auc1": 0.993354,
    "auc2": 0.99256,
    "auc_diff": 0.000794,
    "z": 0.519322,
    "p_value": 0.6035364839591166,
    "significant_005": false
  },
  "corrected_cv_test": {
    "available": true,
    "mean_diff": -0.001052,
    "t": -0.362533,
    "p_value": 0.722368305114639,
    "significant_005": false,
    "correction": "Nadeau-Bengio corrected resampled t-test",
    "k_folds": 5,
    "n_train": 456,
    "n_test": 113
  },
  "bootstrap": {
    "logistic_regression": {
      "metric": "auroc",
      "point": 0.9925597073210835,
      "ci_low": 0.9877210680832356,
      "ci_high": 0.9963823506093671,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "random_forest": {
      "metric": "auroc",
      "point": 0.9879418224312787,
      "ci_low": 0.9831704795382021,
      "ci_high": 0.9920445595676118,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "xgboost": {
      "metric": "auroc",
      "point": 0.9918785124112538,
      "ci_low": 0.9876876181645746,
      "ci_high": 0.9952765668846163,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "rbf_svm": {
      "metric": "auroc",
      "point": 0.9932130084738298,
      "ci_low": 0.9902340995702038,
      "ci_high": 0.9957413161634895,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "matched_mlp": {
      "metric": "auroc",
      "point": 0.9888204757794103,
      "ci_low": 0.9823409896532337,
      "ci_high": 0.9944402986900094,
    
```

## Entanglement ablation

```json
{
  "available": true,
  "model": "vqc",
  "entangled": {
    "auroc": 0.9973714953271028,
    "n_parameters": 25,
    "circuit_depth": 6,
    "entangling_gates": 21
  },
  "non_entangled_parameter_matched": {
    "auroc": 0.9851051401869159,
    "n_parameters": 25,
    "circuit_depth": 4,
    "entangling_gates": 0
  },
  "auroc_delta_entangled_minus_nonentangled": 0.012266,
  "capacity_preserved": true,
  "entanglement_contribution": "positive contribution",
  "interpretation": "Same-parameter-count comparison; differences within noise are reported as no measurable contribution. The result is whatever the experiment actually shows."
}
```

## Dequantization

```json
{
  "quantum_kernel_auroc": 0.992553,
  "rff": {
    "method": "Random Fourier Features (surrogate of quantum kernel distances)",
    "n_components": 200,
    "gamma_median_heuristic": 0.522463,
    "auroc": 0.9918224299065421,
    "note": "classical kernel approximation evaluated on identical test folds"
  },
  "nystrom": {
    "method": "Nystrom low-rank approximation of quantum kernel",
    "n_components": 100,
    "auroc": 0.9919684579439252,
    "kernel_relative_reconstruction_error": 0.094724,
    "note": "classical low-rank surrogate evaluated on identical test folds"
  }
}
```

## Bottleneck honesty meter

```json
{
  "compressed_space_baseline": {
    "model": "logistic regression on squashed compressed features",
    "auroc": 0.9970794392523364,
    "n_dims": 8
  },
  "quantum_score": {
    "model": "vqc",
    "auroc": 0.9973714953271028
  },
  "performance_gap_quantum_minus_baseline": 0.000292,
  "reading": "most predictive performance comes from the classical compression; quantum layer adds no measurable value",
  "purpose": "measures how much predictive performance comes from classical feature compression versus the quantum layer"
}
```

## Resources

- Qubits: 8  
- Circuit depth: 8  
- Quantum parameters: 126  
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
  "config_hash": "d540d564dc27",
  "created_unix": 1789674252
}
```

---

Research and educational platform — not a medical device, not clinically validated, not for diagnosis.
