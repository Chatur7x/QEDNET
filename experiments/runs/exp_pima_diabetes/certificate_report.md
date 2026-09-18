# QED-Net 2.0 Quantum Advantage Certificate

**Experiment:** `exp_pima_diabetes`  
**Dataset:** pima_diabetes  
**Quantum model:** qsvm  
**Configuration hash:** `15ed6bf597f3`  
**Seeds:** [42, 7, 2026]

## Verdict

**No Demonstrated Advantage** — best quantum AUROC 0.8248 <= best classical 0.8370; classical models remain competitive or superior on this dataset

## Performance (mean AUROC over folds/seeds)

| Model | Family | AUROC | 95% CI |
| --- | --- | --- | --- |
| logistic_regression | classical | 0.8370 | [0.8195, 0.8520] |
| matched_mlp | classical | 0.8281 | [0.7966, 0.8319] |
| qsvm | quantum | 0.8248 | [0.8036, 0.8375] |
| rbf_svm | classical | 0.8247 | [0.8061, 0.8404] |
| vqc | quantum | 0.8215 | [0.7991, 0.8336] |
| random_forest | classical | 0.8179 | [0.7992, 0.8333] |
| xgboost | classical | 0.7950 | [0.7755, 0.8107] |
| reupload | quantum | 0.7941 | [0.7602, 0.7978] |
| hybrid_qnn | quantum | 0.7793 | [0.7577, 0.7971] |

## Statistical evidence

```json
{
  "delong_test": {
    "delong_available": true,
    "auc1": 0.820883,
    "auc2": 0.835681,
    "auc_diff": -0.014798,
    "z": -3.293341,
    "p_value": 0.000990042688950421,
    "significant_005": true
  },
  "corrected_cv_test": {
    "available": true,
    "mean_diff": -0.012252,
    "t": -1.155468,
    "p_value": 0.26723880790093846,
    "significant_005": false,
    "correction": "Nadeau-Bengio corrected resampled t-test",
    "k_folds": 5,
    "n_train": 615,
    "n_test": 153
  },
  "bootstrap": {
    "logistic_regression": {
      "metric": "auroc",
      "point": 0.8356807628524047,
      "ci_low": 0.8195197336652452,
      "ci_high": 0.8519600874562342,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "random_forest": {
      "metric": "auroc",
      "point": 0.8164780265339968,
      "ci_low": 0.7992456893524902,
      "ci_high": 0.8333247273683417,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "xgboost": {
      "metric": "auroc",
      "point": 0.7930762852404644,
      "ci_low": 0.7754814517281322,
      "ci_high": 0.8106859482919414,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "rbf_svm": {
      "metric": "auroc",
      "point": 0.8235920398009949,
      "ci_low": 0.8060832082495794,
      "ci_high": 0.8403557012485382,
      "n_resamples": 2000,
      "alpha": 0.05
    },
    "matched_mlp": {
      "metric": "auroc",
      "point": 0.8148250414593698,
      "ci_low": 0.7966178499870417,
      "ci_high": 0.8319325417599055
```

## Entanglement ablation

```json
{
  "available": true,
  "model": "vqc",
  "entangled": {
    "auroc": 0.8203292181069958,
    "n_parameters": 25,
    "circuit_depth": 6,
    "entangling_gates": 21
  },
  "non_entangled_parameter_matched": {
    "auroc": 0.7860905349794238,
    "n_parameters": 25,
    "circuit_depth": 4,
    "entangling_gates": 0
  },
  "auroc_delta_entangled_minus_nonentangled": 0.034239,
  "capacity_preserved": true,
  "entanglement_contribution": "positive contribution",
  "interpretation": "Same-parameter-count comparison; differences within noise are reported as no measurable contribution. The result is whatever the experiment actually shows."
}
```

## Dequantization

```json
{
  "quantum_kernel_auroc": 0.821481,
  "rff": {
    "method": "Random Fourier Features (surrogate of quantum kernel distances)",
    "n_components": 200,
    "gamma_median_heuristic": 0.634922,
    "auroc": 0.7546913580246913,
    "note": "classical kernel approximation evaluated on identical test folds"
  },
  "nystrom": {
    "method": "Nystrom low-rank approximation of quantum kernel",
    "n_components": 100,
    "auroc": 0.822962962962963,
    "kernel_relative_reconstruction_error": 0.039444,
    "note": "classical low-rank surrogate evaluated on identical test folds"
  }
}
```

## Bottleneck honesty meter

```json
{
  "compressed_space_baseline": {
    "model": "logistic regression on squashed compressed features",
    "auroc": 0.8323456790123457,
    "n_dims": 8
  },
  "quantum_score": {
    "model": "vqc",
    "auroc": 0.8203292181069958
  },
  "performance_gap_quantum_minus_baseline": -0.012016,
  "reading": "simple classical model on the same compressed representation outperforms the quantum layer",
  "purpose": "measures how much predictive performance comes from classical feature compression versus the quantum layer"
}
```

## Resources

- Qubits: 8  
- Circuit depth: 8  
- Quantum parameters: 347  
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
  "config_hash": "15ed6bf597f3",
  "created_unix": 1789676572
}
```

---

Research and educational platform — not a medical device, not clinically validated, not for diagnosis.
