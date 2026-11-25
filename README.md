# QED-Net: Hybrid Quantum–Classical ML Platform for Early Disease Detection

> ⚠️ **RESEARCH ONLY**: This platform is a research and educational artifact. It is not a medical device and must not be used for real patient care.

QED-Net is an **experimental research codebase** for biomedical classification studies. It contains quantum and classical estimators, preprocessing and evaluation modules, a FastAPI research API, and a Vite dashboard. These components are not all wired into one validated production workflow; this repository is not production-ready and must not be used for clinical decisions.

## Conceptual research workflow

```
Raw Data → Validation → Preprocessing → Compression → Encoding
                                                    ↓
                                              Model (Quantum/Classical)
                                                    ↓
                                              Post-processing → Calibration
                                                    ↓
                                              Uncertainty (N5: Shot Noise)
                                                    ↓
                                              Conformal Prediction (F6)
                                                    ↓
                                              Quantum Second-Opinion Cascade (F1)
                                                    ↓
                                              Final Result
```

## What is implemented

- The `/research` API exposes the supported classical models, QSVM, VQC, and data-reupload classifier, plus dataset, experiment, prediction, explanation, benchmark, and certificate routes.
- The Vite dashboard calls the `/research` API. It is the active dashboard.
- The Python package contains additional experimental components for preprocessing, evaluation, explainability, calibration, uncertainty, conformal prediction, and certificates. Presence in the package does not mean every component is wired into the API or independently validated.
- The legacy Streamlit page is informational only. It does not train models or return predictions or measured results.
- `qednet.models.factory` and the API have different model surfaces. In particular, the API intentionally does not expose the hybrid mock as a quantum model.

The test suite exercises library and API behavior, but passing tests do not establish clinical validity, security certification, or production readiness.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Chatur7x/QEDNET.git
cd QEDNET

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Configuration

Runnable example configurations are in `configs/experiments/`. The API
research runner and the older `qednet.train` configuration classes use
different schemas. Use the format expected by the command you run;
`configs/experiments/smoke_tiny.yaml` is the compact example below.

### Training

```bash
# Run a real benchmark using the repository script and an existing config
python scripts/reproduce.py --config configs/experiments/smoke_tiny.yaml --compare logistic_regression --seeds 42 --folds 2
```

The experiment script needs the dataset files for the selected dataset. Run `python scripts/download_datasets.py` to fetch the public files supported by that script. The default breast-cancer loader uses scikit-learn's bundled dataset.

### API Server

```bash
# Start API server
python -m api.main

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Research API endpoints use the `/research` prefix, for example `/research/health`, `/research/models`, `/research/train`, `/research/predict`, `/research/explain`, `/research/benchmark/{id}`, and `/research/certificate/{id}`. The root `/health`, `/models`, and `/predict` routes are a separate legacy serving interface; `/live` checks process liveness and `/ready` checks whether a model is available for serving. The legacy `/experiments/{id}` route returns `501` because experiment lookup is not implemented in that interface.

### Dashboard

```bash
# Run the connected frontend; the Streamlit page is informational only.
cd frontend
npm install
npm run dev
```

### Docker

```bash
# Build and run the local demo stack
docker compose up --build

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:5173
```

This Compose setup is a local research demo. It does not provide authentication, tenant isolation, clinical controls, or a production job queue.

## Project Structure

```
qednet/
├── __init__.py
├── data/              # Data loading
├── preprocess/        # Preprocessing pipelines (leakage-free)
├── encoding/          # Quantum encodings (angle, amplitude, basis)
├── models/
│   ├── __init__.py
│   ├── base.py        # BaseEstimator, BaseTransformer interfaces
│   ├── factory.py     # ModelFactory
│   ├── quantum/       # VQC, QSVM, Reupload, Hybrid QNN
│   └── classical/     # XGBoost, LightGBM, Logistic, MLP
├── train/
│   ├── config.py      # ExperimentConfig, DatasetConfig, ModelConfig
│   └── runner.py      # TrainingRunner
├── evaluation/
│   └── metrics.py     # AUROC, AUPRC, Sens@Spec95, MCC, Brier, ECE, bootstrap CI, DeLong
├── explain/
│   ├── shap_explainer.py
│   └── quantum_shap.py
├── deploy/
│   ├── serving.py     # ServingPipeline
│   ├── cascade.py     # QuantumSecondOpinionCascade (F1)
│   ├── conformal.py   # ConformalPredictor (F6)
│   └── uncertainty.py # UncertaintyQuantifier (N5)
api/
│   └── main.py        # FastAPI app
dashboard/
│   └── app.py         # Streamlit dashboard
tests/
├── conftest.py
├── test_config.py
├── test_model_interface.py
├── test_pipeline.py
├── test_deploy.py
└── test_api.py
configs/
├── default.yaml
docker/
├── Dockerfile
├── docker-compose.yml
└── .env.example
.github/workflows/
└── ci.yml
```

## Interface Contract

All models implement the scikit-learn compatible interface:

```python
model.fit(X, y)           # X: (n_samples, n_features), y: (n_samples,)
model.predict_proba(X)    # Returns (n_samples, n_classes)
model.predict(X)          # Returns (n_samples,)
```

All transformers implement:

```python
transformer.fit(X, y=None)
transformer.transform(X)
transformer.fit_transform(X, y=None)
```

## Quantum Second-Opinion Cascade (F1)

The cascade routes ambiguous classical predictions to a quantum model:

```python
from qednet.deploy.cascade import QuantumSecondOpinionCascade

cascade = QuantumSecondOpinionCascade(
    classical_model=xgb_model,
    quantum_model=vqc_model,
    ambiguity_threshold=(0.35, 0.65),  # Configurable
)

result = cascade.predict(X)
# result.routing: "classical" or "quantum_second_opinion"
# result.reason: "high_confidence" or "ambiguous_probability"
```

## Conformal Prediction (F6)

Model-agnostic conformal prediction with finite-sample coverage:

```python
from qednet.deploy.conformal import ConformalPredictor

predictor = ConformalPredictor(model, alpha=0.10)
predictor.calibrate(X_cal, y_cal)

result = predictor.predict(X_test)
# result.prediction_set: [0], [1], or [0, 1]
# result.abstained: True if prediction_set == [0, 1]
```

## Shot-Noise Uncertainty (N5)

Properly separates shot/sampling uncertainty from model uncertainty:

```python
from qednet.deploy.uncertainty import UncertaintyQuantifier

quantifier = UncertaintyQuantifier(shots=1024)
result = quantifier.quantify(probability=0.72)
# result.variance = p(1-p)/S
# result.std = sqrt(variance)
# result.ci_lower, result.ci_upper = 95% CI

# For epistemic uncertainty, provide ensemble predictions
result = quantifier.quantify(0.72, ensemble_probs=np.array([...]))
```

## Leakage Guard

All preprocessing transformers are fitted only within CV folds:

```python
from qednet.preprocess.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(compression="pca", n_components=8)
# In CV: pipeline.fit(X_train, y_train).transform(X_test)
# NEVER: pipeline.fit(X_full, y_full).transform(X_test)
```

## Reproducibility

- Configuration hashing: `config.compute_hash()` returns deterministic SHA256
- Pinned dependencies in `requirements.txt` and `pyproject.toml`
- MLflow tracking for all experiments
- Random seeds controlled throughout pipeline

## Benchmark snapshot

The v2 exploratory benchmark compared four classical baselines with three PennyLane quantum classifiers (QSVM, VQC, and data re-upload) on 80-row stratified samples from four biomedical datasets. It used three-fold cross-validation, fold-local preprocessing, and fit seeds 42, 1729, and 2024. The quantum models ran on the analytic `default.qubit` simulator with four qubits; no quantum hardware was used.

On these samples, classical models had the higher mean AUC on breast cancer, heart disease, and diabetes. Quantum AUC was lower on those datasets. The paired p-values are exploratory because the same folds were reused across seeds; they are not evidence of general model superiority.

Parkinson's random-row folds leaked subject identity across train and test. On the 80-row sample, setting A RandomForest mean AUC was 0.8634920634920635 (sample SD 0.09222733607382565) on random folds and 0.7067460317460317 (sample SD 0.2754061883729273) on subject-grouped folds. The grouped folds had zero subject overlap. These are sample-specific exploratory results with substantial fold variation, not a claim of subject-independent performance from the random-row evaluation.

See [the full v2 report](results-v2/RESULTS_FINAL.md) for per-model tables, paired comparisons, methods, and limitations. The supporting run records and audits are in [`results-v2/`](results-v2/).

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
pytest tests/ -m "leakage" -v
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check qednet tests api dashboard
ruff format --check qednet tests api dashboard

# Type checking
mypy qednet

# Pre-commit hooks
pre-commit install
```

## Branch Workflow

- `main`: Stable releases only
- `member-c-platform`: Member C development branch
- Feature branches: `feat/description`, `fix/description`, `docs/description`

## License

MIT License - See LICENSE for details.

## Citation

If you use QED-Net in your research, please cite:

```bibtex
@software{qednet2024,
  title = {QED-Net: A Hybrid Quantum–Classical Machine Learning Platform for Early Disease Detection},
  author = {QED-Net Team},
  year = {2024},
  url = {https://github.com/Chatur7x/QEDNET}
}
```

## Disclaimer

**This platform is a research and educational artifact. It is not a medical device and must not be used for real patient care.** All predictions are "research predictions" or "benchmark results" and carry no clinical validity.
The root API accepts `QEDNET_CORS_ORIGINS` as a comma-separated list of allowed browser origins. The default permits only the local Vite development origins. CORS is a browser policy, not authentication; do not expose this demo API to the public internet or send patient data to it.

For a non-local frontend build, set `VITE_API_BASE_URL` to the API origin before building the Vite dashboard. Never put API secrets in Vite variables; they are embedded in browser JavaScript.
