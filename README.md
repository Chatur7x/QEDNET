# QED-Net: Hybrid Quantum–Classical ML Platform for Early Disease Detection

> ⚠️ **RESEARCH ONLY**: This platform is a research and educational artifact. It is not a medical device and must not be used for real patient care.

QED-Net is a hybrid quantum-classical machine learning platform for biomedical classification research. It provides a complete pipeline from data ingestion through model training, evaluation, uncertainty quantification, conformal prediction, and deployment.

## Architecture

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

## Key Features

- **Config-driven training** (D7): YAML-based experiment configuration with full reproducibility
- **Model factory** (D7): Support for VQC, QSVM, Reupload, Hybrid QNN, XGBoost, LightGBM, Logistic, MLP
- **MLflow integration** (D7): Experiment tracking with parameters, metrics, and artifacts
- **FastAPI backend** (D11): RESTful API with `/health`, `/models`, `/predict`, `/batch-predict`
- **Streamlit dashboard** (D12): Interactive research interface
- **Quantum Second-Opinion Cascade** (F1): Classical screener routes ambiguous cases to quantum model
- **Conformal prediction** (F6): Finite-sample coverage guarantees with abstention
- **Shot-noise uncertainty** (N5): Proper separation of sampling vs model uncertainty
- **Docker deployment** (D15): Multi-service deployment with docker-compose
- **Test suite** (D16): Unit, integration, and leakage guard tests
- **CI/CD** (D17): GitHub Actions with linting, type checking, and testing

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

Create an experiment configuration (or use the default):

```yaml
# configs/my_experiment.yaml
dataset:
  name: breast_cancer
  compression: pca
  target_dim: 8

model:
  type: vqc
  n_qubits: 8
  layers: 3
  ansatz: strongly_entangling
  encoding: angle
  optimizer: adam
  learning_rate: 0.01

training:
  outer_folds: 5
  inner_folds: 3
  seeds: 10
  shots: 1024

evaluation:
  metrics:
    - auroc
    - auprc
    - sens_at_spec95
    - mcc
    - brier
    - ece
  bootstrap_ci: 2000
  delong_test: true
  decision_curve: true
```

### Training

```bash
# Run training from config
python -m qednet.train.runner --config configs/my_experiment.yaml

# Or programmatically
from qednet.train.runner import run_experiment
result = run_experiment("configs/my_experiment.yaml", use_mock=True)
```

### API Server

```bash
# Start API server
python -m api.main

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health` - Health check
- `GET /models` - List loaded models
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch prediction

### Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py
```

### Docker

```bash
# Build and run with docker-compose
docker-compose -f docker/docker-compose.yml up --build

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

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
