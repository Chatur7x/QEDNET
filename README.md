# QED-Net 2.0

**Hybrid Quantum–Classical Machine Learning Platform for Early Disease Detection**

> **Research and educational platform.** QED-Net 2.0 is **not a medical device**, is **not clinically validated**, and is **not intended for diagnosis or real patient-care decisions**. All experiments use public, de-identified benchmark datasets.

---

## Overview

QED-Net 2.0 investigates where quantum machine learning provides measurable value for biomedical disease-risk classification — **without assuming quantum superiority**. It is a fully integrated research platform: a Python quantum/ML core, a FastAPI service for deployment, and a React research dashboard, all driven by real experiments with complete provenance.

The system combines:

```text
Classical Data Processing
        ↓
Feature Compression (PCA → 8 dims default)
        ↓
Quantum Machine Learning (QSVM · VQC · Data Re-Uploading · Hybrid QNN)
        ↓
Classical Post-Processing (calibration, uncertainty)
        ↓
Evaluation (nested CV, multi-seed, statistics)
        ↓
Explainability (SHAP, sensitivity, counterfactuals, circuit diagnostics)
        ↓
Quantum Advantage Certificate (from measured evidence only)
        ↓
Dashboard + Research Reports
```

**If quantum models perform worse, the platform reports that.** The certificate's verdict comes from deterministic rules applied to measured statistics — never hard-coded.

## The 12 main features (fixed scope)

| ID | Feature | Where |
|----|---------|-------|
| F1 | Biomedical Data Ingestion | `backend/qednet/data/ingestion.py` |
| F2 | Data Validation & Leakage Guard | `backend/qednet/data/validation.py`, `preprocess/guarded.py` |
| F3 | Preprocessing & Feature Compression | `backend/qednet/preprocess/pipeline.py` |
| F4 | Quantum Encoding Library | `backend/qednet/encoding/` (angle, ZZ, amplitude, re-upload) |
| F5 | Quantum Model Zoo | `backend/qednet/models/quantum/` (QSVM, VQC, re-upload, hybrid QNN) |
| F6 | Classical Baseline Suite | `backend/qednet/models/classical/` (LR, RF, XGBoost, RBF-SVM, matched MLP) |
| F7 | Automated Benchmarking Engine | `backend/qednet/evaluation/engine.py` |
| F8 | Explainability Module | `backend/qednet/explain/explainer.py` |
| F9 | Quantum Advantage Certificate | `backend/qednet/certificate/` |
| F10 | Bottleneck Honesty Meter | `backend/qednet/honesty/meter.py` |
| F11 | Uncertainty & Abstention | `backend/qednet/uncertainty/abstention.py` |
| F12 | Classical–Quantum Second-Opinion Cascade | `backend/qednet/cascade/cascade.py` |

## Architecture

```text
QED-Net 2.0 repository
├── backend/
│   ├── qednet/            # the Python research core (F1–F12)
│   │   ├── sim/           # batched statevector simulator (+ PennyLane reference)
│   │   ├── data/          # F1/F2 + built-in open dataset adapters
│   │   ├── preprocess/    # F3 + leakage-guarded transformers
│   │   ├── encoding/      # F4 quantum feature maps
│   │   ├── models/        # F5/F6 model zoo behind one contract
│   │   ├── evaluation/    # F7 metrics/statistics/calibration/engine
│   │   ├── explain/       # F8 SHAP/sensitivity/counterfactuals/diagnostics
│   │   ├── certificate/   # F9 certificate + dequantization + ablation
│   │   ├── honesty/       # F10 bottleneck honesty meter
│   │   ├── uncertainty/   # F11 conformal prediction + abstention
│   │   ├── cascade/       # F12 second-opinion cascade
│   │   ├── tracking/      # experiment store (+ optional MLflow mirror)
│   │   ├── train/         # experiment runner (F1→F12 orchestrator)
│   │   └── cli.py         # JSON-emitting CLI (the integration surface)
│   ├── api/               # FastAPI service (docker profile)
│   └── tests/             # unit / leakage / quantum / integration tests
├── src/                   # Next.js 16 research dashboard (React + TypeScript)
│   ├── app/api/           # bridges the Python core via the CLI + job system
│   ├── components/qednet/ # 8 dashboard views + design system + charts
│   └── lib/               # typed API client, Python bridge
├── configs/experiments/   # YAML experiment configurations
├── data/raw/              # bundled open datasets
├── experiments/runs/      # machine-readable results + certificates + reports
├── scripts/               # reproduce.py, dataset setup, experiment launcher
├── Dockerfile / docker-compose.yml
└── docs/                  # model cards, methodology
```

### Quantum execution backends (dual)

- **`numpy` (default)** — a batched statevector simulator written for this project: exact quantum-circuit simulation vectorised over whole mini-batches, differentiable end-to-end with `autograd` (reverse-mode through the statevector, the same mathematics as PennyLane's `backprop`). This makes the full nested-CV protocol computationally feasible.
- **`pennylane` (reference)** — the same circuits executed on PennyLane `default.qubit` / `lightning.qubit`.
- **Equivalence is enforced by tests**: `backend/tests/quantum/test_statevector.py` asserts the batched simulator reproduces PennyLane to machine precision (≤1e-12) on identical circuits, including angle and ZZ feature-map states, and that QSVM states match across backends.
- Exact statevector simulation is equivalent to infinite shots; resource reports record this honestly (`shots: null`).

## Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ (sandbox: 3.12) · TypeScript 5 |
| Quantum ML | PennyLane 0.45 (+ custom batched statevector engine, PL-verified) |
| Quantum secondary | Qiskit-compatible circuit descriptions; `qiskit` optional |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Data | pandas, Pandera-style validation, imbalanced-learn (SMOTE inside folds only) |
| Explainability | SHAP (KernelExplainer over the full pipeline), permutation, parameter-shift circuit gradients |
| Tracking | JSON experiment store + optional MLflow mirror |
| Backend API | FastAPI + Pydantic (docker profile); Next.js API routes bridge the CLI in the demo |
| Background jobs | Detached process jobs (demo); Celery + Redis (docker profile) |
| Storage | File store (demo); PostgreSQL + MinIO/S3 (docker profile) |
| Dashboard | Next.js 16 · React 19 · Tailwind CSS 4 · shadcn/ui · recharts · Framer Motion |
| Testing | pytest (+ Hypothesis available) |
| Packaging | Docker + docker-compose |

## Quick start (demo / sandbox)

The demo runs the dashboard (Next.js) against the Python core through the CLI bridge — no database required.

```bash
# 1. Python environment (Python 3.11+)
pip install -r requirements.txt

# 2. Fetch the open datasets (sklearn provides breast cancer)
mkdir -p data/raw && cd data/raw
curl -LO https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
curl -LO https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
curl -LO  https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv

# 3. Run the test suite (leakage guard, quantum equivalence, integration)
cd backend && python -m pytest && cd ..

# 4. Run an end-to-end experiment (writes experiments/runs/<id>/)
PYTHONPATH=backend python -m qednet.cli run-experiment \
  --config configs/experiments/breast_cancer.yaml --verbose

# 5. Dashboard
bun install && bun run dev     # http://localhost:3000
```

Every dashboard number comes from `experiments/runs/*/result.json` — real experiment output.

## Full deployment (docker-compose)

```bash
docker compose up --build
```

Starts FastAPI (:8000), Celery worker, Redis, PostgreSQL, and MinIO. See `docker-compose.yml`.

> The Docker profile is provided for deployment completeness and **has not been executed in the development sandbox** — verify on a Docker-equipped machine before relying on it (per the project's no-unverified-claims rule).

## Using the platform

### Run an experiment

```bash
PYTHONPATH=backend python -m qednet.cli run-experiment --config configs/experiments/heart_disease.yaml
```

Or from the dashboard: **Training → launch experiment** (background job with live progress).

Each run produces, under `experiments/runs/<exp_id>/`:

- `result.json` — every metric, fold, seed, curve, statistic (machine-readable)
- `certificate.json` + `certificate_report.md` — the Quantum Advantage Certificate
- `explainability.json` — SHAP global/local, permutation, sensitivity, counterfactuals, quantum diagnostics
- `report.md` — the research report rendered in the dashboard
- `pipeline.pkl`, `models.pkl`, `conformal.pkl` — deployment artifacts for prediction
- `config.yaml`, `progress.json`, plus the reproducibility record inside `result.json`

### Reproduce an experiment

```bash
python scripts/reproduce.py --config configs/experiments/breast_cancer.yaml          # demo protocol
python scripts/reproduce.py --config configs/experiments/breast_cancer.yaml --full   # 10 seeds
```

### Predict (cascade with abstention)

```bash
PYTHONPATH=backend python -m qednet.cli predict --id exp_breast_cancer \
  --input-json '[14.0, 20.5, ...]'    # features in dataset's original scale
```

Or in the dashboard: **Prediction** view → set features → run cascade.

### Datasets

| Tier | Dataset | Samples × features | Status |
|------|---------|--------------------|--------|
| 1 (must) | Wisconsin Breast Cancer | 569 × 30 | bundled (sklearn) |
| 1 (must) | UCI Heart Disease (Cleveland) | 303 × 13 | bundled |
| 1 (must) | UCI Parkinson's (voice) | 195 × 22 | bundled |
| 2 (should) | Pima Indian Diabetes | 768 × 8 | bundled |
| 3 (stretch) | TCGA / MIMIC-IV | — | NOT bundled (restricted; platform works fully without them) |

CSV upload with schema discovery, profiling and validation is available in the **Datasets** view.

## Evaluation protocol

- **Nested cross-validation**: outer 5-fold, inner 3-fold hyperparameter tuning (tuning happens strictly inside the inner loop)
- **Multi-seed**: 3 seeds in the demo protocol (42, 7, 2026); 10 seeds with `--full` — mean ± std and bootstrap CIs are always reported, never a single lucky seed
- **Metrics**: AUROC, AUPRC, Sensitivity, Specificity, Sensitivity@95%Specificity, F1, MCC, Brier, ECE — plus training/inference time, parameter count, qubit count, circuit depth, circuit count, shots, estimated quantum resource cost
- **Statistics**: DeLong paired tests, 2000-resample bootstrap CIs, Nadeau–Bengio corrected resampled t-test, decision curve analysis
- **Calibration**: Platt scaling and isotonic regression, fitted on training-fold predictions only
- **Leakage safety**: every learned transformer is wrapped by a runtime guard — any `fit` outside a training fold raises `LeakageError` (TEST FAILURE / EXPERIMENT INVALID). Deliberate-leakage tests prove the guard works.

## Scientific integrity rules (enforced)

- No fabricated benchmarks — every metric traces to a logged fold/seed
- No cherry-picked seeds, no weakened baselines (XGBoost is treated as a strong baseline)
- Entanglement ablation uses an exactly parameter-matched control (CNOTs carry zero trainable parameters)
- Dequantization: RFF and Nyström classical surrogates of the quantum kernel, evaluated on identical folds
- Certificate verdicts come from documented deterministic rules over measured statistics
- Negative results are valid results and are reported as-is

## Testing

```bash
cd backend
python -m pytest                      # full suite (54 tests)
python -m pytest tests/leakage        # deliberate leakage attempts must be caught
python -m pytest tests/quantum        # simulator == PennyLane, machine precision
python -m pytest tests/integration    # complete F1→F12 end-to-end run
```

## Known limitations

- Demo experiments use a compute-reduced protocol (5-fold × 3 seeds) to fit the sandbox; the full protocol (10 seeds, inner tuning) is one flag away (`--full`) but takes hours with exact simulation.
- Exact statevector simulation means no shot noise; `shots` is recorded as `null` (equivalent to shots=∞). Shot-noise simulation at inference is supported via config.
- The Docker deployment profile (Celery/Redis/Postgres/MinIO) is untested in the sandbox.
- Restricted datasets (TCGA, MIMIC-IV) are not bundled; the platform is designed to work without them.
- QSVM kernels use exact fidelity of simulated states — efficient for ≤12 qubits only (resource limits are enforced).

## Safety statement

QED-Net 2.0 is a research and educational platform. It is not a medical device, not clinically validated, and not intended for diagnosis or real patient-care decisions. The system uses public/de-identified benchmark data; direct identifiers (name, email, phone, patient ID) are flagged during validation and must be removed before research use. Privacy, bias and dataset limitations are documented per dataset and in every generated report.

## License / attribution

Bundled datasets: UCI Machine Learning Repository (Heart Disease, Parkinson's, Pima) and sklearn (Wisconsin Breast Cancer) — public benchmark research datasets. This platform is a research artifact; treat all outputs accordingly.
