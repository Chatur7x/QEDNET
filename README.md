# QED-Net 2.0

**Hybrid Quantum–Classical Machine Learning Platform for Early Disease Detection**

> **Research and educational platform.** QED-Net 2.0 is **not a medical device**, is **not clinically validated**, and is **not intended for diagnosis or real patient-care decisions**. All experiments use public, de-identified benchmark datasets.

---

## Overview

QED-Net 2.0 is a research codebase for comparing quantum and classical machine-learning approaches on biomedical benchmark data. It contains a Python research core, a FastAPI service, and a Next.js dashboard. The Render preview uses Supabase email/password authentication with email confirmation disabled and a single-email allowlist. The current account has signed in successfully; a fresh signup flow has not been independently re-tested. Persistent experiment storage, multi-user data isolation, and clinical validation are not configured.

The product direction is a whole-body health research platform, with dedicated women's-health and menstrual-cycle research alongside other body-system studies. These are planned research areas, not currently implemented trackers, Android features, diagnostic models, or clinical functions. See the [whole-body health research plan](docs/WHOLE_BODY_HEALTH_RESEARCH_PLAN.md).

See the [early-detection product specification](docs/EARLY_DETECTION_PRODUCT_SPEC.md) and [product roadmap](docs/PRODUCT_ROADMAP.md) for the research scope, evidence gates, and unresolved problems. “Early detection” is a research objective, not a demonstrated capability of the current cross-sectional benchmarks.

See [research completion status](docs/RESEARCH_COMPLETION_STATUS.md) for the latest local verification evidence and the remaining production blockers.

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

## Current research-core modules

These modules are present in the repository; they do not constitute the planned
whole-body product or establish clinical validity.

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
├── data/raw/              # optional local datasets (not committed)
├── experiments/runs/      # machine-readable results + certificates + reports
├── scripts/               # reproduce.py, dataset setup, experiment launcher
├── Dockerfile / docker-compose.yml
└── docs/                  # model cards, methodology
```

### Quantum execution backends

- **`numpy` (default)** — the VQC, data re-uploading, and HybridQNN implementations use this project's batched statevector simulator and automatic differentiation.
- **`pennylane`** — QSVM has a PennyLane simulator path. VQC, data re-uploading, and HybridQNN currently reject this backend because their PennyLane execution path is not implemented; they no longer report a requested-but-unused backend as if it ran.
- **Simulator tests are not hardware tests.** The statevector tests compare selected simulator calculations; no real quantum hardware execution is established by this repository review.
- Exact statevector simulation is equivalent to infinite shots; resource reports record this honestly (`shots: null`).

## Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ · TypeScript 5 |
| Quantum ML | PennyLane 0.45 (+ custom batched statevector engine, PL-verified) |
| Quantum secondary | Qiskit-compatible circuit descriptions; `qiskit` optional |
| Classical ML | scikit-learn, XGBoost |
| Data | pandas, built-in schema/leakage validation, imbalanced-learn (SMOTE inside folds only) |
| Explainability | SHAP (KernelExplainer over the full pipeline), permutation, parameter-shift circuit gradients |
| Tracking | JSON experiment store + optional MLflow mirror |
| Backend API | FastAPI + Pydantic (docker profile); Next.js API routes bridge the CLI in the demo |
| Background jobs | Detached process jobs (dashboard demo); optional ARQ + Redis worker when database mode is configured |
| Storage | File store (demo); optional PostgreSQL + MinIO/S3 adapters when configured |
| Dashboard | Next.js 16 · React 19 · Tailwind CSS 4 · shadcn/ui · recharts · Framer Motion |
| Testing | pytest (+ Hypothesis available) |
| Packaging | Docker + docker-compose |

## Quick start (demo / sandbox)

The demo runs the dashboard (Next.js) against the Python core through the CLI bridge — no database required.

```bash
# 1. Python environment (Python 3.11+)
pip install -r requirements.txt

# 2. Add the optional local data files required for heart disease,
#    Parkinson's, and Pima diabetes. Those files are not included in Git.
#    Breast cancer can be loaded from scikit-learn.

# 3. Run the test suite (leakage guard, quantum equivalence, integration)
cd backend && python -m pytest && cd ..

# 4. Run an end-to-end experiment (writes experiments/runs/<id>/)
PYTHONPATH=backend python -m qednet.cli run-experiment \
  --config configs/experiments/breast_cancer.yaml --verbose

# 5. Dashboard
bun install && bun run dev     # http://localhost:3000
```

For production-build smoke checks on Windows, build the app, start the standalone
server in one PowerShell window, then run the scoped app tests in another:

```powershell
bun run build
$env:PORT = "3100"
node .next/standalone/server.js
```

```powershell
$env:QEDNET_BASE_URL = "http://localhost:3100"
bun run test
```

The app tests verify liveness and that protected routes remain behind the auth
gate. They do not verify authenticated research API behavior without configured
Supabase test credentials. Run backend tests from `backend/`; running pytest at the
repository root also discovers duplicate tests under the separate `preserved/` project.

Experiment metrics shown in the dashboard are read from saved run artifacts. Dataset metadata and service health are obtained from their respective backend endpoints. The cloud demo currently has no saved experiments, so it does not show model performance results.

## Integrated deployment (docker-compose)

```bash
docker compose up --build
```

Starts the Next.js dashboard and FastAPI API in the app container, plus an ARQ worker,
Redis, PostgreSQL, and RustFS (the local S3-compatible object store). The app waits for
PostgreSQL and Redis, then runs the Alembic migrations before serving requests. Local
Compose credentials are development defaults; replace them before exposing this stack
outside a trusted machine.

The local Docker stack has been started and checked: app, worker, PostgreSQL, Redis, and
RustFS all ran; database, queue, and object-store readiness passed; and the artifact
adapter passed a write/read/delete smoke check. The sign-in page loads, but a protected
research API redirects to setup because this local Compose environment has no Supabase
URL/key or allowlisted email configured. An authenticated research session was not
verified. The Render deployment uses filesystem-demo mode and does not use the Compose
PostgreSQL, Redis, or RustFS services. Render's free service filesystem is ephemeral,
so uploaded data and run artifacts are not durable there.

### Local Aiven PostgreSQL pilot

Aiven Free PostgreSQL can be tested as a local development database. It is not a
production database tier, and it does not provide QEDNET's queue or artifact storage.
Create `aiven.local.env` from `aiven.local.env.example`, put the Aiven URI there with
`sslmode=require`, and keep the file private (it is gitignored). Run
`powershell -ExecutionPolicy Bypass -File scripts/check-aiven.ps1` to test TLS/database
connectivity without displaying the URI. Only after confirming the database is dedicated
and empty, run the same command with `-ApplySchema` to apply Alembic migrations.
This does not enable `USE_DB_STORAGE` or alter the hosted Render service. Do not use
patient or other sensitive data in this free-tier pilot.

The optional Prisma helper uses a separate SQLite setting, `QEDNET_PRISMA_DATABASE_URL`
(for example, a `file:` URL). It is not the Python research backend's `DATABASE_URL`
and is not the deployed experiment store. The app build generates the Prisma client
but does not migrate or modify a database. Use a disposable database when validating
schema changes; do not point smoke tests at an existing database file.

### Email/password accounts

Production builds lock the dashboard and API unless these settings are configured: `NEXT_PUBLIC_SITE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, and `QEDNET_ALLOWED_EMAIL`. Render Docker builds pass the public Supabase settings to Next.js so the browser auth client is configured at build time. The current Supabase project has email confirmation disabled, so email ownership is not verified during signup. The allowlist permits one email only because experiment files are shared; do not add users until per-user storage isolation exists. Password reset still depends on the configured Supabase email delivery and has not been re-verified in this pass. The publishable key is intended for browser use; never place a Supabase secret/service-role key in the browser or Render web service. Local development (`bun run dev`) bypasses the production gate and must not be exposed publicly.

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
- `pipeline.pkl`, `models.pkl`, `conformal.pkl` — research inference artifacts; the current experiment does not claim measured conformal coverage
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
| 1 (must) | UCI Heart Disease (Cleveland) | 303 × 13 in the dataset adapter documentation | local source file required; not bundled |
| 1 (must) | UCI Parkinson's (voice) | 195 × 22 in the dataset adapter documentation | local source file required; not bundled |
| 2 (should) | Pima Indian Diabetes | 768 × 8 in the dataset adapter documentation | local source file required; not bundled |
| 3 (stretch) | TCGA / MIMIC-IV | — | NOT bundled (restricted; platform works fully without them) |

CSV upload with schema discovery, profiling and validation is available in the **Datasets** view.

## Evaluation protocol

- **Nested cross-validation**: outer 5-fold, inner 3-fold hyperparameter tuning (tuning happens strictly inside the inner loop)
- **Multi-seed**: 3 seeds in the demo protocol (42, 7, 2026); 10 seeds with `--full` — mean ± std and bootstrap CIs are always reported, never a single lucky seed
- **Metrics**: AUROC, AUPRC, Sensitivity, Specificity, Sensitivity@95%Specificity, F1, MCC, Brier, ECE — plus training/inference time, parameter count, qubit count, circuit depth, circuit count, shots, estimated quantum resource cost
- **Statistics**: DeLong paired tests, 2000-resample bootstrap CIs, Nadeau–Bengio corrected resampled t-test, decision curve analysis
- **Calibration**: Platt scaling and isotonic regression, fit from cross-fitted predictions within each outer training fold
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
python -m pytest                      # full suite
python -m pytest tests/leakage        # deliberate leakage attempts must be caught
python -m pytest tests/quantum        # simulator == PennyLane, machine precision
python -m pytest tests/integration    # complete F1→F12 end-to-end run
```

## Known limitations

- Demo configs use a compute-reduced protocol (5 outer folds × 3 seeds); inner-CV tuning is already enabled. `scripts/reproduce.py --full` replaces the configured seed list with 10 seeds and can take substantially longer with exact simulation.
- Exact statevector simulation means no shot noise; `shots` is recorded as `null` (equivalent to shots=∞). Shot-noise simulation at inference is supported via config.
- The local Docker Compose stack now runs with PostgreSQL, Redis, and RustFS, but local Supabase authentication is not configured and the protected research UI has not been verified with an authenticated session. The dashboard's job route still starts local child processes; it is not integrated with the durable ARQ queue. The current Render service uses filesystem-demo mode and has no durable database/object store configured.
- Restricted datasets (TCGA, MIMIC-IV) are not bundled; the platform is designed to work without them.
- QSVM kernels use exact fidelity of simulated states — efficient for ≤12 qubits only (resource limits are enforced).

## Safety statement

QED-Net 2.0 is a research and educational platform. It is not a medical device, not clinically validated, and not intended for diagnosis or real patient-care decisions. The system uses public/de-identified benchmark data; direct identifiers (name, email, phone, patient ID) are flagged during validation and must be removed before research use. Privacy, bias and dataset limitations are documented per dataset and in every generated report.

## License / attribution

Dataset attribution: the breast-cancer adapter loads scikit-learn's Wisconsin Diagnostic Breast Cancer dataset; the other adapters expect local source files. The repository does not bundle those raw files. No repository license file is present, so the project license is not established by this README. Do not assume the project or third-party datasets are licensed for commercial or clinical use.
