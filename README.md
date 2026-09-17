# QED-Net 2.0 — Hybrid Quantum–Classical ML for Early Disease Detection

> **Research and educational platform. NOT a medical device. NOT clinically
> validated. NOT for diagnosis or patient-care decisions.** Uses only public /
> de-identified benchmark datasets.

Hybrid quantum–classical disease-risk classification with rigorous benchmarking,
explainability, uncertainty, and an automated Quantum Advantage Certificate.
Never assumes quantum superiority — real results are reported as measured.

## 12 features (F1–F12)

F1 ingestion · F2 validation + leakage guard · F3 preprocessing/compression ·
F4 quantum encodings (angle/ZZ/amplitude/re-uploading) · F5 quantum zoo
(QSVM, VQC, Reupload, HybridQNN) · F6 classical baselines (logReg, RF, XGBoost,
RBF-SVM, param-matched MLP) · F7 benchmarking (nested CV, multi-seed, DeLong,
bootstrap, calibration) · F8 explainability (SHAP, sensitivity, counterfactuals,
quantum diagnostics, entanglement ablation) · F9 advantage certificate · F10
bottleneck honesty meter · F11 uncertainty/conformal/abstention · F12
classical→quantum second-opinion cascade.

## Setup

```powershell
python --version            # 3.11 per spec (dev ran on 3.14)
python -m pip install -r requirements.txt
python scripts/download_datasets.py   # real UCI data -> data/raw/
```

## Run

```powershell
python -m pytest tests -q              # 52 tests
uvicorn api.main:app --port 8000       # backend: http://localhost:8000/health
Set-Location frontend; npm install; npm run dev   # dashboard :5173
python scripts/reproduce.py --config configs/experiments/smoke_tiny.yaml `
  --compare logistic_regression vqc --seeds 42 --out reports/out.json
docker compose config                  # validated; full `up --build` untested
```

## Real measured results (smoke scale, 2-fold, seed 42)

- `reports/benchmark_breast_tiny.json`: logistic AUROC **0.988**, VQC-tiny **0.721**
  → quantum worse here; reported honestly, no advantage claimed.
- `reports/benchmark_heart_tiny.json`: logistic **0.899**, XGBoost **0.875**.

Full-scale protocol (5-fold × 10 seeds, 8-qubit VQC) is compute-bound on CPU
simulators — use `configs/experiments/breast_cancer_vqc.yaml` overnight.

## Layout

`qednet/` (data, preprocess, encoding, models, evaluation, explain,
certificate, deploy, diagnostics, hardware, train, utils) · `api/` (FastAPI +
`schemas.py`) · `frontend/` (React+TS+Vite) · `configs/experiments/` ·
`scripts/` · `tests/` (unit, leakage, integration, quantum, certificate) ·
`runs/` (experiment artifacts) · `reports/`.

## Notes

- Preprocessing (scaler/imputer/SMOTE/PCA) is fit inside CV folds only;
  `tests/leakage/` fails loudly on violation.
- Long API jobs use FastAPI BackgroundTasks (Celery/Redis upgrade path noted
  in code). MLflow logging is optional (skipped if uninstalled).
- `docker compose up --build` not yet executed; compose file validated only.
```

NOTE: keep exactly 12 main features — everything else is implementation detail.
