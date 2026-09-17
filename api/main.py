"""QED-Net 2.0 FastAPI backend (D11). Real pipeline, no mock results.

Long jobs run via FastAPI BackgroundTasks (ponytail: no Celery/Redis until
throughput demands it). Runs persist under ./runs/{id}/ as JSON + pickle.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.schemas import DatasetName, ExperimentConfig, ModelType
from qednet.certificate.generate import certificate_to_markdown, generate_certificate
from qednet.data import DatasetRegistry
from qednet.data.validation import DATASET_METADATA
from qednet.evaluation.benchmark import make_preprocess_pipeline, nested_cv
from qednet.models.classical import LogisticRegressionModel
from qednet.preprocess import PreprocessConfig
from qednet.train.config import create_model

RUNS = Path("runs")
UPLOADS = Path("data/uploads")
RUNS.mkdir(exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

SAFETY = "Research use only. Not a medical device; not for diagnosis."
IDENTIFIER_HINTS = ("name", "email", "phone", "patient", "address", "ssn")

app = FastAPI(title="QED-Net 2.0", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _status(exp_id: str) -> dict:
    p = RUNS / exp_id / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "unknown"}


def _write_status(exp_id: str, **kw) -> None:
    d = RUNS / exp_id
    d.mkdir(parents=True, exist_ok=True)
    cur = _status(exp_id) if (d / "status.json").exists() else {}
    cur.update(kw)
    (d / "status.json").write_text(json.dumps(cur, default=str))


def _auroc_summary(res: dict) -> dict:
    m = res["aggregated_metrics"]["auroc"]
    return {"mean": m["mean"], "ci_lower": m["ci_lower"], "ci_upper": m["ci_upper"]}


def _run_train(exp_id: str, cfg: dict, compare: list[str]) -> None:
    t0 = time.time()
    try:
        _write_status(exp_id, status="running", started=time.time())
        config = ExperimentConfig(**cfg)
        bundle = DatasetRegistry.load(config.dataset.value)
        X, y = bundle.X, bundle.y
        pre_cfg = PreprocessConfig(**config.preprocess.model_dump())
        seeds = config.training.seeds or [config.seed]

        results: dict = {}
        for mtype in compare:
            per_seed = []
            for s in seeds:
                mc = config.model.model_copy(update={"type": mtype})
                nq = min(pre_cfg.compress_dim, 8)
                model = create_model(mc, nq, X.shape[1], s)  # type: ignore[arg-type]
                r = nested_cv(
                    model, X, y, lambda: make_preprocess_pipeline(pre_cfg),
                    outer_folds=config.training.outer_folds, random_state=s,
                    calibrate=False, n_bootstrap=200,
                )
                per_seed.append(_auroc_summary(r))
            means = [p["mean"] for p in per_seed]
            results[mtype] = {
                "mean": float(np.mean(means)),
                "std": float(np.std(means)) if len(means) > 1 else 0.0,
                "ci_lower": min(p["ci_lower"] for p in per_seed),
                "ci_upper": max(p["ci_upper"] for p in per_seed),
                "per_seed": per_seed,
            }

        # final artifacts: fit primary model on full compressed data (for /predict)
        primary = config.model.type.value
        mc = config.model
        nq = min(pre_cfg.compress_dim, 8)
        pre = make_preprocess_pipeline(pre_cfg)
        Xc = pre.fit_transform(X, y)
        final_model = create_model(mc, nq, X.shape[1], seeds[0])  # type: ignore[arg-type]
        final_model.fit(Xc, y)
        d = RUNS / exp_id
        with open(d / "model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(d / "preprocess.pkl", "wb") as f:
            pickle.dump(pre, f)

        # honesty meter: logistic on the same compressed features
        honest = LogisticRegressionModel()
        honest.fit(Xc, y)
        q_key = primary if primary in results else list(results)[0]
        classical = {k: v for k, v in results.items() if k not in ("vqc", "qsvm", "reupload", "hybrid_qnn")}
        best_c = max(classical.values(), key=lambda v: v["mean"]) if classical else results[q_key]
        (d / "results.json").write_text(json.dumps({
            "experiment_id": exp_id, "dataset": config.dataset.value,
            "models": results, "honesty_gap": results[q_key]["mean"] - best_c["mean"],
            "n_train": len(X), "feature_names": bundle.feature_names,
            "safety_note": SAFETY,
        }, default=str))

        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
        cert = generate_certificate(
            experiment_id=exp_id, dataset=config.dataset.value, model=q_key,
            config_hash=cfg_hash, seeds=seeds, quantum=results[q_key],
            classical_best={"mean": best_c["mean"], "ci_lower": best_c["ci_lower"], "ci_upper": best_c["ci_upper"]},
            param_matched=results.get("parameter_matched_mlp", best_c),
            n_train=len(X),
            resources={"compressed_dim": Xc.shape[1], "qubits": nq},
        )
        (d / "certificate.json").write_text(json.dumps(cert, default=str))
        _write_status(exp_id, status="completed", elapsed_s=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001 — surfaced via status, logged
        _write_status(exp_id, status="failed", error=str(e)[:500])


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "safety": SAFETY,
            "available_datasets": DatasetRegistry.available(),
            "available_models": [m.value for m in ModelType]}


@app.get("/datasets")
def datasets():
    out = []
    for name in DatasetRegistry.available():
        try:
            b = DatasetRegistry.load(name)
            out.append({"name": name, "samples": len(b.y), "features": len(b.feature_names),
                        "pos_rate": round(float(b.y.mean()), 4), "meta": DATASET_METADATA.get(name, {})})
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "error": str(e)[:200]})
    return {"datasets": out, "safety": SAFETY}


@app.post("/datasets/upload")
async def upload(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV supported")
    uid = uuid.uuid4().hex[:8]
    dest = UPLOADS / f"{uid}_{Path(file.filename).name}"
    dest.write_bytes(await file.read())
    try:
        df = pd.read_csv(dest)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid CSV: {e}")
    if df.empty:
        raise HTTPException(400, "Empty dataset")
    profile = {"rows": len(df), "columns": list(df.columns),
               "missing": {c: int(df[c].isna().sum()) for c in df.columns},
               "duplicates": int(df.duplicated().sum()),
               "suspected_identifiers": [c for c in df.columns if any(h in c.lower() for h in IDENTIFIER_HINTS)]}
    (UPLOADS / f"{uid}.profile.json").write_text(json.dumps(profile))
    return {"file_id": uid, "filename": file.filename, "profile": profile}


@app.post("/datasets/validate")
def validate_dataset(payload: dict):
    target = payload.get("target")
    if payload.get("file_id"):
        df = pd.read_csv(next(UPLOADS.glob(f"{payload['file_id']}_*.csv")))
        name = "uploaded_csv"
    elif payload.get("dataset"):
        b = DatasetRegistry.load(payload["dataset"])
        return {"dataset": payload["dataset"], "samples": len(b.y),
                "target": b.target_name, "valid": True, "safety": SAFETY}
    else:
        raise HTTPException(400, "Provide file_id or dataset")
    errors = []
    if not target or target not in df.columns:
        errors.append("missing/invalid target column")
    if df.empty:
        errors.append("empty dataset")
    return {"file_id": payload.get("file_id"), "target": target,
            "missing": {c: int(df[c].isna().sum()) for c in df.columns},
            "duplicates": int(df.duplicated().sum()), "valid": not errors,
            "errors": errors, "safety": SAFETY}


@app.get("/models")
def models():
    return {"models": [m.value for m in ModelType],
            "quantum": ["vqc", "qsvm", "reupload", "hybrid_qnn"],
            "classical": ["logistic_regression", "random_forest", "xgboost", "rbf_svm",
                          "parameter_matched_mlp"], "safety": SAFETY}


@app.post("/train")
def train(payload: dict, background: BackgroundTasks):
    try:
        cfg = ExperimentConfig(**payload.get("config", payload))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid config: {e}")
    compare = payload.get("compare", [cfg.model.type.value])
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    _write_status(exp_id, status="queued")
    background.add_task(_run_train, exp_id, cfg.model_dump(), compare)
    return {"experiment_id": exp_id, "status": "queued"}


@app.get("/experiments")
def experiments():
    return {"experiments": [{**{"id": p.name}, **_status(p.name)}
                            for p in RUNS.iterdir() if p.is_dir()]}


@app.get("/experiments/{exp_id}")
def experiment(exp_id: str):
    st = _status(exp_id)
    if st.get("status") == "unknown":
        raise HTTPException(404, "Unknown experiment")
    out = {"id": exp_id, **st}
    rp = RUNS / exp_id / "results.json"
    if rp.exists():
        out["results"] = json.loads(rp.read_text())
    return out


def _load_artifacts(exp_id: str):
    d = RUNS / exp_id
    if not (d / "model.pkl").exists():
        raise HTTPException(400, "Experiment has no fitted model yet")
    with open(d / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(d / "preprocess.pkl", "rb") as f:
        pre = pickle.load(f)
    results = json.loads((d / "results.json").read_text())
    return model, pre, results


@app.post("/predict")
def predict(payload: dict):
    try:
        exp_id, features = payload["experiment_id"], np.array(payload["features"], dtype=float)
    except KeyError as e:
        raise HTTPException(400, f"Missing field: {e}")
    model, pre, results = _load_artifacts(exp_id)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    proba = float(model.predict_proba(pre.transform(features))[0, 1])
    low, high = 0.35, 0.65  # configurable cascade gate; recorded here
    abstain = low <= proba <= high
    return {"prediction": int(proba >= 0.5), "probability": proba,
            "abstained": abstain, "ambiguity_range": [low, high],
            "model": results.get("models", {}).keys().__iter__().__next__(),
            "safety": SAFETY}


@app.post("/explain")
def explain(payload: dict):
    exp_id, features = payload["experiment_id"], np.array(payload["features"], dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    model, pre, results = _load_artifacts(exp_id)
    names = results["feature_names"]
    base = float(model.predict_proba(pre.transform(features))[0, 1])
    sens = []
    for i, name in enumerate(names):
        delta = max(abs(features[0, i]) * 0.1, 1e-3)
        up = features.copy(); up[0, i] += delta
        dn = features.copy(); dn[0, i] -= delta
        pu = float(model.predict_proba(pre.transform(up))[0, 1])
        pd_ = float(model.predict_proba(pre.transform(dn))[0, 1])
        sens.append({"feature": name, "sensitivity": (pu - pd_) / (2 * delta), "base_proba": base})
    sens.sort(key=lambda s: abs(s["sensitivity"]), reverse=True)
    return {"base_proba": base, "sensitivity": sens, "safety": SAFETY}


@app.get("/benchmark/{exp_id}")
def benchmark(exp_id: str):
    rp = RUNS / exp_id / "results.json"
    if not rp.exists():
        raise HTTPException(404, "No benchmark for this id")
    return json.loads(rp.read_text())


@app.get("/certificate/{exp_id}")
def certificate(exp_id: str, format: str = "json"):
    cp = RUNS / exp_id / "certificate.json"
    if not cp.exists():
        raise HTTPException(404, "No certificate for this id")
    cert = json.loads(cp.read_text())
    if format == "markdown":
        return PlainTextResponse(certificate_to_markdown(cert))
    return cert
