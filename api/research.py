"""QED-Net 2.0 FastAPI backend (D11). Real pipeline, no mock results.

Long jobs run via FastAPI BackgroundTasks (ponytail: no Celery/Redis until
throughput demands it). Runs persist under ./runs/{id}/ as JSON + pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.schemas import ExperimentConfig, ModelType
from qednet.certificate.generate import certificate_to_markdown, generate_certificate
from qednet.data import DatasetRegistry
from qednet.data.validation import DATASET_METADATA
from qednet.evaluation.benchmark import make_preprocess_pipeline, nested_cv
from qednet.models.classical import (
    LogisticRegressionModel,
    ParameterMatchedMLP,
    RandomForestModel,
    RBFSVMModel,
    XGBoostModel,
)
from qednet.models.quantum import QSVM, VQC, ReuploadClassifier
from qednet.preprocess import PreprocessConfig


def create_model(mc, n_qubits: int, n_features: int, seed: int = 42):
    """Create a model supported by this research API."""
    mtype = mc.type.value if hasattr(mc.type, "value") else str(mc.type)
    if mtype == "vqc":
        return VQC(
            n_qubits=n_qubits,
            n_layers=mc.n_layers,
            encoding=mc.encoding,
            device=mc.device,
            shots=mc.shots,
            diff_method=mc.diff_method,
            optimizer=mc.optimizer,
            lr=mc.learning_rate,
            max_iter=mc.max_iter,
            seed=seed,
        )
    if mtype == "qsvm":
        return QSVM(
            n_qubits=n_qubits, encoding=mc.encoding, device=mc.device, shots=mc.shots, C=mc.C
        )
    if mtype == "reupload":
        return ReuploadClassifier(
            n_qubits=n_qubits,
            n_layers=mc.n_layers,
            device=mc.device,
            shots=mc.shots,
            diff_method=mc.diff_method,
            optimizer=mc.optimizer,
            lr=mc.learning_rate,
            max_iter=mc.max_iter,
            seed=seed,
        )
    if mtype == "logistic_regression":
        return LogisticRegressionModel(C=mc.C, random_state=seed)
    if mtype == "random_forest":
        return RandomForestModel(
            n_estimators=mc.n_estimators, max_depth=mc.max_depth, random_state=seed
        )
    if mtype == "xgboost":
        return XGBoostModel(
            n_estimators=mc.n_estimators,
            max_depth=mc.max_depth,
            learning_rate=mc.learning_rate,
            random_state=seed,
        )
    if mtype == "rbf_svm":
        return RBFSVMModel(C=mc.C, random_state=seed)
    if mtype == "parameter_matched_mlp":
        # Preprocessing may change feature dimensionality (for example PCA), so
        # let the estimator infer the transformed width from its training fold.
        return ParameterMatchedMLP(
            n_params_target=mc.n_params_target,
            input_dim=None,
            hidden_dims=tuple(mc.hidden_dims),
            epochs=mc.epochs,
            random_state=seed,
        )
    raise ValueError(f"Unknown model type: {mtype}")


RUNS = Path("runs")
UPLOADS = Path("data/uploads")
RUNS.mkdir(exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_EXP_ID_RE = re.compile(r"exp_[0-9a-f]{8}\Z")
_UPLOAD_ID_RE = re.compile(r"[0-9a-f]{8}\Z")

SAFETY = "Research use only. Not a medical device; not for diagnosis."
IDENTIFIER_HINTS = ("name", "email", "phone", "patient", "address", "ssn")

app = FastAPI(title="QED-Net 2.0", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "QEDNET_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _status(exp_id: str) -> dict:
    p = _run_dir(exp_id) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "unknown"}


def _run_dir(exp_id: str) -> Path:
    """Resolve only API-generated run IDs and keep the path inside RUNS."""
    if not isinstance(exp_id, str) or not _EXP_ID_RE.fullmatch(exp_id):
        raise HTTPException(404, "Unknown experiment")
    root = RUNS.resolve()
    path = (root / exp_id).resolve()
    if path.parent != root:
        raise HTTPException(404, "Unknown experiment")
    return path


def _write_status(exp_id: str, **kw) -> None:
    d = _run_dir(exp_id)
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
        x, y = bundle.X, bundle.y
        pre_cfg = PreprocessConfig(**config.preprocess.model_dump())
        seeds = config.training.seeds or [config.seed]

        results: dict = {}
        for mtype in compare:
            per_seed = []
            for s in seeds:
                mc = config.model.model_copy(update={"type": mtype})
                nq = min(pre_cfg.compress_dim, 8)
                model = create_model(mc, nq, x.shape[1], s)  # type: ignore[arg-type]
                r = nested_cv(
                    model,
                    x,
                    y,
                    lambda: make_preprocess_pipeline(pre_cfg),
                    outer_folds=config.training.outer_folds,
                    random_state=s,
                    calibrate=False,
                    n_bootstrap=200,
                    groups=bundle.groups,
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
        x_compressed = pre.fit_transform(x, y)
        final_model = create_model(mc, nq, x.shape[1], seeds[0])  # type: ignore[arg-type]
        final_model.fit(x_compressed, y)
        d = RUNS / exp_id
        with open(d / "model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(d / "preprocess.pkl", "wb") as f:
            pickle.dump(pre, f)

        # honesty meter: logistic on the same compressed features
        honest = LogisticRegressionModel()
        honest.fit(x_compressed, y)
        q_key = primary if primary in results else list(results)[0]
        classical = {k: v for k, v in results.items() if k not in ("vqc", "qsvm", "reupload")}
        best_c = max(classical.values(), key=lambda v: v["mean"]) if classical else results[q_key]
        (d / "results.json").write_text(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "dataset": config.dataset.value,
                    "models": results,
                    "honesty_gap": results[q_key]["mean"] - best_c["mean"],
                    "n_train": len(x),
                    "feature_names": bundle.feature_names,
                    "safety_note": SAFETY,
                },
                default=str,
            )
        )

        cfg_hash = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        cert = generate_certificate(
            experiment_id=exp_id,
            dataset=config.dataset.value,
            model=q_key,
            config_hash=cfg_hash,
            seeds=seeds,
            quantum=results[q_key],
            classical_best={
                "mean": best_c["mean"],
                "ci_lower": best_c["ci_lower"],
                "ci_upper": best_c["ci_upper"],
            },
            param_matched=results.get("parameter_matched_mlp", best_c),
            n_train=len(x),
            resources={"compressed_dim": x_compressed.shape[1], "qubits": nq},
        )
        (d / "certificate.json").write_text(json.dumps(cert, default=str))
        _write_status(exp_id, status="completed", elapsed_s=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001 ÔÇö surfaced via status, logged
        _write_status(exp_id, status="failed", error=str(e)[:500])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "safety": SAFETY,
        "available_datasets": DatasetRegistry.available(),
        "available_models": [m.value for m in ModelType],
    }


@app.get("/datasets")
def datasets():
    out = []
    for name in DatasetRegistry.available():
        try:
            b = DatasetRegistry.load(name)
            out.append(
                {
                    "name": name,
                    "samples": len(b.y),
                    "features": len(b.feature_names),
                    "pos_rate": round(float(b.y.mean()), 4),
                    "meta": DATASET_METADATA.get(name, {}),
                }
            )
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "error": str(e)[:200]})
    return {"datasets": out, "safety": SAFETY}


@app.post("/datasets/upload")
async def upload(file: UploadFile):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Only CSV supported")
    uid = uuid.uuid4().hex[:8]
    dest = UPLOADS / f"{uid}_{filename}"
    size = 0
    try:
        with dest.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "CSV exceeds the upload size limit")
                output.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    try:
        df = pd.read_csv(dest)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid CSV: {e}") from e
    if df.empty:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty dataset")
    profile = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "suspected_identifiers": [
            c for c in df.columns if any(h in c.lower() for h in IDENTIFIER_HINTS)
        ],
    }
    (UPLOADS / f"{uid}.profile.json").write_text(json.dumps(profile))
    return {"file_id": uid, "filename": filename, "profile": profile}


@app.post("/datasets/validate")
def validate_dataset(payload: dict):
    target = payload.get("target")
    if payload.get("file_id"):
        file_id = payload["file_id"]
        if not isinstance(file_id, str) or not _UPLOAD_ID_RE.fullmatch(file_id):
            raise HTTPException(400, "Invalid file_id")
        matches = list(UPLOADS.glob(f"{file_id}_*.csv"))
        if not matches:
            raise HTTPException(404, "Uploaded dataset not found")
        df = pd.read_csv(matches[0])
    elif payload.get("dataset"):
        b = DatasetRegistry.load(payload["dataset"])
        return {
            "dataset": payload["dataset"],
            "samples": len(b.y),
            "target": b.target_name,
            "valid": True,
            "safety": SAFETY,
        }
    else:
        raise HTTPException(400, "Provide file_id or dataset")
    errors = []
    if not target or target not in df.columns:
        errors.append("missing/invalid target column")
    if df.empty:
        errors.append("empty dataset")
    return {
        "file_id": payload.get("file_id"),
        "target": target,
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "valid": not errors,
        "errors": errors,
        "safety": SAFETY,
    }


@app.get("/models")
def models():
    return {
        "models": [m.value for m in ModelType],
        "quantum": ["vqc", "qsvm", "reupload"],
        "classical": [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "rbf_svm",
            "parameter_matched_mlp",
        ],
        "safety": SAFETY,
    }


@app.post("/train")
def train(payload: dict, background: BackgroundTasks):
    try:
        cfg = ExperimentConfig(**payload.get("config", payload))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid config: {e}") from e
    compare = payload.get("compare", [cfg.model.type.value])
    if not isinstance(compare, list) or not compare:
        raise HTTPException(400, "compare must be a non-empty list of supported model names")
    supported = {model.value for model in ModelType}
    unsupported = [
        model for model in compare if not isinstance(model, str) or model not in supported
    ]
    if unsupported:
        raise HTTPException(400, f"Unsupported model(s): {unsupported}")
    exp_id = f"exp_{uuid.uuid4().hex[:8]}"
    _write_status(exp_id, status="queued")
    background.add_task(_run_train, exp_id, cfg.model_dump(), compare)
    return {"experiment_id": exp_id, "status": "queued"}


@app.get("/experiments")
def experiments():
    return {
        "experiments": [
            {**{"id": p.name}, **_status(p.name)}
            for p in RUNS.iterdir()
            if p.is_dir() and _EXP_ID_RE.fullmatch(p.name)
        ]
    }


@app.get("/experiments/{exp_id}")
def experiment(exp_id: str):
    st = _status(exp_id)
    if st.get("status") == "unknown":
        raise HTTPException(404, "Unknown experiment")
    out = {"id": exp_id, **st}
    rp = _run_dir(exp_id) / "results.json"
    if rp.exists():
        out["results"] = json.loads(rp.read_text())
    return out


def _load_artifacts(exp_id: str):
    d = _run_dir(exp_id)
    if not (d / "model.pkl").exists():
        raise HTTPException(400, "Experiment has no fitted model yet")
    with open(d / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(d / "preprocess.pkl", "rb") as f:
        pre = pickle.load(f)
    results = json.loads((d / "results.json").read_text())
    return model, pre, results


def _request_features(payload: dict) -> tuple[str, np.ndarray]:
    try:
        exp_id = payload["experiment_id"]
        features = np.asarray(payload["features"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "Provide experiment_id and numeric features") from exc
    if features.ndim != 1 or features.size == 0 or not np.isfinite(features).all():
        raise HTTPException(422, "features must be a non-empty finite one-dimensional list")
    return exp_id, features.reshape(1, -1)


def _transform_features(features: np.ndarray, pre) -> np.ndarray:
    expected = getattr(pre, "n_features_in_", None)
    if expected is not None and features.shape[1] != expected:
        raise HTTPException(422, f"Expected {expected} input features, got {features.shape[1]}")
    return pre.transform(features)


@app.post("/predict")
def predict(payload: dict):
    exp_id, features = _request_features(payload)
    model, pre, results = _load_artifacts(exp_id)
    proba = float(model.predict_proba(_transform_features(features, pre))[0, 1])
    low, high = 0.35, 0.65  # configurable cascade gate; recorded here
    abstain = low <= proba <= high
    return {
        "prediction": int(proba >= 0.5),
        "probability": proba,
        "abstained": abstain,
        "ambiguity_range": [low, high],
        "model": results.get("models", {}).keys().__iter__().__next__(),
        "safety": SAFETY,
    }


@app.post("/explain")
def explain(payload: dict):
    exp_id, features = _request_features(payload)
    model, pre, results = _load_artifacts(exp_id)
    names = results["feature_names"]
    if len(names) != features.shape[1]:
        raise HTTPException(500, "Stored feature metadata does not match the fitted preprocessor")
    base = float(model.predict_proba(_transform_features(features, pre))[0, 1])
    sens = []
    for i, name in enumerate(names):
        delta = max(abs(features[0, i]) * 0.1, 1e-3)
        up = features.copy()
        up[0, i] += delta
        dn = features.copy()
        dn[0, i] -= delta
        pu = float(model.predict_proba(_transform_features(up, pre))[0, 1])
        pd_ = float(model.predict_proba(_transform_features(dn, pre))[0, 1])
        sens.append({"feature": name, "sensitivity": (pu - pd_) / (2 * delta), "base_proba": base})
    sens.sort(key=lambda s: abs(s["sensitivity"]), reverse=True)
    return {"base_proba": base, "sensitivity": sens, "safety": SAFETY}


@app.get("/benchmark/{exp_id}")
def benchmark(exp_id: str):
    rp = _run_dir(exp_id) / "results.json"
    if not rp.exists():
        raise HTTPException(404, "No benchmark for this id")
    return json.loads(rp.read_text())


@app.get("/certificate/{exp_id}")
def certificate(exp_id: str, format: str = "json"):
    cp = _run_dir(exp_id) / "certificate.json"
    if not cp.exists():
        raise HTTPException(404, "No certificate for this id")
    cert = json.loads(cp.read_text())
    if format == "markdown":
        return PlainTextResponse(certificate_to_markdown(cert))
    return cert
