"""QED-Net 2.0 FastAPI application (docker-compose deployment profile).

In the sandbox the Next.js dashboard talks to the Python core directly via
the CLI; in the full docker-compose deployment this FastAPI service exposes
the same capabilities over HTTP with ARQ+Redis handling durable training
jobs and PostgreSQL/MinIO providing persistent metadata and artifacts.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qednet import RESEARCH_STATUS  # noqa: E402
from qednet.data.ingestion import safe_filename  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qednet.api")

app = FastAPI(
    title="QED-Net 2.0 API",
    description="Hybrid quantum-classical ML research platform API. "
    + RESEARCH_STATUS,
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "QEDNET_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def _validate_id(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise HTTPException(400, f"Invalid {field}")
    return value


def _cli(args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """Invoke the qednet CLI and parse its JSON stdout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "qednet.cli", *args],
        cwd=str(ROOT / "backend"), capture_output=True, text=True, timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "Research command exceeded its time limit") from exc
    except OSError as exc:
        raise HTTPException(503, "Research command could not be started") from exc
    for line in reversed(proc.stdout.strip().split("\n")):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise HTTPException(status_code=500, detail=proc.stderr[-800:] or "backend error")


class ValidateRequest(BaseModel):
    dataset: str


class PredictRequest(BaseModel):
    experiment_id: str = Field(alias="experimentId", pattern=r"^[A-Za-z0-9_-]{1,128}$")
    features: List[float]

    model_config = {"populate_by_name": True}


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness — process is up. Use /ready for dependency readiness."""
    from qednet.api.health import health as liveness
    payload = liveness()
    payload.update({"ok": True, "kind": "liveness"})
    return payload


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """Readiness — Postgres + Redis + S3 when USE_DB_STORAGE is enabled (503 if unready)."""
    from fastapi.responses import JSONResponse
    from qednet.api.health import readiness
    from qednet.db.settings import StorageSettings

    payload, code = readiness(StorageSettings.from_env())
    return JSONResponse(status_code=code, content=payload)


@app.get("/datasets")
def datasets() -> Dict[str, Any]:
    return _cli(["list-datasets"], timeout=120)


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...),
                         target: Optional[str] = Form(default=None),
                         name: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in (".csv", ".tsv"):
        raise HTTPException(400, "Only CSV/TSV files are accepted")
    safe_name = safe_filename(filename)
    dest = ROOT / "data" / "uploads" / f"{uuid.uuid4().hex}_{safe_name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with dest.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File exceeds the upload size limit")
                output.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    args = ["upload", "--csv", str(dest)]
    if target:
        args += ["--target", target]
    if name:
        args += ["--name", name]
    return _cli(args, timeout=180)


@app.post("/datasets/validate")
def validate(req: ValidateRequest) -> Dict[str, Any]:
    return _cli(["validate", "--dataset", req.dataset], timeout=300)


@app.get("/models")
def models() -> Dict[str, Any]:
    from qednet.models.registry import MODEL_REGISTRY, model_families
    return {"models": sorted(MODEL_REGISTRY), "families": model_families()}


@app.post("/train")
def train(background_tasks: BackgroundTasks, config: str = Form(...),
          experiment_id: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    """Start a training job. With USE_DB_STORAGE the job is durably queued via
    ARQ+Redis; otherwise it degrades to an in-process background task."""
    config = _validate_id(config, "config name")
    if experiment_id is not None:
        experiment_id = _validate_id(experiment_id, "experiment ID")
    cfg_path = ROOT / "configs" / "experiments" / f"{config}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"Unknown config: {config}")
    exp_id = experiment_id or f"exp_{config}_{int(__import__('time').time())}"

    # Phase 1: durable DB + ARQ path when enabled. Falls back to the Phase 0
    # in-process background task only when the flag is off (demo mode).
    from qednet.db.settings import StorageSettings
    settings = StorageSettings.from_env()
    if settings.use_db_storage:
        from qednet.config import config_to_dict, load_config
        from qednet.api.integration import start_db_job

        cfg = load_config(cfg_path)
        cfg.experiment_id = exp_id
        try:
            return start_db_job(
                settings,
                cfg_path=cfg_path,
                experiment_id=exp_id,
                config_json=json.dumps(config_to_dict(cfg)),
                config_hash=cfg.config_hash(),
                dataset=cfg.dataset.name,
            )
        except Exception as exc:  # noqa: BLE001 - surface queue/DB failure
            raise HTTPException(503, f"training queue unavailable: {exc}")

    queued: Any = None
    if queued is None:
        import time
        from qednet.config import load_config
        from qednet.data.datasets import register_builtins
        from qednet.data.ingestion import DatasetRegistry
        from qednet.train.runner import run_experiment
        from qednet.tracking.store import ExperimentStore

        def _run():
            cfg = load_config(cfg_path)
            cfg.experiment_id = exp_id
            store = ExperimentStore()
            reg = DatasetRegistry()
            register_builtins(reg)
            run_experiment(cfg, store, reg)

        background_tasks.add_task(_run)
        queued = "in-process background task"
    return {"ok": True, "experiment_id": exp_id, "execution": queued}


@app.get("/experiments")
def experiments() -> Dict[str, Any]:
    return _cli(["list-experiments"], timeout=120)


@app.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str) -> Dict[str, Any]:
    _validate_id(experiment_id, "experiment ID")
    return _cli(["get-experiment", "--id", experiment_id], timeout=300)


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    return _cli(["predict", "--id", req.experiment_id,
                 "--input-json", json.dumps(req.features)], timeout=300)


@app.post("/explain")
def explain(experiment_id: str = Form(...),
            model: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    _validate_id(experiment_id, "experiment ID")
    args = ["explain", "--id", experiment_id]
    if model:
        args += ["--model", model]
    return _cli(args, timeout=600)


@app.get("/benchmark/{experiment_id}")
def benchmark(experiment_id: str) -> Dict[str, Any]:
    _validate_id(experiment_id, "experiment ID")
    data = _cli(["get-experiment", "--id", experiment_id], timeout=300)
    return {"ok": True, "benchmark": (data or {}).get("result", {}).get("models")}


@app.get("/certificate/{experiment_id}")
def certificate(experiment_id: str) -> Dict[str, Any]:
    _validate_id(experiment_id, "experiment ID")
    return _cli(["certificate", "--id", experiment_id], timeout=300)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
