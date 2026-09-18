"""QED-Net 2.0 FastAPI application (docker-compose deployment profile).

In the sandbox the Next.js dashboard talks to the Python core directly via
the CLI; in the full docker-compose deployment this FastAPI service exposes
the same capabilities over HTTP with Celery+Redis handling long-running
training jobs and PostgreSQL/MinIO providing durable storage.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qednet import RESEARCH_STATUS  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qednet.api")

app = FastAPI(
    title="QED-Net 2.0 API",
    description="Hybrid quantum-classical ML research platform API. "
    + RESEARCH_STATUS,
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[2]


def _cli(args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """Invoke the qednet CLI and parse its JSON stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "qednet.cli", *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
        env={"PYTHONPATH": str(ROOT / "backend"), "PATH": "/usr/bin:/usr/local/bin"},
    )
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
    experiment_id: str = Field(alias="experimentId")
    features: List[float]

    model_config = {"populate_by_name": True}


@app.get("/health")
def health() -> Dict[str, Any]:
    data = _cli(["env"], timeout=60)
    return {"ok": True, "backend": data, "research_status": RESEARCH_STATUS}


@app.get("/datasets")
def datasets() -> Dict[str, Any]:
    return _cli(["list-datasets"], timeout=120)


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...),
                         target: Optional[str] = Form(default=None),
                         name: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    if not file.filename.lower().endswith((".csv", ".tsv")):
        raise HTTPException(400, "Only CSV/TSV files are accepted")
    dest = ROOT / "data" / "uploads" / file.filename.replace("/", "_")
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(400, "File exceeds 25 MB limit")
    dest.write_bytes(content)
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
def train(background_tasks: Any, config: str = Form(...),
          experiment_id: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    """Start a training job. With Celery+Redis available the job is queued;
    otherwise it degrades to a background task in this process."""
    cfg_path = ROOT / "configs" / "experiments" / f"{config}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"Unknown config: {config}")
    exp_id = experiment_id or f"exp_{config}_{int(__import__('time').time())}"

    try:
        from celery import Celery  # noqa: F401
        queued = _queue_celery(str(cfg_path), exp_id)
    except Exception:  # noqa: BLE001 - Celery optional in this profile
        queued = None
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


def _queue_celery(cfg_path: str, exp_id: str):
    """Queue through Celery if a broker is reachable (best effort)."""
    import time
    from celery import Celery
    celery_app = Celery(__name__, broker="redis://redis:6379/0",
                        backend="redis://redis:6379/1")
    celery_app.send_task("qednet.train.run_experiment", args=[cfg_path, exp_id])
    return "celery queue"


@app.get("/experiments")
def experiments() -> Dict[str, Any]:
    return _cli(["list-experiments"], timeout=120)


@app.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str) -> Dict[str, Any]:
    return _cli(["get-experiment", "--id", experiment_id], timeout=300)


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    return _cli(["predict", "--id", req.experiment_id,
                 "--input-json", json.dumps(req.features)], timeout=300)


@app.post("/explain")
def explain(experiment_id: str = Form(...),
            model: Optional[str] = Form(default=None)) -> Dict[str, Any]:
    args = ["explain", "--id", experiment_id]
    if model:
        args += ["--model", model]
    return _cli(args, timeout=600)


@app.get("/benchmark/{experiment_id}")
def benchmark(experiment_id: str) -> Dict[str, Any]:
    data = _cli(["get-experiment", "--id", experiment_id], timeout=300)
    return {"ok": True, "benchmark": (data or {}).get("result", {}).get("models")}


@app.get("/certificate/{experiment_id}")
def certificate(experiment_id: str) -> Dict[str, Any]:
    return _cli(["certificate", "--id", experiment_id], timeout=300)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
