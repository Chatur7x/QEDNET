"""Durable-job worker tasks for QED-Net 2.0.

The lifecycle (job -> running -> succeeded/failed plus audit rows) lives in
``run_job_with_session``. The ARQ entrypoint and the real training routine
just delegate to it, so the state machine has one implementation and can be
unit-tested with an injected ``run_fn`` and no Redis/training infrastructure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from qednet.db import repositories as repo
from qednet.db.settings import StorageSettings

logger = logging.getLogger("qednet.worker")


def _session():
    """Return a sessionmaker bound to StorageSettings.database_url."""
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(bind=_engine(), expire_on_commit=False, future=True)


@lru_cache(maxsize=1)
def _engine():
    from sqlalchemy import create_engine
    return create_engine(
        StorageSettings.from_env().database_url, pool_pre_ping=True, future=True
    )


def _config_from_json(config_json: str):
    """Rebuild an ExperimentConfig from the stored JSON snapshot."""
    from qednet.config import load_config

    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(config_json)
        return load_config(Path(path))
    finally:
        os.unlink(path)


def _do_run(ctx: dict[str, Any]) -> dict[str, Any]:
    """Real heavy work: train, persist artifacts, attach them to the run.

    Runs in a worker thread with its own DB session. Heavy/optional imports
    are local so importing this module (and the unit tests) stays cheap.
    """
    from qednet.data.datasets import register_builtins
    from qednet.data.ingestion import DatasetRegistry
    from qednet.storage.artifacts import get_store
    from qednet.train.runner import run_experiment
    from qednet.tracking.store import ExperimentStore

    org_id = ctx["org_id"]
    experiment_id = ctx["experiment_id"]
    run_id = ctx.get("run_id")

    cfg = _config_from_json(ctx["config_json"])
    cfg.experiment_id = experiment_id

    store = ExperimentStore()
    registry = DatasetRegistry()
    register_builtins(registry)

    result = run_experiment(cfg, store, registry, run_explainability=True)

    artifacts = get_store()
    SessionLocal = _session()
    with SessionLocal() as session:
        if run_id is None:  # ponytail: helper normally creates it; defensive only
            exp = repo.get_experiment(session, org_id, experiment_id)
            run = repo.create_run(session, org_id, exp.id,
                                  config_hash=ctx.get("config_hash"))
            run_id = run.id

        # Result report
        result_json = json.dumps(result, default=str)
        result_key = f"runs/{experiment_id}/result.json"
        artifacts.put(result_key, result_json.encode(), "application/json")
        repo.add_report(session, org_id, run_id, report_type="result",
                        artifact_key=result_key,
                        size_bytes=len(result_json.encode()))

        # Certificate (produced by the runner when certificate.enabled)
        cert_path = store.experiment_dir(experiment_id) / "certificate.json"
        if cert_path.exists():
            cert = json.loads(cert_path.read_text())
            cert_json = json.dumps(cert, default=str)
            cert_key = f"runs/{experiment_id}/certificate.json"
            artifacts.put(cert_key, cert_json.encode(), "application/json")
            repo.upsert_certificate(
                session, org_id, run_id,
                classification=cert.get("final_evidence_classification", "unknown"),
                payload_json=cert_json, artifact_key=cert_key,
            )
            repo.add_report(session, org_id, run_id, report_type="certificate",
                            artifact_key=cert_key,
                            size_bytes=len(cert_json.encode()))

        repo.set_run_status(session, org_id, run_id, "succeeded",
                            runtime_s=result.get("total_runtime_s"))
        session.commit()

    return {**result, "ok": True, "run_id": str(run_id),
            "experiment_id": experiment_id}


async def run_job_with_session(
    session_factory: Callable[[], Any],
    *,
    org_id: str,
    job_id: str,
    experiment_id: str | None = None,
    config_hash: str | None = None,
    config_json: str | None = None,
    dataset: str | None = None,
    worker_id: str | None = None,
    run_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full durable-job lifecycle. Single source of truth for state + audit.

    ``run_fn`` is a sync callable receiving one context dict and returning a
    result dict; it is executed in a thread. Inject a fake to test without
    Redis, training or any external infrastructure.
    """
    # ARQ serializes job args to strings; SQLAlchemy's Uuid filter requires
    # UUID objects. Normalize at the boundary (idempotent for direct callers).
    import uuid

    if isinstance(org_id, str):
        org_id = uuid.UUID(org_id)
    if isinstance(job_id, str):
        job_id = uuid.UUID(job_id)

    run_fn = run_fn or _do_run
    session = session_factory()
    job = None
    run_id = None
    try:
        job = repo.get_job(session, org_id, job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found for org {org_id}")

        run_id = job.run_id
        if run_id is None:
            exp = repo.get_experiment(session, org_id, experiment_id)
            if exp is None:
                raise ValueError(
                    f"experiment {experiment_id} not found for org {org_id}"
                )
            run = repo.create_run(session, org_id, exp.id, config_hash=config_hash)
            run_id = run.id
            job.run_id = run.id

        repo.set_job_status(session, org_id, job_id, "running",
                            attempt=1, started=True, worker_id=worker_id)
        repo.append_audit(session, org_id, "job_started",
                          target_type="job", target_id=str(job_id))
        session.commit()  # durable "running" before the long-running work

        ctx = {
            "org_id": org_id, "job_id": job_id, "run_id": run_id,
            "experiment_id": experiment_id, "config_hash": config_hash,
            "config_json": config_json, "dataset": dataset,
            "worker_id": worker_id, "attempt": 1,
        }
        result = await asyncio.to_thread(run_fn, ctx)

        repo.set_run_status(
            session, org_id, run_id, "succeeded",
            runtime_s=result.get("runtime_s") or result.get("total_runtime_s"),
        )
        job.error = None  # a retried success must clear a prior failure's error
        repo.set_job_status(session, org_id, job_id, "succeeded", finished=True)
        repo.append_audit(session, org_id, "job_succeeded",
                          target_type="job", target_id=str(job_id))
        session.commit()
        return result
    except BaseException as exc:
        session.rollback()
        if job is not None:
            if run_id is not None:
                repo.set_run_status(session, org_id, run_id, "failed")
            repo.set_job_status(session, org_id, job_id, "failed",
                                error=str(exc), finished=True)
            repo.append_audit(
                session, org_id, "job_failed", target_type="job",
                target_id=str(job_id),
                metadata_json=json.dumps({"error": str(exc)}),
            )
            session.commit()
        raise  # let ARQ apply its own retry policy
    finally:
        session.close()


async def run_experiment_task(
    ctx: dict[str, Any],
    org_id: str,
    job_id: str,
    experiment_id: str,
    config_hash: str,
    config_json: str,
    dataset: str,
) -> dict[str, Any]:
    """ARQ job entrypoint. ``ctx`` is arq's per-worker context."""
    worker_id = str(ctx.get("job_id")) if isinstance(ctx, dict) and ctx.get("job_id") else None
    return await run_job_with_session(
        _session(),
        org_id=org_id, job_id=job_id, experiment_id=experiment_id,
        config_hash=config_hash, config_json=config_json, dataset=dataset,
        worker_id=worker_id,
    )
