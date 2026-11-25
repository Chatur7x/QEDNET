"""Phase 1 API integration — DB/queue path gated by USE_DB_STORAGE.

Phase 0 demo mode keeps the CLI/in-process path untouched. When USE_DB_STORAGE
is set, training creates durable org/experiment/run/job rows and enqueues an
ARQ task; the worker owns execution and status transitions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qednet.db.settings import StorageSettings
from qednet.db import repositories as repo

logger = logging.getLogger("qednet.api.integration")
DEMO_ORG_SLUG = "qednet"
DEMO_ORG_NAME = "QED-Net Research (demo)"


def session_factory(settings: StorageSettings) -> sessionmaker:
    return sessionmaker(
        bind=create_engine(settings.database_url, pool_pre_ping=True, future=True),
        expire_on_commit=False,
        future=True,
    )


def start_db_job(
    settings: StorageSettings,
    *,
    cfg_path: Path,
    experiment_id: str,
    config_json: str,
    config_hash: str,
    dataset: str,
) -> dict[str, Any]:
    """Persist experiment/run/job rows and enqueue the ARQ task.

    Returns {"ok": True, "experiment_id":..., "job_id":..., "execution":"arq"}.
    Raises on enqueue failure (the caller surfaces a 503) — no orphaned
    "queued" job is left behind: the job row is created only after enqueue.
    """
    factory = session_factory(settings)
    session = factory()
    try:
        org = repo.ensure_org(session, slug=DEMO_ORG_SLUG, name=DEMO_ORG_NAME)
        exp = repo.create_experiment(
            session, org.id,
            experiment_id=experiment_id,
            config_hash=config_hash,
            config_json=config_json,
            status="queued",
        )
        run = repo.create_run(
            session, org.id, exp.id,
            status="queued", config_hash=config_hash,
            protocol_json=json.dumps({
                "config_path": str(cfg_path), "dataset": dataset,
            }),
        )
        session.commit()
        # Keep UUID objects for repository filters (SQLite Uuid type requires
        # them); stringify only at the ARQ/enqueue and response boundary.
        org_uuid, run_uuid = org.id, run.id
        # Job row is created before enqueue so the worker can find it; on
        # enqueue failure we mark it failed rather than leaving it dangling.
        job = repo.create_job(
            session, org.id,
            job_type="run_experiment", run_id=run.id,
            payload_json=json.dumps({"experiment_id": experiment_id}),
        )
        session.commit()
        job_uuid = job.id
    finally:
        session.close()

    try:
        enqueue_arq(settings, str(org_uuid), str(job_uuid), experiment_id, config_hash, config_json, dataset)
    except Exception as exc:  # noqa: BLE001
        s2 = factory()
        try:
            repo.set_job_status(s2, org_uuid, job_uuid, "failed", error=f"enqueue failed: {exc}", finished=True)
            s2.commit()
        finally:
            s2.close()
        raise

    return {
        "ok": True,
        "experiment_id": experiment_id,
        "job_id": str(job_uuid),
        "run_id": str(run_uuid),
        "execution": "arq",
        "research_status": "research only; not a medical device",
    }


def enqueue_arq(
    settings: StorageSettings,
    org_id: str,
    job_id: str,
    experiment_id: str,
    config_hash: str,
    config_json: str,
    dataset: str,
) -> None:
    """Enqueue the durable task. Requires arq + a reachable Redis."""
    from arq import create_pool
    from arq.connections import RedisSettings
    from qednet.worker.tasks import run_experiment_task

    rs = RedisSettings.from_dsn(settings.redis_url)
    import asyncio

    async def _enqueue() -> None:
        redis = await create_pool(rs)
        try:
            await redis.enqueue_job(
                run_experiment_task.__name__,
                org_id, job_id, experiment_id, config_hash, config_json, dataset,
            )
        finally:
            await redis.close()

    asyncio.run(_enqueue())
