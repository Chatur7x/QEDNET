"""Infra-free unit tests for the durable-job worker lifecycle.

No Redis, no training, no external services: in-memory SQLite plus an injected
fake ``run_fn``. The real (heavy) ARQ task path is deliberately never called.
"""
import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qednet.db import repositories as repo
from qednet.db.models import AuditLog, Base, Job, Run
from qednet.worker.settings import WorkerSettings
from qednet.worker.tasks import run_experiment_task, run_job_with_session


def _fixture():
    """Create org + experiment + run + queued job in a fresh in-memory DB."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    seed = factory()
    org = repo.ensure_org(seed, "acme", "Acme")
    exp = repo.create_experiment(
        seed, org.id, experiment_id="exp-1",
        config_hash="hash1", config_json='{"experiment_id": "exp-1"}',
    )
    run = repo.create_run(seed, org.id, exp.id, config_hash="hash1")
    job = repo.create_job(
        seed, org.id, job_type="run_experiment", run_id=run.id,
        payload_json=json.dumps({"experiment_id": "exp-1"}),
    )
    seed.commit()
    ids = (org.id, exp.id, run.id, job.id)
    seed.close()
    return factory, ids


def _audit_actions(session, org_id):
    return [
        a.action for a in session.query(AuditLog)
        .filter(AuditLog.organization_id == org_id)
        .order_by(AuditLog.timestamp, AuditLog.id).all()
    ]


def test_success_lifecycle():
    factory, (org_id, _exp_id, run_id, job_id) = _fixture()

    def fake_run_fn(ctx):
        # The helper has marked the job running and handed over a run id.
        assert ctx["run_id"] == run_id
        assert ctx["experiment_id"] == "exp-1"
        assert ctx["attempt"] == 1
        return {"ok": True, "experiment_id": "exp-1"}

    result = asyncio.run(run_job_with_session(
        factory, org_id=org_id, job_id=job_id, experiment_id="exp-1",
        config_hash="hash1", config_json='{"experiment_id": "exp-1"}',
        dataset="breast_cancer", run_fn=fake_run_fn,
    ))

    assert result["ok"] is True
    session = factory()
    job = repo.get_job(session, org_id, job_id)
    assert job.status == "succeeded"            # queued -> running -> succeeded
    assert job.attempts == 1
    assert job.started_at is not None and job.finished_at is not None
    assert job.error is None
    run = session.query(Run).filter(Run.id == run_id).one()
    assert run.status == "succeeded"
    assert _audit_actions(session, org_id) == ["job_started", "job_succeeded"]
    session.close()


def test_string_ids_from_arq_are_normalized():
    """ARQ serializes job args to strings; ensure the boundary converts them."""
    factory, (org_id, _exp_id, run_id, job_id) = _fixture()

    def fake_run_fn(ctx):
        # Boundary normalizes ARQ strings to UUID objects before DB/ctx use.
        assert ctx["org_id"] == org_id
        assert ctx["job_id"] == job_id
        return {"ok": True}

    asyncio.run(run_job_with_session(
        factory, org_id=str(org_id), job_id=str(job_id), experiment_id="exp-1",
        config_hash="hash1", config_json="{}", dataset="breast_cancer",
        run_fn=fake_run_fn,
    ))
    session = factory()
    assert repo.get_job(session, org_id, job_id).status == "succeeded"
    session.close()


def test_failure_lifecycle_reraises():
    factory, (org_id, _exp_id, run_id, job_id) = _fixture()

    def boom(ctx):
        raise RuntimeError("training exploded")

    with pytest.raises(RuntimeError, match="training exploded"):
        asyncio.run(run_job_with_session(
            factory, org_id=org_id, job_id=job_id, experiment_id="exp-1",
            run_fn=boom,
        ))

    session = factory()
    job = repo.get_job(session, org_id, job_id)
    assert job.status == "failed"               # queued -> running -> failed
    assert job.attempts == 1
    assert "training exploded" in job.error
    assert job.finished_at is not None
    run = session.query(Run).filter(Run.id == run_id).one()
    assert run.status == "failed"
    assert _audit_actions(session, org_id) == ["job_started", "job_failed"]
    session.close()


def test_worker_settings_shape():
    assert WorkerSettings.functions == [run_experiment_task]
    assert WorkerSettings.job_timeout == 24 * 3600
    assert isinstance(WorkerSettings().redis_url, str)
