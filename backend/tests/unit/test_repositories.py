"""Unit tests for the session-based repository layer."""
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qednet.db import repositories as repo
from qednet.db.models import (AuditLog, Base, Certificate, Dataset, Experiment,
                              Job, Organization, Report, Run)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    org = repo.ensure_org(session, "acme", "Acme")
    session.commit()
    yield session, org
    session.close()


def _experiment(session, org_id, eid="exp-1"):
    return repo.create_experiment(
        session, org_id, experiment_id=eid,
        config_hash="hash1", config_json='{"a": 1}',
    )


def _run(session, org_id, exp_id_db):
    return repo.create_run(session, org_id, exp_id_db, config_hash="hash1")


# ------------------------------------------------------------------- identity
def test_ensure_org_idempotent(db):
    session, org = db
    again = repo.ensure_org(session, "acme", "Renamed")
    assert again.id == org.id
    assert session.query(Organization).count() == 1
    assert again.name == "Acme"


# ------------------------------------------------------------------- datasets
def test_upsert_dataset_idempotent(db):
    session, org = db
    first = repo.upsert_dataset(session, org.id, name="ds1", source="csv", target="y")
    second = repo.upsert_dataset(
        session, org.id, name="ds1", source="csv", target="label",
        sha256="abc", artifact_key="k",
    )
    assert first.id == second.id
    assert session.query(Dataset).count() == 1
    assert second.target == "label"
    assert second.sha256 == "abc"


# ---------------------------------------------------------------- experiments
def test_create_experiment_idempotent(db):
    session, org = db
    first = _experiment(session, org.id)
    second = repo.create_experiment(
        session, org.id, experiment_id="exp-1",
        config_hash="different", config_json="{}",
    )
    assert first.id == second.id
    assert session.query(Experiment).count() == 1
    assert second.config_hash == "hash1"  # existing row untouched
    assert repo.get_experiment(session, org.id, "exp-1").id == first.id


# ----------------------------------------------------------------------- runs
def test_create_run_and_set_status(db):
    session, org = db
    exp = _experiment(session, org.id)
    run = _run(session, org.id, exp.id)
    assert run.id is not None
    assert run.status == "queued"

    updated = repo.set_run_status(session, org.id, run.id, "succeeded", runtime_s=1.25)
    assert updated.status == "succeeded"
    assert updated.runtime_s == 1.25
    session.commit()
    fresh = session.query(Run).filter(Run.id == run.id).one()
    assert fresh.status == "succeeded" and fresh.runtime_s == 1.25


# ---------------------------------------------------------------- certificates
def test_upsert_certificate_idempotent(db):
    session, org = db
    exp = _experiment(session, org.id)
    run = _run(session, org.id, exp.id)
    first = repo.upsert_certificate(
        session, org.id, run.id, classification="robust", payload_json='{"v":1}',
    )
    second = repo.upsert_certificate(
        session, org.id, run.id, classification="fragile", payload_json='{"v":2}',
        artifact_key="cert-key",
    )
    assert first.id == second.id
    assert session.query(Certificate).count() == 1
    assert second.classification == "fragile"
    assert second.artifact_key == "cert-key"


# --------------------------------------------------------------------- reports
def test_add_report(db):
    session, org = db
    exp = _experiment(session, org.id)
    run = _run(session, org.id, exp.id)
    report = repo.add_report(
        session, org.id, run.id, report_type="result",
        artifact_key="reports/1.pdf", checksum_sha256="deadbeef", size_bytes=42,
    )
    assert report.id is not None
    assert session.query(Report).count() == 1
    fresh = session.query(Report).filter(Report.id == report.id).one()
    assert fresh.report_type == "result"
    assert fresh.checksum_sha256 == "deadbeef" and fresh.size_bytes == 42


# ----------------------------------------------------------------------- jobs
def test_jobs(db):
    session, org = db
    job = repo.create_job(
        session, org.id, job_type="run_experiment", payload_json='{"x":1}',
    )
    assert job.id is not None
    assert repo.get_job(session, org.id, job.id).id == job.id

    updated = repo.set_job_status(
        session, org.id, job.id, "running",
        progress_json='{"pct":50}', worker_id="w1", attempt=2, started=True,
    )
    assert updated.status == "running"
    assert updated.progress_json == '{"pct":50}'
    assert updated.attempts == 2
    assert updated.worker_id == "w1"
    assert updated.started_at is not None

    finished = repo.set_job_status(
        session, org.id, job.id, "succeeded", finished=True,
    )
    assert finished.finished_at is not None
    assert len(repo.list_jobs(session, org.id)) == 1
    assert repo.list_jobs(session, org.id, status="succeeded")[0].id == job.id
    assert repo.list_jobs(session, org.id, status="failed") == []


# ---------------------------------------------------------------------- audit
def test_append_audit(db):
    session, org = db
    log = repo.append_audit(
        session, org.id, "experiment.created",
        target_type="experiment", target_id="exp-1", metadata_json='{"k":1}',
    )
    assert log.id is not None
    assert log.timestamp is not None
    assert session.query(AuditLog).count() == 1
    fresh = session.query(AuditLog).filter(AuditLog.id == log.id).one()
    assert fresh.action == "experiment.created"
    assert fresh.target_id == "exp-1"


# ---------------------------------------------------------- cross-org isolation
def test_cross_org_isolation(db):
    session, org = db
    other = repo.ensure_org(session, "other", "Other Org")
    session.commit()

    exp = _experiment(session, org.id)
    job = repo.create_job(session, org.id, job_type="t")

    assert repo.get_experiment(session, other.id, "exp-1") is None
    assert repo.get_job(session, other.id, job.id) is None
    assert repo.list_jobs(session, other.id) == []
    assert repo.list_jobs(session, other.id, status="queued") == []
