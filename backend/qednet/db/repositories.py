"""Session-based repository functions for the Phase 1 persistence layer.

Every public function takes ``session`` first and ``org_id`` second so the
Phase 2 tenancy-scoping hook has one uniform place to land. Idempotent
functions use query-then-insert, every writer calls ``session.flush()`` so
generated IDs are available on the returned object, and NO function commits:
the caller owns the transaction boundary. Timestamps from model defaults are
left alone.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from qednet.db.models import (
    AuditLog,
    Certificate,
    Dataset,
    Experiment,
    Job,
    Organization,
    Report,
    Run,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_org(session: Session, slug: str, name: str) -> Organization:
    """Idempotent by slug."""
    org = session.query(Organization).filter(Organization.slug == slug).first()
    if org is None:
        org = Organization(slug=slug, name=name)
        session.add(org)
        session.flush()
    return org


def upsert_dataset(
    session: Session,
    org_id,
    *,
    name: str,
    source: str,
    target: str,
    sha256: str | None = None,
    profile_json: str | None = None,
    artifact_key: str | None = None,
) -> Dataset:
    """Idempotent by (org_id, name); refreshes the supplied fields."""
    ds = (
        session.query(Dataset)
        .filter(Dataset.organization_id == org_id, Dataset.name == name)
        .first()
    )
    if ds is None:
        ds = Dataset(organization_id=org_id, name=name)
        session.add(ds)
    ds.source = source
    ds.target = target
    ds.sha256 = sha256
    ds.profile_json = profile_json
    ds.artifact_key = artifact_key
    session.flush()
    return ds


def create_experiment(
    session: Session,
    org_id,
    *,
    experiment_id: str,
    config_hash: str,
    config_json: str,
    dataset_id=None,
    created_by=None,
    status: str = "pending",
) -> Experiment:
    """Idempotent by (org_id, experiment_id); existing rows are returned untouched."""
    exp = (
        session.query(Experiment)
        .filter(
            Experiment.organization_id == org_id,
            Experiment.experiment_id == experiment_id,
        )
        .first()
    )
    if exp is None:
        exp = Experiment(
            organization_id=org_id,
            experiment_id=experiment_id,
            config_hash=config_hash,
            config_json=config_json,
            dataset_id=dataset_id,
            created_by=created_by,
            status=status,
        )
        session.add(exp)
        session.flush()
    return exp


def get_experiment(session: Session, org_id, experiment_id: str) -> Experiment | None:
    return (
        session.query(Experiment)
        .filter(
            Experiment.organization_id == org_id,
            Experiment.experiment_id == experiment_id,
        )
        .first()
    )


def create_run(
    session: Session,
    org_id,
    experiment_id_db,
    *,
    status: str = "queued",
    protocol_json: str | None = None,
    config_hash: str | None = None,
) -> Run:
    run = Run(
        organization_id=org_id,
        experiment_id_db=experiment_id_db,
        status=status,
        protocol_json=protocol_json,
        config_hash=config_hash,
    )
    session.add(run)
    session.flush()
    return run


def set_run_status(
    session: Session,
    org_id,
    run_id,
    status: str,
    *,
    runtime_s: float | None = None,
) -> Run:
    run = session.query(Run).filter(Run.organization_id == org_id, Run.id == run_id).one()
    run.status = status
    if runtime_s is not None:
        run.runtime_s = runtime_s
    session.flush()
    return run


def upsert_certificate(
    session: Session,
    org_id,
    run_id,
    *,
    classification: str,
    payload_json: str,
    artifact_key: str | None = None,
) -> Certificate:
    """Idempotent by run_id; refreshes the supplied fields."""
    cert = (
        session.query(Certificate)
        .filter(Certificate.organization_id == org_id, Certificate.run_id == run_id)
        .first()
    )
    if cert is None:
        cert = Certificate(organization_id=org_id, run_id=run_id)
        session.add(cert)
    cert.classification = classification
    cert.payload_json = payload_json
    cert.artifact_key = artifact_key
    session.flush()
    return cert


def add_report(
    session: Session,
    org_id,
    run_id,
    *,
    report_type: str,
    artifact_key: str,
    checksum_sha256: str | None = None,
    size_bytes: int | None = None,
) -> Report:
    report = Report(
        organization_id=org_id,
        run_id=run_id,
        report_type=report_type,
        artifact_key=artifact_key,
        checksum_sha256=checksum_sha256,
        size_bytes=size_bytes,
    )
    session.add(report)
    session.flush()
    return report


def create_job(
    session: Session,
    org_id,
    *,
    job_type: str,
    run_id=None,
    payload_json: str | None = None,
) -> Job:
    job = Job(
        organization_id=org_id,
        job_type=job_type,
        run_id=run_id,
        payload_json=payload_json,
    )
    session.add(job)
    session.flush()
    return job


def get_job(session: Session, org_id, job_id) -> Job | None:
    return (
        session.query(Job)
        .filter(Job.organization_id == org_id, Job.id == job_id)
        .first()
    )


def set_job_status(
    session: Session,
    org_id,
    job_id,
    status: str,
    *,
    progress_json: str | None = None,
    error: str | None = None,
    worker_id: str | None = None,
    attempt: int | None = None,
    started: bool = False,
    finished: bool = False,
) -> Job:
    job = session.query(Job).filter(Job.organization_id == org_id, Job.id == job_id).one()
    job.status = status
    if progress_json is not None:
        job.progress_json = progress_json
    if error is not None:
        job.error = error
    if worker_id is not None:
        job.worker_id = worker_id
    if attempt is not None:
        job.attempts = attempt
    if started and job.started_at is None:
        job.started_at = _utcnow()
    if finished and job.finished_at is None:
        job.finished_at = _utcnow()
    session.flush()
    return job


def list_jobs(session: Session, org_id, status: str | None = None) -> list[Job]:
    query = session.query(Job).filter(Job.organization_id == org_id)
    if status is not None:
        query = query.filter(Job.status == status)
    return query.order_by(Job.created_at).all()


def append_audit(
    session: Session,
    org_id,
    action: str,
    *,
    actor_id=None,
    target_type: str | None = None,
    target_id: str | None = None,
    metadata_json: str | None = None,
) -> AuditLog:
    log = AuditLog(
        organization_id=org_id,
        action=action,
        actor_id=actor_id,
        target_type=target_type,
        target_id=target_id,
        metadata_json=metadata_json,
    )
    session.add(log)
    session.flush()
    return log
