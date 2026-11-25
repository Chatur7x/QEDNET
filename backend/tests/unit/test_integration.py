"""Phase 1 API integration tests — DB job lifecycle with enqueue mocked."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from qednet.api import integration
from qednet.db.models import Base, Experiment, Job, Run
from qednet.db.settings import StorageSettings

import pytest


@pytest.fixture()
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'integration.sqlite'}"
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    settings = StorageSettings(use_db_storage=True, database_url=url)
    return factory, settings


def _call(start, factory, settings, *, fail_enqueue=False):
    with patch.object(integration, "enqueue_arq") as enq:
        if fail_enqueue:
            enq.side_effect = RuntimeError("redis down")
        return start(
            settings,
            cfg_path=Path("configs/experiments/breast_cancer.yaml"),
            experiment_id="exp_integration_test",
            config_json=json.dumps({"dataset": {"name": "breast_cancer"}}),
            config_hash="deadbeef1234",
            dataset="breast_cancer",
        )


def test_start_db_job_persists_rows_and_enqueues(db):
    factory, settings = db
    result = _call(integration.start_db_job, factory, settings)
    assert result["ok"] is True and result["execution"] == "arq"
    s = factory()
    try:
        assert s.scalar(select(func.count()).select_from(Experiment)) == 1
        assert s.scalar(select(func.count()).select_from(Run)) == 1
        jobs = s.scalars(select(Job)).all()
        assert len(jobs) == 1 and jobs[0].status == "queued"
        assert jobs[0].job_type == "run_experiment"
    finally:
        s.close()


def test_enqueue_failure_marks_job_failed(db):
    factory, settings = db
    with pytest.raises(RuntimeError, match="redis down"):
        _call(integration.start_db_job, factory, settings, fail_enqueue=True)
    s = factory()
    try:
        job = s.scalar(select(Job))
        assert job.status == "failed" and "redis down" in job.error
    finally:
        s.close()
