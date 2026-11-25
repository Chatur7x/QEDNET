from __future__ import annotations

import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as api


client = TestClient(api.app)


def test_cors_has_no_wildcard_origin():
    cors = next(m for m in api.app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert "*" not in cors.kwargs["allow_origins"]


def test_experiment_identifiers_reject_path_fragments():
    with pytest.raises(HTTPException) as exc:
        api._validate_id("../../outside", "experiment ID")
    assert exc.value.status_code == 400
    assert client.get("/experiments/invalid.id").status_code == 400


def test_train_rejects_path_like_config_names():
    response = client.post("/train", data={"config": "../secrets"})
    assert response.status_code == 400


def test_upload_uses_safe_unique_path_and_calls_real_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    called: dict[str, object] = {}

    def fake_cli(args, timeout=300):
        called["args"] = args
        called["timeout"] = timeout
        return {"ok": True, "dataset": "uploaded"}

    monkeypatch.setattr(api, "_cli", fake_cli)
    response = client.post(
        "/datasets/upload",
        files={"file": (r"..\outside.csv", b"a,target\n1,0\n", "text/csv")},
        data={"target": "target", "name": "demo"},
    )
    assert response.status_code == 200
    uploads = list((tmp_path / "data" / "uploads").iterdir())
    assert len(uploads) == 1
    assert uploads[0].parent == (tmp_path / "data" / "uploads")
    assert ".." not in uploads[0].name
    assert called["args"][0:2] == ["upload", "--csv"]


def test_upload_limit_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 3)
    response = client.post(
        "/datasets/upload",
        files={"file": ("large.csv", b"a,target\n1,0\n", "text/csv")},
    )
    assert response.status_code == 413
    assert not list((tmp_path / "data" / "uploads").glob("*"))


def test_cli_preserves_windows_path(monkeypatch):
    monkeypatch.setenv("PATH", "sentinel-path")
    seen: dict[str, object] = {}

    class Completed:
        stdout = '{"ok":true}\n'
        stderr = ""

    def fake_run(*args, **kwargs):
        seen["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    assert api._cli(["list-datasets"])["ok"] is True
    assert seen["env"]["PATH"] == "sentinel-path"


def test_cli_timeout_returns_gateway_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(api.subprocess, "run", timeout)
    with pytest.raises(HTTPException) as exc:
        api._cli(["list-datasets"], timeout=1)
    assert exc.value.status_code == 504
