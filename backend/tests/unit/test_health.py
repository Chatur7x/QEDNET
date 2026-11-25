"""Unit tests for the /health (liveness) and /ready (readiness) probes."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qednet.api import health as health_mod  # noqa: E402
from qednet.db.settings import StorageSettings  # noqa: E402


def test_health_is_liveness_only():
    payload = health_mod.health()

    assert payload["status"] == "ok"
    assert payload["qednet_version"] == health_mod.QEDNET_VERSION
    assert "not a medical device" in payload["research_status"]
    assert "not clinically validated" in payload["research_status"]
    assert "dependencies" not in payload
    assert "use_db_storage" not in payload


def test_readiness_demo_mode_skips_all_connections():
    settings = replace(StorageSettings.from_env(), use_db_storage=False)

    with patch("sqlalchemy.create_engine") as create_engine, \
            patch("redis.from_url") as from_url, \
            patch("qednet.storage.artifacts.get_store") as get_store:
        payload, status = health_mod.readiness(settings)

    assert status == 200
    assert payload["status"] == "ready"
    assert payload["mode"] == "filesystem_demo"
    assert payload["use_db_storage"] is False
    assert set(payload["dependencies"]) == {"postgres", "redis", "s3"}
    for dep in payload["dependencies"].values():
        assert dep["ok"] is True
        assert dep["status"] == "not_configured"
    assert "not a medical device" in payload["research_status"]
    create_engine.assert_not_called()
    from_url.assert_not_called()
    get_store.assert_not_called()


def test_readiness_db_mode_unreachable_returns_503():
    settings = replace(
        StorageSettings.from_env(),
        use_db_storage=True,
        database_url="postgresql+psycopg://qednet:qednet@127.0.0.1:1/qednet",
        redis_url="redis://127.0.0.1:1/0",
    )

    class _FakeStore:
        def exists(self, key):
            return False

    # Keep S3 deterministic; the unreachable Postgres/Redis checks are real.
    with patch("qednet.storage.artifacts.get_store", return_value=_FakeStore()):
        payload, status = health_mod.readiness(settings)

    assert status == 503
    assert payload["status"] == "unready"
    assert payload["use_db_storage"] is True
    for name in ("postgres", "redis"):
        dep = payload["dependencies"][name]
        assert dep["ok"] is False
        assert isinstance(dep["error"], str) and dep["error"]
    assert payload["dependencies"]["s3"]["ok"] is True
