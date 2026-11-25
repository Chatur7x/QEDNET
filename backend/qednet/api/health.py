"""Health and readiness probes for QED-Net 2.0.

Liveness (`health`) never touches dependencies. Readiness (`readiness`) checks
Postgres, Redis and S3/MinIO when USE_DB_STORAGE is enabled; in Phase 0
filesystem-demo mode it reports them as not configured without connecting.
"""
from __future__ import annotations

from qednet import RESEARCH_STATUS, __version__ as QEDNET_VERSION
from qednet.db.settings import StorageSettings

HEALTHCHECK_KEY = "__healthcheck__"


def health(settings: StorageSettings | None = None) -> dict:
    """Liveness probe: process is up. No dependency I/O, always 200-capable."""
    return {
        "status": "ok",
        "research_status": RESEARCH_STATUS,
        "qednet_version": QEDNET_VERSION,
    }


def _check_postgres(settings: StorageSettings) -> dict:
    from sqlalchemy import create_engine

    url = settings.database_url
    if url.startswith("postgresql"):
        engine = create_engine(url, connect_args={"connect_timeout": 2})
    else:
        engine = create_engine(url)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    finally:
        engine.dispose()
    return {"ok": True, "error": None}


def _check_redis(settings: StorageSettings) -> dict:
    import redis

    client = redis.from_url(
        settings.redis_url, socket_connect_timeout=1, socket_timeout=1)
    client.ping()
    return {"ok": True, "error": None}


def _check_s3(settings: StorageSettings) -> dict:
    from qednet.storage.artifacts import get_store

    # A missing object is still a live connection: exists() returns False
    # without raising, so returning at all means the check passed.
    get_store(settings).exists(HEALTHCHECK_KEY)
    return {"ok": True, "error": None}


def _not_configured() -> dict:
    return {"ok": True, "error": None, "status": "not_configured"}


def readiness(settings: StorageSettings | None = None) -> tuple[dict, int]:
    """Readiness probe: verify required dependencies, return (payload, status)."""
    settings = settings or StorageSettings.from_env()
    if not settings.use_db_storage:
        payload = {
            "status": "ready",
            "research_status": RESEARCH_STATUS,
            "dependencies": {
                name: _not_configured() for name in ("postgres", "redis", "s3")
            },
            "use_db_storage": False,
            "mode": "filesystem_demo",
        }
        return payload, 200

    dependencies: dict[str, dict] = {}
    for name, check in (
        ("postgres", _check_postgres),
        ("redis", _check_redis),
        ("s3", _check_s3),
    ):
        try:
            dependencies[name] = check(settings)
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            dependencies[name] = {"ok": False, "error": str(exc)}

    ok = all(dep["ok"] for dep in dependencies.values())
    payload = {
        "status": "ready" if ok else "unready",
        "research_status": RESEARCH_STATUS,
        "dependencies": dependencies,
        "use_db_storage": True,
    }
    return payload, 200 if ok else 503
