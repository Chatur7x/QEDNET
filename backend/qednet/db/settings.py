"""Environment-driven settings for persistence/queue/object storage.

Phase 0 used filesystem JSON and detached-process jobs. Phase 1 introduces
Postgres (metadata), Redis/ARQ (durable queue), and MinIO/S3 (artifacts),
gated by USE_DB_STORAGE so the research CLI keeps working unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StorageSettings:
    use_db_storage: bool = False
    database_url: str = "postgresql+psycopg://qednet:qednet@localhost:5432/qednet"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "qednet-artifacts"
    s3_secure: bool = False
    signed_url_ttl_s: int = 300

    @classmethod
    def from_env(cls) -> "StorageSettings":
        return cls(
            use_db_storage=_flag("USE_DB_STORAGE"),
            database_url=os.environ.get("DATABASE_URL", cls.database_url),
            redis_url=os.environ.get("REDIS_URL", cls.redis_url),
            s3_endpoint=os.environ.get("S3_ENDPOINT", cls.s3_endpoint),
            s3_access_key=os.environ.get("S3_ACCESS_KEY", cls.s3_access_key),
            s3_secret_key=os.environ.get("S3_SECRET_KEY", cls.s3_secret_key),
            s3_bucket=os.environ.get("S3_BUCKET", cls.s3_bucket),
            s3_secure=_flag("S3_SECURE"),
            signed_url_ttl_s=int(os.environ.get("SIGNED_URL_TTL_S", cls.signed_url_ttl_s)),
        )


@dataclass(frozen=True)
class Database:
    """SQLAlchemy engine/session factory holder (created lazily)."""

    url: str

    def engine(self):  # pragma: no cover - exercised with a real DB in Phase 1 DoD
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(self.url, pool_pre_ping=True, future=True)
        return engine, sessionmaker(bind=engine, expire_on_commit=False, future=True)
