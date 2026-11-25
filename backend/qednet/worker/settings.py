"""ARQ worker settings for the durable-job queue.

Importing this module never requires arq to be installed: the optional import
is guarded so unit tests (and the research CLI) stay infrastructure-free.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .tasks import run_experiment_task

logger = logging.getLogger("qednet.worker")

try:  # arq lives in the optional `api` extra
    from arq.connections import RedisSettings  # type: ignore
except Exception:  # pragma: no cover - only on installs without arq
    class RedisSettings:  # plain fallback; never used to open a connection
        @classmethod
        def from_dsn(cls, dsn: str):  # noqa: N805
            return dsn

DEFAULT_REDIS_URL = "redis://localhost:6379/0"


async def _on_startup(ctx) -> None:
    logger.info("qednet worker started")


async def _on_shutdown(ctx) -> None:
    logger.info("qednet worker stopped")


@dataclass
class WorkerSettings:
    """ARQ settings class; class attributes below are read by arq's CLI/Worker."""

    redis_url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    )

    functions = [run_experiment_task]
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    job_timeout = 24 * 3600
    redis_settings = RedisSettings.from_dsn(
        os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    )
