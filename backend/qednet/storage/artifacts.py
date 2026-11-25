"""Artifact storage adapters.

Keys are opaque strings; content-type is passed through unchanged. The
InMemory store needs no infrastructure (unit tests / research CLI default),
the MinIO store persists to S3 when USE_DB_STORAGE is set.
"""
from __future__ import annotations

import io
from datetime import timedelta
from typing import Protocol

from minio import Minio
from minio.error import S3Error

from ..db.settings import StorageSettings


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes,
            content_type: str = "application/octet-stream") -> str: ...

    def get(self, key: str) -> bytes: ...

    def signed_url(self, key: str, expires_s: int | None = None) -> str: ...

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...


class InMemoryArtifactStore:
    DEFAULT_TTL_S = 300

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}

    def put(self, key: str, data: bytes,
            content_type: str = "application/octet-stream") -> str:
        self._objects[key] = (data, content_type)
        return key

    def get(self, key: str) -> bytes:
        return self._objects[key][0]

    def signed_url(self, key: str, expires_s: int | None = None) -> str:
        ttl = self.DEFAULT_TTL_S if expires_s is None else expires_s
        return f"mem://{key}?expires={ttl}"

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._objects


class MinioArtifactStore:
    def __init__(self, settings: StorageSettings):
        self._bucket = settings.s3_bucket
        self._ttl = settings.signed_url_ttl_s
        self._client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put(self, key: str, data: bytes,
            content_type: str = "application/octet-stream") -> str:
        self._client.put_object(self._bucket, key, io.BytesIO(data), len(data),
                                content_type=content_type)
        return key

    def get(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def signed_url(self, key: str, expires_s: int | None = None) -> str:
        ttl = self._ttl if expires_s is None else expires_s
        return self._client.presigned_get_object(
            self._bucket, key, expires=timedelta(seconds=ttl))

    def delete(self, key: str) -> None:
        self._client.remove_object(self._bucket, key)

    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise


def get_store(settings: StorageSettings | None = None) -> ArtifactStore:
    settings = settings or StorageSettings.from_env()
    if settings.use_db_storage:
        return MinioArtifactStore(settings)
    return InMemoryArtifactStore()
