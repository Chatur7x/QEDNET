"""Unit tests for the artifact storage adapters (no infrastructure needed)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qednet.storage import InMemoryArtifactStore  # noqa: E402


def test_inmemory_round_trip():
    store = InMemoryArtifactStore()
    key = "run/1/certificate.json"
    payload = b'{"ok": true}'

    assert store.put(key, payload, "application/json") == key
    assert store.get(key) == payload
    assert store.exists(key) is True

    url = store.signed_url(key)
    assert url.startswith("mem://")
    assert key in url

    store.delete(key)
    assert store.exists(key) is False


def test_inmemory_signed_url_ttl():
    store = InMemoryArtifactStore()
    assert store.signed_url("k") == (
        f"mem://k?expires={InMemoryArtifactStore.DEFAULT_TTL_S}"
    )
    assert store.signed_url("k", expires_s=60) == "mem://k?expires=60"
