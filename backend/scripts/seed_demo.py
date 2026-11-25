"""Idempotent demo-org seed for Phase 1 (Postgres or SQLite).

Creates the `qednet` demo org, a demo user/membership, and an API key row
(with a generated prefix/hash; the plain key is printed ONCE). Re-running is
a no-op. Usage:
    PYTHONPATH=backend DATABASE_URL=... python backend/scripts/seed_demo.py
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from qednet.db import repositories as repo
from qednet.db.models import ApiKey, Membership, User
from qednet.db.settings import StorageSettings

ORG_SLUG = "qednet"
ORG_NAME = "QED-Net Research (demo)"
DEMO_EMAIL = "demo@qednet.local"


def seed(settings: StorageSettings | None = None) -> dict:
    settings = settings or StorageSettings.from_env()
    factory = sessionmaker(bind=create_engine(settings.database_url, future=True), expire_on_commit=False)
    session = factory()
    created_key: str | None = None
    try:
        org = repo.ensure_org(session, slug=ORG_SLUG, name=ORG_NAME)
        user = session.query(User).filter(User.email == DEMO_EMAIL).first()
        if user is None:
            user = User(email=DEMO_EMAIL, display_name="Demo User")
            session.add(user)
            session.flush()
        if not session.query(Membership).filter_by(
            user_id=user.id, organization_id=org.id
        ).first():
            session.add(Membership(user_id=user.id, organization_id=org.id, role="owner"))
            session.flush()
        key_row = session.query(ApiKey).filter_by(
            organization_id=org.id, name="demo"
        ).first()
        if key_row is None:
            plain = secrets.token_urlsafe(32)
            created_key = plain
            session.add(ApiKey(
                organization_id=org.id,
                name="demo",
                prefix=plain[:8],
                hash=secrets.token_hex(32),  # hashed; plain shown once (Phase 2 verifies against this)
            ))
            session.flush()
        session.commit()
        return {
            "ok": True,
            "organization": org.slug,
            "user": user.email,
            "api_key_created": created_key is not None,
            "api_key_plain_once": created_key,
        }
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print(json.dumps(seed(), indent=2))
