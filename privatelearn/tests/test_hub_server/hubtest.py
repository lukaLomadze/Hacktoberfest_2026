"""Shared helpers for the platform-server tests (not a test module itself).

Run from privatelearn/:  python -m unittest discover -s tests/test_hub_server -v
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]  # privatelearn
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from hub.server.app import create_app  # noqa: E402
from hub.server.context import Settings  # noqa: E402
from hub.server.seed import ADMIN_EMAIL, COMPANIES  # noqa: E402

TBILISI, BATUMI, KUTAISI, RUSTAVI, GORI = (c[2] for c in COMPANIES)


def make_client(seed_mode="dev", settings=None, federation=None, db_path=":memory:"):
    app = create_app(db_path, seed_mode=seed_mode, settings=settings or Settings(), federation=federation)
    return TestClient(app)


def login(client, email=ADMIN_EMAIL) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": email})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def err(r) -> dict:
    """The §1.1 error object of a failed response."""
    body = r.json()
    assert set(body) == {"error"}, body
    assert set(body["error"]) == {"code", "message", "field"}, body
    return body["error"]
