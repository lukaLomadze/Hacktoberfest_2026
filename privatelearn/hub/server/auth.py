"""Demo authentication: sign in with an email, get a bearer token.

There is no password (PLATFORM-API §5.1): this is a judges' demo, not a production
login, and the docs say so. What is still done properly: tokens are random
(`secrets`), only their SHA-256 is stored, sessions expire, and every route checks the
role and the company before touching data.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass

from fastapi import Request

from .context import hub_of
from .db import now_iso
from .errors import forbidden, invalid_transition, unauthorized


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(db, user_id: str, ttl_s: float) -> str:
    token = new_token()
    with db.tx() as c:
        c.execute("DELETE FROM sessions WHERE expires_epoch < ?", (time.time(),))
        c.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_epoch) VALUES (?,?,?,?)",
                  (hash_token(token), user_id, now_iso(), time.time() + ttl_s))
    return token


def bearer_token(request: Request):
    value = request.headers.get("authorization", "")
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


@dataclass
class Principal:
    user: dict
    company: dict  # None for the admin
    token_hash: str

    @property
    def is_admin(self) -> bool:
        return self.user["role"] == "admin"

    @property
    def company_id(self):
        return self.company["id"] if self.company else None


def current(request: Request, *, allow_query: bool = False) -> Principal:
    """The signed-in user. `allow_query` accepts `?token=` (EventSource cannot set headers)."""
    token = bearer_token(request)
    if token is None and allow_query:
        token = request.query_params.get("token") or None
    if token is None:
        raise unauthorized()
    db = hub_of(request).db
    th = hash_token(token)
    row = db.one("SELECT s.expires_epoch, u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                 "WHERE s.token_hash = ?", (th,))
    if row is None:
        raise unauthorized("session not found; sign in again")
    if row.pop("expires_epoch") < time.time():
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (th,))
        raise unauthorized("session expired; sign in again")
    company = None
    if row["company_id"]:
        company = db.one("SELECT * FROM companies WHERE id = ?", (row["company_id"],))
    return Principal(user=row, company=company, token_hash=th)


def require_admin(p: Principal) -> Principal:
    if not p.is_admin:
        raise forbidden("admin only")
    return p


def require_company(p: Principal) -> dict:
    if p.is_admin or p.company is None:
        raise forbidden("company accounts only")
    return p.company


def require_approved_company(p: Principal) -> dict:
    company = require_company(p)
    if company["status"] != "approved":
        raise invalid_transition(f"your company is {company['status']}; an admin must approve it first")
    return company
