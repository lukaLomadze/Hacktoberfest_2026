"""PLATFORM-API §5.1: demo login, logout, me."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request, Response

from .. import serialize as ser
from .. import validate as v
from ..auth import create_session, current
from ..context import hub_of
from ..errors import unauthorized

router = APIRouter()


@router.post("/auth/login")
def login(request: Request, payload: Any = Body(None)):
    body = v.body_object(payload)
    email = v.email(body.get("email"), "email")
    hub = hub_of(request)
    user = hub.db.one("SELECT * FROM users WHERE email = ?", (email,))
    if user is None:
        # the same answer for every unknown email: the login does not reveal who is registered
        raise unauthorized("unknown email")
    token = create_session(hub.db, user["id"], hub.settings.session_ttl_s)
    company = hub.db.one("SELECT * FROM companies WHERE id = ?", (user["company_id"],)) if user["company_id"] else None
    return {"token": token, "user": ser.user(user), "company": ser.company(company)}


@router.post("/auth/logout", status_code=204)
def logout(request: Request):
    p = current(request)
    hub_of(request).db.execute("DELETE FROM sessions WHERE token_hash = ?", (p.token_hash,))
    return Response(status_code=204)


@router.get("/auth/me")
def me(request: Request):
    p = current(request)
    return {"user": ser.user(p.user), "company": ser.company(p.company)}
