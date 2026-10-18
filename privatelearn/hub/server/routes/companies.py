"""PLATFORM-API §5.2: company registration and the admin's approval decisions.

Company lifecycle (§3): pending → approved | rejected; approved ⇄ suspended.
`approve` also reinstates a suspended company. A rejected company stays rejected:
it can still sign in to see why, but cannot act.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Body, Request

from .. import serialize as ser
from .. import validate as v
from ..auth import create_session, current, require_admin
from ..context import hub_of
from ..db import new_id, now_iso
from ..enums import COMPANY_STATUS
from ..errors import conflict, forbidden, invalid_transition
from ..queries import company_or_404

router = APIRouter()


def _name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


@router.post("/companies/register", status_code=201)
def register(request: Request, payload: Any = Body(None)):
    body = v.body_object(payload)
    v.reject_unknown(body, ("name", "field", "contact_email", "wallet_address"))
    name = " ".join(v.string(body.get("name"), "name", min_len=2, max_len=120).split())
    field = v.string(body.get("field"), "field", max_len=60)
    email = v.email(body.get("contact_email"), "contact_email")
    wallet = v.wallet(body.get("wallet_address"), "wallet_address")
    hub = hub_of(request)
    cid, uid, t = new_id("co"), new_id("usr"), now_iso()
    try:
        with hub.db.tx() as c:
            if c.execute("SELECT 1 FROM companies WHERE name_key = ?", (_name_key(name),)).fetchone():
                raise conflict(f"a company called {name!r} is already registered", "name")
            if c.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                raise conflict("this email is already registered; sign in instead", "contact_email")
            c.execute("INSERT INTO companies (id, name, name_key, field, contact_email, status, wallet_address, "
                      "created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (cid, name, _name_key(name), field, email, "pending", wallet, t))
            c.execute("INSERT INTO users (id, email, role, company_id, created_at) VALUES (?,?,?,?,?)",
                      (uid, email, "company", cid, t))
    except sqlite3.IntegrityError:  # a concurrent registration won the race
        raise conflict("this company or email was registered at the same moment; sign in instead")
    token = create_session(hub.db, uid, hub.settings.session_ttl_s)
    return {"token": token, "user": ser.user(hub.db.one("SELECT * FROM users WHERE id = ?", (uid,))),
            "company": ser.company(company_or_404(hub.db, cid))}


@router.get("/companies")
def list_companies(request: Request, status: Optional[str] = None):
    require_admin(current(request))
    db = hub_of(request).db
    if status is None:
        rows = db.all("SELECT * FROM companies ORDER BY created_at DESC, name")
    else:
        v.enum(status, "status", COMPANY_STATUS)
        rows = db.all("SELECT * FROM companies WHERE status = ? ORDER BY created_at DESC, name", (status,))
    return {"items": [ser.company(r) for r in rows]}


@router.get("/companies/{company_id}")
def get_company(company_id: str, request: Request):
    p = current(request)
    if not p.is_admin and p.company_id != company_id:
        raise forbidden("you can only see your own company")
    return ser.company(company_or_404(hub_of(request).db, company_id))


def _transition(request, company_id, allowed_from, to, payload):
    require_admin(current(request))
    body = v.body_object(payload, required=False)
    v.reject_unknown(body, ("reason",))
    reason = v.opt_string(body.get("reason"), "reason", max_len=500) or None
    db = hub_of(request).db
    with db.tx() as c:
        row = c.execute("SELECT status FROM companies WHERE id = ?", (company_id,)).fetchone()
        if row is None:
            company_or_404(db, company_id)  # raises 404
        if row["status"] not in allowed_from:
            raise invalid_transition(f"company is {row['status']!r}; this needs one of: {', '.join(allowed_from)}")
        c.execute("UPDATE companies SET status = ?, status_reason = ? WHERE id = ?", (to, reason, company_id))
    return ser.company(company_or_404(db, company_id))


@router.post("/companies/{company_id}/approve")
def approve(company_id: str, request: Request, payload: Any = Body(None)):
    return _transition(request, company_id, ("pending", "suspended"), "approved", payload)


@router.post("/companies/{company_id}/reject")
def reject(company_id: str, request: Request, payload: Any = Body(None)):
    return _transition(request, company_id, ("pending",), "rejected", payload)


@router.post("/companies/{company_id}/suspend")
def suspend(company_id: str, request: Request, payload: Any = Body(None)):
    return _transition(request, company_id, ("approved",), "suspended", payload)
