"""PLATFORM-API §5.4: a company's membership in a group, and its state machine (§3).

    requested → approved → node_connected → schema_proposed → schema_confirmed → ready
              ↘ rejected                                            training ↔ ready/active
    any state except training → left (the company withdraws or leaves)

The node token is shown once, when it is issued; only its SHA-256 is stored. Issuing a
new one revokes the old one, so a leaked command line can be killed by re-issuing.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from fastapi import APIRouter, Body, Request

from .. import validate as v
from ..auth import current, hash_token, new_token, require_admin, require_approved_company, require_company
from ..context import hub_of
from ..db import new_id, now_iso
from ..enums import MEMBERSHIP_STATUS
from ..errors import conflict, invalid_transition
from ..membership import WIRED, move
from ..queries import membership_or_404, membership_out, own_membership, visible_group_or_404

router = APIRouter()

JOINABLE_GROUP_STATUS = ("open", "training", "trained")


def _reason(payload):
    body = v.body_object(payload, required=False)
    v.reject_unknown(body, ("reason",))
    return v.opt_string(body.get("reason"), "reason", max_len=500) or None


@router.post("/groups/{group_id}/join", status_code=201)
def join(group_id: str, request: Request):
    p = current(request)
    company = require_approved_company(p)
    hub = hub_of(request)
    mid, t = new_id("mem"), now_iso()
    try:
        with hub.db.tx() as c:
            group = visible_group_or_404(hub.db, p, group_id)
            if group["status"] not in JOINABLE_GROUP_STATUS:
                raise invalid_transition(f"group is {group['status']!r}; it is not taking members")
            open_row = c.execute("SELECT id, status FROM memberships WHERE company_id = ? AND group_id = ? "
                                 "AND status NOT IN ('rejected', 'left')", (company["id"], group_id)).fetchone()
            if open_row:
                raise conflict(f"you already have membership {open_row['id']} ({open_row['status']}) in this group")
            c.execute("INSERT INTO memberships (id, company_id, group_id, status, requested_at, updated_at) "
                      "VALUES (?,?,?,?,?,?)", (mid, company["id"], group_id, "requested", t, t))
    except sqlite3.IntegrityError:  # two join requests at the same moment
        raise conflict("you already have an open membership in this group")
    return membership_out(hub, membership_or_404(hub.db, mid))


@router.get("/memberships")
def list_memberships(request: Request, group_id: Optional[str] = None, status: Optional[str] = None):
    p = current(request)
    hub = hub_of(request)
    where, args = [], []
    if not p.is_admin:
        where.append("company_id = ?")
        args.append(p.company_id)
    if group_id is not None:
        where.append("group_id = ?")
        args.append(group_id)
    if status is not None:
        where.append("status = ?")
        args.append(v.enum(status, "status", MEMBERSHIP_STATUS))
    sql = "SELECT * FROM memberships" + (" WHERE " + " AND ".join(where) if where else "")
    rows = hub.db.all(sql + " ORDER BY requested_at DESC, id", args)
    return {"items": [membership_out(hub, r) for r in rows]}


@router.get("/memberships/{membership_id}")
def get_membership(membership_id: str, request: Request):
    p = current(request)
    hub = hub_of(request)
    return membership_out(hub, own_membership(hub.db, p, membership_id))


@router.post("/memberships/{membership_id}/approve")
def approve(membership_id: str, request: Request):
    require_admin(current(request))
    hub = hub_of(request)
    with hub.db.tx() as c:
        m = membership_or_404(hub.db, membership_id)
        company = c.execute("SELECT status FROM companies WHERE id = ?", (m["company_id"],)).fetchone()
        if company["status"] != "approved":
            raise invalid_transition(f"the company is {company['status']!r}; approve the company first")
        group = c.execute("SELECT status FROM ml_groups WHERE id = ?", (m["group_id"],)).fetchone()
        if group["status"] == "archived":
            raise invalid_transition("the group is archived")
        move(c, membership_id, ("requested",), "approved")
        on_enrolled = getattr(hub, "on_enrolled", None)  # usage ledger hook (D8)
        if on_enrolled:
            on_enrolled(c, m)
    return membership_out(hub, membership_or_404(hub.db, membership_id))


@router.post("/memberships/{membership_id}/reject")
def reject(membership_id: str, request: Request, payload: Any = Body(None)):
    require_admin(current(request))
    reason = _reason(payload)
    hub = hub_of(request)
    with hub.db.tx() as c:
        membership_or_404(hub.db, membership_id)
        move(c, membership_id, ("requested",), "rejected", reason)
    return membership_out(hub, membership_or_404(hub.db, membership_id))


@router.post("/memberships/{membership_id}/node-token")
def issue_node_token(membership_id: str, request: Request):
    p = current(request)
    require_approved_company(p)
    hub = hub_of(request)
    token = new_token()
    with hub.db.tx() as c:
        m = own_membership(hub.db, p, membership_id)
        if m["status"] == "training":
            raise invalid_transition("a job is running with this node; wait for it to finish")
        if m["status"] not in WIRED:
            raise invalid_transition(f"membership is {m['status']!r}; it must be approved first")
        group = c.execute("SELECT status FROM ml_groups WHERE id = ?", (m["group_id"],)).fetchone()
        if group["status"] == "archived":
            raise invalid_transition("the group is archived")
        c.execute("UPDATE memberships SET node_token_hash = ?, updated_at = ? WHERE id = ?",
                  (hash_token(token), now_iso(), membership_id))
    base = hub.settings.public_url or str(request.base_url).rstrip("/")
    return {"node_token": token,
            "command": f"python hub/node/node.py --server {base} --token {token} --db <your connection>"}


@router.post("/memberships/{membership_id}/leave")
def leave(membership_id: str, request: Request):
    p = current(request)
    require_company(p)  # leaving is the company's own decision
    hub = hub_of(request)
    with hub.db.tx() as c:
        m = own_membership(hub.db, p, membership_id)
        if m["status"] == "training":
            raise invalid_transition("a job is running with your data; wait for it to finish")
        move(c, membership_id, tuple(s for s in MEMBERSHIP_STATUS if s not in ("training", "left", "rejected")),
             "left")
        c.execute("UPDATE memberships SET node_token_hash = NULL WHERE id = ?", (membership_id,))
    return membership_out(hub, membership_or_404(hub.db, membership_id))
