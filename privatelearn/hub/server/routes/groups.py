"""PLATFORM-API §5.3: groups, one model each.

Group lifecycle (§3): draft → open → training → trained (→ training for a fine-tune);
archived from any state except while a job is running. The TaskSpec can only change
while the group is a draft: once companies have mapped their data to it, changing it
would silently invalidate their mappings.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request

from .. import serialize as ser
from .. import taskspec
from .. import validate as v
from ..auth import current, require_admin
from ..context import hub_of
from ..db import dumps, loads, new_id, now_iso
from ..enums import JOB_LIVE
from ..errors import invalid_transition
from ..queries import (PUBLIC_GROUP_STATUS, group_out, is_group_member, membership_out,
                       visible_group_or_404)

router = APIRouter()

REVENUE_SHARE = ("none", "equal", "by_rows")


def validate_pricing(value, field="pricing"):
    """Reserved for payments (PLATFORM-API §4.3): stored now, charged by nothing yet."""
    if value is None:
        return None
    value = v.obj(value, field)
    v.reject_unknown(value, ("currency", "enrollment_fee", "per_prediction", "revenue_share"), f"{field}.")
    out = {"currency": v.string(value.get("currency", "SOL"), f"{field}.currency", max_len=10)}
    for k in ("enrollment_fee", "per_prediction"):
        out[k] = v.number(value.get(k, 0), f"{field}.{k}", 0, 1e9)
    out["revenue_share"] = v.enum(value.get("revenue_share", "none"), f"{field}.revenue_share", REVENUE_SHARE)
    return out


@router.get("/groups")
def list_groups(request: Request):
    p = current(request)
    db = hub_of(request).db
    if p.is_admin:
        rows = db.all("SELECT * FROM ml_groups ORDER BY created_at DESC, name")
    else:
        marks = ",".join("?" * len(PUBLIC_GROUP_STATUS))
        rows = db.all(f"SELECT * FROM ml_groups WHERE status IN ({marks}) OR id IN "
                      f"(SELECT group_id FROM memberships WHERE company_id = ?) ORDER BY created_at DESC, name",
                      (*PUBLIC_GROUP_STATUS, p.company_id))
    return {"items": [group_out(db, r) for r in rows]}


@router.get("/groups/{group_id}")
def get_group(group_id: str, request: Request):
    p = current(request)
    hub = hub_of(request)
    db = hub.db
    row = visible_group_or_404(db, p, group_id)
    out = group_out(db, row)
    if p.is_admin:
        members = db.all("SELECT * FROM memberships WHERE group_id = ? ORDER BY requested_at", (group_id,))
    else:
        members = db.all("SELECT * FROM memberships WHERE group_id = ? AND company_id = ? ORDER BY requested_at",
                         (group_id, p.company_id))
    out["members"] = [membership_out(hub, m) for m in members]
    versions = (db.all("SELECT * FROM model_versions WHERE group_id = ? ORDER BY version DESC", (group_id,))
                if is_group_member(db, p, group_id) else [])
    out["versions"] = [ser.version_summary(r) for r in versions]
    return out


@router.post("/groups", status_code=201)
def create_group(request: Request, payload: Any = Body(None)):
    require_admin(current(request))
    hub = hub_of(request)
    body = v.body_object(payload)
    v.reject_unknown(body, ("name", "description", "task_spec", "pricing"))
    name = v.string(body.get("name"), "name", min_len=2, max_len=120)
    description = v.opt_string(body.get("description"), "description", max_len=2000)
    spec = taskspec.normalize(body.get("task_spec"), hub.settings.available_model_types)
    pricing = validate_pricing(body.get("pricing"))
    gid, t = new_id("grp"), now_iso()
    hub.db.execute("INSERT INTO ml_groups (id, name, description, status, task_spec, pricing, created_at, "
                   "updated_at) VALUES (?,?,?,?,?,?,?,?)",
                   (gid, name, description, "draft", dumps(spec), dumps(pricing) if pricing else None, t, t))
    return group_out(hub.db, hub.db.one("SELECT * FROM ml_groups WHERE id = ?", (gid,)))


@router.patch("/groups/{group_id}")
def update_group(group_id: str, request: Request, payload: Any = Body(None)):
    p = require_admin(current(request))
    hub = hub_of(request)
    body = v.body_object(payload)
    v.reject_unknown(body, ("name", "description", "task_spec", "pricing"))
    changes = {}
    if "name" in body:
        changes["name"] = v.string(body["name"], "name", min_len=2, max_len=120)
    if "description" in body:
        changes["description"] = v.opt_string(body["description"], "description", max_len=2000)
    if "task_spec" in body:
        changes["task_spec"] = dumps(taskspec.normalize(body["task_spec"], hub.settings.available_model_types))
    if "pricing" in body:
        pricing = validate_pricing(body["pricing"])
        changes["pricing"] = dumps(pricing) if pricing else None
    with hub.db.tx() as c:
        row = visible_group_or_404(hub.db, p, group_id)
        if row["status"] == "archived":
            raise invalid_transition("an archived group cannot be edited")
        if "task_spec" in changes and row["status"] != "draft":
            raise invalid_transition("the task spec can only change while the group is a draft")
        if changes:
            sets = ", ".join(f"{k} = ?" for k in changes)
            c.execute(f"UPDATE ml_groups SET {sets}, updated_at = ? WHERE id = ?",
                      (*changes.values(), now_iso(), group_id))
    return group_out(hub.db, hub.db.one("SELECT * FROM ml_groups WHERE id = ?", (group_id,)))


@router.post("/groups/{group_id}/publish")
def publish(group_id: str, request: Request):
    p = require_admin(current(request))
    hub = hub_of(request)
    with hub.db.tx() as c:
        row = visible_group_or_404(hub.db, p, group_id)
        if row["status"] != "draft":
            raise invalid_transition(f"group is {row['status']!r}; only a draft can be published")
        # re-check: the model type may have become unavailable since the draft was saved
        spec = taskspec.normalize(loads(row["task_spec"]), hub.settings.available_model_types)
        c.execute("UPDATE ml_groups SET status = 'open', task_spec = ?, updated_at = ? WHERE id = ?",
                  (dumps(spec), now_iso(), group_id))
    return group_out(hub.db, hub.db.one("SELECT * FROM ml_groups WHERE id = ?", (group_id,)))


@router.post("/groups/{group_id}/archive")
def archive(group_id: str, request: Request):
    p = require_admin(current(request))
    hub = hub_of(request)
    with hub.db.tx() as c:
        row = visible_group_or_404(hub.db, p, group_id)
        if row["status"] == "archived":
            raise invalid_transition("group is already archived")
        marks = ",".join("?" * len(JOB_LIVE))
        if c.execute(f"SELECT 1 FROM jobs WHERE group_id = ? AND status IN ({marks})",
                     (group_id, *JOB_LIVE)).fetchone():
            raise invalid_transition("a job is running in this group; cancel it first")
        c.execute("UPDATE ml_groups SET status = 'archived', updated_at = ? WHERE id = ?", (now_iso(), group_id))
        # open join requests can no longer be granted
        c.execute("UPDATE memberships SET status = 'rejected', status_reason = 'group archived', updated_at = ? "
                  "WHERE group_id = ? AND status = 'requested'", (now_iso(), group_id))
    return group_out(hub.db, hub.db.one("SELECT * FROM ml_groups WHERE id = ?", (group_id,)))
