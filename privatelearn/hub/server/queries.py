"""Lookups and access rules shared by the routes.

Visibility (PLATFORM-API §5.3): the admin sees everything. A company sees groups that
are `open`, `training` or `trained`, plus any group it has (or had) a membership in.
A group the caller cannot see is reported as not found, so draft groups do not leak.
"""

from __future__ import annotations

from . import serialize as ser
from .errors import forbidden, not_found
from .nodes import node_status

PUBLIC_GROUP_STATUS = ("open", "training", "trained")
# memberships that count as members of a group (requests and closed ones do not)
MEMBER_STATUS = ("approved", "node_connected", "schema_proposed", "schema_confirmed", "ready",
                 "training", "active")


def company_or_404(db, company_id):
    row = db.one("SELECT * FROM companies WHERE id = ?", (company_id,))
    if row is None:
        raise not_found(f"company {company_id}")
    return row


def has_membership(db, company_id, group_id, *, exclude=("rejected",)) -> bool:
    marks = ",".join("?" * len(exclude))
    return db.one(f"SELECT 1 AS x FROM memberships WHERE company_id = ? AND group_id = ? "
                  f"AND status NOT IN ({marks}) LIMIT 1", (company_id, group_id, *exclude)) is not None


def can_see_group(db, p, group_row) -> bool:
    if p.is_admin:
        return True
    if group_row["status"] in PUBLIC_GROUP_STATUS:
        return True
    return has_membership(db, p.company_id, group_row["id"], exclude=())


def visible_group_or_404(db, p, group_id):
    row = db.one("SELECT * FROM ml_groups WHERE id = ?", (group_id,))
    if row is None or not can_see_group(db, p, row):
        raise not_found(f"group {group_id}")
    return row


def is_group_member(db, p, group_id) -> bool:
    """Admin, or a company with a non-rejected membership: may see jobs, versions, events."""
    return p.is_admin or has_membership(db, p.company_id, group_id)


def require_group_member(db, p, group_id):
    if not is_group_member(db, p, group_id):
        raise forbidden("only members of this group can see this")


def member_count(db, group_id) -> int:
    marks = ",".join("?" * len(MEMBER_STATUS))
    return db.one(f"SELECT count(*) AS n FROM memberships WHERE group_id = ? AND status IN ({marks})",
                  (group_id, *MEMBER_STATUS))["n"]


def current_version_row(db, group_id):
    return db.one("SELECT * FROM model_versions WHERE group_id = ? ORDER BY version DESC LIMIT 1", (group_id,))


def group_out(db, row) -> dict:
    return ser.group(row, member_count(db, row["id"]), current_version_row(db, row["id"]))


def membership_or_404(db, membership_id):
    row = db.one("SELECT * FROM memberships WHERE id = ?", (membership_id,))
    if row is None:
        raise not_found(f"membership {membership_id}")
    return row


def own_membership(db, p, membership_id):
    """The membership, if the caller is the admin or the company that owns it."""
    row = membership_or_404(db, membership_id)
    if not p.is_admin and row["company_id"] != p.company_id:
        raise forbidden("this membership belongs to another company")
    return row


def membership_out(hub, row) -> dict:
    db = hub.db
    names = db.one("SELECT c.name AS company_name, g.name AS group_name FROM companies c, ml_groups g "
                   "WHERE c.id = ? AND g.id = ?", (row["company_id"], row["group_id"]))
    node_row = db.one("SELECT * FROM nodes WHERE membership_id = ?", (row["id"],))
    return ser.membership(row, names["company_name"], names["group_name"],
                          ser.node(node_row, node_status(hub, node_row, row["status"])))
