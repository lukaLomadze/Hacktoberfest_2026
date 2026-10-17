"""Database rows → the JSON objects of PLATFORM-API §4. The only place shapes are built."""

from __future__ import annotations

from datetime import datetime, timezone

from .db import loads


def iso_from_epoch(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def user(row) -> dict:
    return {"id": row["id"], "email": row["email"], "role": row["role"], "company_id": row["company_id"]}


def company(row) -> dict:
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "field": row["field"],
            "contact_email": row["contact_email"], "status": row["status"],
            "status_reason": row["status_reason"], "wallet_address": row["wallet_address"],
            "created_at": row["created_at"]}


def version_summary(row) -> dict:
    if row is None:
        return None
    return {"id": row["id"], "version": row["version"], "metrics": loads(row["metrics"]),
            "created_at": row["created_at"]}


def group(row, member_count: int, current_version_row=None) -> dict:
    return {"id": row["id"], "name": row["name"], "description": row["description"], "status": row["status"],
            "task_spec": loads(row["task_spec"]), "pricing": loads(row["pricing"]),
            "member_count": member_count, "current_version": version_summary(current_version_row),
            "created_at": row["created_at"], "updated_at": row["updated_at"]}


def node(row, status: str) -> dict:
    if row is None:
        return None
    return {"id": row["id"], "status": status, "last_seen": iso_from_epoch(row["last_seen_epoch"]),
            "node_version": row["node_version"], "agent": loads(row["agent"]),
            "connector": loads(row["connector"])}


def membership(row, company_name: str, group_name: str, node_obj=None) -> dict:
    return {"id": row["id"], "company_id": row["company_id"], "company_name": company_name,
            "group_id": row["group_id"], "group_name": group_name, "status": row["status"],
            "status_reason": row["status_reason"], "node": node_obj,
            "data": loads(row["data"]), "schema": loads(row["schema_"]), "validation": loads(row["validation"]),
            "contribution": {"rows": row["contribution_rows"], "rounds": row["contribution_rounds"]},
            "has_node_token": row["node_token_hash"] is not None,
            "activated_at": row["activated_at"], "requested_at": row["requested_at"],
            "updated_at": row["updated_at"]}
