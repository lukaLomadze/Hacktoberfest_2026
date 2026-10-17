"""Database rows → the JSON objects of PLATFORM-API §4. The only place shapes are built."""

from __future__ import annotations

def user(row) -> dict:
    return {"id": row["id"], "email": row["email"], "role": row["role"], "company_id": row["company_id"]}


def company(row) -> dict:
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "field": row["field"],
            "contact_email": row["contact_email"], "status": row["status"],
            "status_reason": row["status_reason"], "wallet_address": row["wallet_address"],
            "created_at": row["created_at"]}
