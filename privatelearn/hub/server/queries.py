"""Lookups shared by the routes."""

from __future__ import annotations

from .errors import not_found


def company_or_404(db, company_id):
    row = db.one("SELECT * FROM companies WHERE id = ?", (company_id,))
    if row is None:
        raise not_found(f"company {company_id}")
    return row
