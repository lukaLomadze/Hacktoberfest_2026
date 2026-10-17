"""Membership rules shared by the seed and (from D4) the membership routes."""

from __future__ import annotations

def missing_required(task_spec, mapping) -> list:
    mapped = {m["feature"] for m in mapping or []}
    return [i["name"] for i in task_spec["input_schema"] if i["required"] and i["name"] not in mapped]
