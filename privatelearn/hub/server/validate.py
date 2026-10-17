"""Request-body checks that fail with `422 validation_error` naming the exact field.

Bodies are parsed by hand rather than with pydantic models so every failure uses the
PLATFORM-API §1.1 shape and names the field the way the frontend displays it
(`task_spec.input_schema[2].unit`). Two traps these helpers close:
  - `bool` is a subclass of `int` in Python, so `true` would pass as the integer 1;
  - Python's json accepts NaN and Infinity, which are not valid JSON for the frontend.
Patterns use `fullmatch`: with `match`, `$` also matches before a trailing newline, so
"glucose\\n" would pass as a name distinct from "glucose".
"""

from __future__ import annotations

import math
import re

from .errors import bad_request, validation

EMAIL_RE = re.compile(r"[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}")
# Solana addresses are base58 public keys: 32-44 characters, no 0, O, I or l.
SOLANA_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")
NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def body_object(payload, *, required=True) -> dict:
    """The request body as a dict. A missing body is `{}` unless `required`."""
    if payload is None:
        if required:
            raise bad_request("a JSON object body is required")
        return {}
    if not isinstance(payload, dict):
        raise bad_request("the request body must be a JSON object sent with Content-Type: application/json")
    return payload


def reject_unknown(d: dict, allowed, path=""):
    for k in d:
        if k not in allowed:
            raise validation(f"{path}{k}", "unknown field")


def obj(v, field) -> dict:
    if not isinstance(v, dict):
        raise validation(field, "must be an object")
    return v


def lst(v, field, *, min_len=0, max_len=10_000) -> list:
    if not isinstance(v, list):
        raise validation(field, "must be a list")
    if not min_len <= len(v) <= max_len:
        raise validation(field, f"must have {min_len}-{max_len} items")
    return v


def string(v, field, *, min_len=1, max_len=200) -> str:
    if not isinstance(v, str):
        raise validation(field, "must be a string")
    v = v.strip()
    if not min_len <= len(v) <= max_len:
        raise validation(field, f"must be {min_len}-{max_len} characters")
    return v


def opt_string(v, field, *, max_len=2000) -> str:
    """An optional free-text field: missing/null becomes ''."""
    if v is None:
        return ""
    return string(v, field, min_len=0, max_len=max_len)


def integer(v, field, lo=None, hi=None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise validation(field, "must be an integer")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        bound = (f"between {lo} and {hi}" if lo is not None and hi is not None
                 else f">= {lo}" if lo is not None else f"<= {hi}")
        raise validation(field, f"must be {bound}")
    return v


def number(v, field, lo=None, hi=None, *, lo_open=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise validation(field, "must be a number")
    v = float(v)
    if not math.isfinite(v):
        raise validation(field, "must be a finite number")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise validation(field, f"must be {'>' if lo_open else '>='} {lo}")
    if hi is not None and v > hi:
        raise validation(field, f"must be <= {hi}")
    return v


def opt_number(v, field, **kw):
    return None if v is None else number(v, field, **kw)


def boolean(v, field) -> bool:
    if not isinstance(v, bool):
        raise validation(field, "must be true or false")
    return v


def enum(v, field, allowed) -> str:
    if not isinstance(v, str) or v not in allowed:
        raise validation(field, f"must be one of {', '.join(allowed)}")
    return v


def email(v, field) -> str:
    v = string(v, field, max_len=254).lower()
    if not EMAIL_RE.fullmatch(v):
        raise validation(field, "must be an email address")
    return v


def wallet(v, field):
    if v is None:
        return None
    v = string(v, field, max_len=64)
    if not SOLANA_RE.fullmatch(v):
        raise validation(field, "must be a Solana address (base58, 32-44 characters)")
    return v


def name_key(v, field) -> str:
    """A machine name: lower snake_case, starts with a letter (feature and label names)."""
    if not isinstance(v, str) or not NAME_RE.fullmatch(v):
        raise validation(field, "must be lower snake_case: a letter, then letters, digits or _ (max 64)")
    return v


def json_value(v, field, depth=0):
    """Any JSON value, checked for finite numbers and bounded nesting."""
    if depth > 20:
        raise validation(field, "is nested too deeply")
    if isinstance(v, float) and not math.isfinite(v):
        raise validation(field, "must not contain NaN or Infinity")
    if isinstance(v, dict):
        for k, x in v.items():
            json_value(x, f"{field}.{k}", depth + 1)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            json_value(x, f"{field}[{i}]", depth + 1)
    return v
