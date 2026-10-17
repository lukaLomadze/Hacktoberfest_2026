"""Where things live, and access to the engine in `privatelearn/src`.

The server reuses the engine (the bus's `sanitize`, the logistic model, the column
profiler) instead of copying it, so `src/` is put on `sys.path` exactly the way
`src/pipeline/run_loop.py` does it.
"""

from __future__ import annotations

import pathlib
import sys

HUB = pathlib.Path(__file__).resolve().parents[1]  # privatelearn/hub
ROOT = HUB.parent  # privatelearn
SRC = ROOT / "src"
WEB = HUB / "web"
DEFAULT_DB = HUB / "hub.db"
TRACE = ROOT / "trace.json"
MANIFEST = ROOT / "data" / "hospitals" / "manifest.json"
HOSPITALS_DIR = ROOT / "data" / "hospitals"


def hospital_csv(index: int) -> pathlib.Path:
    return HOSPITALS_DIR / f"hospital_{index}.csv"


def use_engine() -> None:
    """Make `agents`, `model` and `shared` (privatelearn/src) importable. Idempotent."""
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
