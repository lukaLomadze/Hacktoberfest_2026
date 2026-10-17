"""Small JSON helpers shared across the project."""

from __future__ import annotations

import json
import pathlib


def load_json(path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path, obj, indent: int = 2) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")


def set_seed(seed: int) -> None:
    """Deterministic RNG for the numpy and random modules used across the project."""
    import random

    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass
