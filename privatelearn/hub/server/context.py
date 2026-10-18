"""The objects every route needs, hung on `app.state.hub`."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request

from .enums import AVAILABLE_MODEL_TYPES


@dataclass
class Settings:
    session_ttl_s: float = 7 * 24 * 3600
    # A node's GET /node/inbox blocks for at most poll_wait_s. A node counts as online
    # while a poll is open or within node_offline_s of its last poll, so the threshold
    # must exceed poll_wait_s or a healthy, long-polling node would flicker offline.
    poll_wait_s: float = 10.0
    node_offline_s: float = 15.0
    available_model_types: tuple = AVAILABLE_MODEL_TYPES
    public_url: str = ""  # used in the node command; "" = derive from the request


@dataclass
class Hub:
    db: object
    federation: object
    settings: Settings = field(default_factory=Settings)
    nodes: object = None  # NodeRegistry (in-memory liveness), set by the app


def hub_of(request: Request) -> Hub:
    return request.app.state.hub
