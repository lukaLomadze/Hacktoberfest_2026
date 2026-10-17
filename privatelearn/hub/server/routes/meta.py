"""GET /meta (PLATFORM-API §5.1): enums and what the engine can train, so the frontend
never hardcodes them."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import enums
from ..context import hub_of

router = APIRouter()


@router.get("/meta")
def meta(request: Request):
    return {"version": enums.API_VERSION, "enums": enums.as_dict(),
            "available_model_types": list(hub_of(request).settings.available_model_types)}
