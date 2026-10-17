"""The single error shape of PLATFORM-API.md §1.1, for every failure path.

Routes raise `ApiError`; FastAPI's own errors (unknown route, wrong method, a body
that is not JSON) and unexpected exceptions are converted to the same shape, so the
frontend has exactly one error format to handle.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("hub.server")

CODE_FOR_STATUS = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                   405: "method_not_allowed", 409: "conflict", 422: "validation_error",
                   503: "node_offline"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field


def bad_request(message, field=None):
    return ApiError(400, "bad_request", message, field)


def unauthorized(message="sign in first"):
    return ApiError(401, "unauthorized", message)


def forbidden(message):
    return ApiError(403, "forbidden", message)


def not_found(what):
    return ApiError(404, "not_found", f"{what} not found")


def invalid_transition(message):
    return ApiError(409, "invalid_transition", message)


def conflict(message, field=None):
    return ApiError(409, "conflict", message, field)


def validation(field, message):
    return ApiError(422, "validation_error", f"{field}: {message}" if field else message, field)


def body(status, code, message, field=None):
    return JSONResponse({"error": {"code": code, "message": message, "field": field}}, status_code=status)


def _field_from_loc(loc) -> str | None:
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "header")]
    return ".".join(parts) or None


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return body(exc.status, exc.code, exc.message, exc.field)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = CODE_FOR_STATUS.get(exc.status_code, "bad_request" if exc.status_code < 500 else "internal")
        message = exc.detail if isinstance(exc.detail, str) else code
        return body(exc.status_code, code, message, None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        if any(str(e.get("type", "")).startswith("json") for e in errors):
            return body(400, "bad_request", "request body is not valid JSON")
        first = errors[0] if errors else {}
        field = _field_from_loc(first.get("loc", ()))
        return body(422, "validation_error", f"{field}: {first.get('msg', 'invalid value')}", field)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return body(500, "internal", "internal server error")
