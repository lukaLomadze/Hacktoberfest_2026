"""The platform server: `create_app()` wires the database, the federation and the routes.

    cd privatelearn
    python -m hub.server                 # http://127.0.0.1:8100  (API under /api/v1)

Environment: HUB_DB (database file, default hub/hub.db), HUB_SEED (dev | demo | none).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Match, Route

from . import errors, paths
from .context import Hub, Settings
from .db import Database
from .enums import API_VERSION
from .routes import auth, meta
from .seed import seed

log = logging.getLogger("hub.server")

ROUTERS = [meta.router, auth.router]


class WebFiles(StaticFiles):
    """hub/web at `/`, except that /api/... never reaches the static files.

    Starlette prefers this mount's full match over a route that matches the path but not
    the method, so `GET /api/v1/auth/login` lands here: answer it with the API's 405, and
    any other /api/... path with a JSON 404. `path` comes with OS separators ("api\\v1" on
    Windows), so it is normalised before the check.
    """

    async def get_response(self, path: str, scope):
        route_path = path.replace(os.sep, "/")
        if route_path == "api" or route_path.startswith("api/"):
            probe = {"type": "http", "path": "/" + route_path, "method": scope["method"], "root_path": ""}
            if any(isinstance(r, Route) and r.matches(probe)[0] == Match.PARTIAL
                   for r in scope["app"].router.routes):
                raise errors.ApiError(405, "method_not_allowed", f"{scope['method']} is not allowed on /{route_path}")
            raise errors.not_found(f"route /{route_path}")
        return await super().get_response(path, scope)


def create_app(db_path=None, *, federation=None, seed_mode=None, settings=None) -> FastAPI:
    settings = settings or Settings()
    db = Database(db_path or os.environ.get("HUB_DB") or paths.DEFAULT_DB)
    seed(db, seed_mode or os.environ.get("HUB_SEED", "dev"), settings.available_model_types)
    hub = Hub(db=db, federation=federation, settings=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        stop = getattr(hub.federation, "shutdown", None)
        if callable(stop):
            stop()
        db.close()

    app = FastAPI(title="PrivateLearn platform", version=API_VERSION, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.hub = hub
    errors.install(app)
    # The UI is served by this server (same origin); this only helps a frontend dev server
    # on another localhost port. No credentials cross origins: auth is a bearer header.
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
                       allow_methods=["*"], allow_headers=["*"])
    for r in ROUTERS:
        app.include_router(r, prefix="/api/v1")

    if paths.WEB.is_dir():
        app.mount("/", WebFiles(directory=str(paths.WEB), html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return {"message": "PrivateLearn platform API. The web app (hub/web) is not built yet.",
                    "api": "/api/v1", "docs": "/api/docs"}
    return app
