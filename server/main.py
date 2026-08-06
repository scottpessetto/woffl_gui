"""WOFFL web app - ASGI entrypoint.

Run locally (repo root):
    uvicorn server.main:app --reload --port 8000

Databricks Apps runs ``uvicorn server.main:app`` and injects
UVICORN_HOST / UVICORN_PORT automatically (all port vars are set to
DATABRICKS_APP_PORT by the Apps runtime).

Serves:
- ``/api/*``  JSON API (routers below)
- ``/``       the built React SPA from web/dist (SPA fallback to index.html)
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Repo root on sys.path so `woffl` imports resolve when uvicorn is launched
# from a different cwd (same guard as the Streamlit entrypoint).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.config import WEB_DIST
from server.routers import compute, database, history, meta, pumps, well_sort, wells

log = logging.getLogger("woffl.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _warm_caches() -> None:
    """Fire-and-forget startup warm of the expensive Databricks fetches,
    mirroring the Streamlit startup threads. Per-fetch failures are
    swallowed - a cold cache degrades to lazy loading, never to a crash."""
    from functools import partial

    from server.config import DEFAULT_TEST_MONTHS
    from server.services import datasources
    from server.services import tests as tests_svc

    def _warm(label: str, fn) -> None:
        try:
            fn()
            log.info("cache warm ok: %s", label)
        except Exception as exc:  # noqa: BLE001 - warmers must never raise
            log.warning("cache warm failed (%s): %s", label, exc)

    targets = [
        ("well_characteristics", datasources.well_chars_safe),
        ("pf_latest", datasources.pf_latest_safe),
        ("jp_history", datasources.jp_history_safe),
        ("well_tests", partial(tests_svc.fetch_all_well_tests, DEFAULT_TEST_MONTHS)),
    ]
    # Well Sort pulls (producers, catalog, shut-in log, 180-day tests) warm
    # in their own daemon threads; failures degrade to lazy loading too.
    from server.services import well_sort as well_sort_svc

    well_sort_svc.warm()
    for label, fn in targets:
        threading.Thread(target=_warm, args=(label, fn), daemon=True, name=f"warm-{label}").start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_caches()
    yield


def _version() -> str:
    try:
        import woffl

        return getattr(woffl, "__version__", "0.0.0")
    except Exception:  # noqa: BLE001
        return "0.0.0"


app = FastAPI(title="WOFFL", version=_version(), lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=1500)

app.include_router(meta.router, prefix="/api")
app.include_router(wells.router, prefix="/api")
app.include_router(compute.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(pumps.router, prefix="/api")
app.include_router(database.router, prefix="/api")
app.include_router(well_sort.router, prefix="/api")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    if request.url.path.startswith("/api/"):
        log.exception("unhandled API error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "internal", "message": f"{type(exc).__name__}: {exc}"},
        )
    raise exc


# --- SPA static hosting -----------------------------------------------------
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = (WEB_DIST / full_path).resolve()
        # Serve real files (favicon etc.); everything else falls back to the
        # SPA shell so client-side routes deep-link correctly.
        if full_path and candidate.is_file() and candidate.is_relative_to(WEB_DIST):
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def no_spa():
        return JSONResponse(
            {
                "app": "WOFFL API",
                "hint": "web/dist not found - build the SPA with `npm run build` in web/, or use the Vite dev server (npm run dev) which proxies /api here.",
                "docs": "/api/docs",
            }
        )
