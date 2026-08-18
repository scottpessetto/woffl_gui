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
from server.routers import compute, database, gauge, history, meta, optimize, pumps, well_sort, wells

log = logging.getLogger("woffl.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the CPU pool and the fleet cache warmup, stop both on shutdown.

    Everything about what gets warmed, on what cadence, and with how many
    warehouse connections lives in ``server.warmup`` - see its module docstring.

    ORDER MATTERS: the pool starts FIRST, before the warm loop's threads
    exist. Its workers re-import the app stack, and doing that from a
    still-quiet process is both faster and safer than doing it once the
    server is holding warehouse sockets - see ``server.pool``.
    """
    from server import pool, warmup

    pool.start()
    warmup.start()
    try:
        yield
    finally:
        warmup.stop()
        pool.stop()


def _version() -> str:
    try:
        import woffl

        return getattr(woffl, "__version__", "0.0.0")
    except Exception:  # noqa: BLE001
        return "0.0.0"


app = FastAPI(title="WOFFL", version=_version(), lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
# compresslevel: starlette defaults to 9, which recompresses the 720 KB
# echarts chunk at maximum effort on the event loop for every cold fetch.
# Level 6 is roughly a third of the CPU for ~2% more bytes - a trade worth
# taking on a 2-vCPU tier, where that CPU is competing with real work.
app.add_middleware(GZipMiddleware, minimum_size=1500, compresslevel=6)

app.include_router(meta.router, prefix="/api")
app.include_router(wells.router, prefix="/api")
app.include_router(compute.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(pumps.router, prefix="/api")
app.include_router(database.router, prefix="/api")
app.include_router(well_sort.router, prefix="/api")
app.include_router(gauge.router, prefix="/api")
app.include_router(optimize.router, prefix="/api")


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

    class _HashedAssets(StaticFiles):
        """Vite emits content-hashed filenames under /assets - a changed file
        is a NEW URL, so the browser may cache forever. Through the Databricks
        Apps proxy this removes ~10 conditional round trips per page load."""

        async def get_response(self, path: str, scope):  # type: ignore[override]
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return resp

    app.mount("/assets", _HashedAssets(directory=WEB_DIST / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa(full_path: str):
        candidate = (WEB_DIST / full_path).resolve()
        # Serve real files (favicon etc.); everything else falls back to the
        # SPA shell so client-side routes deep-link correctly.
        if full_path and candidate.is_file() and candidate.is_relative_to(WEB_DIST):
            return FileResponse(candidate)
        # The SPA shell must always revalidate: it names the current asset
        # hashes, and a stale shell would 404 on redeployed assets.
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
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
