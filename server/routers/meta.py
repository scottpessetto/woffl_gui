"""Meta endpoint - version, identity, and deployment info for the SPA header."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server import config, identity
from server.schemas import MetaResponse, WarmupStatus

router = APIRouter(tags=["meta"])


@router.get("/meta", response_model=MetaResponse)
def get_meta(request: Request) -> MetaResponse:
    """App version, request user, and deployment flags."""
    from woffl.assembly import databricks_client

    try:
        import woffl

        version = getattr(woffl, "__version__", "0.0.0")
    except Exception:
        version = "0.0.0"

    return MetaResponse(
        version=version,
        user=identity.request_user(request),
        writes_enabled=config.writes_enabled(),
        warehouse_id=databricks_client.DEFAULT_WAREHOUSE_ID,
        deployed=config.is_deployed(),
    )


@router.get("/meta/warmup", response_model=WarmupStatus)
def get_warmup_status() -> WarmupStatus:
    """Fleet cache warmup progress. `wells_ok == wells_total` after the first
    pass means no user will pay a cold per-well Databricks query.
    `fleet_history_ok` says the fleet's history came from the two fleet
    statements; False means that pass fell back to the per-well fan-out, which
    `statements` (the pass's warehouse-statement count) shows the cost of."""
    from server import warmup

    return WarmupStatus(**warmup.status())
