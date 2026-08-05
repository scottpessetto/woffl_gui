"""Meta endpoint - version, identity, and deployment info for the SPA header."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server import config, identity
from server.schemas import MetaResponse

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
