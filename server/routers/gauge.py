"""Memory-gauge upload parsing (read-only, stateless).

The Streamlit app kept parsed gauges in session state; the SPA keeps them in
a client store and re-sends EVERY file of a well when one is added or
removed. Parsing + multi-file combination (timestamp dedupe -> daily
medians) therefore always run server-side through woffl.gui.memory_gauge -
the same code path as Streamlit, byte-identical daily medians, no client
math. Nothing is persisted: gauge data lives exactly as long as the
engineer's browser session, matching the Streamlit contract.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from server import schemas

router = APIRouter(prefix="/gauge", tags=["gauge"])

# One gauge XLSX is a few MB (90 days of 5 s samples); anything bigger than
# this is not a gauge export and only wastes parse time.
_MAX_FILE_BYTES = 50 * 1024 * 1024


@router.post("/parse", response_model=schemas.GaugeParseResponse)
async def parse_gauge(files: list[UploadFile] = File(...)) -> Any:
    """Parse + combine one well's memory-gauge XLSX files.

    422 with the offending filename on any parse failure - the client keeps
    its previous gauge state untouched in that case.
    """
    from woffl.gui.memory_gauge import MemoryGaugeData, parse_xlsx

    if not files:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid", "message": "no files uploaded"},
        )

    parsed = []
    for f in files:
        blob = await f.read()
        if len(blob) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid",
                    "message": f"{f.filename}: file exceeds {_MAX_FILE_BYTES // (1024 * 1024)} MB",
                },
            )
        try:
            parsed.append(parse_xlsx(blob, f.filename or "gauge.xlsx"))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid", "message": f"{f.filename}: {exc}"},
            ) from exc

    combined = MemoryGaugeData(well_name="_upload", files=parsed)
    return {
        "files": [
            {
                "filename": p.source_filename,
                "start_date": p.start_date.strftime("%Y-%m-%d"),
                "end_date": p.end_date.strftime("%Y-%m-%d"),
                "sample_count": p.sample_count,
                "pressure_min": p.pressure_min,
                "pressure_max": p.pressure_max,
            }
            for p in parsed
        ],
        "daily": [
            {"date": ts.strftime("%Y-%m-%d"), "bhp": float(bhp)}
            for ts, bhp in zip(combined.daily_df["tag_date"], combined.daily_df["bhp"])
        ],
        "start_date": combined.start_date.strftime("%Y-%m-%d"),
        "end_date": combined.end_date.strftime("%Y-%m-%d"),
        "sample_count": combined.sample_count,
    }
