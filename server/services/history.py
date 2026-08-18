"""JP install history + extended production window (the history chart data).

Server-side port of the JP History tab's data layer: install rows from the
enriched tracker frame, well tests back to the earliest install, and the
daily BHP series. v1 serves the Databricks path only - the Streamlit
memory-gauge overlay layers have no server equivalent.

mirrors woffl/gui/tabs/jp_history_tab.py (queries + fetch flow)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import datasources, frames

# mirrors woffl/gui/tabs/jp_history_tab.py:_EXTENDED_TEST_QUERY
# (lifted verbatim; the em dash in the original SQL comment is written as
# "-" per the ASCII house rule)
_EXTENDED_TEST_QUERY = """\
SELECT
    vwt.well_name,
    vwt.wt_date,
    vwt.form_oil AS oil_rate,
    vwt.form_wat AS fwat_rate,
    vwt.lift_wat,
    round(vbdc.bhp_cln_value, 2) AS bhp,
    vpd.tubing_prs AS pf_tubing_prs,
    vpd.inn_ann_prs AS pf_inn_ann_prs
FROM mpu.wells.vw_well_test vwt
LEFT JOIN mpu.wells.vw_bhp_daily_clean vbdc
    ON vwt.enthid = vbdc.enthid
    AND to_date(vwt.wt_date) = vbdc.tag_date
LEFT JOIN (
    -- Test-day PF pressure (same aggregated join as well_test_client) -
    -- feeds the Pump Report Card's per-era PF context and the chart overlay.
    SELECT
        enthid,
        sample_date,
        max(tubing_prs) AS tubing_prs,
        max(inn_ann_prs) AS inn_ann_prs
    FROM mpu.wells.vw_pressure_daily
    GROUP BY enthid, sample_date
) vpd
    ON vwt.enthid = vpd.enthid
    AND to_date(vwt.wt_date) = vpd.sample_date
WHERE vwt.well_name = '{well_name}'
    AND vwt.wt_date BETWEEN '{start_date}' AND '{end_date}'
    AND vwt.allocated = True
ORDER BY vwt.wt_date
"""

# mirrors woffl/gui/tabs/jp_history_tab.py:_BHP_DAILY_QUERY
_BHP_DAILY_QUERY = """\
SELECT
    vbdc.tag_date,
    round(vbdc.bhp_cln_value, 2) AS bhp
FROM mpu.wells.vw_bhp_daily_clean vbdc
WHERE vbdc.enthid = (
    SELECT DISTINCT enthid
    FROM mpu.wells.vw_well_test
    WHERE well_name = '{well_name}'
    LIMIT 1
)
AND vbdc.tag_date BETWEEN '{start_date}' AND '{end_date}'
ORDER BY vbdc.tag_date
"""


# mirrors woffl/gui/tabs/jp_history_tab.py:_cached_extended_tests
#
# maxsize sizing: the key carries `end` = today, so every well gets a NEW key at
# midnight. It must therefore hold at least TWO days x the fleet, or the day's
# fresh entries evict each other while yesterday's are still resident.
@ttl_cache(config.TTL_EXTENDED_TESTS, maxsize=512)
def extended_tests(db_name: str, start: str, end: str) -> pd.DataFrame:
    """Well tests for an extended date range without requiring BHP.

    Args:
        db_name: Databricks-format well name (e.g. "B-028" - the output of
            well_test_client._denormalize_well_name).
        start: window start, YYYY-MM-DD.
        end: window end, YYYY-MM-DD.

    Returns:
        Test frame sorted by WtDate with resolved pf_press/pf_source columns.
        Raises on query failure (failures are never cached).
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.pf_pressure import add_pf_columns
    from woffl.assembly.sql_guards import validate_iso_date, validate_well_name
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(
        _EXTENDED_TEST_QUERY.format(
            well_name=validate_well_name(db_name),
            start_date=validate_iso_date(start),
            end_date=validate_iso_date(end),
        )
    )
    if df.empty:
        return df

    rename = {
        "well_name": "well",
        "wt_date": "WtDate",
        "bhp": "BHP",
        "oil_rate": "WtOilVol",
        "fwat_rate": "WtWaterVol",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "well" in df.columns:
        df["well"] = df["well"].apply(_normalize_well_name)
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], utc=True).dt.tz_localize(None)

    for col in ["BHP", "WtOilVol", "WtWaterVol", "lift_wat", "pf_tubing_prs", "pf_inn_ann_prs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resolve test-day PF (annulus vs tubing per circulation direction).
    df = add_pf_columns(df)

    # Only require date + at least one rate - NOT BHP.
    df = df.dropna(subset=["WtDate"])
    return df.sort_values("WtDate")


# mirrors woffl/gui/tabs/jp_history_tab.py:_cached_bhp_daily
# Same rolling-`end` key as extended_tests - see its maxsize note.
@ttl_cache(config.TTL_EXTENDED_TESTS, maxsize=512)
def bhp_daily(db_name: str, start: str, end: str) -> pd.DataFrame:
    """Daily BHP for all dates (not just well-test dates).

    Args:
        db_name: Databricks-format well name (e.g. "B-028").
        start: window start, YYYY-MM-DD.
        end: window end, YYYY-MM-DD.

    Returns:
        Frame with tag_date + bhp columns, NaN rows dropped, date-sorted.
        Raises on query failure (failures are never cached).
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.sql_guards import validate_iso_date, validate_well_name

    df = execute_query(
        _BHP_DAILY_QUERY.format(
            well_name=validate_well_name(db_name),
            start_date=validate_iso_date(start),
            end_date=validate_iso_date(end),
        )
    )
    if df.empty:
        return df

    if "tag_date" in df.columns:
        df["tag_date"] = pd.to_datetime(df["tag_date"], utc=True).dt.tz_localize(None)
    if "bhp" in df.columns:
        df["bhp"] = pd.to_numeric(df["bhp"], errors="coerce")

    return df.dropna(subset=["tag_date", "bhp"]).sort_values("tag_date")


# DataFrame column -> JSON key for the install rows (JpInstallRow contract).
# Column names are the enriched tracker frame's (pump_identity.enrich_jp_history
# adds Circ Direction / Raw Pump / Pump Converted on top of the tracker's
# Date Set / Date Pulled / Nozzle Number / Throat Ratio / Tubing Diameter /
# Manufacturer).
_INSTALL_COLUMNS: dict[str, str] = {
    "Date Set": "date_set",
    "Date Pulled": "date_pulled",
    "Nozzle Number": "nozzle",
    "Throat Ratio": "throat",
    "Tubing Diameter": "tubing_od",
    "Circ Direction": "circulating",
    "Manufacturer": "manufacturer",
    "Raw Pump": "raw_pump",
    "Pump Converted": "pump_converted",
}

# DataFrame column -> JSON key for the extended test rows.
_TEST_COLUMNS: dict[str, str] = {
    "WtDate": "date",
    "WtOilVol": "oil_rate",
    "WtWaterVol": "fwat_rate",
    "lift_wat": "lift_wat",
    "BHP": "bhp",
    "pf_press": "pf_press",
    "pf_source": "pf_source",
}

_BHP_COLUMNS: dict[str, str] = {"tag_date": "date", "bhp": "bhp"}


def _code_str(value: Any) -> Optional[str]:
    """Nozzle/throat cell -> clean string code ('12.0' -> '12'), None if blank.

    mirrors woffl/assembly/pump_report.py:format_pump (per-part tolerance) -
    the JSON contract types nozzle/throat as strings, but the tracker stores
    float nozzles.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _current_pump_caption(jp_hist: pd.DataFrame, well: str) -> Optional[str]:
    """'13C set 2026-01-15' caption for the well's current pump, or None."""
    from woffl.assembly.jp_history import get_current_pump
    from woffl.assembly.pump_report import format_pump

    pump = get_current_pump(jp_hist, well)
    if pump is None:
        return None
    label = format_pump(pump.get("nozzle_no"), pump.get("throat_ratio"))
    date_set = pump.get("date_set")
    if date_set is not None and pd.notna(date_set):
        return f"{label} set {pd.Timestamp(date_set).strftime('%Y-%m-%d')}"
    return label


def _installs_for(jp_hist: pd.DataFrame, well: str) -> pd.DataFrame:
    """The well's dated installs, oldest first (empty frame when it has none)."""
    if "Well Name" in jp_hist.columns:
        well_jp = jp_hist[jp_hist["Well Name"] == well].copy()
    else:
        well_jp = jp_hist.iloc[0:0].copy()
    if "Date Set" in well_jp.columns:
        well_jp = well_jp.dropna(subset=["Date Set"]).sort_values("Date Set")
    return well_jp


def _query_window(well: str, well_jp: pd.DataFrame) -> tuple[str, str, str]:
    """(db_name, start, end) for this well's extended_tests / bhp_daily pulls.

    THE single source of those two cache keys: ``warm_well`` and
    ``jp_history_payload`` must agree exactly or the warmup fills entries no
    request ever reads. ``end`` is today, so the keys roll over at midnight -
    see the maxsize notes on the two fetchers.
    """
    from woffl.assembly.well_test_client import _denormalize_well_name

    return (
        _denormalize_well_name(well),
        pd.Timestamp(well_jp["Date Set"].min()).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d"),
    )


def warm_well(well: str) -> bool:
    """Pre-pay the two per-well Databricks pulls behind /wells/{name}/jp-history.

    These are the app's only genuinely per-well warehouse queries, and they are
    why a well used to be slow on its first open and instant for the rest of the
    day. Called by ``server.warmup`` for every well in the fleet.

    Returns:
        True when the tracker had a dated install and both fetchers were
        touched; False when this well has nothing to query.

    Raises:
        RuntimeError: when no JP-history source is reachable. Fetcher errors
        propagate too - failures are never cached and the warmup driver logs
        them and moves on.
    """
    jp_hist, source = datasources.jp_history_safe()
    if jp_hist is None or source is None:
        raise RuntimeError("JP history unavailable (Databricks and Excel fallback both failed)")
    well_jp = _installs_for(jp_hist, well)
    if well_jp.empty:
        return False
    db_name, start, end = _query_window(well, well_jp)
    # Sequential on purpose: the warmup already runs wells in parallel and each
    # of its threads owns one thread-local warehouse connection, so a nested
    # pool here would double warehouse concurrency for no wall-clock win.
    #
    # cache_refresh, not a plain call: a plain call returns a still-fresh entry
    # and queries nothing, which made the warm cadence the TTL rather than the
    # loop's interval. Forcing it also stores with the warm retention floor, so
    # the entry cannot be deleted between two passes (server/cache.py).
    extended_tests.cache_refresh(db_name, start, end)  # type: ignore[attr-defined]
    bhp_daily.cache_refresh(db_name, start, end)  # type: ignore[attr-defined]
    return True


def jp_history_payload(well: str) -> dict[str, Any]:
    """JpHistoryResponse payload for one well.

    mirrors woffl/gui/tabs/jp_history_tab.py:render_tab +
    _fetch_extended_well_tests (Databricks path): installs filtered/sorted by
    Date Set, and the test/BHP window running from the earliest install to
    today. The tab pads only the chart's x-range (15 days, display-only);
    the queries themselves start AT the earliest Date Set, so this does too.

    Args:
        well: canonical GUI well name (e.g. "MPB-28").

    Returns:
        Dict matching schemas.JpHistoryResponse. A well with no install rows
        returns empty installs/tests/bhp_daily and skips the SQL entirely.

    Raises:
        RuntimeError: when both JP-history sources are unavailable.
    """
    jp_hist, source = datasources.jp_history_safe()
    if jp_hist is None or source is None:
        raise RuntimeError("JP history unavailable (Databricks and Excel fallback both failed)")

    well_jp = _installs_for(jp_hist, well)

    payload: dict[str, Any] = {
        "well": well,
        "installs": [],
        "tests": [],
        "bhp_daily": [],
        "current_pump": None,
        "source": source,
    }
    if well_jp.empty:
        return payload

    # Stringify pump codes before projection: the tracker stores float
    # nozzles (12.0) and the JSON contract types nozzle/throat as strings.
    for col in ("Nozzle Number", "Throat Ratio"):
        if col in well_jp.columns:
            well_jp[col] = well_jp[col].map(_code_str)
    payload["installs"] = frames.records(well_jp, _INSTALL_COLUMNS)
    payload["current_pump"] = _current_pump_caption(jp_hist, well)

    db_name, start, end = _query_window(well, well_jp)

    # Fail-soft per series - the tab warns and renders what it has. The two
    # pulls are independent Databricks queries at seconds each, so they run
    # in PARALLEL (thread-local warehouse connections; ttl_cache is locked):
    # total = max(t1, t2) instead of t1 + t2.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="jp-hist") as pool:
        tests_future = pool.submit(extended_tests, db_name, start, end)
        bhp_future = pool.submit(bhp_daily, db_name, start, end)
        try:
            payload["tests"] = frames.records(tests_future.result(), _TEST_COLUMNS)
        except Exception:
            payload["tests"] = []
        try:
            payload["bhp_daily"] = frames.records(bhp_future.result(), _BHP_COLUMNS)
        except Exception:
            payload["bhp_daily"] = []
    return payload
