"""JP install history + extended production window (the history chart data).

Server-side port of the JP History tab's data layer: install rows from the
enriched tracker frame, well tests back to the earliest install, and the
daily BHP series. v1 serves the Databricks path only - the Streamlit
memory-gauge overlay layers have no server equivalent.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import datasources, frames

log = logging.getLogger(__name__)

# Lifted verbatim from the retired Streamlit app; the em dash in the
# original SQL comment is written as "-" per the ASCII house rule.
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

# --- Fleet forms of the same two pulls -------------------------------------
#
# Identical SELECT / JOIN shape to the two per-well queries above, widened to
# an IN list and the fleet's outer date window. `warm_fleet` runs each ONCE and
# slices the result per well, so a warm pass costs 2 statements instead of
# 2 x ~90. That matters because the SQL warehouse bills per WAKE WINDOW, not
# per statement: ~180 serialized per-well queries held the warehouse up for
# minutes on every pass.
_FLEET_TEST_QUERY = """\
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
WHERE vwt.well_name IN ({well_list})
    AND vwt.wt_date BETWEEN '{start_date}' AND '{end_date}'
    AND vwt.allocated = True
ORDER BY vwt.well_name, vwt.wt_date
"""

# The per-well form resolves enthid with a scalar subquery; the fleet form
# needs the mapping BACK to well_name to slice the result, so the same
# vw_well_test lookup joins as a distinct (well_name, enthid) table.
#
# One deliberate difference: the per-well subquery is `LIMIT 1`, so a well with
# two enthids in vw_well_test gets ONE of them, arbitrarily (no ORDER BY). The
# join takes all of them, which for such a well means more rows per tag_date
# than the per-well pull would have cached. Every well seen in-tree maps to a
# single enthid (prop_hist_client raises on >1 match when it resolves one), so
# this is a note for whoever meets the first counter-example, not a known bug.
_FLEET_BHP_QUERY = """\
SELECT
    vwt_map.well_name,
    vbdc.tag_date,
    round(vbdc.bhp_cln_value, 2) AS bhp
FROM mpu.wells.vw_bhp_daily_clean vbdc
JOIN (
    SELECT DISTINCT well_name, enthid
    FROM mpu.wells.vw_well_test
    WHERE well_name IN ({well_list})
) vwt_map
    ON vbdc.enthid = vwt_map.enthid
WHERE vbdc.tag_date BETWEEN '{start_date}' AND '{end_date}'
ORDER BY vwt_map.well_name, vbdc.tag_date
"""

# --- Shut-in windows (the chart's zero-rate periods) ------------------------
#
# Source: the daily downtime log mpu.wells.vw_shut_in, the same log Well Sort
# reads. There is no daily allocated-production view in mpu.wells, so a logged
# down day is the only daily on/off signal the app has. Like the BHP pull, the
# log is keyed by the well's vw_well_test enthid (dthid = enthid): that keeps a
# dual-purpose well's injector-side downtime out of the producer's history.
# Hours are summed per day across the log's rows (Well Sort's rule). Only days
# with notable downtime come back; `_shape_shut_in` collapses them to windows,
# so the cached value and the payload are a handful of rows, not one per day.
_SHUT_IN_HOURS_SQL = "SUM(CAST(s.down_hours AS DOUBLE))"

_SHUT_IN_QUERY = f"""\
SELECT
    s.dtdate,
    {_SHUT_IN_HOURS_SQL} AS hrs,
    MAX_BY(s.down_code, s.down_hours) AS down_code,
    MAX_BY(s.down_reason, s.down_hours) AS down_reason
FROM mpu.wells.vw_shut_in s
WHERE s.dthid IN (
    SELECT DISTINCT enthid
    FROM mpu.wells.vw_well_test
    WHERE well_name = '{{well_name}}'
)
AND s.dtdate BETWEEN '{{start_date}}' AND '{{end_date}}'
GROUP BY s.dtdate
HAVING {_SHUT_IN_HOURS_SQL} >= {{min_hours}}
ORDER BY s.dtdate
"""

# Fleet form, same shape as _FLEET_BHP_QUERY's join-back (named si_map so the
# two fleet joins stay distinguishable in logs and test fakes).
_FLEET_SHUT_IN_QUERY = f"""\
SELECT
    si_map.well_name,
    s.dtdate,
    {_SHUT_IN_HOURS_SQL} AS hrs,
    MAX_BY(s.down_code, s.down_hours) AS down_code,
    MAX_BY(s.down_reason, s.down_hours) AS down_reason
FROM mpu.wells.vw_shut_in s
JOIN (
    SELECT DISTINCT well_name, enthid
    FROM mpu.wells.vw_well_test
    WHERE well_name IN ({{well_list}})
) si_map
    ON s.dthid = si_map.enthid
WHERE s.dtdate BETWEEN '{{start_date}}' AND '{{end_date}}'
GROUP BY si_map.well_name, s.dtdate
HAVING {_SHUT_IN_HOURS_SQL} >= {{min_hours}}
ORDER BY si_map.well_name, s.dtdate
"""

# A day counts as SHUT IN at >= 20 logged down hours (Well Sort's "fully shut
# in" threshold). Partial days (>= 8 h, Well Sort's notable-down threshold)
# never start or end a window, but up to SHUT_IN_BRIDGE_DAYS of them between
# two full-down runs are bridged: a casing-leak shutdown logged 24, 19.7, 14.6,
# 24, 24 ... is one shut-in, not two with a phantom two-day restart.
SHUT_IN_BRIDGE_DAYS = 3

# Warehouse statements one well's history costs when warmed or opened cold
# (tests + daily BHP + shut-in log). server.warmup counts its fallback with it.
PER_WELL_STATEMENTS = 3


# ---------------------------------------------------------------------------
# Post-query shaping
#
# Factored out of the two fetchers so the FLEET pull can produce byte-identical
# per-well frames from a slice of one wide result. Every caller (per-well query,
# fleet slice) runs the same function, which is the only reason a primed entry
# is safe to hand to the request path unchanged.
# ---------------------------------------------------------------------------


def _shape_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Raw ``_EXTENDED_TEST_QUERY`` rows -> the frame `extended_tests` returns.

    An empty frame is returned untouched (that is what a well with no tests in
    the window has always produced, and `frames.records` handles it).
    """
    from woffl.assembly.pf_pressure import add_pf_columns
    from woffl.assembly.well_test_client import _normalize_well_name

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


def _shape_bhp(df: pd.DataFrame) -> pd.DataFrame:
    """Raw ``_BHP_DAILY_QUERY`` rows -> the frame `bhp_daily` returns."""
    if df.empty:
        return df

    if "tag_date" in df.columns:
        df["tag_date"] = pd.to_datetime(df["tag_date"], utc=True).dt.tz_localize(None)
    if "bhp" in df.columns:
        df["bhp"] = pd.to_numeric(df["bhp"], errors="coerce")

    return df.dropna(subset=["tag_date", "bhp"]).sort_values("tag_date")


_SHUT_IN_COLUMNS_OUT = ["start", "end", "days", "code", "reason"]


def _shut_in_thresholds() -> tuple[float, float]:
    """(full-day, partial-day) down-hour thresholds, shared with Well Sort."""
    from woffl.assembly.well_sort_client import FULL_DAY_HOURS_THRESHOLD, NOTABLE_DOWN_HOURS

    return float(FULL_DAY_HOURS_THRESHOLD), float(NOTABLE_DOWN_HOURS)


def _shape_shut_in(df: pd.DataFrame) -> pd.DataFrame:
    """Raw ``_SHUT_IN_QUERY`` day rows -> shut-in windows, oldest first.

    A window runs from its first to its last full-down day (inclusive). Days
    must be consecutive calendar days; any day missing from the rows (no
    notable downtime) ends the run. Partial days bridge two full-down runs
    only when there are at most ``SHUT_IN_BRIDGE_DAYS`` of them in a row.

    Returns:
        Frame with start/end (Timestamp, day resolution), days (calendar days
        in the window), code/reason (the first full-down day's cause). Empty
        (with those columns) when there is no full-down day.
    """
    full, partial = _shut_in_thresholds()
    empty = pd.DataFrame(columns=_SHUT_IN_COLUMNS_OUT)
    if df is None or df.empty or "dtdate" not in df.columns or "hrs" not in df.columns:
        return empty
    days = pd.DataFrame(
        {
            "day": pd.to_datetime(df["dtdate"], utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.normalize(),
            "hrs": pd.to_numeric(df["hrs"], errors="coerce"),
            "code": df["down_code"] if "down_code" in df.columns else None,
            "reason": df["down_reason"] if "down_reason" in df.columns else None,
        }
    )
    days = days.dropna(subset=["day", "hrs"])
    days = days[days["hrs"] >= partial].sort_values("day").drop_duplicates("day", keep="last")
    if days.empty:
        return empty

    windows: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    pending_partial = 0  # partial days since the current window's last full day
    prev_day: Optional[pd.Timestamp] = None
    for row in days.itertuples(index=False):
        contiguous = prev_day is not None and row.day - prev_day == pd.Timedelta(days=1)
        if not contiguous:
            current, pending_partial = None, 0
        if row.hrs >= full:
            if current is not None and pending_partial <= SHUT_IN_BRIDGE_DAYS:
                current["end"] = row.day
            else:
                current = {"start": row.day, "end": row.day, "code": row.code, "reason": row.reason}
                windows.append(current)
            pending_partial = 0
        elif current is not None:
            pending_partial += 1
            if pending_partial > SHUT_IN_BRIDGE_DAYS:
                current = None
        prev_day = row.day

    out = pd.DataFrame(windows, columns=["start", "end", "code", "reason"])
    out["days"] = ((out["end"] - out["start"]).dt.days + 1).astype(int)
    return out[_SHUT_IN_COLUMNS_OUT].reset_index(drop=True)


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
    from woffl.assembly.sql_guards import validate_iso_date, validate_well_name

    return _shape_tests(
        execute_query(
            _EXTENDED_TEST_QUERY.format(
                well_name=validate_well_name(db_name),
                start_date=validate_iso_date(start),
                end_date=validate_iso_date(end),
            )
        )
    )


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

    return _shape_bhp(
        execute_query(
            _BHP_DAILY_QUERY.format(
                well_name=validate_well_name(db_name),
                start_date=validate_iso_date(start),
                end_date=validate_iso_date(end),
            )
        )
    )


# Same rolling-`end` key as extended_tests - see its maxsize note.
@ttl_cache(config.TTL_EXTENDED_TESTS, maxsize=512)
def shut_in_windows(db_name: str, start: str, end: str) -> pd.DataFrame:
    """Shut-in windows from the daily downtime log (vw_shut_in).

    Args:
        db_name: Databricks-format well name (e.g. "L-006").
        start: window start, YYYY-MM-DD.
        end: window end, YYYY-MM-DD.

    Returns:
        ``_shape_shut_in`` frame (start, end, days, code, reason). Raises on
        query failure (failures are never cached).
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.sql_guards import validate_iso_date, validate_well_name

    return _shape_shut_in(
        execute_query(
            _SHUT_IN_QUERY.format(
                well_name=validate_well_name(db_name),
                start_date=validate_iso_date(start),
                end_date=validate_iso_date(end),
                min_hours=float(_shut_in_thresholds()[1]),
            )
        )
    )


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

# start/end are inclusive shut-in DAYS (YYYY-MM-DD); the chart zeroes rates
# from the start of `start` to the end of `end`.
_SHUT_IN_COLUMNS: dict[str, str] = {
    "start": "start",
    "end": "end",
    "days": "days",
    "code": "code",
    "reason": "reason",
}


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
    """(db_name, start, end) for this well's extended_tests / bhp_daily /
    shut_in_windows pulls.

    THE single source of those cache keys: ``warm_well``, ``warm_fleet``
    and ``jp_history_payload`` must agree exactly or the warmup fills entries
    no request ever reads. ``end`` is today, so the keys roll over at midnight -
    see the maxsize notes on the fetchers.
    """
    from woffl.assembly.well_test_client import _denormalize_well_name

    return (
        _denormalize_well_name(well),
        pd.Timestamp(well_jp["Date Set"].min()).strftime("%Y-%m-%d"),
        datetime.now().strftime("%Y-%m-%d"),
    )


def warm_well(well: str) -> bool:
    """Pre-pay the per-well Databricks pulls behind /wells/{name}/jp-history
    (tests, daily BHP and the shut-in log - ``PER_WELL_STATEMENTS``).

    These are the app's only genuinely per-well warehouse queries, and they are
    why a well used to be slow on its first open and instant for the rest of the
    day. ``server.warmup`` now covers the fleet with ``warm_fleet``'s
    statements and calls this only as the FALLBACK when that fails; it is also
    the on-demand path for warming a single well.

    Returns:
        True when the tracker had a dated install and every fetcher was
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
    shut_in_windows.cache_refresh(db_name, start, end)  # type: ignore[attr-defined]
    return True


def _fleet_slice(
    df: pd.DataFrame, well_col: str, date_col: str, db_name: str, start: str, end: str
) -> pd.DataFrame:
    """One well's rows out of a fleet frame, over ITS OWN [start, end] window.

    Reproduces the per-well query's WHERE clause in pandas: the same
    ``BETWEEN`` bounds (inclusive, midnight-anchored like the SQL string
    comparison) against the same UTC-naive date normalisation the shaping
    functions apply, on the raw column names, before any renaming. The index is
    reset so the primed frame is indistinguishable from a fresh query result.
    """
    if df.empty or well_col not in df.columns or date_col not in df.columns:
        return df.iloc[0:0].reset_index(drop=True)
    dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    mask = (df[well_col] == db_name) & dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def warm_fleet(wells: list[str]) -> dict[str, Any]:
    """Warm every well's per-well caches with THREE fleet statements.

    ``warm_well`` per well is 3 warehouse queries x ~90 wells = ~270 statements
    holding the SQL warehouse awake for minutes on every pass, and the warehouse
    bills per WAKE WINDOW rather than per statement. The windowed pulls are
    the same SELECT/JOIN shape widened to an IN list, so one statement each
    answers the whole fleet; every well's slice is shaped by the SAME
    ``_shape_tests`` / ``_shape_bhp`` / ``_shape_shut_in`` the per-well fetchers
    use and primed under the exact key ``jp_history_payload`` reads. The
    shut-in statement alone is fail-soft: its failure primes the other two and
    leaves the shut-in entries cold.

    A well with no rows in a fleet frame is primed with an EMPTY frame - that is
    precisely what its per-well query would have returned, and priming it is
    what keeps the request path off the warehouse.

    Args:
        wells: canonical GUI well names (e.g. "MPB-28").

    Returns:
        {"wells": primed, "skipped": no dated install, "statements": issued}.

    Raises:
        RuntimeError: when no JP-history source is reachable. Query failures
        propagate too (failures are NEVER cached), so the caller can fall back
        to the per-well path.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.sql_guards import validate_iso_date, validate_well_name

    jp_hist, source = datasources.jp_history_safe()
    if jp_hist is None or source is None:
        raise RuntimeError("JP history unavailable (Databricks and Excel fallback both failed)")

    # Same skip rule as warm_well: a well with no dated install has nothing to
    # query, so it must not widen the IN list either.
    windows: list[tuple[str, str, str]] = []
    skipped = 0
    for well in wells:
        well_jp = _installs_for(jp_hist, well)
        if well_jp.empty:
            skipped += 1
            continue
        windows.append(_query_window(well, well_jp))
    if not windows:
        return {"wells": 0, "skipped": skipped, "statements": 0}

    # The outer window: earliest install anywhere in the fleet, through today.
    # Each well is then sliced back to its own [start, end], so a well never
    # gets rows from before its first pump.
    fleet_start = min(start for _db, start, _end in windows)
    fleet_end = max(end for _db, _start, end in windows)
    sql_args = {
        "well_list": ", ".join(
            f"'{validate_well_name(db)}'" for db in sorted({db for db, _s, _e in windows})
        ),
        "start_date": validate_iso_date(fleet_start),
        "end_date": validate_iso_date(fleet_end),
    }

    # Captured BEFORE the fetch: a write that clears these caches while the
    # fleet query is in flight must discard the primes, exactly as it discards
    # an in-flight per-well fetch (server/cache.py clear() version guard).
    tests_version = extended_tests.cache_version()  # type: ignore[attr-defined]
    bhp_version = bhp_daily.cache_version()  # type: ignore[attr-defined]
    shut_in_version = shut_in_windows.cache_version()  # type: ignore[attr-defined]

    tests_raw = execute_query(_FLEET_TEST_QUERY.format(**sql_args))
    bhp_raw = execute_query(_FLEET_BHP_QUERY.format(**sql_args))
    # The shut-in log only decorates the chart, so its failure must not throw
    # away the two primes above (nor push the pass onto the per-well fallback):
    # those wells' shut-in entries stay cold and the request path fetches them
    # on demand, fail-soft.
    shut_in_raw: Optional[pd.DataFrame]
    try:
        shut_in_raw = execute_query(
            _FLEET_SHUT_IN_QUERY.format(**sql_args, min_hours=float(_shut_in_thresholds()[1]))
        )
    except Exception as exc:  # noqa: BLE001 - optional decoration, see above
        log.warning("warm: fleet shut-in log pull failed: %s", exc)
        shut_in_raw = None

    primed = 0
    for db_name, start, end in windows:
        tests_df = _shape_tests(
            _fleet_slice(tests_raw, "well_name", "wt_date", db_name, start, end)
        )
        # Drop the join-back column: bhp_daily's own query selects only
        # (tag_date, bhp), and a primed frame must match it column for column.
        bhp_df = _shape_bhp(
            _fleet_slice(bhp_raw, "well_name", "tag_date", db_name, start, end).drop(
                columns=["well_name"], errors="ignore"
            )
        )
        stored_tests = extended_tests.cache_prime(  # type: ignore[attr-defined]
            tests_df, db_name, start, end, version=tests_version
        )
        stored_bhp = bhp_daily.cache_prime(  # type: ignore[attr-defined]
            bhp_df, db_name, start, end, version=bhp_version
        )
        if shut_in_raw is not None:
            shut_in_windows.cache_prime(  # type: ignore[attr-defined]
                _shape_shut_in(
                    _fleet_slice(shut_in_raw, "well_name", "dtdate", db_name, start, end)
                ),
                db_name,
                start,
                end,
                version=shut_in_version,
            )
        if stored_tests and stored_bhp:
            primed += 1
    return {"wells": primed, "skipped": skipped, "statements": PER_WELL_STATEMENTS}


def jp_history_payload(well: str) -> dict[str, Any]:
    """JpHistoryResponse payload for one well.

    The Databricks path: installs filtered/sorted by Date Set, and the
    test/BHP window running from the earliest install to today. The retired
    Streamlit tab padded only the chart's x-range (15 days, display-only);
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
        "shut_in": [],
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

    # Fail-soft per series - the tab warns and renders what it has. The three
    # pulls are independent Databricks queries at seconds each, so they run
    # in PARALLEL (thread-local warehouse connections; ttl_cache is locked):
    # total = max(t1, t2, t3) instead of the sum. A missing shut-in log just
    # leaves `shut_in` empty and the chart interpolates between tests as before.
    with ThreadPoolExecutor(max_workers=PER_WELL_STATEMENTS, thread_name_prefix="jp-hist") as pool:
        tests_future = pool.submit(extended_tests, db_name, start, end)
        bhp_future = pool.submit(bhp_daily, db_name, start, end)
        shut_in_future = pool.submit(shut_in_windows, db_name, start, end)
        try:
            payload["tests"] = frames.records(tests_future.result(), _TEST_COLUMNS)
        except Exception:
            payload["tests"] = []
        try:
            payload["bhp_daily"] = frames.records(bhp_future.result(), _BHP_COLUMNS)
        except Exception:
            payload["bhp_daily"] = []
        try:
            payload["shut_in"] = frames.records(shut_in_future.result(), _SHUT_IN_COLUMNS)
        except Exception as exc:  # noqa: BLE001 - optional decoration
            log.warning("shut-in log unavailable for %s: %s", well, exc)
            payload["shut_in"] = []
    return payload
