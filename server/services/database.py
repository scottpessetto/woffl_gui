"""Well Database page data: chars table, aging jet pumps, prop_hist audit.

Server-side port of woffl/gui/well_database_page.py (table + aging filters)
and woffl/gui/prop_history.py (save-history fetch + shaping). Read-only.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import datasources, frames

# DataFrame column -> JSON key for the chars table rows.
# mirrors woffl/gui/well_database_page.py:run_well_database_page display_cols
_CHARS_COLUMNS: dict[str, str] = {
    "Well": "well",
    "is_sch": "is_sch",
    "tvd_estimated": "tvd_estimated",
    "out_dia": "tubing_od",
    "thick": "tubing_thickness",
    "casing_out_dia": "casing_out_dia",
    "casing_inn_dia": "casing_inn_dia",
    "JP_MD": "jp_md",
    "JP_TVD": "jp_tvd",
    "res_pres": "res_pres",
    "form_temp": "form_temp",
    "oil_api": "oil_api",
    "gas_sg": "gas_sg",
    "wat_sg": "wat_sg",
    "bubble_point": "bubble_point",
}


def database_rows() -> dict[str, Any]:
    """WellDatabaseResponse payload: chars rows + missing-survey wells.

    Returns:
        Dict matching schemas.WellDatabaseResponse.
    """
    df, source = datasources.well_chars_safe()
    return {
        "rows": frames.records(df, _CHARS_COLUMNS),
        "source": source,
        "missing_surveys": datasources.missing_surveys_safe(),
    }


# mirrors woffl/gui/well_database_page.py:_latest_test_dates
_LATEST_TEST_QUERY = """\
SELECT well_name, max(wt_date) AS last_test
FROM mpu.wells.vw_well_test
WHERE allocated = True
GROUP BY well_name
"""


# mirrors woffl/gui/well_database_page.py:_latest_test_dates
@ttl_cache(config.TTL_CHARS, maxsize=1)
def latest_test_dates() -> dict[str, Any]:
    """{well: latest ALLOCATED test date} - the recency proxy for "online".

    Fail-soft to {}: the aging filter then disables itself rather than
    silently dropping every well (the empty result is cached like the
    Streamlit site cached it).
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    try:
        df = execute_query(_LATEST_TEST_QUERY)
        return {
            _normalize_well_name(str(r["well_name"]).strip()): r["last_test"]
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


# DataFrame column -> JSON key for the aging-pump rows.
_AGING_COLUMNS: dict[str, str] = {
    "Well Name": "well",
    "Pump": "pump",
    "Date Set": "date_set",
    "Days In Hole": "days_in_hole",
    "Installs": "installs",
    "Last Test": "last_test",
    "Online": "online",
}


def aging_pumps(known_only: bool, online_only: bool, online_days: int, min_days: int) -> dict[str, Any]:
    """AgingPumpsResponse payload: current-pump tenure per well, filtered.

    mirrors woffl/gui/well_database_page.py:run_well_database_page (the
    "Aging jet pumps" section): pump_ages over the enriched JP-history
    frame, the wells-in-chars filter, and the recently-online filter. The
    page's age threshold flags rows; here `min_days` filters them (the API
    consumer renders exactly what it asked for).

    Args:
        known_only: drop tracker wells absent from the chars table
            (converted to ESP, retired, or never characterized).
        online_only: keep only wells with an ALLOCATED well test inside the
            `online_days` window. Skipped when test dates are unavailable
            (mirrors filter_recently_online's empty-map behavior).
        online_days: recency window for the online proxy, days.
        min_days: keep pumps with days_in_hole >= this.

    Returns:
        Dict matching schemas.AgingPumpsResponse; rows sorted oldest first.
    """
    from woffl.assembly.jp_history import pump_ages
    from woffl.assembly.pump_report import format_pump

    jp_hist, _source = datasources.jp_history_safe()
    if jp_hist is None:
        return {"rows": []}
    ages = pump_ages(jp_hist)
    if ages.empty:
        return {"rows": []}

    if known_only:
        chars, _chars_source = datasources.well_chars_safe()
        if "Well" in chars.columns:
            ages = ages[ages["Well Name"].isin(set(chars["Well"].astype(str)))]
        if ages.empty:
            return {"rows": []}

    # Attach Last Test / Online for every row (the response always carries
    # the proxy), then filter. mirrors
    # woffl/assembly/jp_history.py:filter_recently_online - vw_well_test
    # dates arrive TZ-AWARE (Etc/UTC) while today is naive, so coerce
    # through UTC and strip the tz before comparing.
    last_tests = latest_test_dates()
    ages = ages.copy()
    ages["Last Test"] = pd.to_datetime(
        ages["Well Name"].map(last_tests), utc=True, errors="coerce"
    ).dt.tz_localize(None)
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=int(online_days))
    ages["Online"] = ages["Last Test"].notna() & (ages["Last Test"] >= cutoff)

    if online_only and last_tests:
        ages = ages[ages["Online"]]
    ages = ages[ages["Days In Hole"] >= int(min_days)].copy()
    if ages.empty:
        return {"rows": []}

    nozzles = ages.get("Nozzle Number", pd.Series(index=ages.index, dtype=object))
    throats = ages.get("Throat Ratio", pd.Series(index=ages.index, dtype=object))
    ages["Pump"] = [format_pump(n, t) for n, t in zip(nozzles, throats)]
    return {"rows": frames.records(ages, _AGING_COLUMNS)}


# mirrors woffl/gui/prop_history.py:fetch_prop_history (query lifted verbatim,
# f-string braces made str.format-safe)
_PROP_HISTORY_QUERY = """\
SELECT ph.prop_id,
       coalesce(x.prop_name, ph.prop_id) AS prop_name,
       x.units,
       coalesce(x.category, 'other') AS category,
       ph.prop_value,
       ph.entry_datetime,
       ph.entry_user,
       c.comment_text AS comment
FROM mpu.wells.prop_hist ph
LEFT JOIN mpu.wells.prop_xref x ON ph.prop_id = x.prop_id
-- The engineer's note for the SAVE this row belongs to. prop_hist has
-- no batch id, so the shared entry_datetime is the join key (every
-- prop of one save is written with one stamp -- see
-- prop_hist_client.push_prop). Grouped so a retried comment write can
-- never fan a prop row out into duplicates.
LEFT JOIN (
    SELECT enthid, entry_datetime, MAX(comment_text) AS comment_text
    FROM mpu.wells.woffl_eng_comment
    GROUP BY enthid, entry_datetime
) c ON ph.enthid = c.enthid AND ph.entry_datetime = c.entry_datetime
WHERE ph.enthid = {enthid}
ORDER BY ph.entry_datetime DESC, ph.prop_id
"""


# mirrors woffl/gui/prop_history.py:fetch_prop_history (5 min TTL - the page
# an engineer refreshes right after a save, so a long TTL would show a lie)
@ttl_cache(config.TTL_PROP_HISTORY, maxsize=64)
def _prop_history(enthid: int) -> pd.DataFrame:
    """Every prop_hist row for one enthid, newest first. Raises on failure."""
    from woffl.assembly.databricks_client import execute_query

    # int() is the numeric-interpolation guard (see sql_guards module doc).
    return execute_query(_PROP_HISTORY_QUERY.format(enthid=int(enthid)))


def evict_prop_history(well: str) -> None:
    """Drop one well's audit-trail entry after a prop_hist write, so the
    prop-history page shows the new rows on its next poll instead of after
    the 5-minute TTL. Called by server.services.ipr._invalidate_after_write.

    The enthid map is process-cached and warm after any successful write
    (the push resolved this same well through it), so the lookup is a dict
    read. If it fails anyway, fall back to clearing the whole 64-entry
    cache - a successful write must never fail on eviction, and correctness
    beats one page's worth of re-SELECTs.
    """
    from woffl.assembly.prop_hist_client import well_enthid_map

    try:
        enthid = well_enthid_map().get(well)
    except Exception:  # noqa: BLE001 - see docstring
        _prop_history.cache_clear()
        return
    if enthid is not None:
        _prop_history.cache_evict(int(enthid))


# DataFrame column -> JSON key for prop-history rows (current + history).
_PROP_COLUMNS: dict[str, str] = {
    "prop_id": "prop_id",
    "prop_name": "prop_name",
    "units": "units",
    "category": "category",
    "prop_value": "prop_value",
    "entry_user": "entry_user",
    "entry_datetime": "entry_datetime",
    "comment": "comment_text",
}


def prop_history_payload(well: str) -> Optional[dict[str, Any]]:
    """PropHistoryResponse payload, or None when the well has no enthid.

    mirrors woffl/gui/prop_history.py:shape_history (the ordering + latest
    per-prop marking; display strings and Alaska-time columns stay
    client-side): history = all rows newest-first, current = each prop's
    live row ordered by category then name.

    Args:
        well: canonical GUI well name (e.g. "MPB-28").

    Returns:
        Dict matching schemas.PropHistoryResponse, or None for a well absent
        from vw_well_header's enthid map (the router 404s).
    """
    from woffl.assembly.prop_hist_client import well_enthid_map

    enthid = well_enthid_map().get(well)
    if enthid is None:
        return None

    df = _prop_history(int(enthid))
    if df is None or df.empty:
        return {"well": well, "current": [], "history": []}

    d = df.copy()
    if "comment" not in d.columns:
        d["comment"] = None
    d["entry_datetime"] = pd.to_datetime(d["entry_datetime"], utc=True)
    d = d.sort_values(["entry_datetime", "prop_id"], ascending=[False, True]).reset_index(drop=True)
    is_current = ~d["prop_id"].duplicated()
    # Keep the full UTC timestamp: entry_datetime is the ordering key and the
    # client shows wall-clock time, but frames.records date-truncates
    # Timestamps - so pre-format to strings before projection.
    d["entry_datetime"] = d["entry_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    latest = d[is_current].sort_values(["category", "prop_name"]).reset_index(drop=True)
    return {
        "well": well,
        "current": frames.records(latest, _PROP_COLUMNS),
        "history": frames.records(d, _PROP_COLUMNS),
    }
