"""Well Sort page data: online/offline/LTSI tables, down events, marginal WC,
triage decisions, bench workbook.

The decision/marginal math is the SINGLE canonical implementation in
woffl.assembly.well_sort_engine; this module only fetches (TTL-cached),
composes via well_sort_client, and serializes. TTLs are carried over from
the retired Streamlit app: 1 h for the Databricks pulls, 5 min for live XV
status.

The Wells tab always queries a 180-day test window (the Streamlit tab's
fixed "Tests window: 180 d" caption), not the client default of 120.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import frames
from woffl.assembly import well_sort_engine as engine
from woffl.assembly.well_sort_client import (
    apply_pops_pad,
    build_online_table,
    build_shut_in_table,
    classify_wells,
    compute_recent_down_events,
    export_bench_xlsx,
    fetch_current_shut_in_history,
    fetch_last_tests_ever,
    fetch_mpu_producers,
    fetch_producer_catalog,
    fetch_recent_shut_in_history,
    fetch_recent_tests,
    fetch_xv_status,
    split_offline_ltsi,
)

TESTS_WINDOW_DAYS = 180  # the Streamlit Wells tab's fixed window


# ---------------------------------------------------------------------------
# Cached fetchers (failures never cached; XV soft-fails inside the client)
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_WELL_SORT, maxsize=1)
def _shut_in_history() -> pd.DataFrame:
    return fetch_current_shut_in_history()


@ttl_cache(config.TTL_WELL_SORT, maxsize=2)
def _recent_shut_in_history(days: int) -> pd.DataFrame:
    return fetch_recent_shut_in_history(days=days)


@ttl_cache(config.TTL_WELL_SORT, maxsize=2)
def _recent_tests(days: int) -> pd.DataFrame:
    return fetch_recent_tests(days=days)


@ttl_cache(config.TTL_WELL_SORT, maxsize=1)
def _producers() -> list[str]:
    return fetch_mpu_producers()


@ttl_cache(config.TTL_WELL_SORT, maxsize=1)
def _catalog() -> pd.DataFrame:
    return fetch_producer_catalog()


@ttl_cache(config.TTL_WELL_SORT, maxsize=1)
def _last_tests_ever() -> pd.DataFrame:
    return fetch_last_tests_ever()


@ttl_cache(config.TTL_XV_STATUS, maxsize=1)
def _xv_status() -> pd.DataFrame:
    return fetch_xv_status()


_CACHED_FETCHERS = (
    _shut_in_history,
    _recent_shut_in_history,
    _recent_tests,
    _producers,
    _catalog,
    _last_tests_ever,
    _xv_status,
)


def refresh() -> int:
    """Clear every Well Sort fetch cache (the page's Refresh button)."""
    for fn in _CACHED_FETCHERS:
        fn.cache_clear()  # type: ignore[attr-defined]
    return len(_CACHED_FETCHERS)


def warm_targets() -> list[tuple[str, Any]]:
    """(label, thunk) for the slow pulls a first page load would otherwise eat.

    Threading and retry cadence belong to ``server.warmup``, which owns the one
    warm loop for the whole app - this module only names its own fetchers so the
    warmup never has to guess which of them are worth a query. Each thunk is a
    ``cache.refresher``: a forced re-query that overwrites the entry, because a
    plain call returns a still-fresh entry and warms nothing.

    ``_xv_status`` is deliberately absent. It is live safety-valve state on a
    5 min TTL - warming it every few hours would only ever serve a reading old
    enough to be wrong, and the page already renders without it.
    """
    from server.cache import refresher

    return [
        ("shut_in_history", refresher(_shut_in_history)),
        ("producers", refresher(_producers)),
        ("producer_catalog", refresher(_catalog)),
        ("last_tests_ever", refresher(_last_tests_ever)),
        ("recent_tests", refresher(_recent_tests, TESTS_WINDOW_DAYS)),
    ]


# ---------------------------------------------------------------------------
# Pipeline (table build -> online/offline/LTSI split)
# ---------------------------------------------------------------------------


def _pops_config(pops_pads: list[str], force_true: list[str]) -> tuple[set[str], dict[str, bool]]:
    return set(pops_pads), {w: True for w in force_true}


def _build_tables(
    mode: str,
    stale_days: int,
    pops_pads: list[str],
    force_true: list[str],
) -> dict[str, Any]:
    """online/offline/ltsi frames + shared context, POPs applied."""
    shut_hist = _shut_in_history()
    tests = _recent_tests(TESTS_WINDOW_DAYS)
    producers = _producers()
    catalog = _catalog()
    last_tests = _last_tests_ever()
    xv = _xv_status()

    online_set, shut_set = classify_wells(producers, shut_hist, xv_df=xv, trust_xv=True)
    online_df = build_online_table(
        tests,
        shut_hist,
        producers,
        mode="allocated" if mode == "allocated" else "any",
        stale_days=stale_days,
        xv_df=xv,
        online_wells=online_set,
        catalog_df=catalog,
    )
    shut_df = build_shut_in_table(
        shut_hist,
        tests,
        xv_df=xv,
        shut_in_wells=shut_set,
        catalog_df=catalog,
        last_tests_df=last_tests,
    )
    pads, overrides = _pops_config(pops_pads, force_true)
    online_df = apply_pops_pad(online_df, pads, overrides)
    shut_df = apply_pops_pad(shut_df, pads, overrides)
    offline_df, ltsi_df = split_offline_ltsi(shut_df)

    all_pads: list[str] = (
        sorted(catalog["well_pad"].dropna().unique().tolist()) if not catalog.empty else []
    )
    return {
        "online": online_df,
        "offline": offline_df,
        "ltsi": ltsi_df,
        "producers": producers,
        "all_pads": all_pads,
        "xv_available": not xv.empty,
    }


def _online_full(stale_days: int, pops_pads: list[str], force_true: list[str]) -> pd.DataFrame:
    """The marginal-WC feed: online table, ALWAYS allocated mode (the Wells
    radio never affects the marginal calcs)."""
    return _build_tables("allocated", stale_days, pops_pads, force_true)["online"]


# ---------------------------------------------------------------------------
# Serialization column maps (DataFrame column -> JSON key)
# ---------------------------------------------------------------------------

_ONLINE_COLUMNS: dict[str, str] = {
    "Well": "well",
    "Pad": "pad",
    "Reservoir": "reservoir",
    "LiftType": "lift_type",
    "PopsPad": "pops_pad",
    "TestDate": "test_date",
    "DaysSinceTest": "days_since_test",
    "StaleTest": "stale_test",
    "Allocated": "allocated",
    "FallbackUsed": "fallback_used",
    "Oil": "oil",
    "Water": "water",
    "Gas": "gas",
    "LiftWater": "lift_water",
    "LiftGas": "lift_gas",
    "TotalWater": "total_water",
    "TotalGas": "total_gas",
    "EspHz": "esp_hz",
    "EspAmps": "esp_amps",
    "WC": "wc",
    "TotalWC": "total_wc",
    "GOR": "gor",
    "TotalGOR": "total_gor",
    "BHP": "bhp",
    "WHP": "whp",
    "Oil_2moAvg": "oil_2mo_avg",
    "Wat_2moAvg": "wat_2mo_avg",
    "OilDev": "oil_dev",
    "WatDev": "wat_dev",
    "FlagOutlier": "flag_outlier",
    "AllocVsInfoOilPct": "alloc_vs_info_oil_pct",
    "LatestAllocDate": "latest_alloc_date",
    "LatestInfoDate": "latest_info_date",
    "ProdXV": "prod_xv",
    "PFXV": "pf_xv",
    "XVTime": "xv_time",
    "JustRestarted": "just_restarted",
}

_SHUT_COLUMNS: dict[str, str] = {
    "Well": "well",
    "Pad": "pad",
    "Reservoir": "reservoir",
    "LiftType": "lift_type",
    "PopsPad": "pops_pad",
    "ShutInSince": "shut_in_since",
    "CurrentCode": "current_code",
    "CurrentReason": "current_reason",
    "Notes": "notes",
    "DownHours": "down_hours",
    "LastOnlineDate": "last_online_date",
    "LastTestDate": "last_test_date",
    "Oil": "oil",
    "Water": "water",
    "Gas": "gas",
    "LiftWater": "lift_water",
    "LiftGas": "lift_gas",
    "TotalWater": "total_water",
    "TotalGas": "total_gas",
    "EspHz": "esp_hz",
    "EspAmps": "esp_amps",
    "WC": "wc",
    "TotalWC": "total_wc",
    "GOR": "gor",
    "TotalGOR": "total_gor",
    "NearAvgOil": "near_avg_oil",
    "NearAvgWater": "near_avg_water",
    "NearAvgGas": "near_avg_gas",
    "NTestsNear": "n_tests_near",
    "ProdXV": "prod_xv",
    "PFXV": "pf_xv",
    "XVTime": "xv_time",
}

_TRIAGE_EXTRA_ONLINE: dict[str, str] = {
    "DecisionCode": "decision_code",
    "Why": "why",
    "WCvsMarginal": "wc_vs_marginal",
    "WCBasis": "wc_basis",
    "_rank": "rank",
}

_TRIAGE_EXTRA_SHUT: dict[str, str] = {
    "DecisionCode": "decision_code",
    "Why": "why",
    "WCvsMarginal": "wc_vs_marginal",
    "WCBasis": "wc_basis",
    "NearAvgWC": "near_avg_wc",
    "NearAvgWCBasis": "near_avg_wc_basis",
    "_rank": "rank",
}

_EVENT_COLUMNS: dict[str, str] = {
    "Well": "well",
    "Pad": "pad",
    "Reservoir": "reservoir",
    "Started": "started",
    "Ended": "ended",
    "Days": "days",
    "MaxHrs": "max_hrs",
    "TotalHrs": "total_hrs",
    "Code": "code",
    "Reason": "reason",
    "Notes": "notes",
    "Ongoing": "ongoing",
}

_RANKED_COLUMNS: dict[str, str] = {
    "Well": "well",
    "Pad": "pad",
    "Reservoir": "reservoir",
    "Oil": "oil",
    "TotalWater": "total_water",
    "TotalWC": "total_wc",
    "CumWater": "cum_water",
    "CumWaterPct": "cum_water_pct",
}

_PAD_RANKED_COLUMNS: dict[str, str] = {
    "Well": "well",
    "Reservoir": "reservoir",
    "Oil": "oil",
    "LiftWater": "lift_water",
    "TotalWater": "total_water",
    "TotalWC": "total_wc",
    "WC_pad": "wc_pad",
}


def _fmt_xv_time(df: pd.DataFrame) -> pd.DataFrame:
    """XVTime keeps its clock (frames.json_value truncates datetimes to a
    date; the old UI shows `MM-DD HH:mm` because same-day readings matter)."""
    if df.empty or "XVTime" not in df.columns:
        return df
    df = df.copy()
    df["XVTime"] = pd.to_datetime(df["XVTime"], errors="coerce").dt.strftime("%m-%d %H:%M")
    return df


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def tables_payload(
    mode: str, stale_days: int, pops_pads: list[str], force_true: list[str]
) -> dict[str, Any]:
    ctx = _build_tables(mode, stale_days, pops_pads, force_true)
    online = ctx["online"]
    return {
        "online": frames.records(_fmt_xv_time(online), _ONLINE_COLUMNS),
        "offline": frames.records(_fmt_xv_time(ctx["offline"]), _SHUT_COLUMNS),
        "ltsi": frames.records(_fmt_xv_time(ctx["ltsi"]), _SHUT_COLUMNS),
        "all_pads": ctx["all_pads"],
        "producers": ctx["producers"],
        "xv_available": ctx["xv_available"],
        "tests_window_days": TESTS_WINDOW_DAYS,
        "outliers_flagged": int(online["FlagOutlier"].sum()) if not online.empty else 0,
        "just_restarted": int(online["JustRestarted"].sum()) if not online.empty else 0,
        "default_pops_pads": list(engine.DEFAULT_POPS_PADS),
        "pump_limit_presets": dict(engine.PUMP_LIMIT_PRESETS),
        "pops_pump_handles": dict(engine.POPS_PUMP_HANDLES),
    }


def events_payload(window_days: int, down_hours: float) -> dict[str, Any]:
    events = compute_recent_down_events(
        _recent_shut_in_history(60),
        _producers(),
        catalog_df=_catalog(),
        window_days=window_days,
        down_hours_threshold=float(down_hours),
    )
    return {"rows": frames.records(events, _EVENT_COLUMNS)}


def _marginal_result(
    threshold_pct: float, stale_days: int, pops_pads: list[str], force_true: list[str]
) -> Optional[dict[str, Any]]:
    full = _online_full(stale_days, pops_pads, force_true)
    if full.empty:
        return None
    non_pops = full[~full["PopsPad"]].copy()
    return engine.field_marginal_wc(non_pops, threshold_pct)


def marginal_payload(
    threshold_pct: float, stale_days: int, pops_pads: list[str], force_true: list[str]
) -> Optional[dict[str, Any]]:
    result = _marginal_result(threshold_pct, stale_days, pops_pads, force_true)
    if result is None:
        return None
    ranked = result["ranked_df"]
    return {
        "marginal_wc": result["marginal_wc"],
        "well": result["well"],
        "pad": result["pad"],
        "total_field_water": result["total_field_water"],
        "well_count": result["well_count"],
        "threshold_pct": result["threshold_pct"],
        "marg_idx": result["marg_idx"],
        "cum_water_at_marginal": frames.opt_float(ranked.iloc[result["marg_idx"]]["CumWater"]),
        "rows": frames.records(ranked, _RANKED_COLUMNS),
    }


def pad_marginal_payload(
    pad: str,
    pump_limit: float,
    stale_days: int,
    pops_pads: list[str],
    force_true: list[str],
) -> Optional[dict[str, Any]]:
    full = _online_full(stale_days, pops_pads, force_true)
    result = engine.pad_marginal_wc(full, pad, pump_limit)
    if result is None:
        return None
    return {
        "marginal_wc": result["marginal_wc"],
        "well": result["well"],
        "pad": result["pad"],
        "pad_water": result["pad_water"],
        "pump_limit": result["pump_limit"],
        "headroom": result["headroom"],
        "well_count": result["well_count"],
        "water_basis": result["water_basis"],
        "rows": frames.records(result["ranked_df"], _PAD_RANKED_COLUMNS),
    }


def triage_payload(
    threshold_pct: float, stale_days: int, pops_pads: list[str], force_true: list[str]
) -> Optional[dict[str, Any]]:
    ctx = _build_tables("allocated", stale_days, pops_pads, force_true)
    non_pops = ctx["online"][~ctx["online"]["PopsPad"]].copy() if not ctx["online"].empty else ctx["online"]
    marg = engine.field_marginal_wc(non_pops, threshold_pct)
    if marg is None:
        return None

    online_dec = engine.add_online_decision(ctx["online"], marg["marginal_wc"])
    shut_dec = engine.add_shut_decision(ctx["offline"], marg["marginal_wc"])

    ranked = marg["ranked_df"]
    raw = ranked.iloc[0] if len(ranked) else None
    return {
        "marginal_wc": marg["marginal_wc"],
        "well": marg["well"],
        "pad": marg["pad"],
        "threshold_pct": marg["threshold_pct"],
        "raw_worst_wc": frames.opt_float(raw["TotalWC"]) if raw is not None else None,
        "raw_worst_well": str(raw["Well"]) if raw is not None else None,
        "raw_worst_water": frames.opt_float(raw["TotalWater"]) if raw is not None else None,
        "xv_available": ctx["xv_available"],
        "online": frames.records(
            _fmt_xv_time(online_dec), {**_ONLINE_COLUMNS, **_TRIAGE_EXTRA_ONLINE}
        ),
        "shut": frames.records(
            _fmt_xv_time(shut_dec), {**_SHUT_COLUMNS, **_TRIAGE_EXTRA_SHUT}
        ),
    }


def bench_xlsx(
    mode: str, stale_days: int, pops_pads: list[str], force_true: list[str]
) -> bytes:
    ctx = _build_tables(mode, stale_days, pops_pads, force_true)
    return export_bench_xlsx(ctx["online"], ctx["offline"], ctx["ltsi"])
