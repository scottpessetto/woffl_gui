"""Header page service: per-well relations and IPRs, the impact run, and saves.

Three operations, each a small layer over :mod:`header_model` (the math):

``build_board``
    Every producer on the selected pads with its lift type, recent test,
    current WHP/BHP (historian), measured closed-loop BHP~WHP and WHP~Header
    relations, saved values from ``prop_hist``, a Vogel IPR fit from gauged
    tests, and the lift-group correlations that gaugeless wells borrow.

``run_impact``
    The oil/liquid response to a production-header change per pad, either a
    typed scenario or one measured around an observed event (a bring-online).
    Jet pumps are solved with the WOFFL model at both wellhead pressures from
    the same saved inputs an optimization run uses; every other lift type uses
    its relation and IPR. Event runs also compare predicted and measured BHP
    changes on gauged wells.

``save``
    Persists chosen relations and IPRs to ``mpu.wells.prop_hist`` through the
    one gated writer, one statement per well. Values come from the completed
    board job on the server, never from the client, except explicit manual
    entries.

"Header" here is always the PRODUCTION header (wellhead back-pressure),
never the power-fluid header the pad optimizer calls "header".
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from server import config, jobs
from server.cache import ttl_cache
from server.services import datasources, header_model as hm, tests as tests_svc

log = logging.getLogger(__name__)

# prop_hist ids owned by this page (added to mpu.wells.prop_xref 2026-09-29).
HDR_SLOPE = "hdr_bhp_whp_slope"
HDR_WHP_HDR = "hdr_whp_hdr_slope"
HDR_R2 = "hdr_fit_r2"
HDR_DAYS = "hdr_fit_days"
HDR_REL_SOURCE = "hdr_rel_source"
HDR_IPR_SOURCE = "hdr_ipr_source"
HDR_PROP_IDS = (HDR_SLOPE, HDR_WHP_HDR, HDR_R2, HDR_DAYS, HDR_REL_SOURCE, HDR_IPR_SOURCE)
# The well's one IPR lives in the existing curve props.
IPR_PROP_IDS = ("ipr_qwf_liq", "ipr_pwf", "resvr_press")
SAVED_PROP_IDS = HDR_PROP_IDS + IPR_PROP_IDS

FIT_DAYS_DEFAULT = 120
TEST_MONTHS = 12
ONLINE_MAX_TEST_AGE_DAYS = 45
NOW_WINDOW_HOURS = 72
# A gauged well whose BHP now sits this far above its latest test is treated
# as down (an ESP trip builds BHP toward reservoir pressure).
DOWN_BHP_RISE_PSI = 300.0
DOWN_BHP_RISE_FRAC = 1.5
# Event validation: an error beyond max(this, 3x the header change, 3x the
# prediction) is a well event inside the window, not a relation error.
OPERATIONAL_ERR_PSI = 15.0
# Measured wells needed before a lift + reservoir correlation replaces the
# lift-only one.
CORR_MIN_WELLS = 4
# Gauged, flowing wells before a pad + reservoir BHP/ResP group replaces the
# reservoir-wide one.
IPR_GROUP_MIN = 3
# BHP/ResP for a gaugeless well when no gauged well gives one.
DEFAULT_BHP_RATIO = 0.35
# A well without a saved ResP runs on the documented default +/- this band.
DEFAULT_PRES_BAND = 0.20
# Board cache: long enough for a string of "what if" runs, cleared on save.
BOARD_TTL = 900.0
# Response curve: uniform header change on every selected pad (psi). Jet
# pumps are solved on the coarser grid and interpolated.
CURVE_GRID = (-30, -20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40)
CURVE_JP_GRID = (-30, -10, 10, 20, 40)
# Downtime-log POP suggestions around an event.
POP_LOOKBACK_DAYS = 3
POP_AFTER_DAYS = 2
POP_DOWN_HOURS = 12.0
POP_FULL_DAY_HOURS = 23.5
POP_UP_HOURS = 4.0

_PAD_RE = re.compile(r"^[A-Z]$")
_CACHE_TTL = 300.0
_cache_lock = threading.Lock()
_saved_cache: dict[tuple, tuple[float, dict]] = {}
_board_cache: dict[tuple, tuple[float, dict]] = {}

ProgressFn = Callable[[str], None]


def _noop(_text: str) -> None:
    return None


def clean_pads(pads: list[str]) -> list[str]:
    """Upper-cased unique pad letters; raises on anything else (SQL safety)."""
    out: list[str] = []
    for p in pads:
        s = str(p).strip().upper()
        if not _PAD_RE.match(s):
            raise ValueError(f"invalid pad '{p}' - expected one letter")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("select at least one pad")
    return sorted(out)


def clear_caches() -> None:
    with _cache_lock:
        _saved_cache.clear()
        _board_cache.clear()


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _r(x: Any, nd: int = 1) -> Optional[float]:
    v = _f(x)
    return None if v is None else round(v, nd)


# ── reads ────────────────────────────────────────────────────────────────────


def fetch_saved(pads: tuple[str, ...]) -> dict[str, dict[str, dict[str, Any]]]:
    """Latest saved header/IPR props per well on ``pads`` (5-minute cache).

    Returns ``{well: {prop_id: {"value", "at", "by"}}}`` with normalized well
    names. Fail-soft: a read failure returns {} and the board says nothing is
    saved rather than failing.
    """
    key = tuple(pads)
    with _cache_lock:
        hit = _saved_cache.get(key)
        if hit and time.monotonic() - hit[0] < _CACHE_TTL:
            return hit[1]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        from woffl.assembly.databricks_client import execute_query
        from woffl.assembly.well_test_client import _normalize_well_name

        ids = ",".join(f"'{p}'" for p in SAVED_PROP_IDS)
        pad_list = ",".join(f"'{p}'" for p in clean_pads(list(pads)))
        df = execute_query(
            f"""
            SELECT h.well_name, p.prop_id, p.prop_value, p.entry_datetime, p.entry_user
            FROM mpu.wells.vw_well_header h
            JOIN (
                SELECT enthid, prop_id, prop_value, entry_datetime, entry_user,
                       ROW_NUMBER() OVER (PARTITION BY enthid, prop_id
                                          ORDER BY entry_datetime DESC) AS rn
                FROM mpu.wells.prop_hist
                WHERE prop_id IN ({ids})
            ) p ON p.enthid = h.enthid AND p.rn = 1
            WHERE h.well_pad IN ({pad_list})
            """
        )
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                well = _normalize_well_name(str(r["well_name"]).strip())
                at = r.get("entry_datetime")
                out.setdefault(well, {})[str(r["prop_id"])] = {
                    "value": _f(r.get("prop_value")),
                    "at": str(at)[:19] if at is not None and not pd.isna(at) else None,
                    "by": str(r.get("entry_user") or "") or None,
                }
    except Exception:  # noqa: BLE001 - a board without saved values still works
        log.warning("header: saved props unavailable", exc_info=True)
        return {}
    with _cache_lock:
        _saved_cache[key] = (time.monotonic(), out)
    return out


@ttl_cache(config.TTL_WELL_TESTS, maxsize=1)
def fetch_reservoirs() -> dict[str, str]:
    """{well: reservoir name} for producers, from vw_well_header."""
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(
        "SELECT well_name, reservoir FROM mpu.wells.vw_well_header WHERE well_type = 'prod'"
    )
    if df is None or df.empty:
        return {}
    return {_normalize_well_name(str(r["well_name"]).strip()): str(r["reservoir"] or "")
            for _, r in df.iterrows()}


def _reservoirs() -> dict[str, str]:
    try:
        return fetch_reservoirs()
    except Exception:  # noqa: BLE001 - reservoir only picks defaults
        log.warning("header: reservoir names unavailable", exc_info=True)
        return {}


def _latest_tests(tests: pd.DataFrame, wells: list[str]) -> dict[str, dict[str, Any]]:
    """Latest test per well with the fields the board uses."""
    out: dict[str, dict[str, Any]] = {}
    if tests is None or tests.empty or "well" not in tests.columns:
        return out
    sub = tests[tests["well"].isin(wells)].copy()
    if sub.empty:
        return out
    sub["WtDate"] = pd.to_datetime(sub["WtDate"], errors="coerce")
    sub = sub.sort_values("WtDate")
    for well, g in sub.groupby("well"):
        pos = (g[pd.to_numeric(g["WtTotalFluid"], errors="coerce") > 0]
               if "WtTotalFluid" in g.columns else g.iloc[0:0])
        row = (pos if not pos.empty else g).iloc[-1]
        wc = _f(row.get("form_wc"))
        if wc is not None and wc > 1.0:
            wc = wc / 100.0
        out[str(well)] = {
            "date": row["WtDate"],
            "oil": _f(row.get("WtOilVol")),
            "liquid": _f(row.get("WtTotalFluid")),
            "wc": wc,
            "gor": _f(row.get("fgor")),
            "bhp": _f(row.get("BHP")),
            "whp": _f(row.get("whp")),
        }
    return out


def _recent_median(s: Optional[pd.Series], hours: int = NOW_WINDOW_HOURS) -> Optional[float]:
    if s is None:
        return None
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    idx = pd.DatetimeIndex(s.index)
    cut = idx.max() - pd.Timedelta(hours=hours)
    return _f(s[idx >= cut].median())


def _trends(wells: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    from server.services.tools import header_trend as ht

    if not wells:
        return {}
    dfs, _missing = ht.fetch_header_trends(tuple(sorted(wells)), start.isoformat(), end.isoformat())
    return dfs or {}


def _relation_fit(df: Optional[pd.DataFrame], y: str, x: str) -> dict[str, Any]:
    from server.services.tools import header_trend as ht

    if df is None or y not in df.columns or x not in df.columns:
        return hm.summarize_daily(None)
    fit = ht.fit_within_day(df, y_name=y, x_name=x, robust=True)
    return hm.summarize_daily(fit.daily if fit is not None else None)


# ── board ────────────────────────────────────────────────────────────────────


def build_board(pads: list[str], fit_days: int = FIT_DAYS_DEFAULT,
                progress: ProgressFn = _noop) -> dict[str, Any]:
    """Per-well relations, IPR candidates and the correlations wells can borrow.

    Cached per (pads, fit_days, day) for BOARD_TTL and cleared by a save, so
    repeated questions ("what if it is 15 psi?") reuse the same fits.
    """
    pads = clean_pads(pads)
    key = (tuple(pads), int(fit_days), date.today().isoformat())
    with _cache_lock:
        hit = _board_cache.get(key)
        if hit and time.monotonic() - hit[0] < BOARD_TTL:
            return hit[1]

    from woffl.assembly.jp_history import get_current_pump
    from server.services.tools import header_impact as hi

    progress("loading producers and tests")
    overview = hi.fetch_well_overview(6)
    if overview is None or overview.empty:
        raise ValueError("no producer tests available - Databricks unreachable?")
    ov = overview[overview["well_pad"].isin(pads)].copy()
    wells = sorted(ov["well"].astype(str))
    jp_hist, _ = datasources.jp_history_safe()
    reservoirs = _reservoirs()
    tests = tests_svc.fetch_all_well_tests(TEST_MONTHS)
    latest = _latest_tests(tests, wells)
    today = date.today()

    progress(f"reading {fit_days} days of hourly BHP/WHP/header for {len(wells)} wells")
    trends = _trends(wells, today - timedelta(days=int(fit_days)), today)
    saved = fetch_saved(tuple(pads))

    progress("fitting relations and IPRs")
    rows: list[dict[str, Any]] = []
    for _, o in ov.iterrows():
        well = str(o["well"])
        lift = hi._classify_lift(well, jp_hist, o)
        pump = get_current_pump(jp_hist, well) if jp_hist is not None else None
        t = latest.get(well, {})
        df = trends.get(well)
        reservoir = hm.reservoir_key(reservoirs.get(well))
        wt_date = t.get("date") if t.get("date") is not None else o.get("wt_date")
        age = None
        if wt_date is not None and not pd.isna(wt_date):
            ts = pd.Timestamp(wt_date)
            ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
            age = (pd.Timestamp(today) - ts.normalize()).days
        bhp_now = _recent_median(df["BHP"]) if df is not None and "BHP" in df else None
        whp_now = _recent_median(df["WHP"]) if df is not None and "WHP" in df else None
        hdr_now = _recent_median(df["HeaderP"]) if df is not None and "HeaderP" in df else None
        gauge_note = hm.gauge_problem(bhp_now, whp_now, reservoir,
                                      _recent(df["BHP"]) if df is not None and "BHP" in df else None)

        rel = _relation_fit(df, "BHP", "WHP")
        whp_hdr = _relation_fit(df, "WHP", "HeaderP")

        # Gauge-test Vogel fit (non-JP wells; JP wells use their Solver IPR).
        ipr_fit_stats = None
        if lift != "JP" and gauge_note is None and tests is not None and not tests.empty:
            tw = tests[(tests["well"] == well)].copy()
            tw = tw[(pd.to_numeric(tw["BHP"], errors="coerce") > 50)
                    & (pd.to_numeric(tw["WtTotalFluid"], errors="coerce") > 0)]
            if len(tw) >= 2:
                fit = hm.fit_pseudo_pr(tw["BHP"].astype(float).tolist(),
                                       tw["WtTotalFluid"].astype(float).tolist(),
                                       hm.pr_cap(reservoir))
                if fit is not None:
                    ipr_fit_stats = {"pres": _r(fit["pr"], 0), "qmax": _r(fit["qmax"], 0),
                                     "n": fit["n"], "spread": _r(fit["pwf_spread"], 0),
                                     "rmse": _r(fit["rmse"], 1), "usable": fit["usable"],
                                     "why_not": fit["why_not"]}
        liquid = _r(t.get("liquid"), 0)
        bhp_test = _r(t.get("bhp"), 0)
        anchor_pwf = bhp_test or _r(bhp_now, 0)
        fit_anchor = ({"qwf": liquid, "pwf": anchor_pwf, "pres": ipr_fit_stats["pres"]}
                      if ipr_fit_stats and liquid and anchor_pwf else None)
        if fit_anchor and hm.ipr_valid(fit_anchor):
            fit_anchor = None
        # The default ladder uses only a usable fit; an engineer who has looked
        # at the test points can still pick (and save) the flagged one.
        ipr_fit = fit_anchor if ipr_fit_stats and ipr_fit_stats["usable"] else None
        looks_down = gauge_note is None and _looks_down(bhp_now, t.get("bhp"))
        # Each well's own reservoir pressure: the latest prop_hist value
        # (a Solver save, a header-page save or a bulk load), else the
        # documented default for its reservoir.
        sp = saved.get(well, {}).get("resvr_press") or {}
        pres_saved = sp.get("value") or _f(o.get("resvr_press"))
        age_ok = age is not None and age <= ONLINE_MAX_TEST_AGE_DAYS

        rows.append({
            "well": well,
            "pad": str(o["well_pad"]),
            "lift": lift,
            "reservoir": reservoir,
            "pump": (f"{pump['nozzle_no']}{pump['throat_ratio']}"
                     if lift == "JP" and pump and pump.get("nozzle_no") else None),
            "test_date": str(pd.Timestamp(wt_date).date()) if wt_date is not None and not pd.isna(wt_date) else None,
            "test_age_days": age,
            "age_ok": age_ok,
            "looks_down": looks_down,
            "online_default": age_ok and not looks_down,
            "down_note": _down_note(age, bhp_now if looks_down else None, t.get("bhp")),
            "oil": _r(t.get("oil", o.get("oil")), 0),
            "liquid": liquid,
            "wc": _r(t.get("wc"), 3),
            "gor": _r(t.get("gor"), 0),
            "whp_test": _r(t.get("whp", o.get("whp")), 0),
            "bhp_test": bhp_test,
            "whp_now": _r(whp_now, 0),
            "bhp_now": _r(bhp_now, 0),
            "header_now": _r(hdr_now, 0),
            "has_gauge": bhp_now is not None,
            "gauge_auto_bad": gauge_note is not None,
            "gauge_note": gauge_note,
            "resvr_press": _r(o.get("resvr_press"), 0),
            "pres_well": _r(pres_saved if pres_saved else hm.default_pres(reservoir), 0),
            "pres_basis": "saved" if pres_saved else "default",
            "pres_saved_at": sp.get("at"),
            "pres_saved_by": sp.get("by"),
            "measured": {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in rel.items()},
            "whp_hdr": {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in whp_hdr.items()},
            "ipr_fit_stats": ipr_fit_stats,
            "ipr_fit": ipr_fit,
            "ipr_fit_any": fit_anchor,
            "saved": _saved_relation(saved.get(well, {})),
            "saved_ipr": _saved_ipr(saved.get(well, {}), lift),
        })

    correlations = _correlations(rows)
    ipr_groups = _ipr_groups(rows)
    for g in ipr_groups.values():
        g["shut_in"] = _shut_in_readings(rows, g["reservoir"], g["pad"])
    for r in rows:
        _attach_options(r, correlations, ipr_groups)
    rows.sort(key=lambda r: (r["pad"], r["lift"], r["well"]))
    headers = {}
    for p in pads:
        vals = [r["header_now"] for r in rows if r["pad"] == p and r["header_now"] is not None]
        headers[p] = _r(float(np.median(vals)), 0) if vals else None
    board = {
        "pads": pads,
        "fit_days": int(fit_days),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "correlations": correlations,
        "ipr_groups": ipr_groups,
        "header_now": headers,
        "defaults": {"res_pres": hm.RES_DEFAULT_PRES, "online_max_test_age_days": ONLINE_MAX_TEST_AGE_DAYS},
    }
    with _cache_lock:
        _board_cache[key] = (time.monotonic(), board)
    return board


def _recent(s: pd.Series, hours: int = NOW_WINDOW_HOURS) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    idx = pd.DatetimeIndex(s.index)
    return s[idx >= idx.max() - pd.Timedelta(hours=hours)]


def _saved_relation(sv: dict[str, Any]) -> Optional[dict[str, Any]]:
    if sv.get(HDR_SLOPE, {}).get("value") is None:
        return None
    code = int(sv.get(HDR_REL_SOURCE, {}).get("value") or 0)
    return {
        "slope": sv[HDR_SLOPE]["value"],
        "whp_hdr": sv.get(HDR_WHP_HDR, {}).get("value"),
        "source": hm.REL_SOURCE_NAME.get(code, "measured"),
        "r2": sv.get(HDR_R2, {}).get("value"),
        "days": sv.get(HDR_DAYS, {}).get("value"),
        "at": sv[HDR_SLOPE]["at"], "by": sv[HDR_SLOPE]["by"],
    }


def _saved_ipr(sv: dict[str, Any], lift: str) -> Optional[dict[str, Any]]:
    if not all(sv.get(p, {}).get("value") is not None for p in IPR_PROP_IDS):
        return None
    code = int(sv.get(HDR_IPR_SOURCE, {}).get("value") or 0)
    return {
        "qwf": sv["ipr_qwf_liq"]["value"], "pwf": sv["ipr_pwf"]["value"],
        "pres": sv["resvr_press"]["value"],
        # No header-page source code: saved in Solver (JP) or by hand.
        "source": hm.IPR_SOURCE_NAME.get(code, "jp" if lift == "JP" else "manual"),
        "at": sv["ipr_pwf"]["at"], "by": sv["ipr_pwf"]["by"],
    }


def _looks_down(bhp_now: Optional[float], bhp_test: Optional[float]) -> bool:
    """True when the gauge has built far above the latest flowing test."""
    if bhp_now is None or bhp_test is None or bhp_test <= 0:
        return False
    return bhp_now > max(DOWN_BHP_RISE_FRAC * bhp_test, bhp_test + DOWN_BHP_RISE_PSI)


def _down_note(age: Optional[int], bhp_now: Optional[float], bhp_test: Optional[float]) -> Optional[str]:
    if age is None:
        return "no recent test"
    if age > ONLINE_MAX_TEST_AGE_DAYS:
        return f"latest test {age} days old"
    if _looks_down(bhp_now, bhp_test):
        return f"BHP now {bhp_now:.0f} psi vs {bhp_test:.0f} at test - looks shut in"
    return None


# ── correlations ─────────────────────────────────────────────────────────────


def _correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """BHP~WHP slope correlations per lift type, and per lift type + reservoir.

    Reservoir matters as much as rate: on F/L/R (2026-09-29) Schrader ESPs
    sat at 0.05-0.27 where Kuparuk ESPs read ~0.8, so a pooled line would hand
    gaugeless Schrader wells several times their peers' slope. A lift +
    reservoir group is formed once it has CORR_MIN_WELLS measured wells; the
    lift-only group stays as the fallback. Only wells with a plausible gauge
    and a measured relation contribute.
    """
    out: dict[str, Any] = {}
    for lift in hm.LIFT_GROUPS:
        members = [r for r in rows if r["lift"] == lift]
        if not members:
            continue
        out[lift] = _group_correlation(lift, None, members)
        for res in sorted({r["reservoir"] for r in members if r["reservoir"]}):
            sub = [r for r in members if r["reservoir"] == res]
            if sum(1 for r in sub if _measured_ok(r)) >= CORR_MIN_WELLS:
                out[f"{lift} {res}"] = _group_correlation(lift, res, sub)
    return out


def _measured_ok(r: dict[str, Any]) -> bool:
    return not r["gauge_auto_bad"] and r["measured"]["status"] == "measured"


def _group_correlation(lift: str, reservoir: Optional[str], members: list[dict[str, Any]]) -> dict[str, Any]:
    pts = [{"well": r["well"], "q_liq": r["liquid"], "slope": r["measured"]["slope"],
            "reservoir": r["reservoir"]}
           for r in members if _measured_ok(r)]
    return {
        "lift": lift,
        "reservoir": reservoir,
        "correlation": hm.fit_correlation(pts),
        "points": [{**p, "slope": _r(p["slope"], 3)} for p in pts],
        "n_wells": len(members),
    }


def _ratio_ok(r: dict[str, Any]) -> bool:
    """A flowing well whose plausible gauge can say where BHP sits vs its ResP."""
    return bool(r["bhp_now"]) and not r["gauge_auto_bad"] and r.get("age_ok") and not r.get("looks_down") \
        and r["pres_well"] and r["bhp_now"] < r["pres_well"]


def _ipr_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reservoir groups for wells WITHOUT a working gauge.

    Every well keeps its own reservoir pressure - the value saved in
    prop_hist, else the documented default (Schrader 1,800) - so a group never
    sets ResP. What a gaugeless well lacks is a flowing BHP; the group gives
    it the typical BHP/ResP of gauged, flowing wells in the same pad and
    reservoir (pad + reservoir once IPR_GROUP_MIN wells have one, else the
    reservoir). Shut-in gauge readings are listed as evidence only.
    """
    samples: dict[str, list[tuple[str, str, float]]] = {}
    for r in rows:
        if r["reservoir"] and _ratio_ok(r):
            samples.setdefault(r["reservoir"], []).append((r["well"], r["pad"], r["bhp_now"] / r["pres_well"]))
    all_ratios = [s[2] for v in samples.values() for s in v]
    out: dict[str, Any] = {}
    for res in sorted({r["reservoir"] for r in rows if r["reservoir"]}):
        members = [r for r in rows if r["reservoir"] == res]
        pts = samples.get(res, [])
        out[res] = _ipr_group(res, None, members, pts, all_ratios)
        for pad in sorted({p for _, p, _ in pts}):
            sub = [s for s in pts if s[1] == pad]
            if len(sub) >= IPR_GROUP_MIN:
                out[f"{pad} {res}"] = _ipr_group(res, pad, [r for r in members if r["pad"] == pad], sub, all_ratios)
    return out


def _ipr_group(res: str, pad: Optional[str], members: list[dict[str, Any]],
               pts: list[tuple[str, str, float]], all_ratios: list[float]) -> dict[str, Any]:
    ratio = hm.spread([p[2] for p in pts])
    note = None
    if ratio is None:
        med = float(np.median(all_ratios)) if all_ratios else DEFAULT_BHP_RATIO
        ratio = {"med": med, "q25": med, "q75": med, "n": 0}
        note = (f"no gauged, flowing {res} well to take a BHP/ResP from; using "
                f"{'the field median' if all_ratios else 'a default'} {med:.2f}")
    return {
        "reservoir": res, "pad": pad,
        "default_pres": hm.default_pres(res),
        "ratio": ratio,
        "wells": [p[0] for p in pts],
        "n_saved_pres": sum(1 for r in members if r.get("pres_basis") == "saved"),
        "n_members": len(members),
        "note": note,
    }


def _shut_in_readings(rows: list[dict[str, Any]], res: str, pad: Optional[str] = None) -> list[dict[str, Any]]:
    """Plausible gauges on wells that look shut in or have no recent test.

    A shut-in gauge builds toward reservoir pressure, so these are lower
    bounds on ResP near that well - evidence to weigh when saving a well's
    ResP, never an input.
    """
    out = []
    for r in rows:
        if r["reservoir"] != res or (pad and r["pad"] != pad) or not r["bhp_now"] or r["gauge_auto_bad"]:
            continue
        if r.get("looks_down") or not r.get("age_ok"):
            out.append({"well": r["well"], "bhp": r["bhp_now"]})
    return sorted(out, key=lambda x: -x["bhp"])


def _attach_options(r: dict[str, Any], correlations: dict[str, Any], ipr_groups: dict[str, Any]) -> None:
    """Every BHP~WHP correlation and BHP-ratio group this well could borrow.

    The page picks among these without a round trip; run and save resolve the
    same numbers through :func:`effective`. The IPR option always uses the
    well's own ResP (saved, else default +/-20%) and its test rate; the group
    only supplies BHP when the well has no working gauge.
    """
    r["corr_options"], r["ipr_options"] = {}, {}
    r["corr_group"] = r["ipr_group"] = None
    if r["lift"] == "JP":
        return
    for key, g in correlations.items():
        corr = g.get("correlation")
        s = hm.predict_slope(corr, r["liquid"])
        if s is None:
            continue
        mad = corr["resid_mad"] if corr else 0.0
        r["corr_options"][key] = {"slope": _r(s, 3), "lo": _r(hm.clip_slope(s - mad), 3),
                                  "hi": _r(hm.clip_slope(s + mad), 3), "same_lift": g["lift"] == r["lift"]}
    own = f"{r['lift']} {r['reservoir']}"
    r["corr_group"] = own if own in r["corr_options"] else (r["lift"] if r["lift"] in r["corr_options"] else None)
    pres = r["pres_well"]
    band = 0.0 if r["pres_basis"] == "saved" else DEFAULT_PRES_BAND
    for key, g in ipr_groups.items():
        opt: dict[str, Any] = {"same_reservoir": g["reservoir"] == r["reservoir"]}
        for variant, bhp in (("gauge", r["bhp_now"]), ("nogauge", None)):
            pwf = bhp if variant == "gauge" else (g["ratio"]["med"] * pres if pres else None)
            if not (r["liquid"] and pwf and pres):
                opt[variant] = None
                continue
            opt[variant] = {"qwf": _r(r["liquid"], 0), "pwf": _r(pwf, 0), "pres": _r(pres, 0),
                            "pres_lo": _r(pres * (1 - band), 0), "pres_hi": _r(pres * (1 + band), 0),
                            "pres_basis": r["pres_basis"]}
        r["ipr_options"][key] = opt
    pad_key = f"{r['pad']} {r['reservoir']}"
    r["ipr_group"] = pad_key if pad_key in ipr_groups else (r["reservoir"] if r["reservoir"] in ipr_groups else None)


# ── choices ──────────────────────────────────────────────────────────────────


def _ch(ch: Any, name: str) -> Any:
    return getattr(ch, name, None) if ch is not None else None


def effective(r: dict[str, Any], ch: Any = None) -> dict[str, Any]:
    """The relation, IPR and gauge state a well runs with, given a choice.

    One resolver for the run, the save and (mirrored in web/src) the table.
    Defaults: relation saved > measured (working gauge) > correlation;
    IPR saved > gauge-test fit (working gauge) > reservoir correlation.
    """
    gb = _ch(ch, "gauge_bad")
    gauge_bad = bool(gb) if gb is not None else bool(r.get("gauge_auto_bad"))
    gauge_ok = bool(r.get("has_gauge")) and not gauge_bad
    online = _ch(ch, "online")
    if online is None:
        online = bool(r.get("age_ok")) and (not r.get("looks_down") or gauge_bad) \
            if "age_ok" in r else bool(r.get("online_default"))
    corr_key = _ch(ch, "corr_group")
    corr_key = corr_key if corr_key in (r.get("corr_options") or {}) else r.get("corr_group")
    ipr_key = _ch(ch, "ipr_group")
    ipr_key = ipr_key if ipr_key in (r.get("ipr_options") or {}) else r.get("ipr_group")
    rel = _resolve_relation(r, _ch(ch, "relation"), _ch(ch, "slope"), gauge_ok, corr_key)
    ipr_choice = _ch(ch, "ipr")
    manual = ({k: _ch(ch, k) for k in ("qwf", "pwf", "pres")} if ipr_choice == "manual" else None)
    ipr = _resolve_ipr(r, ipr_choice, manual, gauge_ok, ipr_key)
    return {"gauge_ok": gauge_ok, "gauge_bad": gauge_bad, "online": bool(online),
            "bhp_now": r.get("bhp_now") if gauge_ok else None, "rel": rel, "ipr": ipr}


def _resolve_relation(r: dict[str, Any], choice: Optional[str], manual: Optional[float],
                      gauge_ok: bool, corr_key: Optional[str]) -> dict[str, Any]:
    """dBHP/dWHP with its range and source for a non-JP well."""
    m = r["measured"]
    choice = choice or "auto"
    if choice == "auto":
        if r.get("saved"):
            choice = "saved"
        elif gauge_ok and m["status"] == "measured":
            choice = "measured"
        elif corr_key:
            choice = "correlation"
        else:
            choice = "none"
    wh = r["whp_hdr"]
    whp_hdr = (hm.clip_slope(wh["slope"]) if wh.get("status") == "measured" else None) or 1.0
    out: dict[str, Any] = {"slope": None, "lo": None, "hi": None, "whp_hdr": whp_hdr,
                           "source": choice, "saved": False, "group": None}
    if choice == "saved" and r.get("saved"):
        s = hm.clip_slope(r["saved"]["slope"])
        lo = hi = s
        if gauge_ok and m["slope"] is not None and r["saved"]["source"] == "measured":
            lo, hi = min(s, hm.clip_slope(m["q25"]) or s), max(s, hm.clip_slope(m["q75"]) or s)
        saved_wh = hm.clip_slope(r["saved"].get("whp_hdr"))
        out.update(slope=s, lo=lo, hi=hi, source=r["saved"]["source"], saved=True,
                   whp_hdr=saved_wh or whp_hdr)
    elif choice == "measured" and m["slope"] is not None:
        if not gauge_ok:
            out["reason"] = "the BHP gauge is marked bad, so its measured relation is not used"
            return out
        s = hm.clip_slope(m["slope"])
        out.update(slope=s, lo=min(s, hm.clip_slope(m["q25"]) or s), hi=max(s, hm.clip_slope(m["q75"]) or s),
                   source="measured" if m["status"] == "measured" else "weak_measured")
    elif choice == "correlation" and corr_key and corr_key in r.get("corr_options", {}):
        o = r["corr_options"][corr_key]
        out.update(slope=o["slope"], lo=o["lo"], hi=o["hi"], source="correlation", group=corr_key)
    elif choice == "manual" and manual is not None and 0.0 <= manual <= 1.5:
        out.update(slope=float(manual), lo=float(manual), hi=float(manual), source="manual")
    else:
        out["reason"] = f"no {choice} relation for this well"
    return out


def _resolve_ipr(r: dict[str, Any], choice: Optional[str], manual: Optional[dict[str, Any]],
                 gauge_ok: bool, ipr_key: Optional[str]) -> dict[str, Any]:
    """Vogel anchor with its ResP range and source for a non-JP well."""
    choice = choice or "auto"
    if choice == "assumed":  # pre-2026-09-30 name for the reservoir correlation
        choice = "correlation"
    if choice == "auto":
        # Each well keeps its own ResP: a saved IPR, else its saved ResP (or
        # the default) at today's rate and BHP. A gauge-test fit backs out a
        # pseudo-ResP, so it is used only when chosen (and saving it sets
        # that well's ResP).
        if r.get("saved_ipr"):
            choice = "saved"
        elif ipr_key:
            choice = "correlation"
        else:
            choice = "none"
    ipr, lo, hi, source, saved, group = None, None, None, choice, False, None
    if choice == "saved" and r.get("saved_ipr"):
        s = r["saved_ipr"]
        ipr, source, saved = {k: s[k] for k in ("qwf", "pwf", "pres")}, s["source"], True
    elif choice == "fit" and gauge_ok and (r.get("ipr_fit") or r.get("ipr_fit_any")):
        # Explicitly chosen: a flagged fit is allowed (the engineer reviewed it).
        ipr = dict(r.get("ipr_fit") or r["ipr_fit_any"])
    elif choice == "correlation" and ipr_key:
        opt = (r.get("ipr_options") or {}).get(ipr_key) or {}
        v = opt.get("gauge" if gauge_ok else "nogauge")
        if v:
            ipr = {k: v[k] for k in ("qwf", "pwf", "pres")}
            lo, hi, group = v["pres_lo"], v["pres_hi"], ipr_key
    elif choice == "manual" and manual:
        ipr = {k: _f(manual.get(k)) for k in ("qwf", "pwf", "pres")}
    reason = hm.ipr_valid(ipr)
    if ipr is not None and reason is None:
        lo = lo if lo is not None else float(ipr["pres"])
        hi = hi if hi is not None else float(ipr["pres"])
    return {"ipr": ipr if reason is None else None, "pres_lo": lo, "pres_hi": hi,
            "source": source, "saved": saved, "group": group, "reason": reason}


# Legacy single-choice helpers (tests and callers that pass a bare choice).
def resolve_relation(row: dict[str, Any], choice: Optional[str], manual: Optional[float]) -> dict[str, Any]:
    gauge_ok = bool(row.get("has_gauge", True)) and not row.get("gauge_auto_bad")
    return _resolve_relation(row, choice, manual, gauge_ok, row.get("corr_group"))


def resolve_ipr(row: dict[str, Any], choice: Optional[str], manual: Optional[dict[str, Any]]) -> dict[str, Any]:
    gauge_ok = bool(row.get("has_gauge", True)) and not row.get("gauge_auto_bad")
    return _resolve_ipr(row, choice, manual, gauge_ok, row.get("ipr_group"))


# ── impact run ───────────────────────────────────────────────────────────────


def _pad_deltas(req: Any, board: dict[str, Any], progress: ProgressFn) -> tuple[dict[str, float], Optional[dict]]:
    """Header change per pad and, for event runs, the measured windows."""
    if req.mode == "scenario":
        deltas = {p: float(req.delta_by_pad.get(p, 0.0)) for p in board["pads"]}
        return deltas, None
    if not req.event_time:
        raise ValueError("an observed-event run needs an event time")
    win = hm.event_windows(req.event_time, req.pre_hours, req.post_hours, req.gap_hours)
    start = win["pre_start"].date() - timedelta(days=1)
    end = min(win["post_end"].date() + timedelta(days=1), date.today() + timedelta(days=1))
    wells = [r["well"] for r in board["rows"]]
    progress("reading the event windows from the historian")
    trends = _trends(wells, start, end)
    deltas: dict[str, float] = {}
    pad_meas: dict[str, Any] = {}
    for p in board["pads"]:
        series = [trends[r["well"]]["HeaderP"] for r in board["rows"]
                  if r["pad"] == p and r["well"] in trends and "HeaderP" in trends[r["well"]]]
        m = hm.window_delta(series[0] if series else None, win)
        pad_meas[p] = {k: _r(v, 1) if isinstance(v, float) else v for k, v in m.items()}
        if m["delta"] is not None:
            deltas[p] = float(m["delta"])
    per_well = {}
    for r in board["rows"]:
        df = trends.get(r["well"])
        if df is None:
            continue
        per_well[r["well"]] = {
            "whp": hm.window_delta(df["WHP"] if "WHP" in df else None, win),
            "bhp": hm.window_delta(df["BHP"] if "BHP" in df else None, win),
        }
    event = {
        "time": req.event_time,
        "windows": {k: v.isoformat(timespec="minutes") for k, v in win.items()},
        "pads": pad_meas,
        "wells": per_well,
        "pops": _pop_candidates(board["pads"], req.event_time),
    }
    return deltas, event


def _pop_candidates(pads: list[str], event_time: str) -> list[dict[str, Any]]:
    """Wells the daily downtime log shows coming back on, or going down, near an event.

    A suggestion list, not an attribution: the log is daily, and a header
    step can have other causes (pigging, facility changes). Fail-soft.
    """
    try:
        from woffl.assembly.databricks_client import execute_query
        from woffl.assembly.well_test_client import _normalize_well_name

        ev = pd.Timestamp(event_time).normalize()
        start = (ev - pd.Timedelta(days=POP_LOOKBACK_DAYS)).date().isoformat()
        end = (ev + pd.Timedelta(days=POP_AFTER_DAYS + 1)).date().isoformat()
        pad_list = ",".join(f"'{p}'" for p in clean_pads(pads))
        df = execute_query(
            f"""
            SELECT h.well_name, h.well_pad, s.dtdate, SUM(CAST(s.down_hours AS DOUBLE)) AS hrs
            FROM mpu.wells.vw_shut_in s
            JOIN mpu.wells.vw_well_header h ON s.dthid = h.enthid
            WHERE h.well_pad IN ({pad_list}) AND h.well_type = 'prod'
              AND s.dtdate BETWEEN '{start}' AND '{end}'
            GROUP BY h.well_name, h.well_pad, s.dtdate
            """
        )
    except Exception:  # noqa: BLE001 - suggestions only
        log.warning("header: downtime log unavailable", exc_info=True)
        return []
    if df is None or df.empty:
        return []
    df["dtdate"] = pd.to_datetime(df["dtdate"]).dt.normalize()
    df["hrs"] = pd.to_numeric(df["hrs"], errors="coerce").fillna(0.0)
    # A POP is the first partial day after a fully-down day; a trip is the
    # first logged-down day after an up day. Either within a day or two of
    # the event: the log is daily, and a JP's power fluid reaches the header
    # before the log calls the well up (MPL-20: PF on 2026-09-28 ~12:00, log
    # partial on 09-29).
    days = [ev + pd.Timedelta(days=k) for k in range(-1, POP_AFTER_DAYS + 1)]
    out = []
    for (name, pad), g in df.groupby(["well_name", "well_pad"]):
        down = {d: h for d, h in zip(g["dtdate"], g["hrs"])}
        well = _normalize_well_name(str(name).strip())
        for d in days:
            prev, cur = down.get(d - pd.Timedelta(days=1), 0.0), down.get(d, 0.0)
            if prev >= POP_FULL_DAY_HOURS and cur < POP_FULL_DAY_HOURS:
                out.append({"well": well, "pad": str(pad), "kind": "came on", "day": str(d.date()),
                            "note": f"down all of {(d - pd.Timedelta(days=1)).date()}, "
                                    f"{24 - cur:.1f} h up on {d.date()}"})
                break
            if prev < POP_UP_HOURS and cur >= POP_DOWN_HOURS:
                out.append({"well": well, "pad": str(pad), "kind": "went down", "day": str(d.date()),
                            "note": f"up on {(d - pd.Timedelta(days=1)).date()}, {cur:.1f} h down on {d.date()}"})
                break
    return sorted(out, key=lambda x: (x["kind"], x["day"], x["well"]))


def _jp_solve(pads: list[str], jp_rows: list[dict[str, Any]], scenarios: dict[Any, dict[str, float]],
              progress: ProgressFn) -> tuple[dict[str, dict[str, Any]], dict[Any, dict[str, float]], list[str]]:
    """Installed jet pumps at the model WHP and at each scenario's WHP, PF held.

    ``scenarios`` maps a key to {pad: header change}. Returns per-well detail
    for the key "run", oil changes per well for every key, and notes.
    """
    from server.services.optimizer_runs import _build_configs
    from woffl.gui.pad_optimize import _model_at_forced_header

    notes: list[str] = []
    detail: dict[str, dict[str, Any]] = {}
    oil: dict[Any, dict[str, float]] = {k: {} for k in scenarios}
    if not jp_rows:
        return detail, oil, notes
    progress(f"solving {len(jp_rows)} jet pumps at {len(scenarios) + 1} wellhead pressures")
    prov: dict[str, dict[str, Any]] = {}
    configs = _build_configs(pads, set(), [], notes, prov=prov, only={r["well"] for r in jp_rows})
    by_name = {c.well_name: c for c in configs}
    runnable = [c for c in configs if c.installed_nozzle and c.installed_throat and c.ppf_surf_well]
    for r in jp_rows:
        if r["well"] not in by_name:
            detail[r["well"]] = {"error": "jet-pump inputs could not be loaded"}
        elif by_name[r["well"]] not in runnable:
            detail[r["well"]] = {"error": "installed pump or PF pressure unknown"}
    if not runnable:
        return detail, oil, notes
    rows_by = {r["well"]: r for r in jp_rows}
    ratio = {}
    for c in runnable:
        wh = rows_by[c.well_name]["whp_hdr"]
        ratio[c.well_name] = (hm.clip_slope(wh["slope"]) if wh["status"] == "measured" else None) or 1.0
    current = {c.well_name: (c.installed_nozzle, c.installed_throat) for c in runnable}
    pf = {c.well_name: float(c.ppf_surf_well) for c in runnable}
    base = _model_at_forced_header([copy(c) for c in runnable], pf, current)
    scen_out: dict[Any, dict] = {}
    for key, deltas in scenarios.items():
        cfgs = []
        for c in runnable:
            cc = copy(c)
            cc.surf_pres = float(c.surf_pres) + ratio[c.well_name] * deltas.get(c.pad or rows_by[c.well_name]["pad"], 0.0)
            cfgs.append(cc)
        scen_out[key] = _model_at_forced_header(cfgs, pf, current)
        for c in runnable:
            b, s = base.get(c.well_name), scen_out[key].get(c.well_name)
            if b is not None and s is not None:
                oil[key][c.well_name] = s[0] - b[0]
    run = scen_out.get("run", {})
    for c in runnable:
        b, s = base.get(c.well_name), run.get(c.well_name)
        p = prov.get(c.well_name, {})
        if b is None or s is None:
            detail[c.well_name] = {"error": "jet-pump model did not solve at one of the pressures",
                                   "ipr": {"qwf": c.qwf, "pwf": c.pwf, "pres": c.res_pres}}
            continue
        wc = float(c.form_wc)
        d_oil = s[0] - b[0]
        d_whp = ratio[c.well_name] * scenarios["run"].get(rows_by[c.well_name]["pad"], 0.0)
        detail[c.well_name] = {
            "whp_model": _r(c.surf_pres, 0), "pf": _r(c.ppf_surf_well, 0), "d_whp": d_whp,
            "oil_base": b[0], "oil_scen": s[0], "d_oil": d_oil,
            "d_liq": d_oil / (1.0 - wc) if wc < 1.0 else None,
            "d_bhp": (s[2] - b[2]) if b[2] is not None and s[2] is not None else None,
            "sonic": bool(b[3]), "wc": wc,
            "ipr": {"qwf": c.qwf, "pwf": c.pwf, "pres": c.res_pres},
            "ipr_source": p.get("ipr_source"),
            "pump_calibration": (p.get("pump_calibration") or {}).get("status"),
            "hydraulics": p.get("hydraulics_model"),
        }
    return detail, oil, notes


def run_impact(req: Any, progress: ProgressFn = _noop) -> dict[str, Any]:
    """Per-well and per-pad response to the requested header change, with a
    low/high range and the pad response curve."""
    board = build_board(req.pads, req.fit_days, progress)
    deltas, event = _pad_deltas(req, board, progress)
    choices = {c.well: c for c in req.wells}
    event_well = (req.event_well or "").strip().upper() or None

    rows: list[dict[str, Any]] = []
    jp_rows: list[dict[str, Any]] = []
    eff_by: dict[str, dict[str, Any]] = {}
    for r in board["rows"]:
        ch = choices.get(r["well"])
        eff = effective(r, ch)
        eff_by[r["well"]] = eff
        online = eff["online"]
        is_event = bool(event_well and r["well"].upper() == event_well)
        if is_event:
            online = False
        out = {"well": r["well"], "pad": r["pad"], "lift": r["lift"], "reservoir": r["reservoir"],
               "online": bool(online), "d_header": deltas.get(r["pad"]), "oil": r["oil"],
               "liquid": r["liquid"], "wc": r["wc"], "whp_now": r["whp_now"], "bhp_now": eff["bhp_now"],
               "gauge_bad": eff["gauge_bad"], "pump": r["pump"], "measured_slope": r["measured"]["slope"],
               "outcome": "event_well" if is_event else "offline"}
        rows.append(out)
        if not online:
            continue
        if out["d_header"] is None:
            out.update(outcome="missing_inputs", reason=f"no header change for pad {r['pad']}")
            continue
        if r["lift"] == "JP":
            jp_rows.append(r)
            continue
        _nonjp_row(out, r, eff)

    scenarios: dict[Any, dict[str, float]] = {"run": deltas}
    for dh in CURVE_JP_GRID:
        scenarios[dh] = {p: float(dh) for p in board["pads"]}
    jp, jp_oil, notes = _jp_solve(board["pads"], jp_rows, scenarios, progress)
    by_well = {o["well"]: o for o in rows}
    for r in jp_rows:
        out = by_well[r["well"]]
        res = jp.get(r["well"]) or {"error": "not solved"}
        if "error" in res:
            _jp_fallback(out, r, eff_by[r["well"]], res)
            continue
        d_oil = _r(res["d_oil"], 1)
        out.update(
            outcome="modeled", relation_source="physics", ipr_source="jp",
            relation_saved=False, ipr_saved=res.get("ipr_source") == "saved",
            slope=None, whp_hdr=None, d_whp=_r(res["d_whp"], 1), d_bhp=_r(res["d_bhp"], 1),
            d_liq=_r(res["d_liq"], 1), d_oil=d_oil, d_oil_lo=d_oil, d_oil_hi=d_oil, sonic=res["sonic"],
            model_oil=_r(res["oil_base"], 0), pf=res["pf"], whp_model=res["whp_model"],
            ipr=res["ipr"], jp_ipr_source=res.get("ipr_source"),
            pump_calibration=res.get("pump_calibration"), hydraulics=res.get("hydraulics"),
        )

    validation = _validation(rows, event) if event else None
    curve = _curve(board, rows, eff_by, jp_oil)
    pads_out = []
    for p in board["pads"]:
        pr = [o for o in rows if o["pad"] == p and o["online"]]
        mod = [o for o in pr if o["outcome"] == "modeled"]
        d_oil = sum(o["d_oil"] or 0.0 for o in mod)
        dh = deltas.get(p)
        pads_out.append({
            "pad": p, "d_header": _r(dh, 1), "online": len(pr), "modeled": len(mod),
            "d_oil": _r(d_oil, 1), "d_liq": _r(sum(o["d_liq"] or 0.0 for o in mod), 1),
            "d_oil_lo": _r(sum(o.get("d_oil_lo") or 0.0 for o in mod), 1),
            "d_oil_hi": _r(sum(o.get("d_oil_hi") or 0.0 for o in mod), 1),
            "oil_per_10psi": _r(d_oil / dh * 10.0, 1) if dh else None,
            "by_lift": {lift: _r(sum(o["d_oil"] or 0.0 for o in mod if o["lift"] == lift), 1)
                        for lift in sorted({o["lift"] for o in mod})},
        })
    modeled = [o for o in rows if o["online"] and o["outcome"] == "modeled"]
    total_oil = sum(o["d_oil"] or 0.0 for o in modeled)
    status = hm.run_status(rows)
    return {
        "mode": req.mode,
        "pads": pads_out,
        "rows": rows,
        "totals": {
            "d_oil": _r(total_oil, 1),
            "d_oil_lo": _r(sum(o.get("d_oil_lo") or 0.0 for o in modeled), 1),
            "d_oil_hi": _r(sum(o.get("d_oil_hi") or 0.0 for o in modeled), 1),
            "d_liq": _r(sum(o["d_liq"] or 0.0 for o in modeled), 1),
            "modeled": len(modeled),
            "online": status["online"],
            "event_well_oil": _r(req.event_well_oil, 0) if req.event_well_oil is not None else None,
            "net_oil": _r((req.event_well_oil or 0.0) + total_oil, 1) if req.event_well_oil is not None else None,
        },
        "curve": curve,
        "status": status,
        "event": event,
        "event_well": event_well,
        "validation": validation,
        "notes": notes,
        "board_built_at": board["built_at"],
        "board": board,
        "assumptions": (
            "Header change passes to WHP by each well's measured WHP~Header slope (1.0 when not "
            "measured). Non-JP wells: dBHP = closed-loop BHP~WHP slope x dWHP (already includes the "
            "rate response); liquid change from the Vogel IPR at current BHP; oil = liquid x (1 - WC) "
            "at the latest test. Jet pumps: installed pump solved at the model WHP and WHP + dWHP "
            "with saved inputs and PF held. The range pairs each well's slope spread (measured IQR "
            "or correlation scatter) with its reservoir-correlation ResP spread, all wells at the "
            "same end together - an envelope, not a confidence interval."
        ),
    }


def _nonjp_row(out: dict[str, Any], r: dict[str, Any], eff: dict[str, Any]) -> None:
    """Fill ``out`` from the relation/IPR chain, with its low/high range."""
    rel, ip = eff["rel"], eff["ipr"]
    out.update(relation_source=rel["source"], relation_saved=rel["saved"], relation_group=rel["group"],
               ipr_source=ip["source"], ipr_saved=ip["saved"], ipr_group=ip["group"],
               slope=rel["slope"], whp_hdr=rel["whp_hdr"], ipr=ip["ipr"])
    reason = None
    if rel["slope"] is None:
        reason = rel.get("reason") or "no relation"
    elif ip["ipr"] is None:
        reason = ip["reason"] or "no IPR"
    elif (eff["bhp_now"] or 0.0) >= 0.97 * float(ip["ipr"]["pres"]):
        reason = (f"current BHP {eff['bhp_now']:.0f} psi is at the IPR reservoir pressure "
                  f"{float(ip['ipr']['pres']):.0f} - the well looks shut in")
    if reason:
        out.update(outcome="missing_inputs", reason=reason)
        return
    bhp = eff["bhp_now"] or ip["ipr"]["pwf"]
    wc = r["wc"] if r["wc"] is not None else 0.0
    d = hm.nonjp_delta(out["d_header"], rel["whp_hdr"], rel["slope"], ip["ipr"], bhp, wc)
    lo, hi = hm.range_delta(out["d_header"], rel["whp_hdr"], ip["ipr"], bhp, wc,
                            rel["lo"], rel["hi"], ip["pres_lo"], ip["pres_hi"])
    out.update(outcome="modeled", d_whp=_r(d["d_whp"], 1), d_bhp=_r(d["d_bhp"], 1),
               d_liq=_r(d["d_liq"], 1), d_oil=_r(d["d_oil"], 1), d_oil_lo=_r(lo, 1), d_oil_hi=_r(hi, 1),
               pi=_r(d["pi"], 2), liq_model=_r(d["liq_now"], 0), bhp_used=_r(bhp, 0),
               bhp_basis="gauge" if eff["bhp_now"] else "IPR anchor")


def _jp_fallback(out: dict[str, Any], r: dict[str, Any], eff: dict[str, Any], res: dict[str, Any]) -> None:
    """A jet pump whose model failed: its own MEASURED relation on the pump
    model's IPR, never a weak slope or an ESP correlation."""
    err = res["error"]
    m = r["measured"]
    if not (eff["gauge_ok"] and m["status"] == "measured"):
        out.update(outcome="missing_inputs", relation_source="none", ipr_source="jp",
                   reason=f"jet-pump model: {err}; no measured BHP~WHP relation to fall back on")
        return
    s = hm.clip_slope(m["slope"])
    rel = {"slope": s, "lo": min(s, hm.clip_slope(m["q25"]) or s), "hi": max(s, hm.clip_slope(m["q75"]) or s),
           "whp_hdr": eff["rel"]["whp_hdr"], "source": "measured", "saved": False, "group": None}
    ipr = res.get("ipr")
    if hm.ipr_valid(ipr):
        out.update(outcome="missing_inputs", relation_source="measured", ipr_source="jp",
                   reason=f"jet-pump model: {err}; its IPR is not usable either")
        return
    ip = {"ipr": ipr, "pres_lo": ipr["pres"], "pres_hi": ipr["pres"], "source": "jp", "saved": False,
          "group": None, "reason": None}
    _nonjp_row(out, r, {**eff, "rel": rel, "ipr": ip})
    if out["outcome"] == "modeled":
        out["note"] = f"jet-pump model: {err}; measured relation used"


def _curve(board: dict[str, Any], rows: list[dict[str, Any]], eff_by: dict[str, dict[str, Any]],
           jp_oil: dict[Any, dict[str, float]]) -> dict[str, Any]:
    """Oil change vs a uniform header change on every selected pad.

    Non-JP wells are exact on the grid (arithmetic). Jet pumps are solved on
    CURVE_JP_GRID and interpolated linearly, through zero at zero.
    """
    grid = list(CURVE_GRID)
    by_row = {r["well"]: r for r in board["rows"]}
    pads = {p: {"base": [0.0] * len(grid), "lo": [0.0] * len(grid), "hi": [0.0] * len(grid)} for p in board["pads"]}
    jp_x = [0.0] + [float(k) for k in CURVE_JP_GRID]
    order = np.argsort(jp_x)
    for o in rows:
        if not o["online"] or o["outcome"] != "modeled":
            continue
        pc = pads[o["pad"]]
        if o["relation_source"] == "physics":
            ys = [0.0] + [jp_oil.get(k, {}).get(o["well"], np.nan) for k in CURVE_JP_GRID]
            xs, ys = np.asarray(jp_x)[order], np.asarray(ys, dtype=float)[order]
            ok = np.isfinite(ys)
            vals = np.interp(grid, xs[ok], ys[ok]) if ok.sum() >= 2 else np.zeros(len(grid))
            for i, v in enumerate(vals):
                pc["base"][i] += float(v)
                pc["lo"][i] += float(v)
                pc["hi"][i] += float(v)
            continue
        r = by_row[o["well"]]
        eff = eff_by[o["well"]]
        rel, ip = eff["rel"], eff["ipr"]
        if o.get("note"):  # JP fallback on its measured relation
            m = r["measured"]
            s = hm.clip_slope(m["slope"])
            rel = {**rel, "slope": s, "lo": s, "hi": s}
            ip = {**ip, "ipr": o["ipr"], "pres_lo": o["ipr"]["pres"], "pres_hi": o["ipr"]["pres"]}
        bhp = o["bhp_used"]
        wc = r["wc"] if r["wc"] is not None else 0.0
        for i, dh in enumerate(grid):
            if dh == 0:
                continue
            d = hm.nonjp_delta(dh, rel["whp_hdr"], rel["slope"], ip["ipr"], bhp, wc)["d_oil"]
            lo, hi = hm.range_delta(dh, rel["whp_hdr"], ip["ipr"], bhp, wc,
                                    rel["lo"], rel["hi"], ip["pres_lo"], ip["pres_hi"])
            pc["base"][i] += d
            pc["lo"][i] += lo
            pc["hi"][i] += hi
    total = {k: [sum(pads[p][k][i] for p in pads) for i in range(len(grid))] for k in ("base", "lo", "hi")}
    rnd = lambda xs: [_r(x, 1) for x in xs]  # noqa: E731
    return {
        "grid": grid,
        "pads": {p: {k: rnd(v) for k, v in d.items()} for p, d in pads.items()},
        "total": {k: rnd(v) for k, v in total.items()},
    }


def _validation(rows: list[dict[str, Any]], event: dict[str, Any]) -> dict[str, Any]:
    """Predicted vs measured dBHP on gauged wells around the event."""
    out = []
    for o in rows:
        if not o["online"] or o["outcome"] != "modeled" or o.get("gauge_bad"):
            continue
        m = event["wells"].get(o["well"]) or {}
        mb, mw = (m.get("bhp") or {}).get("delta"), (m.get("whp") or {}).get("delta")
        if mb is None:
            continue
        # The relation alone, driven by the MEASURED WHP change: isolates the
        # BHP~WHP link from the header->WHP transfer.
        rel_pred = (o["slope"] * mw) if (o.get("slope") is not None and mw is not None) else None
        pred = o.get("d_bhp")
        err = (pred - mb) if pred is not None else None
        # A BHP move far beyond anything the header could cause is a well
        # event inside the window (restart, trip, speed change), not a test
        # of the relation. Shown, flagged, and left out of the statistics.
        limit = max(OPERATIONAL_ERR_PSI, 3.0 * abs(o.get("d_header") or 0.0), 3.0 * abs(pred or 0.0))
        operational = err is not None and abs(err) > limit
        out.append({"well": o["well"], "pad": o["pad"], "lift": o["lift"],
                    "d_bhp_pred": pred, "d_bhp_meas": _r(mb, 1),
                    "d_whp_pred": o.get("d_whp"), "d_whp_meas": _r(mw, 1),
                    "d_bhp_from_meas_whp": _r(rel_pred, 1), "error": _r(err, 1),
                    "operational": bool(operational)})
    errs = [v["error"] for v in out if v["error"] is not None and not v["operational"]]
    return {
        "rows": out,
        "n": len(out),
        "n_used": len(errs),
        "n_operational": sum(1 for v in out if v["operational"]),
        "bias": _r(float(np.mean(errs)), 1) if errs else None,
        "mae": _r(float(np.mean(np.abs(errs))), 1) if errs else None,
        "median_abs": _r(float(np.median(np.abs(errs))), 1) if errs else None,
        "within_5": sum(1 for e in errs if abs(e) <= 5.0),
        "operational_err_psi": OPERATIONAL_ERR_PSI,
    }


# ── jobs ─────────────────────────────────────────────────────────────────────

JOB_KINDS = ("header_board", "header_run")


def start_board(pads: list[str], fit_days: int) -> str:
    pads = clean_pads(pads)

    def runner(job: dict[str, Any]) -> dict[str, Any]:
        from server.services.optimizer_runs import _plain
        return _plain(build_board(pads, fit_days, lambda t: jobs.set_progress(job, t)))

    return jobs.start("header_board", runner, "queued: header board")


def start_run(req: Any) -> str:
    req.pads = clean_pads(req.pads)

    def runner(job: dict[str, Any]) -> dict[str, Any]:
        from server.services.optimizer_runs import _plain
        return _plain(run_impact(req, lambda t: jobs.set_progress(job, t)))

    return jobs.start("header_run", runner, "queued: header impact")


# ── save ─────────────────────────────────────────────────────────────────────


def _board_from_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id, JOB_KINDS)
    if job is None:
        raise ValueError("board job expired or unknown - reload the wells and save again")
    if job["status"] != "done" or not job.get("result"):
        raise ValueError("board job has not finished")
    res = job["result"]
    return res if job["kind"] == "header_board" else res["board"]


def save(req: Any) -> dict[str, Any]:
    """Persist chosen relations/IPRs for wells on a completed board or run job.

    One ``push_props`` statement per well (all-or-nothing per well). Numbers
    for measured/correlation/fit choices come from that job on the server,
    resolved with the same :func:`effective` the run uses; only ``manual``
    choices carry client values, validated here. Jet-pump IPRs are refused:
    they are saved in Solver with the pump model.
    """
    from woffl.assembly import prop_hist_client

    board_rows = {r["well"]: r for r in _board_from_job(req.board_job_id)["rows"]}
    user = prop_hist_client.resolve_entry_user()
    results = []
    for w in req.wells:
        r = board_rows.get(w.well)
        if r is None:
            results.append({"well": w.well, "saved": 0, "error": "well is not on this board"})
            continue
        try:
            values = _save_values(r, w)
        except ValueError as exc:
            results.append({"well": w.well, "saved": 0, "error": str(exc)})
            continue
        if not values:
            results.append({"well": w.well, "saved": 0, "error": "nothing selected to save"})
            continue
        try:
            n = prop_hist_client.push_props(w.well, values, entry_user=user)
            results.append({"well": w.well, "saved": n, "error": None, "values": values})
        except Exception as exc:  # noqa: BLE001 - per-well failure is reported, others proceed
            results.append({"well": w.well, "saved": 0, "error": f"{type(exc).__name__}: {exc}"})
    if any(x["saved"] for x in results):
        clear_caches()
    return {"results": results, "saved_wells": sum(1 for x in results if x["saved"]),
            "entry_user": user}


def _save_values(r: dict[str, Any], w: Any) -> dict[str, float]:
    if r["lift"] == "JP":
        if w.relation:
            raise ValueError("jet-pump wells use the pump model; no relation is saved")
        if w.ipr:
            raise ValueError("jet-pump IPRs are saved in Solver with the pump model")
        return {}
    eff = effective(r, w)
    values: dict[str, float] = {}
    if w.relation:
        rel = eff["rel"]
        if rel["slope"] is None:
            raise ValueError(rel.get("reason") or "no relation to save")
        if w.relation == "measured":
            m = r["measured"]
            if m["status"] != "measured" or not eff["gauge_ok"]:
                raise ValueError("measured relation is weak, missing or from a bad gauge - "
                                 "use the correlation or a manual value")
            r2, days = m["r2"] or 0.0, float(m["n_fit"])
        else:
            r2, days = 0.0, 0.0
        values.update({
            HDR_SLOPE: float(rel["slope"]), HDR_WHP_HDR: float(rel["whp_hdr"] or 1.0),
            HDR_R2: float(r2), HDR_DAYS: float(days),
            HDR_REL_SOURCE: hm.REL_SOURCE_CODE[w.relation],
        })
    if w.ipr:
        kind = "correlation" if w.ipr == "assumed" else w.ipr
        ip = eff["ipr"]
        if ip["ipr"] is None:
            raise ValueError(f"IPR not saved: {ip['reason'] or 'no IPR'}")
        values.update({
            "ipr_qwf_liq": float(ip["ipr"]["qwf"]), "ipr_pwf": float(ip["ipr"]["pwf"]),
            "resvr_press": float(ip["ipr"]["pres"]), HDR_IPR_SOURCE: hm.IPR_SOURCE_CODE[kind],
        })
    return values


def _pad_wells(pads: list[str]) -> list[str]:
    """The board's well list for ``pads`` (same order, so the same cache key)."""
    from server.services.tools import header_impact as hi

    ov = hi.fetch_well_overview(6)
    if ov is None or ov.empty:
        return []
    return sorted(ov[ov["well_pad"].isin(pads)]["well"].astype(str))


def well_detail(well: str, fit_days: int = FIT_DAYS_DEFAULT,
                pads: Optional[list[str]] = None) -> dict[str, Any]:
    """Everything the review cards need to judge one well's relation and IPR.

    BHP/WHP/header over the fit window (3-hour medians, for the trend
    chart), the within-day BHP~WHP and WHP~header fits day by day, and the
    well's tests over TEST_MONTHS. Read-only. With ``pads`` it reads the
    board's pad-wide historian pull (a cache hit after a board or Estimate),
    so scrolling through 50 cards costs no extra historian queries.
    """
    from server.services.tools import header_trend as ht

    today = date.today()
    wells = [well]
    if pads:
        pad_wells = _pad_wells(clean_pads(pads))
        if well in pad_wells:
            wells = pad_wells
    dfs = _trends(wells, today - timedelta(days=int(fit_days)), today)
    df = dfs.get(well)
    trend: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    daily_hdr: list[dict[str, Any]] = []
    if df is not None and not df.empty:
        d = df.copy()
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        binned = d.resample("3h").median().dropna(how="all")
        for ts, row in binned.iterrows():
            trend.append({"t": ts.isoformat(timespec="minutes"),
                          **{k.lower(): _r(row.get(k), 1) for k in ("BHP", "WHP", "HeaderP") if k in binned}})
        for y, x, sink in (("BHP", "WHP", daily), ("WHP", "HeaderP", daily_hdr)):
            if y in d and x in d:
                fit = ht.fit_within_day(d, y_name=y, x_name=x, robust=True)
                if fit is not None and fit.daily is not None and not fit.daily.empty:
                    for _, dr in fit.daily.iterrows():
                        sink.append({"day": str(pd.Timestamp(dr["day"]).date()), "slope": _r(dr["slope"], 3),
                                     "r2": _r(dr["r2"], 3), "n": int(dr["n"]),
                                     "x_min": _r(dr["x_min"], 0), "x_max": _r(dr["x_max"], 0)})
    tests_out: list[dict[str, Any]] = []
    tests = tests_svc.fetch_all_well_tests(TEST_MONTHS)
    if tests is not None and not tests.empty and "well" in tests:
        tw = tests[tests["well"] == well].sort_values("WtDate")
        for _, t in tw.iterrows():
            wc = _f(t.get("form_wc"))
            tests_out.append({
                "date": str(pd.Timestamp(t["WtDate"]).date()),
                "liquid": _r(t.get("WtTotalFluid"), 0), "oil": _r(t.get("WtOilVol"), 0),
                "wc": _r(wc / 100.0 if wc is not None and wc > 1 else wc, 3),
                "bhp": _r(t.get("BHP"), 0), "whp": _r(t.get("whp"), 0),
            })
    return {"well": well, "fit_days": int(fit_days), "trend": trend, "daily": daily,
            "daily_whp_hdr": daily_hdr, "tests": tests_out, "r2_day_min": hm.R2_DAY_MIN}


def list_pads() -> list[str]:
    """Pads with producers tested in the last six months."""
    from server.services.tools.header_impact import fetch_well_overview

    ov = fetch_well_overview(6)
    if ov is None or ov.empty or "well_pad" not in ov:
        return []
    return sorted({str(p).strip().upper() for p in ov["well_pad"].dropna() if _PAD_RE.match(str(p).strip().upper())})
