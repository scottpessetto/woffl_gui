"""Multi-point calibration dataset builder (Pillar 1b, P1).

The current pump eras on the M-Pad hold only 0-3 usable well TESTS with
near-zero PF-pressure spread (June-Aug 2026 JPCO wave), so tests alone are
not identifiable. This module builds the fit set the multipoint fitter
(woffl/gui/fric_calibration.calibrate_multipoint) consumes: DAILY triplets
(ppf, bhp, pf_rate) joined from mpu.wells.vw_pressure_daily and
mpu.wells.vw_power_fluid_volume, plus in-era well tests as high-weight
anchor points. Architecture mirrors server/services/evidence.py: cached
fleet frames, a PURE per-well assembly function (the unit-test surface),
and a fail-soft pad-level fetcher.

Point dict contract (plain dicts - woffl/gui never imports server code):

    {"date": iso, "kind": "daily"|"test", "ppf": f, "bhp": f, "pf_rate": f,
     "pwh": f, "qtot": f|None, "oil": f|None, "wc": f|None, "fgor": f|None,
     "weight": f}

Builder result per well:

    {"well", "pump": {"nozzle", "throat", "date_set"}|None,
     "era_start": iso|None, "points": [...], "ppf_spread": f,
     "n_daily": int, "n_test": int, "refusal": str|None}

Refused wells still return the dict with their points included - the P3
harness reports them; refusal only tells the fitter not to trust a fit.

wc is a FRACTION (0-1, form_wc convention); fgor is scf/bbl. Daily points
carry the nearest in-era test's WtTotalFluid/WtOilVol/wc/fgor when one sits
within TEST_ATTACH_DAYS - the fitter anchors per-point Vogel on these -
else the caller-supplied saved-fit fallbacks (fallback_qtot/fallback_wc/
fallback_fgor kwargs; None passes through, the fitter owns final defaults).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache

log = logging.getLogger("woffl.web.calibration_points")

# ---------------------------------------------------------------------------
# Tunables (one block - see the pillar-1b spec)
# ---------------------------------------------------------------------------

BHP_GLITCH_PSI = 50.0        # daily/test BHP at/below this is a dead gauge
PPF_RANGE_PSI = (800.0, 5500.0)  # resolved PF outside this is not a header
PF_RATE_MIN_BPD = 500.0      # pwr_fld_net at/below this is a shut-in day
# Steady-state filter: transients/recoveries dominate the raw daily scatter -
# the response eyeball showed shut-in buildup tails to 1,800 psi. A daily row
# whose BHP sits more than this from the centered 5-day rolling median of the
# well's era BHP series is a transient, not an operating point. The rolling
# median needs >= 3 days in its window, else the row is kept (short eras).
STEADY_STATE_TOL_PSI = 60.0
TEST_ATTACH_DAYS = 30        # nearest-test window for wc/fgor/IPR anchor
TEST_WEIGHT = 3.0            # in-era tests are high-weight anchor points
DAILY_WEIGHT = 1.0
MAX_FIT_POINTS = 20          # cost cap; tests always kept, dailies sampled
MIN_USABLE_POINTS = 10       # fewer -> "young pump era" refusal
MIN_PPF_SPREAD_PSI = 200.0   # smaller -> "not identifiable" refusal
DEFAULT_SURF_PRES = 210.0    # SimParams surf_pres default (server/schemas.py)

# Fleet PF-volume query - same shape as evidence._FLEET_QUERY: max() per
# well+day collapses repeated samples and prefers an operating reading over
# a same-day shut-in zero. pf_well_bore names wells like "M-064".
_PF_VOLUME_QUERY = """\
SELECT pf_well_bore, pfdate,
       max(pwr_fld_net) AS pwr_fld_net
FROM mpu.wells.vw_power_fluid_volume
WHERE pfdate >= date_sub(current_date(), 365)
GROUP BY pf_well_bore, pfdate
"""


# ---------------------------------------------------------------------------
# Fleet frames (cached) - the only I/O besides pad_points' side fetches
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_EXTENDED_TESTS, maxsize=4)
def _fleet_pf_volume() -> pd.DataFrame:
    """365 days of daily net power-fluid rate (bpd) for ALL wells.

    One query for the fleet; the ``well`` column carries the app-normalized
    name ("MPM-64") mapped from pf_well_bore ("M-064") exactly like
    evidence._fleet_pressure_daily maps well_name. Raises on Databricks
    failure - pad_points' caller owns the fail-soft.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(_PF_VOLUME_QUERY)
    if df.empty:
        return pd.DataFrame(columns=["well", "pfdate", "pwr_fld_net"])
    df["pwr_fld_net"] = pd.to_numeric(df["pwr_fld_net"], errors="coerce")
    df["pfdate"] = pd.to_datetime(df["pfdate"], errors="coerce")
    df["well"] = df["pf_well_bore"].astype(str).str.strip().map(_normalize_well_name)
    return df


# ---------------------------------------------------------------------------
# Pure per-well assembly (no I/O - the unit-test surface)
# ---------------------------------------------------------------------------


def _f(value) -> Optional[float]:
    """Cast to float; None for NULL/NaN/non-numeric."""
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(val) else val


def _vogel_scale(
    bhp_test: Optional[float], bhp_day: Optional[float], res_pres: Optional[float]
) -> Optional[float]:
    """Ratio of Vogel deliverability at ``bhp_day`` to that at ``bhp_test``
    on the same curve (same reservoir pressure), i.e. the factor that moves
    the test's rate along ITS OWN inflow curve to the day's drawdown. None
    when any input is missing or the day is at/above reservoir pressure
    (the caller keeps the raw test rate; that point is filtered anyway).
    Floored at 0 (EVID-F6)."""
    if bhp_test is None or bhp_day is None or res_pres is None:
        return None
    try:
        pr = float(res_pres)
        if pr <= 0 or float(bhp_test) >= pr or float(bhp_day) >= pr:
            return None
        rt = float(bhp_test) / pr
        rd = float(bhp_day) / pr
        denom = 1.0 - 0.2 * rt - 0.8 * rt * rt
        if denom <= 0:
            return None
        return max(0.0, (1.0 - 0.2 * rd - 0.8 * rd * rd) / denom)
    except (TypeError, ValueError):
        return None


def _test_wc(row: Any, fallback_wc: Optional[float]) -> Optional[float]:
    """form_wc, else computed WtWaterVol/(WtOilVol+WtWaterVol), else fallback.

    Same chain as scotts_tools/jp_fric_trend; clamped to [0, 0.99] (the
    prop-store MAX_MODELABLE_WC precedent). None when nothing is available.
    """
    wc = _f(row.get("form_wc"))
    if wc is None:
        oil = _f(row.get("WtOilVol")) or 0.0
        wat = _f(row.get("WtWaterVol")) or 0.0
        wc = wat / (oil + wat) if (oil + wat) > 0 else fallback_wc
    if wc is None:
        return None
    return min(max(float(wc), 0.0), 0.99)


def _era_tests(tests_df: Optional[pd.DataFrame], era_start: pd.Timestamp) -> pd.DataFrame:
    """In-era test rows with a parseable date, oldest first; empty fail-soft."""
    if tests_df is None or tests_df.empty or "WtDate" not in tests_df.columns:
        return pd.DataFrame()
    wt = tests_df.copy()
    wt["WtDate"] = pd.to_datetime(wt["WtDate"], errors="coerce")
    wt = wt.dropna(subset=["WtDate"])
    return wt[wt["WtDate"] >= era_start].sort_values("WtDate")


def _stratified_daily_cap(daily_points: list[dict[str, Any]], slots: int) -> list[dict[str, Any]]:
    """Evenly spaced picks across the ppf-sorted dailies so spread survives.

    ``slots`` <= 0 keeps nothing; slots >= len keeps everything. With 2+
    slots the endpoints (min and max ppf) are always among the picks.
    """
    if slots <= 0:
        return []
    if len(daily_points) <= slots:
        return list(daily_points)
    by_ppf = sorted(daily_points, key=lambda p: p["ppf"])
    n = len(by_ppf)
    if slots == 1:
        return [by_ppf[0]]
    # linspace over indices; step > 1 when n > slots, so rounding never
    # collides and every pick is distinct, including both endpoints.
    picks = [round(i * (n - 1) / (slots - 1)) for i in range(slots)]
    return [by_ppf[i] for i in picks]


def points_for_well(
    well: str,
    *,
    jp_hist: Optional[pd.DataFrame],
    tests_df: Optional[pd.DataFrame],
    daily_df: Optional[pd.DataFrame],
    pf_df: Optional[pd.DataFrame],
    res_pres: Optional[float] = None,
    surf_pres: Optional[float] = None,
    fallback_qtot: Optional[float] = None,
    fallback_wc: Optional[float] = None,
    fallback_fgor: Optional[float] = None,
) -> dict[str, Any]:
    """Builder result dict for one well. PURE given its inputs - no I/O.

    Args:
        well: app-normalized name ("MPM-64").
        jp_hist: parse_jp_history frame (Well Name / Date Set / nozzle cols);
            the era is the current pump's Date Set (get_current_pump).
        tests_df: the well's test rows (server.services.tests.tests_for_well
            shape: WtDate, BHP, lift_wat, whp, pf_press, WtTotalFluid,
            WtOilVol, WtWaterVol, form_wc, fgor).
        daily_df: fleet daily-pressure frame (evidence._fleet_pressure_daily
            shape: well, sample_date, tubing_prs, inn_ann_prs, btmhole_prs).
        pf_df: fleet PF-volume frame (_fleet_pf_volume shape: well, pfdate,
            pwr_fld_net).
        res_pres: saved-fit reservoir pressure; daily BHP at/above it is
            shut-in buildup, not flowing (same guard as evidence.py). None
            skips the filter.
        surf_pres: config wellhead pressure - pwh for daily points and the
            whp fallback for test points. None -> DEFAULT_SURF_PRES.
        fallback_qtot/fallback_wc/fallback_fgor: saved-fit liquid rate /
            water cut (fraction) / GOR used when no in-era test sits within
            TEST_ATTACH_DAYS of a daily point. None passes through to the
            point dict (the fitter owns final defaults).
    """
    from woffl.assembly.jp_history import get_current_pump
    from woffl.assembly.pf_pressure import resolve_pf_pressure

    result: dict[str, Any] = {
        "well": well,
        "pump": None,
        "era_start": None,
        "points": [],
        "ppf_spread": 0.0,
        "n_daily": 0,
        "n_test": 0,
        "refusal": None,
    }

    pump = get_current_pump(jp_hist, well) if jp_hist is not None else None
    if pump is None or pump.get("date_set") is None or pd.isna(pump["date_set"]):
        result["refusal"] = "no current pump record in jp_history"
        return result
    era_start = pd.to_datetime(pump["date_set"]).normalize()
    result["pump"] = {
        "nozzle": pump.get("nozzle_no"),
        "throat": pump.get("throat_ratio"),
        "date_set": era_start.date().isoformat(),
    }
    result["era_start"] = era_start.date().isoformat()

    pwh_default = float(surf_pres) if surf_pres is not None else DEFAULT_SURF_PRES
    ppf_lo, ppf_hi = PPF_RANGE_PSI

    # -- test points (in-era, high-weight anchors) --------------------------
    era_tests = _era_tests(tests_df, era_start)
    test_points: list[dict[str, Any]] = []
    for _, row in era_tests.iterrows():
        bhp = _f(row.get("BHP"))
        lift = _f(row.get("lift_wat"))
        ppf = _f(row.get("pf_press"))
        if bhp is None or bhp <= BHP_GLITCH_PSI:
            continue
        if lift is None or lift <= 0:
            continue
        if ppf is None or not (ppf_lo <= ppf <= ppf_hi):
            continue
        whp = _f(row.get("whp"))
        test_points.append(
            {
                "date": row["WtDate"].date().isoformat(),
                "kind": "test",
                "ppf": ppf,
                "bhp": bhp,
                "pf_rate": lift,
                "pwh": whp if whp is not None and whp > 0 else pwh_default,
                "qtot": _f(row.get("WtTotalFluid")),
                "oil": _f(row.get("WtOilVol")),
                "wc": _test_wc(row, fallback_wc),
                "fgor": _f(row.get("fgor")) if _f(row.get("fgor")) is not None else fallback_fgor,
                "weight": TEST_WEIGHT,
            }
        )

    # -- wc/fgor/IPR-anchor attach source: any in-era test with a real rate.
    # Broader than the test-POINT filter on purpose - a test that failed the
    # ppf gate still measured the well's wc/rate that week.
    anchors: list[tuple[pd.Timestamp, pd.Series]] = [
        (row["WtDate"], row)
        for _, row in era_tests.iterrows()
        if _f(row.get("WtTotalFluid")) is not None
    ]

    def _nearest_anchor(when: pd.Timestamp) -> Optional[pd.Series]:
        best, best_days = None, None
        for stamp, row in anchors:
            days = abs((stamp - when).days)
            if days <= TEST_ATTACH_DAYS and (best_days is None or days < best_days):
                best, best_days = row, days
        return best

    # -- daily points (era-gated triplets from the joined fleet frames) -----
    daily_points: list[dict[str, Any]] = []
    if (
        daily_df is not None
        and not daily_df.empty
        and "well" in daily_df.columns
        and pf_df is not None
        and not pf_df.empty
        and "well" in pf_df.columns
    ):
        prs = daily_df[daily_df["well"] == well].copy()
        vol = pf_df[pf_df["well"] == well].copy()
        if not prs.empty and not vol.empty:
            prs["date"] = pd.to_datetime(prs["sample_date"], errors="coerce").dt.normalize()
            vol["date"] = pd.to_datetime(vol["pfdate"], errors="coerce").dt.normalize()
            vol["pwr_fld_net"] = pd.to_numeric(vol["pwr_fld_net"], errors="coerce")
            merged = prs.dropna(subset=["date"]).merge(
                vol.dropna(subset=["date"])[["date", "pwr_fld_net"]], on="date", how="inner"
            )
            merged = merged[merged["date"] >= era_start].sort_values("date")
            # (see _vogel_scale for the per-point rate re-anchoring)
            # Steady-state filter (STEADY_STATE_TOL_PSI): drop transient /
            # recovery days before selection. NaN medians (window < 3 days)
            # and NaN BHPs compare False -> kept for the later row filters.
            if not merged.empty:
                bhp_series = pd.to_numeric(merged["btmhole_prs"], errors="coerce")
                roll_med = bhp_series.rolling(5, center=True, min_periods=3).median()
                merged = merged[~((bhp_series - roll_med).abs() > STEADY_STATE_TOL_PSI)]
            for _, row in merged.iterrows():
                bhp = _f(row.get("btmhole_prs"))
                if bhp is None or bhp <= BHP_GLITCH_PSI:
                    continue
                if res_pres is not None and float(res_pres) > 0 and bhp >= float(res_pres):
                    continue  # shut-in buildup, not flowing (evidence.py guard)
                ppf, pf_src = resolve_pf_pressure(row.get("tubing_prs"), row.get("inn_ann_prs"))
                ppf = _f(ppf)
                if ppf is None or not (ppf_lo <= ppf <= ppf_hi):
                    continue
                # The day's MEASURED wellhead pressure: on a reverse-circ day
                # the tubing gauge IS the production WHP (PF is in the
                # annulus), and vice versa. Holding every daily point at the
                # default WHP dumped 50-150 psi of real variance into the
                # friction coefficients (review 2026-09-01, EVID-F7).
                pwh_day = pwh_default
                prod_side = row.get("tubing_prs") if pf_src == "annulus" else row.get("inn_ann_prs")
                prod_side = _f(prod_side)
                if prod_side is not None and 10.0 <= prod_side <= 600.0:
                    pwh_day = prod_side
                rate = _f(row.get("pwr_fld_net"))
                if rate is None or rate <= PF_RATE_MIN_BPD:
                    continue
                anchor = _nearest_anchor(row["date"])
                if anchor is not None:
                    qtot = _f(anchor.get("WtTotalFluid"))
                    oil = _f(anchor.get("WtOilVol"))
                    wc = _test_wc(anchor, fallback_wc)
                    fgor = _f(anchor.get("fgor"))
                    fgor = fgor if fgor is not None else fallback_fgor
                    # Move the test's rate along ITS OWN Vogel curve to the
                    # day's BHP. The fitter builds an InFlow per point at
                    # (oil, bhp); handing every daily point the raw test oil
                    # at a different BHP asserted a different inflow curve
                    # per day, gave the residual a systematic sign against
                    # PF, and biased the fit toward a pinned (unresponsive)
                    # pump (review 2026-09-01, EVID-F6). Scaled this way,
                    # every point's anchor lies on ONE curve: the test's.
                    scale = _vogel_scale(_f(anchor.get("BHP")), bhp, res_pres)
                    if scale is not None:
                        oil = oil * scale if oil is not None else None
                        qtot = qtot * scale if qtot is not None else None
                else:
                    qtot = fallback_qtot
                    wc = fallback_wc
                    fgor = fallback_fgor
                    oil = qtot * (1.0 - wc) if qtot is not None and wc is not None else None
                daily_points.append(
                    {
                        "date": row["date"].date().isoformat(),
                        "kind": "daily",
                        "ppf": ppf,
                        "bhp": bhp,
                        "pf_rate": rate,
                        "pwh": pwh_day,
                        "qtot": qtot,
                        "oil": oil,
                        "wc": wc,
                        "fgor": fgor,
                        "weight": DAILY_WEIGHT,
                    }
                )

    # -- refusals over the FULL usable set (pre-cap) -------------------------
    n_usable = len(test_points) + len(daily_points)
    all_ppf = [p["ppf"] for p in test_points + daily_points]
    spread = (max(all_ppf) - min(all_ppf)) if len(all_ppf) >= 2 else 0.0
    if n_usable < MIN_USABLE_POINTS:
        result["refusal"] = f"young pump era - {n_usable} usable points"
    elif spread < MIN_PPF_SPREAD_PSI:
        result["refusal"] = (
            f"not identifiable - ppf spread {spread:.0f} psi in this pump era"
        )

    # -- cost cap: ALL tests kept, dailies stratified across the ppf range --
    slots = MAX_FIT_POINTS - len(test_points)
    kept_dailies = _stratified_daily_cap(daily_points, slots)
    points = sorted(test_points + kept_dailies, key=lambda p: (p["date"], p["kind"]))

    result["points"] = points
    result["ppf_spread"] = float(spread)
    result["n_daily"] = len(kept_dailies)
    result["n_test"] = len(test_points)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pad_points(
    well_names: list[str],
    *,
    res_pres: Optional[dict[str, float]] = None,
    surf_pres: Optional[dict[str, float]] = None,
    fallbacks: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, dict[str, Any]]:
    """Builder result dicts for a pad's wells, keyed by app well name.

    Wells whose per-well assembly raises are simply ABSENT - never fatal
    (refused wells are PRESENT with ``refusal`` set; the harness reports
    them). Only the fleet fetches themselves raise; the caller owns that
    fail-soft, mirroring evidence.pad_evidence.

    Args:
        well_names: app-normalized names (the optimizer's WellConfig names).
        res_pres: well -> saved-fit reservoir pressure (buildup filter).
        surf_pres: well -> config wellhead pressure (DEFAULT_SURF_PRES when
            missing).
        fallbacks: well -> {"qtot", "wc", "fgor"} saved-fit values used when
            no in-era test sits within TEST_ATTACH_DAYS of a daily point.
    """
    from server.services import datasources
    from server.services import tests as tests_svc
    from server.services.evidence import _fleet_pressure_daily

    daily = _fleet_pressure_daily()
    pf_vol = _fleet_pf_volume()
    jp_hist, _source = datasources.jp_history_safe()

    out: dict[str, dict[str, Any]] = {}
    for well in well_names:
        try:
            fb = (fallbacks or {}).get(well) or {}
            out[well] = points_for_well(
                well,
                jp_hist=jp_hist,
                tests_df=tests_svc.tests_for_well(well, 24, 0),
                daily_df=daily,
                pf_df=pf_vol,
                res_pres=(res_pres or {}).get(well),
                surf_pres=(surf_pres or {}).get(well),
                fallback_qtot=fb.get("qtot"),
                fallback_wc=fb.get("wc"),
                fallback_fgor=fb.get("fgor"),
            )
        except Exception:
            log.warning("calibration point assembly failed for %s", well, exc_info=True)
    return out
