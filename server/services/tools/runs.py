"""Run orchestration for Scott's Tools.

The ported engine modules keep the tabs' internal helpers; this is the thin
layer that turns them into one call per tool, JSON-ready, off the request
thread where the work is long.

Everything here is READ-ONLY. JP Calibration renders SQL for a human to run;
it does not execute it, and it never has.

Long tools (fric trend, calibration, PF scenario, header impact, the harness)
go through ``server.jobs`` rather than blocking a socket for a minute - the
same pattern the optimizer endpoints use, and the reason the pages poll.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from server import pool
from server.services import frames
from server.services.tools import _common

log = logging.getLogger("woffl.web.tools.runs")


# ---------------------------------------------------------------------------
# JP Friction Trend
# ---------------------------------------------------------------------------


def fric_trend(wells: list[str], months_back: int = 12) -> dict[str, Any]:
    """Fit friction coefficients across each well's test history.

    Args:
        wells: GUI well names to calibrate.
        months_back: Test lookback window.

    Returns:
        dict: ``{"rows": [...], "wells": [...], "months_back": int,
        "skipped": {well: reason}}`` - one row per (well, test).
    """
    from woffl.assembly.jp_history import get_current_pump

    from server.services import datasources
    from server.services.tools import jp_fric_trend as jft

    inputs = jft._build_well_inputs(wells, months_back)
    if not inputs:
        return {"rows": [], "wells": [], "months_back": months_back,
                "skipped": {w: "no tests with the required fields" for w in wells}}

    jp_hist, _src = datasources.jp_history_safe()
    chars_map = _common.well_chars_map()
    vogel_map = _common.get_vogel_for_wells(list(inputs.keys()), months_back)

    jobs, ordered, skipped = [], [], {}
    for well, tests_df in inputs.items():
        pump = get_current_pump(jp_hist, well) if jp_hist is not None else None
        chars = chars_map.get(well)
        if not chars:
            skipped[well] = "not in the well characteristics frame"
            continue
        if not (pump and pump.get("nozzle_no") and pump.get("throat_ratio")):
            skipped[well] = "no current jet pump on record"
            continue
        jobs.append((well, tests_df, chars, pump, vogel_map.get(well)))
        ordered.append(well)

    if not jobs:
        return {"rows": [], "wells": [], "months_back": months_back, "skipped": skipped}

    results = pool.submit_all(jft._calibrate_well, jobs)
    if results is None:  # no pool, or it broke - same work, serially
        results = [jft._calibrate_well(*j) for j in jobs]

    per_well = {w: df for w, df in zip(ordered, results) if df is not None}
    combined = jft.combine_results(per_well)
    return {
        "rows": frames.records(combined) if not combined.empty else [],
        "wells": ordered,
        "months_back": months_back,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# JP Friction Calibration
# ---------------------------------------------------------------------------


def calibration_inputs(months_back: int = 6) -> dict[str, Any]:
    """The per-well calibration input table (no solving yet)."""
    from server.services.tools import jp_calibration as jc

    built = jc._build_calibration_input_table(months_back)
    if not built:
        return {"rows": [], "months_back": months_back}
    df, _chars = built
    return {"rows": frames.records(df), "months_back": months_back}


def run_calibration(wells: Optional[list[str]], months_back: int = 6) -> dict[str, Any]:
    """Fit ken/kth/kdi per well against the measured BHP.

    Args:
        wells: Restrict to these wells; None/empty calibrates everything in
            the input table.
        months_back: Test lookback window.

    Returns:
        dict: ``{"rows": [...], "sql": str, "months_back": int}``. ``sql`` is
        a PREVIEW for a human to run - nothing here writes.
    """
    from server.services.tools import jp_calibration as jc

    built = jc._build_calibration_input_table(months_back)
    if not built:
        return {"rows": [], "sql": "", "months_back": months_back}
    df, pump_info_map = built
    if wells:
        df = df[df["Well"].isin(wells)]
    if df.empty:
        return {"rows": [], "sql": "", "months_back": months_back}

    # pump_info_map is the pump resolved AT the BHP test's date (P1-29) -
    # characteristics and the Vogel row are separate lookups.
    chars_map = _common.well_chars_map()
    vogel_map = _common.get_vogel_for_wells(df["Well"].tolist(), months_back)
    jobs = [
        (row.to_dict(), chars_map.get(row["Well"]),
         pump_info_map.get(row["Well"]), vogel_map.get(row["Well"]))
        for _, row in df.iterrows()
    ]
    results = pool.submit_all(jc.calibrate_one, jobs)
    if results is None:
        results = [jc.calibrate_one(*j) for j in jobs]

    out = pd.DataFrame([r for r in results if r])
    sql = ""
    try:
        sql = jc._format_sql_preview(out, "mpu.wells.prop_hist")
    except Exception:  # noqa: BLE001 - the preview is a convenience
        log.warning("calibration SQL preview failed", exc_info=True)
    return {"rows": frames.records(out), "sql": sql, "months_back": months_back}


# ---------------------------------------------------------------------------
# Header Pressure Impact
# ---------------------------------------------------------------------------


def header_impact_inputs(pads: list[str], months_back: int = 6) -> dict[str, Any]:
    """The per-well input table for the selected pads, all lift types."""
    from server.services.tools import header_impact as hi

    df = hi._build_input_table(pads, months_back)
    if df is None or df.empty:
        return {"rows": [], "pads": pads, "months_back": months_back}
    return {"rows": frames.records(df), "pads": pads, "months_back": months_back}


def header_impact(
    pads: list[str],
    delta_p: float = -50.0,
    months_back: int = 6,
    pad_pf: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Model a header pressure change across every producer on the pads.

    Args:
        pads: Pad letters.
        delta_p: Header change, psi (negative = drawdown, the usual case).
        months_back: Test/trend lookback.
        pad_pf: Per-pad PF pressure overrides (PF-PRESSURE-DEPENDENCY).

    Returns:
        dict: ``{"rows": [...], "totals": {...}, "delta_p", "pads",
        "months_back"}``. Each row carries the physics delta, the empirical
        comparison where tags exist, the lift type, and the verdict.
    """
    from server.services.tools import header_impact as hi

    df = hi._build_input_table(pads, months_back, pad_pf=pad_pf)
    if df is None or df.empty:
        return {"rows": [], "totals": {}, "delta_p": delta_p, "pads": pads,
                "months_back": months_back}

    wells = df["Well"].tolist()
    # NOTE the order: the helper returns (well_dfs, fits, missing), not
    # (fits, dfs). Unpacking it backwards silently hands every well the wrong
    # dict and every row fails identically.
    emp_missing: list[str] = []
    try:
        emp_dfs, emp_fits, emp_missing = hi._fetch_empirical_fits(wells, months_back)
    except Exception:  # noqa: BLE001 - empirical is the optional half
        log.warning("header impact: empirical fits unavailable", exc_info=True)
        emp_dfs, emp_fits = {}, {}

    # Dispatch on lift type. JP wells go through WOFFL physics (two solves,
    # at the current and the scenario WHP); ESP / gas-lift / flowing take the
    # empirical path, which is what `_solve_nonjp_row` is FOR - sending JP
    # wells there yields rows with no DeltaOil at all.
    from woffl.assembly.jp_history import get_current_pump

    from server.services import datasources

    jp_hist, _src = datasources.jp_history_safe()
    chars_map = _common.well_chars_map()
    vogel_map = _common.get_vogel_for_wells(wells, months_back)

    jp_jobs, jp_order, rows = [], [], []
    for _, row in df.iterrows():
        wn = row["Well"]
        if str(row.get("Lift", "JP")) != "JP":
            try:
                rows.append(hi._solve_nonjp_row(wn, row, emp_fits, emp_dfs, delta_p))
            except Exception as exc:  # noqa: BLE001 - one well is a row, not a 500
                log.warning("header impact (non-JP) failed for %s", wn, exc_info=True)
                rows.append({"Well": wn, "Pad": row.get("Pad"), "Lift": row.get("Lift"),
                             "Error": str(exc)[:200], "Verdict": "n/a"})
            continue
        pump = get_current_pump(jp_hist, wn) if jp_hist is not None else None
        jp_jobs.append((row.to_dict(), chars_map.get(wn), pump,
                        vogel_map.get(wn), emp_fits.get(wn), delta_p))
        jp_order.append(wn)

    if jp_jobs:
        solved = pool.submit_all(hi.solve_jp_row, jp_jobs)
        if solved is None:  # no pool, or it broke - same work, serially
            solved = [hi.solve_jp_row(*j) for j in jp_jobs]
        rows.extend(solved)

    out = frames.records(pd.DataFrame(rows))
    d_oil = [r.get("DeltaOil") for r in out if isinstance(r.get("DeltaOil"), (int, float))]
    return {
        "rows": out,
        "totals": {
            "wells": len(out),
            "delta_oil": round(sum(d_oil), 1) if d_oil else 0.0,
            "responsive": sum(1 for r in out if r.get("Verdict") == "responsive"),
            "sonic": sum(1 for r in out if r.get("SonicNow")),
        },
        "delta_p": delta_p,
        "pads": pads,
        "months_back": months_back,
        "no_tags": sorted(emp_missing),
    }


# ---------------------------------------------------------------------------
# PF Scenario
# ---------------------------------------------------------------------------


def pf_scenario(
    wells: list[str],
    pf_a: float,
    pf_b: float,
    months_back: int = 6,
) -> dict[str, Any]:
    """Compare two power-fluid pressures across the selected wells.

    Args:
        wells: GUI well names.
        pf_a: Scenario A PF surface pressure, psi.
        pf_b: Scenario B PF surface pressure, psi.
        months_back: Test lookback for the IPR chain.

    Returns:
        dict: ``{"rows": [...], "totals": {...}, "pf_a", "pf_b"}`` - one row
        per well with oil/BHP/PF at both pressures and the delta.
    """
    from woffl.assembly.jp_history import get_current_pump

    from server.services import datasources
    from server.services.tools import pf_scenario as pfs

    jp_hist, _src = datasources.jp_history_safe()
    chars_map = _common.well_chars_map()
    vogel_map = _common.get_vogel_for_wells(wells, months_back)
    whp_map = _common.get_latest_whp_per_well(months_back)

    jobs = []
    for wn in wells:
        chars = chars_map.get(wn)
        pump = get_current_pump(jp_hist, wn) if jp_hist is not None else None
        if not chars or not (pump and pump.get("nozzle_no") and pump.get("throat_ratio")):
            continue
        jobs.append((wn, chars, pump, vogel_map.get(wn), whp_map.get(wn, 210.0),
                     float(pf_a), float(pf_b)))

    if not jobs:
        return {"rows": [], "totals": {}, "pf_a": pf_a, "pf_b": pf_b}

    results = pool.submit_all(pfs.compare_one, jobs)
    if results is None:
        results = [pfs.compare_one(*j) for j in jobs]

    out = frames.records(pd.DataFrame([r for r in results if r]))
    d_oil = [r.get("DeltaOil") for r in out if isinstance(r.get("DeltaOil"), (int, float))]
    return {
        "rows": out,
        "totals": {"wells": len(out), "delta_oil": round(sum(d_oil), 1) if d_oil else 0.0},
        "pf_a": pf_a,
        "pf_b": pf_b,
    }
