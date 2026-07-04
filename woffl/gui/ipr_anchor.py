"""Anchored Vogel IPR fitting for the single-well solver.

The library's :mod:`woffl.assembly.ipr_analyzer` always anchors the Vogel curve
on the most-recent test and fits reservoir pressure to the *global* test cloud
(it minimizes SSE over every candidate anchor). This GUI module lets the user
anchor the fit on a **specific** or the **median** test instead, and re-fits
reservoir pressure so the Vogel curve best passes through the rest of the cloud
*given that fixed anchor*.

Kept in the GUI layer (not ``woffl/assembly``) so it carries no upstream-library
obligation; it reuses :class:`woffl.flow.inflow.InFlow` and mirrors the Vogel
math in ``ipr_analyzer._calculate_global_sse`` / ``_calculate_r_squared``.

The returned dict is shaped like a row of
``ipr_analyzer.compute_vogel_coefficients`` so callers (Model-vs-Actual chart,
``generate_ipr_curves``, ``build_calibration_inputs``) consume it unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from woffl.flow.inflow import InFlow


def _vogel_denom(pwf: float, pres: float) -> float:
    """Vogel dimensionless inflow term: 1 - 0.2(pwf/pres) - 0.8(pwf/pres)^2."""
    ratio = pwf / pres
    return 1.0 - 0.2 * ratio - 0.8 * ratio**2


def fit_rp_through_anchor(
    bhp_values,
    fluid_values,
    anchor_bhp: float,
    anchor_fluid: float,
    max_pres: float,
    step: int = 5,
) -> int:
    """Reservoir pressure that best-fits the cloud with a *fixed* anchor.

    Sweeps candidate RP from just above the max BHP (or the anchor BHP, whichever
    is higher) up to ``max_pres``. For each candidate, builds the Vogel curve
    anchored on ``(anchor_bhp, anchor_fluid)`` and sums squared error against all
    real test points; returns the RP with the minimum SSE.

    This is the single-fixed-anchor analogue of
    ``ipr_analyzer._calculate_global_sse``, which instead minimizes over every
    possible anchor. Here the anchor is chosen by the user (specific / median),
    so only RP is free. Uses the SAME axis-normalized point-to-curve distance
    as the library fit (``ipr_analyzer._normalized_curve_sse``) — the old
    rate-only SSE railed the RP sweep to the field cap on flat test clouds.
    """
    from woffl.assembly.ipr_analyzer import _axis_scales, _normalized_curve_sse

    bhp_values = np.asarray(bhp_values, dtype=float)
    fluid_values = np.asarray(fluid_values, dtype=float)

    finite = ~(np.isnan(bhp_values) | np.isnan(fluid_values))
    bhp_values = bhp_values[finite]
    fluid_values = fluid_values[finite]

    max_bhp = float(np.max(bhp_values)) if bhp_values.size else float(anchor_bhp)
    floor = max(max_bhp, float(anchor_bhp))
    start_pres = int(floor) + 10
    end_pres = int(max_pres)

    if start_pres >= end_pres:
        return int(floor) + 50

    q_scale, p_scale = _axis_scales(bhp_values, fluid_values)

    best_rp = None
    best_sse = float("inf")
    for pres in range(start_pres, end_pres, step):
        p = float(pres)
        denom_a = _vogel_denom(anchor_bhp, p)
        if denom_a <= 0:
            continue
        qmax = np.array([anchor_fluid / denom_a])
        sse = float(
            _normalized_curve_sse(
                bhp_values, fluid_values, p, qmax, q_scale, p_scale
            )[0]
        )
        if sse < best_sse:
            best_sse = sse
            best_rp = pres

    return int(best_rp) if best_rp is not None else int(floor) + 50


def _r_squared(
    bhp_values, fluid_values, pres: float, anchor_bhp: float, anchor_fluid: float
) -> float:
    """R² of the anchored Vogel curve vs the test cloud (mirrors ipr_analyzer)."""
    bhp_values = np.asarray(bhp_values, dtype=float)
    fluid_values = np.asarray(fluid_values, dtype=float)
    finite = ~(np.isnan(bhp_values) | np.isnan(fluid_values))
    bhp_values = bhp_values[finite]
    fluid_values = fluid_values[finite]

    if fluid_values.size < 2 or anchor_bhp >= pres:
        return 0.0
    denom_a = _vogel_denom(anchor_bhp, pres)
    if denom_a <= 0:
        return 0.0
    qmax = anchor_fluid / denom_a

    ss_res = 0.0
    for j in range(bhp_values.size):
        if bhp_values[j] >= pres:
            continue
        pred = qmax * _vogel_denom(bhp_values[j], pres)
        ss_res += (pred - fluid_values[j]) ** 2

    ss_tot = float(np.sum((fluid_values - np.mean(fluid_values)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _resolve_anchor_row(
    df: pd.DataFrame, anchor_mode: str, anchor_date
) -> tuple[pd.Series, str]:
    """Pick the anchor row and a human label for the given mode.

    Falls back to the most-recent test if a 'specific' date doesn't match any
    row (e.g. the picked test was filtered out by a tighter lookback).
    """
    by_date_desc = df.sort_values("__date", ascending=False)
    recent_row = by_date_desc.iloc[0]

    def _label(row: pd.Series, prefix: str) -> str:
        d = row.get("__date")
        ds = d.strftime("%Y-%m-%d") if pd.notna(d) else "n/a"
        return f"{prefix} ({ds})"

    if anchor_mode == "median":
        median_bhp = df["BHP"].median()
        pos = int((df["BHP"] - median_bhp).abs().values.argmin())
        row = df.iloc[pos]
        return row, _label(row, "median test")

    if anchor_mode == "specific" and anchor_date is not None:
        target = pd.to_datetime(anchor_date, errors="coerce")
        if pd.notna(target):
            mask = df["__date"].dt.normalize() == target.normalize()
            if mask.any():
                row = df[mask].sort_values("__date", ascending=False).iloc[0]
                return row, _label(row, "selected test")

    return recent_row, _label(recent_row, "most recent")


def compute_anchored_vogel(
    test_df: pd.DataFrame,
    *,
    well_name: str | None = None,
    anchor_mode: str = "median",
    anchor_date=None,
    field_max_rp: float = 1800,
    resp_modifier: int = 0,
) -> dict | None:
    """Vogel coefficients anchored on a chosen test, with RP re-fit through it.

    Args:
        test_df: well-test rows (needs ``BHP``, ``WtTotalFluid``, ``WtDate``;
            optionally ``WtWaterVol``, ``fgor``, ``well``).
        well_name: well label for the output row; defaults to ``test_df['well']``.
        anchor_mode: ``"median"``, ``"specific"`` (needs ``anchor_date``), or
            ``"recent"``.
        anchor_date: the test date to anchor on when ``anchor_mode='specific'``.
        field_max_rp: upper bound for the RP sweep (Schrader ~1800, Kuparuk ~3000).
        resp_modifier: psi added to the fitted RP (parity with ipr_analyzer).

    Returns a dict shaped like a ``compute_vogel_coefficients`` row, or ``None``
    when there aren't enough valid points to fit.
    """
    df = test_df.dropna(subset=["BHP", "WtTotalFluid"]).copy()
    if df.empty:
        return None
    df["__date"] = pd.to_datetime(df.get("WtDate"), errors="coerce")

    well = well_name or (
        str(df["well"].iloc[0]) if "well" in df.columns else "Well"
    )

    bhp_values = df["BHP"].values.astype(float)
    fluid_values = df["WtTotalFluid"].values.astype(float)

    anchor_row, anchor_label = _resolve_anchor_row(df, anchor_mode, anchor_date)
    anchor_bhp = float(anchor_row["BHP"])
    anchor_fluid = float(anchor_row["WtTotalFluid"])

    rp = fit_rp_through_anchor(
        bhp_values, fluid_values, anchor_bhp, anchor_fluid, field_max_rp
    ) + int(resp_modifier)

    if anchor_bhp >= rp:
        # Degenerate: anchor BHP at/above the fitted RP. Nudge RP just above it
        # so the Vogel math stays valid rather than returning None.
        rp = int(anchor_bhp) + 50

    qmax = InFlow.vogel_qmax(anchor_fluid, anchor_bhp, rp)

    # Water cut from the anchor row (falls back to 0.5, same default as
    # ipr_analyzer.compute_vogel_coefficients).
    wc = 0.5
    water = anchor_row.get("WtWaterVol")
    if water is not None and not pd.isna(water) and anchor_fluid > 0:
        wc = max(0.0, min(1.0, float(water) / anchor_fluid))

    fgor = anchor_row.get("fgor")
    fgor = int(fgor) if (fgor is not None and not pd.isna(fgor)) else 250

    most_recent = df.sort_values("__date", ascending=False)["__date"].iloc[0]
    r2 = _r_squared(bhp_values, fluid_values, float(rp), anchor_bhp, anchor_fluid)

    anchor_date_val = anchor_row.get("__date")
    return {
        "Well": well,
        "ResP": int(rp),
        "QMax_recent": qmax,
        "qwf": anchor_fluid,
        "pwf": anchor_bhp,
        "form_wc": round(wc, 3),
        "fgor": fgor,
        "num_tests": int(len(df)),
        "most_recent_date": most_recent,
        "R2": round(r2, 3),
        "anchor_label": anchor_label,
        "anchor_date": anchor_date_val if pd.notna(anchor_date_val) else None,
    }
