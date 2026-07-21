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

import os

import numpy as np
import pandas as pd

from woffl.assembly.prop_hist_client import push_prop, resolve_entry_user
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
            _normalized_curve_sse(bhp_values, fluid_values, p, qmax, q_scale, p_scale)[
                0
            ]
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


def find_test_row_by_wt_uid(
    test_df: pd.DataFrame | None, wt_uid: int
) -> pd.Series | None:
    """Row in ``test_df`` whose ``wt_uid`` column matches ``wt_uid``, or ``None``.

    Used by the Solver's saved-IPR pin (``prop_hist``'s ``ipr_wt_uid``, see
    ``woffl.assembly.prop_hist_client``) to check whether the pinned test is
    still present in the well's CURRENT test frame — the pin persists in
    Databricks indefinitely, but the frame is windowed (lookback months,
    memory-gauge coverage, count cap), so a pinned test can "age out" and
    this returns ``None`` for the caller to fall back to the normal default.

    Returns ``None`` when ``test_df`` is empty/``None``, the ``wt_uid``
    column is absent (older cached frames, synthetic test data in call sites
    that don't run through :mod:`woffl.assembly.well_test_client`), or no row
    matches. Manual/provisional test rows always carry ``NaN`` here (see
    ``utils._append_manual_tests``) so they can never match.
    """
    if test_df is None or test_df.empty or "wt_uid" not in test_df.columns:
        return None

    uid_col = pd.to_numeric(test_df["wt_uid"], errors="coerce")
    mask = uid_col == float(wt_uid)
    if not mask.any():
        return None

    matches = test_df[mask]
    if "WtDate" in matches.columns:
        matches = matches.sort_values("WtDate", ascending=False)
    return matches.iloc[0]


# ── Save/clear the IPR-anchor pin (mpu.wells.prop_hist, ipr_wt_uid) ─────────
# W3 of the woffl-prop-hist-persistence plan. This module is already a leaf
# both `tabs/jetpump_solver.py` (the Solver's Save/Clear buttons) and
# `workflow_steps/step_review_wells.py` (the pad-review save hook) import, so
# the actual push mechanics live here ONCE rather than being duplicated at
# both call sites. Only depends on `woffl.assembly.prop_hist_client` (no
# Streamlit, no `woffl.gui` imports) so it stays import-cycle-safe.

_IPR_PIN_PROP_ID = "ipr_wt_uid"
# NOTE: `wt_uid` values in `vw_well_test` are signed and span roughly
# -3.6M to +3.1M (almost all negative in practice) -- there is no numeric
# sentinel that can't collide with a real uid. Un-pinning writes a SQL NULL
# prop_value instead (see `clear_ipr_pin` / `prop_hist_client.push_prop`).

# Message prefixes callers pattern-match on to pick st.caption (expected,
# non-error skip) vs st.warning (an actual push/clear failure) without this
# module needing a Streamlit dependency or a richer return type.
PIN_SKIP_PREFIX = "IPR not saved to Databricks:"
PIN_FAILURE_PREFIX = "Could not save IPR to Databricks:"
UNPIN_FAILURE_PREFIX = "Could not clear saved IPR:"


def writes_enabled() -> bool:
    """Same ALLOW_DATABRICKS_WRITES truthy convention as
    `databricks_client._write_gate_enabled` / `scotts_tools/jp_calibration.py`
    -- re-checked here directly (rather than importing that private helper)
    so callers can decide whether to even SHOW a save/clear control before
    attempting a push (``push_prop`` enforces the same gate again on the
    actual write, via `databricks_client.execute_write`)."""
    return os.environ.get("ALLOW_DATABRICKS_WRITES", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _anchor_row_wt_uid(anchor_row) -> float | None:
    """A usable (non-NaN) ``wt_uid`` from an anchor row, or ``None``.

    Manual/provisional test rows (see ``utils._append_manual_tests``) always
    carry ``wt_uid = NaN`` -- never pinnable. Also guards a row with no
    `wt_uid` column at all (older cached frames, synthetic test data in
    call sites that don't route through `well_test_client`).
    """
    if anchor_row is None:
        return None
    raw = anchor_row.get("wt_uid") if hasattr(anchor_row, "get") else None
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _anchor_date_label(anchor_row) -> str:
    """Best-effort YYYY-MM-DD for an anchor row's WtDate, else 'n/a'."""
    if anchor_row is None:
        return "n/a"
    d = anchor_row.get("WtDate") if hasattr(anchor_row, "get") else None
    if d is None or pd.isna(d):
        return "n/a"
    try:
        return d.strftime("%Y-%m-%d")
    except AttributeError:
        return str(d)


def pin_ipr_anchor(well_name: str, anchor_row) -> tuple[bool, str]:
    """Push ``well_name``'s CURRENTLY-resolved IPR-anchor test as its saved
    default (``mpu.wells.prop_hist``'s ``ipr_wt_uid``).

    Shared by the pad-review save hook (``workflow_steps/step_review_wells.py
    ::_save_and_advance``) and the Solver's "Save IPR as well default" button
    (``tabs/jetpump_solver.py::_render_ipr_pin_controls``) so the two push
    paths can never diverge. ``anchor_row`` should come from
    ``jetpump_solver._resolve_anchor_test_row`` (or an equivalent resolver) --
    the row the anchor CURRENTLY points to, any mode.

    Callers are responsible for:
      * checking `writes_enabled()` BEFORE calling this (a gate-off session
        should show its own UI -- e.g. a hidden button -- rather than this
        function's generic skip message: `push_prop` still enforces the
        gate on the actual write, but a caller that skips the pre-check gets
        a `PIN_FAILURE_PREFIX` message instead of a clean UI state);
      * calling `jetpump_solver._clear_pin_cache(well_name)` immediately
        after a successful push, so the next render re-queries instead of
        replaying the pre-save memo;
      * calling this AFTER any store write / sidebar update has already
        succeeded -- pinning is best-effort and never rolls anything back.

    Returns ``(pushed, message)``:
      * ``(True, "📌 IPR saved to Databricks — test YYYY-MM-DD")`` on success.
      * ``(False, "IPR not saved to Databricks: ...")`` -- an EXPECTED,
        non-error skip (no resolvable test, or the anchor is a manual/
        provisional test with no ``wt_uid``). Show as a caption, never a
        warning -- check ``message.startswith(PIN_SKIP_PREFIX)``.
      * ``(False, "Could not save IPR to Databricks: ...")`` -- ``push_prop``
        itself raised (connection, enthid resolution, unknown prop_id, the
        write gate). Show as ``st.warning``.
    """
    wt_uid = _anchor_row_wt_uid(anchor_row)
    if wt_uid is None:
        reason = (
            "no resolvable test"
            if anchor_row is None
            else "this anchor is a manual/provisional test (no measured well-test ID)"
        )
        return False, f"{PIN_SKIP_PREFIX} {reason}."

    date_label = _anchor_date_label(anchor_row)
    try:
        entry_user = resolve_entry_user()
        push_prop(well_name, _IPR_PIN_PROP_ID, wt_uid, entry_user)
    except Exception as e:  # connection, enthid resolution, unknown prop_id, gate...
        return False, f"{PIN_FAILURE_PREFIX} {e}"

    return True, f"📌 IPR saved to Databricks — test {date_label}"


def clear_ipr_pin(well_name: str) -> tuple[bool, str]:
    """Un-pin ``well_name``'s saved IPR default: push a NULL prop_value.

    ``jetpump_solver._load_pinned_anchor`` treats a NULL/NaN ``prop_value``
    as "no pin" -- there is no numeric sentinel that works, since real
    ``wt_uid`` values are signed and span both positive and negative ranges.
    Corrections/un-pins are new rows (prop_hist is append-only), never a
    DELETE (see the plan's DART review). Same failure handling as
    :func:`pin_ipr_anchor`; callers should also call
    `jetpump_solver._clear_pin_cache(well_name)` after a successful clear.
    """
    try:
        entry_user = resolve_entry_user()
        push_prop(well_name, _IPR_PIN_PROP_ID, None, entry_user)
    except Exception as e:
        return False, f"{UNPIN_FAILURE_PREFIX} {e}"
    return True, "Saved IPR cleared"


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

    well = well_name or (str(df["well"].iloc[0]) if "well" in df.columns else "Well")

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
