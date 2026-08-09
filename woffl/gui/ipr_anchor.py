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
import time

import numpy as np
import pandas as pd

from woffl.assembly.prop_hist_client import (
    next_entry_datetime,
    push_eng_comment,
    push_prop,
    push_props,
    resolve_entry_user,
)
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

# "Cleared" markers. prop_hist.prop_value is DOUBLE **NOT NULL** — a NULL
# INSERT is rejected outright with DELTA_NOT_NULL_CONSTRAINT_VIOLATED. Both
# clear paths here used to push None, so neither had EVER worked: the whole
# table holds zero NULL prop_value rows, un-pinning silently failed, and
# unchecking a 🔒 lock silently failed (Scott on B-28, 2026-08-04 — the WC
# lock came back checked on reload because the unlock never landed).
#
# 0.0 is safe for BOTH, verified against live data 2026-08-04:
#   * `ipr_wt_uid` — `wt_uid = 0` occurs in 0 of 164,452 rows of
#     vw_well_test (range -3,598,569 .. 3,135,427). The old note here said no
#     sentinel was safe; that was true of -1 (a plausible uid) but not of 0.
#   * `*_lock` — only 1.0 has ever been stored, and the read predicate is
#     already `>= 0.5`, so 0.0 reads as unlocked with NO read-side change.
# Neither is a blanket "0 means cleared" rule: 0.0 is a legitimate value for
# props like form_wc, which is why these are named per use.
PIN_CLEARED_VALUE = 0.0
LOCK_UNLOCKED_VALUE = 0.0

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
    """Un-pin ``well_name``'s saved IPR default.

    Writes :data:`PIN_CLEARED_VALUE` (0.0), which readers treat as "no pin".
    This used to push a SQL NULL, which the table's NOT NULL constraint
    rejected — so the 🗑 Clear button had never actually cleared anything (see
    the block comment on PIN_CLEARED_VALUE). Corrections/un-pins are new rows
    (prop_hist is append-only), never a DELETE. Same failure handling as
    :func:`pin_ipr_anchor`; callers should also call
    `jetpump_solver._clear_pin_cache(well_name)` after a successful clear.
    """
    try:
        entry_user = resolve_entry_user()
        push_prop(well_name, _IPR_PIN_PROP_ID, PIN_CLEARED_VALUE, entry_user)
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


# ── Saved IPR VALUES (prop_hist Phase 2: the curve + rate the engineer chose) ─
# Scott's goal, 2026-07-30: "when a user opens a well in single review or
# optimization it has the latest edit so that it keeps the curve and rate the
# engineer saw fit." The five prop_xref ids below were added the same day
# (self-served under his catalog MODIFY — see docs/prop_hist_asks.md ask (c)).
# The pump identity and workflow flags are deliberately NOT here: pump truth
# stays with the JP tracker, per Scott.
#
# Precedence rule against the anchor pin (`ipr_wt_uid`): LATEST TIMESTAMP WINS.
# Same click writes both, so a normal "Save IPR as well default" makes the
# values and the pin agree; a later re-pin without values hands the well back
# to the pinned test; later value edits (a forced/manual IPR) beat the pin.

# The sidebar-seed signature the Solver treats as "already applied — leave the
# sidebar alone". Saved values are seeded UNDER this sig so they ride the
# manual-edit affordance: the anchor selector shows "Most recent" and the
# Solver's sync doesn't reseed until the engineer actively changes the anchor
# (which SHOULD override saved values). The Solver aliases this constant.
IPR_SIDEBAR_DEFAULT_SIG = ("recent", None)

# prop_id -> the load_saved_ipr values key. resvr_press rides along so the
# whole curve (qwf, pwf, ResP) restores atomically; ipr_wt_uid is read only
# for its timestamp (the precedence comparison).
IPR_VALUE_PROPS = {
    "ipr_qwf_liq": "qwf_liq",
    "ipr_pwf": "pwf",
    "form_wc": "form_wc",
    "form_gor": "form_gor",
    "surf_press": "surf_press",
    "resvr_press": "res_pres",
}

VALUES_SAVE_FAILURE_PREFIX = "Could not save IPR values:"

# WC lock (Scott, 2026-07-31): some wells' WELL-TEST water cut is known-bad
# (the washout/MPE-19 class — low rate, the PF return swamps the separator and
# allocation nets formation water to ~0). For those the engineer DECIDES the
# WC, and it must override every test-derived WC — on open, on anchor change,
# everywhere — until explicitly unlocked. 1.0 = locked; NULL = unlocked (the
# ipr_wt_uid un-pin precedent). Unlocked wells keep today's behavior: WC
# follows the chosen anchor test.
WC_LOCK_PROP = "form_wc_lock"

# The lock registry (Scott picked the set 2026-07-31): sidebar key →
# (lock prop_id, value prop_id, human label). Each lock means "the automated
# seed for this field is systematically wrong on this well — my saved value
# stands until I unlock". GOR shares WC's allocation-fiction problem (and
# makes the solver's session-only crash-recovery floor durable); ResP
# completes the weak-RP-fit flow ("DECIDE a pressure" — now the decision
# survives re-anchoring). All sit OUTSIDE the pin-vs-values precedence.
LOCKABLE_FIELDS = {
    "form_wc": ("form_wc_lock", "form_wc", "WC"),
    "form_gor": ("form_gor_lock", "form_gor", "GOR"),
    "res_pres": ("resvr_press_lock", "resvr_press", "ResP"),
}

# BHP-calibration friction coefficients (the Solver's calibrate-to-BHP flow
# writes these into the sidebar; Scott 2026-07-30: they must survive a save +
# reload like everything else). Already-whitelisted canon ids — the pivots
# serve them, so a saved calibration IS the well's characterization.
FRICTION_PROPS = {
    "jpfric_entry": "ken",
    "jpfric_throat": "kth",
    "jpfric_diffuser": "kdi",
}

# The sidebar's uncalibrated defaults (params.py / _render_loss_coefs). A save
# where friction still sits at these AND nothing is stored pushes nothing —
# otherwise every 📌 click would materialize explicit default rows well by
# well, turning "never calibrated" (NULL → library default) into fake data.
SIDEBAR_FRIC_DEFAULTS = {"ken": 0.03, "kth": 0.3, "kdi": 0.4}

_saved_ipr_cache: dict = {}


def clear_saved_ipr_cache(well_name: str | None = None) -> None:
    if well_name is None:
        _saved_ipr_cache.clear()
    else:
        _saved_ipr_cache.pop(well_name, None)


def saved_wins(saved_at, pin_at) -> bool:
    """Latest-timestamp-wins between saved VALUES and the anchor PIN.

    No pin → values win. Same instant (one Save wrote both) → values win,
    which is correct because they were seeded FROM that anchor anyway.
    """
    if saved_at is None:
        return False
    if pin_at is None:
        return True
    return saved_at >= pin_at


def _assemble_saved_ipr(latest: dict):
    """Build one well's saved-IPR record from its latest-row-per-prop_id map.

    Shared by the single-well read (:func:`load_saved_ipr`) and the pad-wide
    warm-up (:func:`warm_saved_ipr_cache`) so the two can never disagree about
    what a set of prop_hist rows MEANS. ``latest`` maps prop_id -> a row-like
    with ``prop_value`` / ``entry_datetime`` / ``entry_user``.
    """
    values, saved_at, saved_by = {}, None, None
    for pid, key in IPR_VALUE_PROPS.items():
        row = latest.get(pid)
        if row is None or pd.isna(row["prop_value"]):
            continue
        values[key] = float(row["prop_value"])
        if pid != "resvr_press":  # canon doesn't date the SAVED set
            ts = row["entry_datetime"]
            if saved_at is None or ts > saved_at:
                saved_at, saved_by = ts, row.get("entry_user")

    # The pin's VALUE and author ride along, not just its timestamp: this row
    # is already in hand, so a second per-well round trip for it
    # (jetpump_solver's old fetch_latest_prop call) was 150 ms of duplication.
    pin_row = latest.get("ipr_wt_uid")
    pin_live = (
        pin_row is not None
        and not pd.isna(pin_row["prop_value"])
        # 0.0 is the explicit "cleared" marker (PIN_CLEARED_VALUE); a real
        # wt_uid is never 0. NaN is still honoured for defensive reads.
        and float(pin_row["prop_value"]) != PIN_CLEARED_VALUE
    )

    friction = {}
    for pid, key in FRICTION_PROPS.items():
        row = latest.get(pid)
        if row is not None and not pd.isna(row["prop_value"]):
            friction[key] = float(row["prop_value"])

    locks, lock_values = {}, {}
    for skey, (lock_id, value_id, _label) in LOCKABLE_FIELDS.items():
        lock_row = latest.get(lock_id)
        locks[skey] = (
            lock_row is not None
            and not pd.isna(lock_row["prop_value"])
            and float(lock_row["prop_value"]) >= 0.5
        )
        value_row = latest.get(value_id)
        if value_row is not None and not pd.isna(value_row["prop_value"]):
            lock_values[skey] = float(value_row["prop_value"])

    # A saved IPR needs the operating pair; friction, the locks, and a bare pin
    # stand on their own (a calibration, a locked field, or a 📌-only well may
    # exist with no saved curve, and all sit OUTSIDE the pin-vs-values
    # precedence). The pin has to count or a pin-only well would return None
    # and lose its anchor.
    has_curve = "qwf_liq" in values and "pwf" in values
    if not (has_curve or friction or any(locks.values()) or pin_live):
        return None
    return {
        "values": values if has_curve else {},
        "friction": friction,
        "locks": locks,
        "lock_values": lock_values,
        # back-compat aliases (pre-registry callers/tests)
        "wc_locked": locks["form_wc"],
        "wc_value": lock_values.get("form_wc"),
        "saved_at": saved_at if has_curve else None,
        "saved_by": saved_by,
        "pin_at": pin_row["entry_datetime"] if pin_live else None,
        "pin_value": float(pin_row["prop_value"]) if pin_live else None,
        "pin_user": str(pin_row.get("entry_user") or "") if pin_live else None,
    }


def _saved_ipr_prop_ids() -> list:
    """Every prop_id a saved-IPR record is assembled from."""
    return (
        list(IPR_VALUE_PROPS)
        + list(FRICTION_PROPS)
        + ["ipr_wt_uid"]
        + [lock_id for lock_id, _, _ in LOCKABLE_FIELDS.values()]
    )


def load_saved_ipr(well_name: str):
    """Latest saved IPR values + the anchor pin for one well, or None.

    One latest-per-prop query over prop_hist (memoized per well per session —
    cleared by :func:`save_ipr_values`, warmed pad-wide by
    :func:`warm_saved_ipr_cache`). Read path: works without the write gate,
    and hosted via the SP's inherited SELECT. A saved-IPR record only COUNTS
    when both ``ipr_qwf_liq`` and ``ipr_pwf`` are present — canonical
    ``resvr_press`` alone is well characterization, not a saved curve.

    Returns the :func:`_assemble_saved_ipr` record (values, friction, locks,
    pin) or None. Fail-soft: any error → None (the sidebar seeds normally).
    """
    if well_name in _saved_ipr_cache:
        return _saved_ipr_cache[well_name]

    result = None
    try:
        from woffl.assembly.databricks_client import execute_query
        from woffl.assembly.prop_hist_client import _resolve_enthid

        enthid = _resolve_enthid(well_name)
        ids = ",".join(f"'{p}'" for p in _saved_ipr_prop_ids())
        df = execute_query(
            f"""
            SELECT prop_id, prop_value, entry_datetime, entry_user FROM (
                SELECT prop_id, prop_value, entry_datetime, entry_user,
                       ROW_NUMBER() OVER (
                           PARTITION BY prop_id ORDER BY entry_datetime DESC
                       ) AS rn
                FROM mpu.wells.prop_hist
                WHERE enthid = {int(enthid)} AND prop_id IN ({ids})
            ) WHERE rn = 1
            """
        )
        if df is not None and not df.empty:
            result = _assemble_saved_ipr(
                {str(r["prop_id"]): r for _, r in df.iterrows()}
            )
    except Exception:
        result = None

    _saved_ipr_cache[well_name] = result
    return result


_WARM_TTL_SECONDS = 300.0
_warmed_at: float = 0.0


def warm_saved_ipr_cache(force: bool = False) -> int:
    """Fill :data:`_saved_ipr_cache` for EVERY well in ONE query.

    ``load_saved_ipr`` is one Databricks round trip per well, and a pad review
    walks ~20 wells — 20 x 150 ms of pure latency (measured 2026-08-04) before
    the engineer touches anything, four pads' worth on the CFP page. This is
    the same latest-per-(enthid, prop_id) window function partitioned across
    every well instead of one, which the pad page already proves is cheap
    (``review_persistence._latest_props`` runs that shape every render).

    Not pad-scoped: the query cost is the same either way (the window function
    reads prop_hist regardless), so scoping would just mean re-running it per
    pad. Guarded by a process-wide TTL instead, which makes the repeat calls
    on every rerun — and the CFP page's four pads — free.

    Wells with NO matching rows are cached as ``None`` too; otherwise each
    would still pay its own round trip to discover it has nothing.

    Never overwrites an existing entry: a live per-well read, or a save that
    just called :func:`clear_saved_ipr_cache`, always wins over this snapshot.

    Fail-soft: any error leaves the cache untouched and the per-well path
    takes over. Returns the number of wells warmed.
    """
    global _warmed_at

    now = time.monotonic()
    if not force and _warmed_at and (now - _warmed_at) < _WARM_TTL_SECONDS:
        return 0

    try:
        from woffl.assembly.databricks_client import execute_query
        from woffl.assembly.well_test_client import _normalize_well_name

        ids = ",".join(f"'{p}'" for p in _saved_ipr_prop_ids())
        df = execute_query(
            f"""
            SELECT h.well_name, p.prop_id, p.prop_value,
                   p.entry_datetime, p.entry_user
            FROM (
                SELECT enthid, prop_id, prop_value, entry_datetime, entry_user,
                       ROW_NUMBER() OVER (
                           PARTITION BY enthid, prop_id
                           ORDER BY entry_datetime DESC
                       ) AS rn
                FROM mpu.wells.prop_hist
                WHERE prop_id IN ({ids})
            ) p
            JOIN mpu.wells.vw_well_header h ON p.enthid = h.enthid
            WHERE p.rn = 1
            """
        )
        # Stamp only on success, so a failed warm-up retries next render
        # instead of going quiet for the whole TTL.
        if df is None or df.empty:
            return 0

        by_well: dict = {}
        for _, r in df.iterrows():
            well = _normalize_well_name(str(r["well_name"]).strip())
            by_well.setdefault(well, {})[str(r["prop_id"])] = r

        warmed = 0
        for well, latest in by_well.items():
            if well in _saved_ipr_cache:
                continue  # a live per-well read (or a save) already won
            _saved_ipr_cache[well] = _assemble_saved_ipr(latest)
            warmed += 1
        _warmed_at = now
        return warmed
    except Exception:
        return 0


def save_ipr_values(
    well_name: str,
    *,
    qwf_liq: float,
    pwf: float,
    res_pres: float,
    form_wc: float,
    form_gor: float,
    surf_pres: float | None = None,
    ken: float | None = None,
    kth: float | None = None,
    kdi: float | None = None,
    bubble_point: float | None = None,
    form_temp: float | None = None,
    comment: str | None = None,
) -> tuple[int, str]:
    """Push the sidebar's CURRENT IPR + fluid values as the well's saved curve.

    ``qwf_liq`` is the TOTAL LIQUID rate at ``pwf`` (BLPD — formation oil plus
    formation water) and lands in ``ipr_qwf_liq`` VERBATIM. It used to be the
    sidebar's OIL rate, converted here as ``oil / (1 - wc)``; that made the
    stored number a DERIVED quantity matching no well test (Scott on B-28,
    2026-08-03: the stored 2135.29 BLPD was really 363 BOPD ÷ (1 − 0.83), and
    because that WC is locked, editing it silently rescaled the engineer's
    anchor). Liquid is what a well test measures, so liquid is what we store;
    oil is derived from it at read time. ``form_wc`` is still capped at 0.99
    for its OWN payload row (the store's MAX_MODELABLE_WC precedent — the
    downstream oil = liq * (1 - wc) degenerates at 1.0).
    ``resvr_press`` is pushed too so the whole curve travels together — note
    that updates the well's CANONICAL reservoir pressure via the pivots, which
    is the point: the reviewed value IS the characterization.
    ``bubble_point`` and ``form_temp`` are the same kind of thing and the same
    deal: canonical characterization (``resvr_bubb`` / ``resvr_temp``), pushed
    ONLY when the caller passes them. The Solver sends one only after the
    engineer moved it off the seeded value - a sensitivity permutation varies
    both - so a routine save never re-writes the characterization it was given.

    Every property of one save shares a single ``entry_datetime``. That stamp
    is the save's only batch identity, and ``comment`` — the engineer's note on
    WHY these values were chosen — is attached to it in
    ``mpu.wells.woffl_eng_comment``. Two saves on the same day stay separate,
    because the grain is the timestamp rather than the date.

    Returns ``(n_pushed, message)``; failure message starts with
    :data:`VALUES_SAVE_FAILURE_PREFIX`. Clears the load memo on success.
    """
    try:
        wc = min(max(float(form_wc), 0.0), 0.99)
        payload = {
            "ipr_qwf_liq": float(qwf_liq),
            "ipr_pwf": float(pwf),
            "form_wc": wc,
            "form_gor": float(form_gor),
            "surf_press": float(surf_pres) if surf_pres is not None else None,
            "resvr_press": float(res_pres),
            # Canonical characterization values, saved ONLY when the caller
            # passes them. The web client sends one only after the engineer
            # moved it off the seeded value (a sensitivity permutation can
            # vary both), so an ordinary save never re-writes the
            # characterization it was handed - the same discipline the
            # friction coefficients follow below.
            "resvr_bubb": float(bubble_point) if bubble_point is not None else None,
            "resvr_temp": float(form_temp) if form_temp is not None else None,
        }
        # Friction rides along only when it carries information: skip when it
        # matches the stored latest (no history noise from an unchanged save)
        # or when nothing is stored AND the sidebar still sits at the
        # uncalibrated defaults (never materialize defaults as canon rows —
        # the BHP-calibration survival fix, Scott 2026-07-30).
        stored = load_saved_ipr(well_name)
        stored_fric = (stored or {}).get("friction", {})
        for key, val in (("ken", ken), ("kth", kth), ("kdi", kdi)):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val = float(val)
            prev = stored_fric.get(key)
            if prev is not None and abs(prev - val) < 1e-9:
                continue
            if prev is None and abs(val - SIDEBAR_FRIC_DEFAULTS[key]) < 1e-9:
                continue
            pid = next(p for p, k in FRICTION_PROPS.items() if k == key)
            payload[pid] = val

        entry_user = resolve_entry_user()
        # ONE stamp for the whole save — the rows' only batch identity, and what
        # the engineer's comment joins on. Allocated (not `datetime.now()`) so a
        # second save inside the same 15.6 ms clock tick gets a strictly LATER
        # stamp instead of colliding with this one: see next_entry_datetime.
        batch_stamp = next_entry_datetime()
        # ONE statement for the whole save. It used to be one INSERT per
        # property in a Python loop: 6-9 serialized Delta commits, which is
        # what made the button hang for seconds (measured 2026-08-08: ~0.16 s
        # of pure protocol per statement on a warm connection, before Delta's
        # own commit cost). push_props validates every row before sending, so
        # a rejected prop can no longer leave a half-written save behind.
        n = push_props(
            well_name,
            {
                pid: val
                for pid, val in payload.items()
                if val is not None and not (isinstance(val, float) and np.isnan(val))
            },
            entry_user,
            entry_datetime=batch_stamp,
        )
        clear_saved_ipr_cache(well_name)

        # Comment LAST and best-effort: the properties are already committed,
        # so a comment failure must degrade to "saved without a note", never
        # undo the save. Skipped when nothing was written — a note explaining
        # changes that didn't happen would be worse than none (decision A).
        note = (comment or "").strip()
        comment_note = ""
        if note and n:
            try:
                push_eng_comment(well_name, batch_stamp, entry_user, note)
                comment_note = " · comment saved"
            except Exception as e:  # noqa: BLE001 — never break a good save
                comment_note = f" · ⚠️ comment not saved ({e})"
        elif note:
            comment_note = " · comment skipped (nothing changed to attach it to)"

        fric_note = (
            " + BHP-calibrated friction"
            if any(pid in payload for pid in FRICTION_PROPS)
            else ""
        )
        return n, (
            f"💾 Saved IPR values for {well_name} "
            # The stored rate is the engineer's MEASURED liquid; the oil it
            # implies at this WC is spelled out so a locked or edited WC is
            # visible in the confirmation instead of quietly moving the
            # anchor (Scott on B-28, 2026-08-03).
            f"(qwf {float(qwf_liq):,.0f} BLPD → "
            f"{float(qwf_liq) * (1.0 - wc):,.0f} BOPD at this WC · "
            f"pwf {pwf:,.0f} · ResP {res_pres:,.0f} "
            f"· WC {wc:.2f} · GOR {form_gor:,.0f}{fric_note}) — restores on "
            f"every open.{comment_note}"
        )
    except Exception as e:
        return 0, f"{VALUES_SAVE_FAILURE_PREFIX} {e}"


def set_prop_lock(
    well_name: str, field: str, locked: bool, value: float | None = None
) -> tuple[bool, str]:
    """Set or clear a per-well field lock (see :data:`LOCKABLE_FIELDS`).

    Locking also pushes the CURRENT sidebar value when given, so the locked
    value is pinned in the same click. Unlocking writes
    :data:`LOCK_UNLOCKED_VALUE` — the field goes back to following the
    automated seed.
    """
    try:
        lock_id, value_id, label = LOCKABLE_FIELDS[field]
    except KeyError:
        return False, f"Could not update the {field} lock: not a lockable field."
    try:
        entry_user = resolve_entry_user()
        if locked:
            # Value and lock flag go in ONE statement, sharing one stamp: they
            # are one click, and a lock row that landed without its value row
            # would pin whatever was there before.
            rows: dict[str, float | None] = {}
            if value is not None:
                v = float(value)
                if field == "form_wc":
                    v = min(max(v, 0.0), 0.99)
                rows[value_id] = v
            rows[lock_id] = 1.0
            push_props(well_name, rows, entry_user)
            message = (
                f"🔒 {label} locked for {well_name} — the saved value now "
                "overrides every test-derived seed until unlocked."
            )
        else:
            # 0.0, not NULL: prop_value is NOT NULL, so the old None push was
            # rejected and the unlock never persisted. The read predicate is
            # `>= 0.5`, so 0.0 already means unlocked.
            push_prop(well_name, lock_id, LOCK_UNLOCKED_VALUE, entry_user)
            message = (
                f"🔓 {label} unlocked for {well_name} — it follows the "
                "chosen anchor / fit again."
            )
        clear_saved_ipr_cache(well_name)
        return True, message
    except Exception as e:
        return False, f"Could not update the {label} lock: {e}"


def set_wc_lock(
    well_name: str, locked: bool, form_wc: float | None = None
) -> tuple[bool, str]:
    """Back-compat wrapper — the WC lock through the generic registry."""
    return set_prop_lock(well_name, "form_wc", locked, value=form_wc)
