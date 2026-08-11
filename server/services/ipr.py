"""IPR Vogel fitting and the saved-IPR anchor pin.

Fit path mirrors the Solver tab's _render_ipr_anchor_and_seed: "recent" uses
the global least-squares library fit (estimate_reservoir_pressure +
compute_vogel_coefficients); "median"/"median_liq"/"specific" use the
GUI-layer anchored fit (ipr_anchor.compute_anchored_vogel) with the field RP
cap.

Pin path reads prop_hist via ipr_anchor.load_saved_ipr (read-only; no write
gate needed) and grades the pin against the current test window.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from server import config, schemas
from server.cache import ttl_cache
from server.services import frames, tests

from woffl.gui import ipr_anchor

# mirrors woffl/gui/tabs/jetpump_solver.py:_WEAK_IPR_R2 - below this the test
# cloud cannot constrain reservoir pressure (negative = worse than a flat
# mean) and the engineer is told to decide a pressure themselves.
_WEAK_IPR_R2 = 0.2

_FIT_ERROR = "need >=2 tests with BHP for a Vogel fit"

# Seed clamp bounds - mirrors woffl/gui/sidebar.py:clamp_seed intent (a seed
# must never land outside its widget's bounds); values are the
# schemas.SimParams Field bounds so a seeded request always re-validates.
_SEED_BOUNDS: dict[str, tuple[float, float]] = {
    "qwf": (10.0, 20000.0),
    "pwf": (100.0, 2500.0),
    "pres": (400.0, 5000.0),
    "form_wc": (0.0, 1.0),
    "form_gor": (20.0, 10000.0),
    "surf_pres": (10.0, 600.0),
}


def _clamp_seed(key: str, value: float) -> float:
    lo, hi = _SEED_BOUNDS[key]
    return min(max(float(value), lo), hi)


def _anchor_test_row(usable: pd.DataFrame, row: dict[str, Any]) -> Optional[pd.Series]:
    """The test row the fit anchored on, for whp/surf_pres seeding.

    The anchored/recent fits both take their (qwf, pwf) pair verbatim from
    one test row, so match it back by that pair; ties fall to the newest.
    """
    matches = usable[
        (usable["BHP"].astype(float) == float(row["pwf"]))
        & (usable["WtTotalFluid"].astype(float) == float(row["qwf"]))
    ]
    if matches.empty:
        return None
    if "WtDate" in matches.columns:
        matches = matches.sort_values("WtDate", ascending=False)
    return matches.iloc[0]


def _apply_bhp_overrides(df: pd.DataFrame, overrides: list[schemas.GaugeDay]) -> pd.DataFrame:
    """Lay memory-gauge daily medians over the test frame's BHP.

    Same mechanics as memory_gauge.apply_to_well_tests - called through it,
    duck-typing the gauge (it only reads ``.daily_df``) - so the stateless
    fit path and the Streamlit session path can never disagree.
    """
    from types import SimpleNamespace

    from woffl.gui.memory_gauge import apply_to_well_tests

    daily = pd.DataFrame(
        {
            "tag_date": [pd.Timestamp(o.date).normalize() for o in overrides],
            "bhp": [float(o.bhp) for o in overrides],
        }
    )
    return apply_to_well_tests(df, SimpleNamespace(daily_df=daily))  # type: ignore[arg-type]


def fit(req: schemas.IprFitRequest) -> dict[str, Any]:
    """Fit a Vogel IPR for one well (IprFitResponse shape).

    Args:
        req: well, anchor mode/date, field model, test window.

    Returns:
        IprFitResponse dict: coeffs + sidebar seeds + weak flag.

    Raises:
        ValueError: fewer than 2 usable (BHP + WtTotalFluid) tests, or the
            fit itself failed (router maps to 422 "invalid").
    """
    df = tests.tests_for_well(req.well, req.months, req.cap)
    if df is None or df.empty:
        raise ValueError(_FIT_ERROR)
    if req.bhp_overrides:
        df = _apply_bhp_overrides(df, req.bhp_overrides)
    usable = df.dropna(subset=["BHP", "WtTotalFluid"])
    if len(usable) < 2:
        raise ValueError(_FIT_ERROR)

    row: Optional[dict[str, Any]] = None
    if req.anchor_mode in ("median", "median_liq", "specific"):
        # mirrors woffl/gui/tabs/jetpump_solver.py:_render_ipr_anchor_and_seed
        # (anchored branch): field RP cap 1800 Schrader / 3000 Kuparuk.
        field_max_rp = 3000 if req.field_model == "Kuparuk" else 1800
        row = ipr_anchor.compute_anchored_vogel(
            df,
            well_name=req.well,
            anchor_mode=req.anchor_mode,
            anchor_date=req.anchor_date,
            field_max_rp=field_max_rp,
        )
        if row is None:
            raise ValueError(_FIT_ERROR)
    else:
        # Most-recent / global least-squares fit (unchanged library path).
        from woffl.assembly.ipr_analyzer import (
            compute_vogel_coefficients,
            estimate_reservoir_pressure,
        )

        merged = estimate_reservoir_pressure(df)
        coeffs = compute_vogel_coefficients(merged)
        if coeffs.empty or "Well" not in coeffs.columns:
            raise ValueError(_FIT_ERROR)
        well_coeffs = coeffs[coeffs["Well"] == req.well]
        if well_coeffs.empty:
            raise ValueError(_FIT_ERROR)
        row = well_coeffs.iloc[0].to_dict()

    res_p = int(row["ResP"])
    qwf = float(row["qwf"])
    pwf = float(row["pwf"])
    form_wc = float(row["form_wc"])
    fgor = float(row["fgor"])
    r2 = frames.opt_float(row.get("R2"))

    qmax: Optional[float] = None
    try:
        from woffl.flow.inflow import InFlow

        qmax = frames.opt_float(InFlow.vogel_qmax(qwf, pwf, res_p))
    except Exception:
        qmax = None

    seeds: dict[str, Any] = {
        "qwf": _clamp_seed("qwf", qwf),
        "pwf": _clamp_seed("pwf", pwf),
        "pres": _clamp_seed("pres", float(res_p)),
        "form_wc": _clamp_seed("form_wc", form_wc),
        "form_gor": _clamp_seed("form_gor", fgor),
    }
    anchor_row = _anchor_test_row(usable, row)
    if anchor_row is not None:
        whp = frames.opt_float(anchor_row.get("whp"))
        if whp is not None and whp > 0:
            seeds["surf_pres"] = _clamp_seed("surf_pres", whp)

    coeffs_out = {
        "res_p": float(res_p),
        "qmax": qmax,
        "qwf": qwf,
        "pwf": pwf,
        "form_wc": form_wc,
        "fgor": fgor,
        "r2": r2,
        "num_tests": int(row["num_tests"]),
        "most_recent_date": frames.json_value(row.get("most_recent_date")),
        "anchor_label": row.get("anchor_label"),
        "anchor_date": frames.json_value(row.get("anchor_date")),
    }

    return {
        "well": req.well,
        "coeffs": coeffs_out,
        "seeds": seeds,
        "weak": r2 is not None and r2 < _WEAK_IPR_R2,
    }


# load_saved_ipr memoizes per-well FOREVER (a Streamlit session cache); on a
# long-lived server that would pin stale forever, so clear its entry and
# re-read under our own 5-minute TTL (config.TTL_SAVED_IPR).
@ttl_cache(config.TTL_SAVED_IPR, maxsize=256)
def _saved_ipr(well: str) -> Optional[dict[str, Any]]:
    ipr_anchor.clear_saved_ipr_cache(well)
    return ipr_anchor.load_saved_ipr(well)


def pin(well: str) -> dict[str, Any]:
    """Saved IPR-anchor pin status for one well (IprPinResponse shape).

    Statuses:
        none: no pin saved (or prop_hist unreachable - fail-soft).
        applied: the pinned wt_uid is in the current 6-month test window.
        stale: a pin exists but its test aged out of the window.

    Args:
        well: GUI well name.

    Returns:
        IprPinResponse dict.
    """
    none_resp: dict[str, Any] = {
        "status": "none",
        "wt_uid": None,
        "date_token": None,
        "entry_user": None,
        "entry_datetime": None,
    }

    try:
        saved = _saved_ipr(well)
    except Exception:
        return none_resp
    if not saved:
        return none_resp

    pin_value = saved.get("pin_value")
    # load_saved_ipr already treats PIN_CLEARED_VALUE as no-pin; re-check
    # defensively so a cleared/NaN value can never read as a live pin.
    if (
        pin_value is None
        or not math.isfinite(float(pin_value))
        or float(pin_value) == ipr_anchor.PIN_CLEARED_VALUE
    ):
        return none_resp

    entry_user = saved.get("pin_user") or None
    pin_at = saved.get("pin_at")
    entry_datetime = None if pin_at is None or pd.isna(pin_at) else str(pin_at)

    df = tests.tests_for_well(well, 6, 0)
    match = ipr_anchor.find_test_row_by_wt_uid(df, int(float(pin_value)))
    if match is not None:
        return {
            "status": "applied",
            "wt_uid": float(pin_value),
            "date_token": frames.json_value(match.get("WtDate")),
            "entry_user": entry_user,
            "entry_datetime": entry_datetime,
        }
    return {
        "status": "stale",
        "wt_uid": float(pin_value),
        "date_token": None,
        "entry_user": entry_user,
        "entry_datetime": entry_datetime,
    }


# ---------------------------------------------------------------------------
# Writes - the Solver's "Save as well default" / "Clear saved IPR"
# ---------------------------------------------------------------------------
#
# All mechanics live in woffl.gui.ipr_anchor (pin_ipr_anchor /
# save_ipr_values / clear_ipr_pin) - the SAME functions the Streamlit button
# calls - so the rules can never diverge: prop_xref whitelist, as-built
# physical properties rejected outright (AS_BUILT_PROP_IDS), WC capped at
# 0.99, friction pushed only when changed and never as materialized
# defaults, one batch stamp per save, comment best-effort. Routers pre-check
# the write gate; push_prop enforces it again on the actual INSERT.


def _invalidate_after_write(well: str) -> None:
    """Drop every cache layer a successful prop_hist write just outdated, so
    the NEXT poll reads the new rows instead of waiting out a 5-minute TTL
    (two-client review sessions poll the pad board while the other client
    saves - client-side cache invalidation can never cover that).

    Call ONLY after the write reported success: a failed write changed
    nothing, and evicting on failure cold-starts reads for no gain.

    - woffl memo: save_ipr_values / set_prop_lock clear it themselves, but a
      pin/un-pin push does not (its Streamlit caller owns that), so clear it
      here too or a _pad_fit recompute after "pin landed, values failed"
      would still read the pre-pin memo. Idempotent when already cleared.
    - _saved_ipr: this well only - a fleet-wide clear made one engineer's
      save cost every other well a cold prop_hist SELECT on its next read.
    - _pad_fit: full clear - a donor well shows up on other pads' boards,
      and a recompute is one warm_saved_ipr_cache snapshot.
    - _prop_history (the audit page): this well's enthid entry.
    """
    ipr_anchor.clear_saved_ipr_cache(well)
    _saved_ipr.cache_evict(well)
    _pad_fit.cache_clear()
    from server.services import database as database_service

    database_service.evict_prop_history(well)


def save(well: str, req: schemas.SaveIprRequest) -> dict[str, Any]:
    """Pin the anchor test (when given) and save the sidebar values.

    Mirrors the Streamlit click exactly: pin first, values second, same
    latest-timestamp precedence. Returns SaveIprResponse shape; per-part
    failures ride in the messages (the Streamlit toast/warning contract)
    rather than failing the whole request.
    """
    pinned = False
    pin_skipped = False
    unpinned = False
    pin_message: Any = None
    if req.unpin:
        # Manual point: the values are about to say "this is the curve", and a
        # surviving pin would make the next open read them as test-anchored.
        # Cleared BEFORE the values so the values carry the later stamp and
        # win the precedence (ipr_anchor.saved_wins).
        unpinned, pin_message = ipr_anchor.clear_ipr_pin(well)
    elif req.pin_wt_uid is not None:
        anchor_row = {"wt_uid": req.pin_wt_uid, "WtDate": req.pin_date}
        pinned, pin_message = ipr_anchor.pin_ipr_anchor(well, anchor_row)
        pin_skipped = not pinned and str(pin_message).startswith(ipr_anchor.PIN_SKIP_PREFIX)

    n_values, values_message = ipr_anchor.save_ipr_values(
        well,
        qwf_liq=req.qwf_liq,
        pwf=req.pwf,
        res_pres=req.res_pres,
        form_wc=req.form_wc,
        form_gor=req.form_gor,
        surf_pres=req.surf_pres,
        ken=req.ken,
        kth=req.kth,
        kdi=req.kdi,
        nozzle_area_factor=req.nozzle_area_factor,
        mach_crit=req.mach_crit,
        bubble_point=req.bubble_point,
        form_temp=req.form_temp,
        comment=req.comment,
    )

    # Invalidate ONLY when something landed: a failed write changed nothing,
    # and evicting on failure would cold-start reads for no gain. Partial
    # success (pin landed, values failed - or the reverse) still outdates
    # every cached read of this well.
    if pinned or unpinned or n_values:
        _invalidate_after_write(well)
    return {
        "pinned": pinned,
        "pin_skipped": pin_skipped,
        "pin_message": pin_message,
        "n_values": n_values,
        "values_message": values_message,
    }


def clear_pin(well: str) -> dict[str, Any]:
    """Un-pin the saved IPR default (appends the cleared marker row)."""
    cleared, message = ipr_anchor.clear_ipr_pin(well)
    if cleared:
        _invalidate_after_write(well)
    return {"cleared": cleared, "message": message}


def set_lock(well: str, req: schemas.PropLockRequest) -> dict[str, Any]:
    """Toggle one of the WC/GOR/ResP field locks.

    Mechanics live in ipr_anchor.set_prop_lock (the Streamlit checkbox's
    function): locking pushes the current sidebar value THEN the 1.0 lock
    row; unlocking pushes the 0.0 unlocked marker. On failure the previous
    state is reported back so the client's toggle stays truthful.
    """
    ok, message = ipr_anchor.set_prop_lock(well, req.field, req.locked, value=req.value)
    if ok:
        _invalidate_after_write(well)
    # Echo the value the way it was actually stored (set_prop_lock caps WC).
    value = req.value
    if value is not None and req.field == "form_wc":
        value = min(max(value, 0.0), 0.99)
    return {
        "ok": ok,
        "message": message,
        "field": req.field,
        "locked": req.locked if ok else not req.locked,
        "value": value if ok and req.locked else None,
    }


# ---------------------------------------------------------------------------
# Optimization pad board - saved-fit readiness per well
# ---------------------------------------------------------------------------


def _fit_row(well: str, pad: str, info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """PadFitWell shape from one _assemble_saved_ipr record (or None)."""
    if not info:
        return {
            "well": well,
            "pad": pad,
            "has_curve": False,
            "saved_at": None,
            "saved_by": None,
            "has_friction": False,
            "friction_keys": [],
            "locks": {},
            "pin_at": None,
            "pin_user": None,
        }
    saved_at = info.get("saved_at")
    pin_at = info.get("pin_at")
    return {
        "well": well,
        "pad": pad,
        "has_curve": bool(info.get("values")),
        "saved_at": None if saved_at is None or pd.isna(saved_at) else str(saved_at),
        "saved_by": info.get("saved_by") or None,
        "has_friction": bool(info.get("friction")),
        "friction_keys": sorted((info.get("friction") or {}).keys()),
        "locks": {k: bool(v) for k, v in (info.get("locks") or {}).items()},
        "pin_at": None if pin_at is None or pd.isna(pin_at) else str(pin_at),
        "pin_user": info.get("pin_user") or None,
    }


@ttl_cache(config.TTL_SAVED_IPR, maxsize=32)
def _pad_fit(pad: str, extra: tuple[str, ...]) -> dict[str, Any]:
    """PadFitStatusResponse payload: every well on ``pad`` + the requested
    donor wells (any pad). One fleet-wide prop_hist snapshot via
    warm_saved_ipr_cache makes the per-well load_saved_ipr calls free; a
    failed warm degrades to per-well reads. Cleared by every write
    (save / clear-pin / lock toggle) so the board reflects saves at once.
    """
    from server.services import wells as wells_svc

    universe = wells_svc.list_wells()["wells"]
    pad_wells = [w for w in universe if w.get("pad") == pad]

    ipr_anchor.warm_saved_ipr_cache()

    def status(name: str, well_pad: str) -> dict[str, Any]:
        try:
            info = ipr_anchor.load_saved_ipr(name)
        except Exception:
            info = None
        return _fit_row(name, well_pad, info)

    pad_by_name = {w["name"]: w.get("pad", "") for w in universe}
    return {
        "pad": pad,
        "wells": [status(w["name"], pad) for w in pad_wells],
        "extras": [status(name, pad_by_name.get(name, "")) for name in extra],
    }


def pad_fit_status(pad: str, extra: list[str]) -> dict[str, Any]:
    """List-arg wrapper (ttl_cache needs hashable keys)."""
    return _pad_fit(pad, tuple(sorted(set(extra))))
