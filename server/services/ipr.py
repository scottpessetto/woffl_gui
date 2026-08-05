"""IPR Vogel fitting and the saved-IPR anchor pin.

Fit path mirrors the Solver tab's _render_ipr_anchor_and_seed: "recent" uses
the global least-squares library fit (estimate_reservoir_pressure +
compute_vogel_coefficients); "median"/"specific" use the GUI-layer anchored
fit (ipr_anchor.compute_anchored_vogel) with the field RP cap.

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
    usable = df.dropna(subset=["BHP", "WtTotalFluid"])
    if len(usable) < 2:
        raise ValueError(_FIT_ERROR)

    row: Optional[dict[str, Any]] = None
    if req.anchor_mode in ("median", "specific"):
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
