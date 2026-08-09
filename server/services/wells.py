"""Well list, selection context, and survey-based well profile.

``well_context`` is the server-side replay of the Streamlit sidebar's
seeding pipeline (woffl/gui/sidebar.py:_update_well_parameters_from_data)
in the SAME order: chars -> pump history -> IPR fit -> saved-IPR overlay ->
live PF. Every numeric seed passes a finite check so NaN never reaches the
JSON response (the SPA applies ``seeds`` wholesale over SimParams defaults).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from server import config
from server.cache import ttl_cache
from server.schemas import NOZZLE_OPTIONS, THROAT_OPTIONS
from server.services import datasources, frames
from server.services import tests as tests_svc

log = logging.getLogger("woffl.web.wells")


# Widget bounds for programmatically seeded fields, keyed by the SimParams
# field name (the sidebar's "res_pres" is SimParams "pres"). Kept in sync with
# schemas.SimParams Field bounds so a seed can never fail response-side
# validation on the client store.
# mirrors woffl/gui/sidebar.py:SEED_BOUNDS
_SEED_BOUNDS: dict[str, tuple[float, float]] = {
    "qwf": (10, 20000),
    "pwf": (100, 2500),
    "pres": (400, 5000),
    "form_wc": (0.0, 1.0),
    "form_gor": (20, 10000),
    "form_temp": (32, 350),
    "surf_pres": (10, 600),
    "ppf_surf": (800, 5500),
    "ken": (0.001, 0.40),
    "kth": (0.05, 1.0),
    "kdi": (0.05, 1.0),
    "jpump_tvd": (2500, 8000),
    "oil_api": (11.0, 39.0),
    "bubble_point": (1001.0, 2999.0),
    "gas_sg": (0.51, 1.19),
    "wat_sg": (0.51, 1.49),
    "tubing_od": (2.0, 9.0),
    "tubing_thickness": (0.1, 2.0),
    "casing_od": (4.0, 17.0),
    "casing_thickness": (0.1, 2.0),
}

# Pad-level default PF surface pressures (psi). C/E/H/I/M/S run at 3400,
# B/G/J at 2200 (booster pads), F at 2800. Pad K has no jet pumps.
# mirrors woffl/gui/utils.py:PAD_PF_DEFAULTS
_PAD_PF_DEFAULTS: dict[str, int] = {
    "B": 2200,
    "C": 3400,
    "E": 3400,
    "F": 2800,
    "G": 2200,
    "H": 3400,
    "I": 3400,
    "J": 2200,
    "M": 3400,
    "S": 3400,
}
_PAD_PF_FALLBACK = 3400

# Sidebar lock key -> the SimParams seed key it overrides.
_LOCK_SEED_KEYS: dict[str, str] = {
    "form_wc": "form_wc",
    "form_gor": "form_gor",
    "res_pres": "pres",
}

# Raw characteristics keys surfaced to the client (NaN -> null).
_CHARS_KEYS: tuple[str, ...] = (
    "Well",
    "is_sch",
    "JP_MD",
    "JP_TVD",
    "tvd_estimated",
    "out_dia",
    "thick",
    "casing_out_dia",
    "casing_inn_dia",
    "res_pres",
    "form_temp",
    "oil_api",
    "gas_sg",
    "wat_sg",
    "bubble_point",
)

_PROFILE_MAX_POINTS = 1500


# ---------------------------------------------------------------------------
# Small mirrored helpers
# ---------------------------------------------------------------------------


def _clamp(key: str, value: float) -> float:
    """Clamp a programmatic seed into its widget's bounds.

    # mirrors woffl/gui/sidebar.py:clamp_seed
    """
    lo, hi = _SEED_BOUNDS.get(key, (None, None))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def _seed(seeds: dict[str, Any], key: str, raw: Any, default: float, cast: type = float) -> None:
    """Seed one field from well data - NaN-safe and bounds-clamped.

    # mirrors woffl/gui/sidebar.py:_seed_param
    """
    num = frames.opt_float(raw)
    try:
        value = cast(num) if num is not None else cast(default)
    except (TypeError, ValueError):
        value = cast(default)
    seeds[key] = _clamp(key, value)


def _need_finite(value: Any) -> float:
    """Finite float or raise - the NaN gate for must-have seed inputs.

    ``int(nan)`` raised in the Streamlit path, dropping the seed pipeline to
    the single-test branch; raising here preserves that fall-through.
    """
    num = frames.opt_float(value)
    if num is None:
        raise ValueError("non-finite seed value")
    return num


def _clean_str(value: Any) -> Optional[str]:
    """Stripped string or None for None/NaN/empty."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _opt_bool(value: Any) -> Optional[bool]:
    """Bool or None for missing/NaN/non-numeric values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    num = frames.opt_float(value)
    return None if num is None else bool(num)


def _pad_from_mp_name(mp_name: str) -> str:
    """MPB-30 -> B, MPI-15 -> I; '' for unknown formats.

    # mirrors woffl/gui/utils.py:pad_from_mp_name
    """
    if not mp_name or "-" not in mp_name:
        return ""
    return mp_name.replace("MP", "").split("-")[0]


def _default_pad_pf(pad: str) -> int:
    """Default PF surface pressure (psi) for a pad letter.

    # mirrors woffl/gui/utils.py:default_pad_pf
    """
    return _PAD_PF_DEFAULTS.get(pad, _PAD_PF_FALLBACK)


def _casing_dims(chars: dict[str, Any]) -> tuple[float, float]:
    """(casing_od, casing_thickness) from chars; fallback 6.875 / 0.5.

    # mirrors woffl/gui/scotts_tools/_common.py:casing_dims_from_chars
    """
    od = frames.opt_float(chars.get("casing_out_dia"))
    inn = frames.opt_float(chars.get("casing_inn_dia"))
    if od is not None and inn is not None and od > inn > 0:
        return od, (od - inn) / 2.0
    return 6.875, 0.5


def _live_pf_seed(well: str, tests_df: Optional[pd.DataFrame]) -> Optional[dict[str, Any]]:
    """Best live PF reading to seed ``ppf_surf`` with, or None.

    Priority: the most recent test's TEST-DAY reading (consistent with the
    qwf/pwf/WC/GOR seeded from that same test) -> the latest daily reading.
    v1 slices the shared test cache directly (no memory-gauge/manual layers).
    # mirrors woffl/gui/utils.py:live_pf_for_seed

    Args:
        well: GUI well name, e.g. "MPB-28".
        tests_df: the well's test slice (with joined pf_press/pf_source).

    Returns:
        {"pf_press", "pf_source", "pf_date", "kind"} or None (caller falls
        back to pad defaults).
    """
    if tests_df is not None and not tests_df.empty and "pf_press" in tests_df.columns:
        recent = tests_df.sort_values("WtDate", ascending=False).iloc[0]
        press = frames.opt_float(recent.get("pf_press"))
        if press is not None:
            # mirrors woffl/gui/utils.py:pf_from_test_row
            return {
                "pf_press": press,
                "pf_source": recent.get("pf_source"),
                "pf_date": recent.get("WtDate"),
                "kind": "test day",
            }
    # mirrors woffl/gui/utils.py:latest_pf_for_well
    pf_df = datasources.pf_latest_safe()
    if pf_df is None or pf_df.empty:
        return None
    rows = pf_df[pf_df["well"] == well]
    if rows.empty:
        return None
    row = rows.iloc[0]
    press = frames.opt_float(row.get("pf_press"))
    if press is None:
        return None
    return {
        "pf_press": press,
        "pf_source": row.get("pf_source"),
        "pf_date": row.get("pf_date"),
        "kind": "latest daily",
    }


# ---------------------------------------------------------------------------
# Well list
# ---------------------------------------------------------------------------


def list_wells() -> dict[str, Any]:
    """All known wells from the characteristics frame (WellsResponse shape).

    Returns:
        {"wells": [WellListItem...], "source": "databricks"|"csv_fallback"}.

    Raises:
        RuntimeError: when both Databricks and the CSV fallback fail.
    """
    df, source = datasources.well_chars_safe()
    wells: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        name = _clean_str(row.get("Well"))
        if not name:
            continue
        tvd_est = _opt_bool(row.get("tvd_estimated"))
        wells.append(
            {
                "name": name,
                "pad": _pad_from_mp_name(name),
                "is_sch": _opt_bool(row.get("is_sch")),
                "jp_tvd": frames.opt_float(row.get("JP_TVD")),
                "tvd_estimated": bool(tvd_est) if tvd_est is not None else False,
                "has_survey": datasources.has_survey(name),
            }
        )
    return {"wells": wells, "source": source}


# ---------------------------------------------------------------------------
# Well context (the sidebar seeding pipeline, replayed)
# ---------------------------------------------------------------------------


def well_context(well: str, months: int = 6, cap: int = 0) -> dict[str, Any]:
    """Everything the client needs when a well is selected (WellContext shape).

    Replays the sidebar's seeding pipeline in order (a) chars, (b) pump
    history, (c) IPR from well tests, (d) saved-IPR overlay, (e) live PF.
    # mirrors woffl/gui/sidebar.py:_update_well_parameters_from_data

    Args:
        well: GUI well name, e.g. "MPB-28".
        months: well-test lookback window in months.
        cap: keep only the newest N tests when > 0.

    Returns:
        Dict matching schemas.WellContext.

    Raises:
        KeyError: when the well is not in the characteristics frame
            (the router maps this to a 404).
    """
    chars_df, chars_source = datasources.well_chars_safe()
    matches = chars_df[chars_df["Well"] == well]
    if matches.empty:
        raise KeyError(well)
    row: dict[str, Any] = matches.iloc[0].to_dict()

    seeds: dict[str, Any] = {}

    # -- (a) chars seeds -----------------------------------------------------
    # mirrors woffl/gui/sidebar.py:_update_well_parameters_from_data
    _seed(seeds, "tubing_od", row.get("out_dia"), 4.5)
    _seed(seeds, "tubing_thickness", row.get("thick"), 0.5)
    casing_od, casing_thick = _casing_dims(row)
    seeds["casing_od"] = _clamp("casing_od", round(casing_od, 3))
    seeds["casing_thickness"] = _clamp("casing_thickness", round(casing_thick, 3))
    _seed(seeds, "form_temp", row.get("form_temp"), 70, cast=int)
    _seed(seeds, "jpump_tvd", row.get("JP_TVD"), 4065, cast=int)
    _seed(seeds, "pres", row.get("res_pres"), 1700, cast=int)

    is_sch_val = _opt_bool(row.get("is_sch"))
    is_sch = True if is_sch_val is None else is_sch_val
    seeds["field_model"] = "Schrader" if is_sch else "Kuparuk"

    # PVT preset by field model, THEN chars overrides (missing/NaN values
    # reset to the preset so the previous well's PVT can't leak across).
    api_default, pbp_default = (22.0, 1750.0) if is_sch else (24.0, 2250.0)
    bubble_raw = row.get("bubble_point")
    if frames.opt_float(bubble_raw) is None:
        bubble_raw = row.get("resvr_bubb")
    _seed(seeds, "oil_api", row.get("oil_api"), api_default)
    _seed(seeds, "bubble_point", bubble_raw, pbp_default)
    _seed(seeds, "gas_sg", row.get("gas_sg"), 0.65)
    _seed(seeds, "wat_sg", row.get("wat_sg"), 1.02)

    # -- (b) pump identity from JP history -----------------------------------
    # mirrors woffl/gui/sidebar.py:_populate_pump_from_history
    pump: Optional[dict[str, Any]] = None
    jp_hist_df, _jp_src = datasources.jp_history_safe()
    if jp_hist_df is not None:
        from woffl.assembly.jp_history import get_current_pump

        current = get_current_pump(jp_hist_df, well)
        if current is not None:
            nozzle_str = _clean_str(current.get("nozzle_no"))
            throat_str = _clean_str(current.get("throat_ratio"))
            throat_str = throat_str.upper() if throat_str else None
            if nozzle_str and throat_str:
                if nozzle_str in NOZZLE_OPTIONS:
                    seeds["nozzle_no"] = nozzle_str
                if throat_str in THROAT_OPTIONS:
                    seeds["area_ratio"] = throat_str
            pump = {
                "nozzle_no": nozzle_str,
                "throat_ratio": throat_str,
                "tubing_od": frames.opt_float(current.get("tubing_od")),
                "date_set": frames.json_value(current.get("date_set")),
            }

    # -- (c) IPR from well tests ----------------------------------------------
    # mirrors woffl/gui/sidebar.py:_auto_populate_from_ipr
    tests_df = tests_svc.tests_for_well(well, months, cap)
    test_count = int(len(tests_df)) if tests_df is not None else 0

    ipr_info: Optional[str] = None
    # Where the IPR numbers below actually came from, and how well the fit
    # held. The optimizer reports this per well: a pump recommendation is
    # only as good as the inflow curve it was chosen against.
    ipr_source: Optional[str] = None
    ipr_r2: Optional[float] = None
    vogel_seeded = False
    if tests_df is not None and len(tests_df) >= 2:
        try:
            from woffl.assembly.ipr_analyzer import (
                compute_vogel_coefficients,
                estimate_reservoir_pressure,
            )

            merged_with_rp = estimate_reservoir_pressure(tests_df)
            vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
            usable = (
                vogel_coeffs is not None
                and not vogel_coeffs.empty
                and "Well" in vogel_coeffs.columns
            )
            well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == well] if usable else None
            if well_coeffs is not None and not well_coeffs.empty:
                coeff = well_coeffs.iloc[0]
                wc = _need_finite(coeff["form_wc"])
                fgor = _need_finite(coeff["fgor"])
                # coeff qwf IS WtTotalFluid - already TOTAL LIQUID (BLPD).
                qwf = _need_finite(coeff["qwf"])
                pwf = _need_finite(coeff["pwf"])
                resp = _need_finite(coeff["ResP"])

                seeds["form_wc"] = _clamp("form_wc", round(wc, 2))
                seeds["form_gor"] = _clamp("form_gor", max(int(fgor), 20))
                seeds["qwf"] = _clamp("qwf", int(qwf))
                seeds["pwf"] = _clamp("pwf", int(pwf))
                seeds["pres"] = _clamp("pres", int(resp))

                recent = tests_df.sort_values("WtDate", ascending=False).iloc[0]
                whp = frames.opt_float(recent.get("whp"))
                if whp is not None:
                    seeds["surf_pres"] = _clamp("surf_pres", int(whp))

                num_tests = int(coeff["num_tests"])
                date_str = frames.json_value(coeff["most_recent_date"]) or str(
                    coeff["most_recent_date"]
                )
                ipr_info = (
                    f"IPR values loaded from {num_tests} well tests "
                    f"(most recent: {date_str})"
                )
                vogel_seeded = True
                ipr_source = "vogel"
                ipr_r2 = frames.opt_float(coeff.get("R2"))
        except Exception as exc:
            # Fall through to the single-test seed path below, but LOG it -
            # a systemic failure degrading every well needs a signal.
            log.warning("IPR auto-populate failed for %s: %s", well, exc)

    if tests_df is not None and not tests_df.empty and not vogel_seeded:
        # Single-test path: 1 test, or 2+ tests where Vogel could not fit.
        recent = tests_df.sort_values("WtDate", ascending=False).iloc[0]
        oil = frames.opt_float(recent.get("WtOilVol"))
        water = frames.opt_float(recent.get("WtWaterVol"))
        total = frames.opt_float(recent.get("WtTotalFluid"))
        bhp = frames.opt_float(recent.get("BHP"))
        fgor = frames.opt_float(recent.get("fgor"))
        whp = frames.opt_float(recent.get("whp"))

        if water is not None and total is not None and total > 0:
            wc = max(0.0, min(1.0, water / total))
            seeds["form_wc"] = _clamp("form_wc", round(wc, 2))
        # qwf is the test's TOTAL LIQUID (WtTotalFluid) - never the oil split.
        if total is not None:
            seeds["qwf"] = _clamp("qwf", int(total))
        elif oil is not None:
            liquid = oil + (water if water is not None else 0.0)
            seeds["qwf"] = _clamp("qwf", int(liquid))
        if bhp is not None:
            seeds["pwf"] = _clamp("pwf", int(bhp))
        if fgor is not None:
            seeds["form_gor"] = _clamp("form_gor", max(int(fgor), 20))
        if whp is not None:
            seeds["surf_pres"] = _clamp("surf_pres", int(whp))

        date_str = frames.json_value(recent.get("WtDate")) or str(recent.get("WtDate"))
        ipr_info = (
            f"Sidebar seeded from 1 well test ({date_str}) - "
            "Vogel IPR fit unavailable"
        )
        ipr_source = "single_test"

    # -- (d) saved-IPR overlay -------------------------------------------------
    # mirrors woffl/gui/sidebar.py:_seed_saved_ipr
    prop_locks: dict[str, dict[str, Any]] = {
        key: {"locked": False, "value": None} for key in _LOCK_SEED_KEYS
    }
    saved_ipr_info: Optional[str] = None
    try:
        from woffl.assembly.prop_hist_client import format_alaska
        from woffl.gui.ipr_anchor import load_saved_ipr, saved_wins

        info = load_saved_ipr(well)
        if info:
            # BHP-calibrated friction seeds INDEPENDENTLY of the pin-vs-values
            # precedence, at FULL precision (rounding broke reload exactness).
            for key, val in (info.get("friction") or {}).items():
                num = frames.opt_float(val)
                if num is not None and key in ("ken", "kth", "kdi"):
                    seeds[key] = _clamp(key, num)

            # Field locks sit OUTSIDE the precedence: a locked WC/GOR/ResP
            # overrides every test-derived seed.
            locks = info.get("locks") or {}
            lock_values = info.get("lock_values") or {}
            for skey, seed_key in _LOCK_SEED_KEYS.items():
                locked = bool(locks.get(skey))
                lock_val = frames.opt_float(lock_values.get(skey))
                prop_locks[skey] = {"locked": locked, "value": lock_val}
                if not locked or lock_val is None:
                    continue
                if skey == "form_wc":
                    seed_val: float = round(min(max(lock_val, 0.0), 0.99), 2)
                else:  # form_gor / res_pres
                    seed_val = int(lock_val)
                seeds[seed_key] = _clamp(seed_key, seed_val)

            # Saved VALUES overlay only when newer than the anchor pin.
            values = info.get("values") or {}
            if values and saved_wins(info.get("saved_at"), info.get("pin_at")):
                wc_val = frames.opt_float(values.get("form_wc"))
                wc_val = 0.5 if wc_val is None else wc_val
                wc_val = min(max(wc_val, 0.0), 0.99)
                seeds["form_wc"] = _clamp("form_wc", round(wc_val, 2))
                gor = frames.opt_float(values.get("form_gor"))
                if gor is not None:
                    seeds["form_gor"] = _clamp("form_gor", int(gor))
                # Stored rate and qwf are BOTH total liquid - no conversion.
                qliq = frames.opt_float(values.get("qwf_liq"))
                if qliq is not None:
                    seeds["qwf"] = _clamp("qwf", int(qliq))
                pwf_val = frames.opt_float(values.get("pwf"))
                if pwf_val is not None:
                    seeds["pwf"] = _clamp("pwf", int(pwf_val))
                resp_val = frames.opt_float(values.get("res_pres"))
                if resp_val is not None:
                    seeds["pres"] = _clamp("pres", int(resp_val))
                sp = frames.opt_float(values.get("surf_press"))
                if sp is not None:
                    seeds["surf_pres"] = _clamp("surf_pres", int(sp))

                ts = info.get("saved_at")
                when = format_alaska(ts, "%Y-%m-%d") if ts is not None else str(ts)
                who = str(info.get("saved_by") or "").split("@")[0]
                # No live anchor pin behind the values means no WELL TEST
                # behind them: the engineer chose the point (a joint match, a
                # backmatched BHP, an applied permutation). Same precedence as
                # a test-anchored save, different truth claim - so it is named
                # differently everywhere it feeds a decision, because a pump
                # recommendation is only as good as the inflow it was chosen
                # against and nobody should read a chosen point as a measured
                # one.
                from_test = info.get("pin_at") is not None
                kind = "saved IPR values" if from_test else "manual IPR point"
                tail = "" if from_test else " - not tied to a well test"
                saved_ipr_info = (
                    f"Restored {kind} ({when} - {who}){tail}"
                    if who
                    else f"Restored {kind} ({when}){tail}"
                )
                # A reviewed save outranks whatever the tests fitted, and its
                # R2 no longer describes the numbers in play.
                ipr_source = "saved" if from_test else "manual"
                ipr_r2 = None
    except Exception:
        log.warning(
            "Saved-IPR seed failed for %s; auto-populated values stand.",
            well,
            exc_info=True,
        )

    # -- (e) live PF seed --------------------------------------------------------
    # mirrors woffl/gui/sidebar.py:_seed_pf_from_live
    from woffl.gui.pump_identity import tracker_direction

    trk = tracker_direction(jp_hist_df, well)
    live = _live_pf_seed(well, tests_df)

    direction: Optional[str]
    if live is None:
        pad = _pad_from_mp_name(well)
        fallback = _default_pad_pf(pad) if pad else 3168
        seeds["ppf_surf"] = _clamp("ppf_surf", int(fallback))
        direction = trk
        pf: dict[str, Any] = {
            "ppf_surf": seeds["ppf_surf"],
            "direction": direction,
            "kind": "fallback",
            "pf_press": float(fallback),
            "pf_source": None,
            "pf_date": None,
        }
    else:
        seeds["ppf_surf"] = _clamp("ppf_surf", int(round(live["pf_press"])))
        source = live.get("pf_source")
        if trk:
            direction = trk
        elif source in ("annulus", "tubing"):
            direction = "reverse" if source == "annulus" else "forward"
        else:
            direction = None
        pf = {
            "ppf_surf": seeds["ppf_surf"],
            "direction": direction,
            "kind": live["kind"],
            "pf_press": float(live["pf_press"]),
            "pf_source": frames.json_value(source),
            "pf_date": frames.json_value(live.get("pf_date")),
        }
    if direction is not None:
        seeds["jpump_direction"] = direction

    # -- as-built locks + raw chars ------------------------------------------------
    # mirrors woffl/gui/sidebar.py:as_built_from_props
    as_built_locks = {
        "tubing": frames.opt_float(row.get("out_dia")) is not None
        and frames.opt_float(row.get("thick")) is not None,
        "casing": frames.opt_float(row.get("casing_out_dia")) is not None
        and frames.opt_float(row.get("casing_inn_dia")) is not None,
        "jpump_tvd": frames.opt_float(row.get("JP_TVD")) is not None,
    }
    chars_json = {key: frames.json_value(row.get(key)) for key in _CHARS_KEYS}

    return {
        "well": well,
        "chars": chars_json,
        "chars_source": chars_source,
        "seeds": seeds,
        "as_built_locks": as_built_locks,
        "prop_locks": prop_locks,
        "pump": pump,
        "pf": pf,
        "ipr_info": ipr_info,
        "ipr_source": ipr_source,
        "ipr_r2": ipr_r2,
        "saved_ipr_info": saved_ipr_info,
        "test_count": test_count,
    }


# ---------------------------------------------------------------------------
# Well profile (survey geometry)
# ---------------------------------------------------------------------------


def _even_indices(size: int, cap: int = _PROFILE_MAX_POINTS) -> np.ndarray:
    """Evenly spaced index array keeping at most ``cap`` of ``size`` points."""
    if size <= cap:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, cap).round().astype(int))


def _chars_jp_tvd(well: str) -> Optional[float]:
    """The well's JP_TVD from the chars frame, or None on any failure."""
    try:
        df, _source = datasources.well_chars_safe()
    except Exception:
        return None
    rows = df[df["Well"] == well]
    if rows.empty:
        return None
    return frames.opt_float(rows.iloc[0].get("JP_TVD"))


@ttl_cache(config.TTL_PROFILES, maxsize=512)
def well_profile_payload(
    well: str, jpump_tvd: Optional[float], field_model: Optional[str]
) -> dict[str, Any]:
    """Survey-based well profile payload (WellProfileResponse shape).

    Builds the profile from the local deviation survey when one exists;
    otherwise falls back to the field-model preset rebuilt at the effective
    jetpump TVD. Raw rays are downsampled evenly to <= 1500 points.
    # mirrors woffl/gui/utils.py:create_well_profile_from_survey

    Args:
        well: GUI well name, e.g. "MPB-28".
        jpump_tvd: jetpump TVD override (ft); falls back to chars JP_TVD,
            then 4065.
        field_model: "Schrader" | "Kuparuk" preset for the no-survey fallback
            (default Schrader).

    Returns:
        Dict matching schemas.WellProfileResponse.
    """
    from woffl.geometry.wellprofile import WellProfile

    tvd_eff = frames.opt_float(jpump_tvd)
    if tvd_eff is None:
        tvd_eff = _chars_jp_tvd(well)
    if tvd_eff is None:
        tvd_eff = 4065.0

    wp = None
    has_survey = False
    survey_df = datasources.survey(well)
    if survey_df is not None and not survey_df.empty:
        try:
            md_list = [float(v) for v in survey_df["meas_depth"].tolist()]
            vd_list = [float(v) for v in survey_df["tvd_depth"].tolist()]
            jetpump_md = float(np.interp(tvd_eff, vd_list, md_list))
            wp = WellProfile(md_list=md_list, vd_list=vd_list, jetpump_md=jetpump_md)
            has_survey = True
        except Exception as exc:
            log.warning("Well profile from survey failed for %s: %s", well, exc)
            wp = None

    if wp is None:
        # mirrors woffl/gui/utils.py:create_well_profile
        model = (field_model or "Schrader").lower()
        wp = WellProfile.kuparuk() if model == "kuparuk" else WellProfile.schrader()
        try:
            jetpump_md = wp.md_interp(tvd_eff)
            wp = WellProfile(
                md_list=wp.md_ray, vd_list=wp.vd_ray, jetpump_md=jetpump_md
            )
        except ValueError:
            # TVD outside the preset's range: keep the preset's own jetpump_md.
            pass

    idx = _even_indices(len(wp.md_ray))
    md = wp.md_ray[idx].astype(float).tolist()
    vd = wp.vd_ray[idx].astype(float).tolist()
    hd = wp.hd_ray[idx].astype(float).tolist()

    try:
        jetpump_vd: Optional[float] = float(wp.vd_interp(float(wp.jetpump_md)))
    except ValueError:
        jetpump_vd = None

    inclination: Optional[dict[str, list[float]]] = None
    if (
        survey_df is not None
        and not survey_df.empty
        and "inclination" in survey_df.columns
    ):
        inc = survey_df[["meas_depth", "inclination"]].dropna()
        if not inc.empty:
            inc_md = inc["meas_depth"].to_numpy(dtype=float)
            inc_deg = inc["inclination"].to_numpy(dtype=float)
            keep = np.isfinite(inc_md) & np.isfinite(inc_deg)
            inc_md, inc_deg = inc_md[keep], inc_deg[keep]
            if inc_md.size:
                inc_idx = _even_indices(int(inc_md.size))
                inclination = {
                    "md": inc_md[inc_idx].tolist(),
                    "deg": inc_deg[inc_idx].tolist(),
                }

    return {
        "well": well,
        "has_survey": has_survey,
        "md": md,
        "vd": vd,
        "hd": hd,
        "md_filtered": [float(v) for v in wp.md_fit],
        "vd_filtered": [float(v) for v in wp.vd_fit],
        "jetpump_md": float(wp.jetpump_md),
        "jetpump_vd": jetpump_vd,
        "inclination": inclination,
    }
