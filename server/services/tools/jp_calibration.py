"""JP Friction Calibration - fit ken/kth/kdi per well against measured BHP.

Port of the retired Streamlit tool (engine only).

READ-ONLY, as the tab was: it renders SQL for a human to run against
vw_prop_mech, it has never executed a write and must not start.
"""


import math
import os

import pandas as pd

from server.services import datasources

from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from server.services.tools._common import (
    PAD_PF_DEFAULTS,
    PAD_PF_FALLBACK,
    default_pad_pf as _default_pad_pf,
    live_pf_for_seed,
    load_well_characteristics,
)

from server.services.tools import _common
from server.services.tools._common import (
    build_well_config,
    casing_dims_from_chars,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_vogel_for_wells,
    has_databricks_casing,
    pad_from_mp_name,
)

# PAD_PF_DEFAULTS / PAD_PF_FALLBACK / _default_pad_pf are imported above from
# server.services.tools._common - the single source of truth for pad PF.


def _latest_bhp_whp_paired_per_well(months_back: int) -> dict[str, dict]:
    """Return ``{well: {"bhp", "whp", "date"}}`` from the SAME test row per well.

    [P1-35 fix] The BHP calibration target and the WHP solver boundary used
    to be picked INDEPENDENTLY (``get_latest_bhp_per_well`` /
    ``get_latest_whp_per_well``, each "latest available"), which could pull
    from two different tests/days when only one of the two was gauged on a
    given day — the mismatch then got silently absorbed into the pushed
    friction coefficients. Pairing them to one row keeps the calibration
    internally consistent; ``whp`` is None (not back-filled from another
    day) when the BHP test's own row has no WHP reading.
    """
    raw = fetch_well_tests_raw(months_back)
    if raw is None or raw.empty or "BHP" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["BHP"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").tail(1)
    out: dict[str, dict] = {}
    for _, row in latest.iterrows():
        whp = row.get("whp")
        out[row["well"]] = {
            "bhp": float(row["BHP"]),
            "whp": float(whp) if pd.notna(whp) else None,
            "date": row["WtDate"],
        }
    return out


def _resolve_pump_for_test(
    jp_hist: pd.DataFrame,
    well_name: str,
    test_date,
    current_pump: dict | None = None,
) -> tuple[dict | None, bool]:
    """Resolve the pump installed AT a historical test's date.

    [P1-29 fix] Historical BHP tests must be paired with the pump that was
    actually in the hole on the test date via ``get_pump_at_date`` (Date Set
    -> next Date Set tenure; JPCOs are same-day pull+set so Date Pulled is
    never used — see ``jp_history.get_pump_at_date``), not today's current
    pump. Mirrors the pattern already used by ``jp_fric_trend.py`` /
    ``jp_washout.py``.

    Falls back to ``current_pump`` (looked up via ``get_current_pump`` when
    not supplied) when no install record covers the test date.

    Returns ``(pump_dict_or_None, pump_changed)``. ``pump_changed`` is True
    when the resolved at-test pump differs from the well's current pump, or
    when the current pump can't be determined at all (so the caller can't
    confirm the fit still applies to what's in the well today).
    """
    if current_pump is None:
        current_pump = get_current_pump(jp_hist, well_name)

    pump_at = get_pump_at_date(jp_hist, well_name, test_date)
    if not (pump_at and pump_at.get("nozzle_no") and pump_at.get("throat_ratio")):
        pump_at = current_pump

    if (
        pump_at is None
        or not pump_at.get("nozzle_no")
        or not pump_at.get("throat_ratio")
    ):
        return None, False

    if current_pump is None:
        return pump_at, True

    pump_changed = str(pump_at["nozzle_no"]) != str(
        current_pump.get("nozzle_no")
    ) or str(pump_at["throat_ratio"]) != str(current_pump.get("throat_ratio"))
    return pump_at, pump_changed


def _denormalize_for_db(well_name: str) -> str:
    """MPB-28 → B-028 (Databricks vw_well_header format)."""
    from woffl.assembly.well_test_client import _denormalize_well_name

    return _denormalize_well_name(well_name)


def _build_calibration_input_table(
    months_back: int,
) -> tuple[pd.DataFrame, dict[str, dict]] | None:
    """Build the per-well input table for the calibration tab.

    Only includes wells that:
      - Have a measured BHP within the lookback window
      - Have a pump resolvable in the JP history (i.e., are jet-pump wells —
        non-JP wells like ESP get filtered out)

    Each row's initial PF Pressure is the well's LIVE value from
    vw_pressure_daily (test-day of its most recent test, else latest daily
    reading), falling back to ``PAD_PF_DEFAULTS`` when the well has no valid
    reading. The user can broadcast a per-pad value via the pad inputs in the
    tab UI, or override individual rows in the table.

    Tubing/casing geometry columns are also surfaced so the user can spot
    wells where casing dims came from the fallback (6.875"/0.5") instead
    of Databricks — those wells will have wrong annulus area and therefore
    wrong PF friction.

    [P1-29 / P1-35 fix] BHP and WHP are read from the SAME test row
    (``_latest_bhp_whp_paired_per_well``) and the pump used is the one
    installed AT that test's date (``_resolve_pump_for_test`` /
    ``get_pump_at_date``), not today's current pump — mirrors
    ``jp_fric_trend.py`` / ``jp_washout.py``. Rows where the test-date pump
    differs from the well's current pump are flagged in a "Pump changed"
    column and default to ``Include=False`` (soft guard — the operator must
    consciously opt a stale-pump fit back in).

    Returns ``(df, pump_info_map)`` where ``df`` has columns: Well, Pad,
    Pump, Test Date, Pump changed, BHP (psi), WHP (psi), PF Pressure (psi),
    Tube OD, Tube ID, Case OD, Case ID, Case src, Ann area, Include, and
    ``pump_info_map`` is ``{well: {"nozzle", "throat", "pump_changed"}}`` for
    use by the run step (so the pump modeled matches the pump shown here).
    Returns None when no eligible wells are found.
    """
    bhp_whp_map = _latest_bhp_whp_paired_per_well(months_back)
    if not bhp_whp_map:
        return None

    jp_hist, _src = datasources.jp_history_safe()
    if jp_hist is None or jp_hist.empty:
        return None

    jp_chars_dict = _common.well_chars_map()

    rows = []
    pump_info_map: dict[str, dict] = {}
    for wn, info in bhp_whp_map.items():
        bhp = info["bhp"]
        whp = info["whp"]
        test_date = info["date"]

        current_pump = get_current_pump(jp_hist, wn)
        pump, pump_changed = _resolve_pump_for_test(
            jp_hist, wn, test_date, current_pump
        )
        if pump is None:
            continue  # non-JP well or no install record at all — skip
        pump_str = f"{pump['nozzle_no']}{pump['throat_ratio']}"
        pump_info_map[wn] = {
            "nozzle": pump["nozzle_no"],
            "throat": pump["throat_ratio"],
            "pump_changed": pump_changed,
        }
        pad = pad_from_mp_name(wn)

        # Geometry being passed into the solver — surface for verification
        chars = jp_chars_dict.get(wn, {})

        def _num(v, default):
            # NaN-safe: `float('nan') or X` returns nan (nan is truthy), so the
            # `or` fallback never fired for a present-but-NaN Databricks value —
            # the verification table then showed NaN geometry.
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return default
            return default if math.isnan(fv) else fv

        tube_od = _num(chars.get("out_dia"), 4.5)
        tube_thk = _num(chars.get("thick"), 0.271)
        tube_id = tube_od - 2 * tube_thk
        case_od, case_thk = casing_dims_from_chars(chars)
        case_id = case_od - 2 * case_thk
        case_src = "DB" if has_databricks_casing(chars) else "fallback"
        # Annulus cross-section (in²) — what the PF friction calc actually uses
        ann_area_in2 = (math.pi / 4) * (case_id**2 - tube_od**2)

        # PF from live daily data (test-day of most recent test, else latest
        # daily reading); pad default only when the well has no valid reading.
        live_pf = live_pf_for_seed(wn)
        pf_seed = int(round(live_pf["pf_press"])) if live_pf else _default_pad_pf(pad)

        rows.append(
            {
                "Well": wn,
                "Pad": pad,
                "Pump": pump_str,
                "Test Date": (
                    test_date.date() if hasattr(test_date, "date") else test_date
                ),
                "Pump changed": pump_changed,
                "BHP (psi)": int(round(bhp)),
                "WHP (psi)": int(round(whp)) if whp is not None else None,
                "PF Pressure (psi)": pf_seed,
                "Tube OD": round(tube_od, 3),
                "Tube ID": round(tube_id, 3),
                "Case OD": round(case_od, 3),
                "Case ID": round(case_id, 3),
                "Case src": case_src,
                "Ann area": round(ann_area_in2, 2),
                # Soft guard: default a stale-pump fit OUT of the run; the
                # operator must consciously re-check it after reading the
                # "Pump changed" warning.
                "Include": not pump_changed,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["Pad", "Well"]).reset_index(drop=True)
    return df, pump_info_map


def _format_sql_preview(results_df: pd.DataFrame, target_table: str) -> str:
    """Render a copy-paste-ready SQL UPDATE block for the data team to review.

    Updates jpfric_entry, jpfric_throat, and jpfric_diffuser per well using
    CASE expressions. Well names are denormalized to Databricks format
    (MPB-28 → B-028). knz (jpfric_nozzle) is NOT touched (held fixed at
    0.01 throughout calibration).

    [P1-29 guard] Wells whose "Pump changed" flag is set (the pump was
    changed out after the calibrated test) are EXCLUDED from the push even
    if converged — the fit used the test-date pump's geometry, which may no
    longer match what's in the well, so it isn't pushed for the current
    pump's coefficients without operator review.
    """
    if "Pump changed" in results_df.columns:
        excluded = results_df[
            (results_df["Status"] == "converged") & (results_df["Pump changed"])
        ]
        converged = results_df[
            (results_df["Status"] == "converged") & (~results_df["Pump changed"])
        ]
    else:
        excluded = results_df.iloc[0:0]
        converged = results_df[results_df["Status"] == "converged"]
    if converged.empty:
        return "-- No converged calibrations to write."

    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    db_names = [_denormalize_for_db(w) for w in converged["Well"]]

    lines = [
        f"-- Friction-coef calibration generated {today}",
        f"-- {len(converged)} wells; ken + kth + kdi (knz unchanged at 0.01)",
        "-- Replace <target_table> with the writable underlying table for vw_prop_mech",
    ]
    if not excluded.empty:
        lines.append(
            f"-- EXCLUDED {len(excluded)} well(s) — pump changed since the "
            f"calibrated test: {', '.join(excluded['Well'].tolist())}"
        )
    # prop_hist is an APPEND-ONLY EAV table (enthid, prop_id, prop_value,
    # entry_datetime, entry_user) that vw_prop_mech pivots - it has no
    # well_name / jpfric_* columns and the house rule forbids UPDATE on it.
    # The preview used to emit exactly that UPDATE, which was both invalid
    # and a rule violation if pasted (review 2026-09-01, EVID-F26). Emit the
    # rows the app's own save path would write, enthid resolved by name.
    # (This module never writes anything itself - the preview is text.)
    table = target_table if "." in target_table else f"mpu.wells.{target_table}"
    lines += [
        "",
        "-- One row per (well, coefficient); latest entry_datetime per",
        "-- (enthid, prop_id) wins in the vw_prop_mech pivot. Append only.",
        f"INSERT INTO {table} (enthid, prop_id, prop_value, entry_datetime, entry_user)",
    ]
    selects: list[str] = []
    for wn_db, ken, kth, kdi in zip(
        db_names, converged["Cal ken"], converged["Cal kth"], converged["Cal kdi"]
    ):
        for prop_id, val in (
            ("jpfric_entry", ken),
            ("jpfric_throat", kth),
            ("jpfric_diffuser", kdi),
        ):
            selects.append(
                f"SELECT h.enthid, '{prop_id}', {float(val):.4f}, current_timestamp(), "
                f"current_user() FROM mpu.wells.vw_well_header h "
                f"WHERE h.wellname = '{wn_db}' AND h.field = 'MPU' AND h.well_type = 'prod'"
            )
    lines.append("\nUNION ALL\n".join(selects) + ";")
    return "\n".join(lines)


def _failed_row(wn: str, row, reason: str) -> dict:
    return {
        "Well": wn,
        "Pad": row.get("Pad"),
        "Pump": row.get("Pump"),
        "Pump changed": bool(row.get("Pump changed", False)),
        "TVD": None,
        "Actual BHP": int(row.get("BHP (psi)", 0)),
        "Modeled BHP": None,
        "BHP err": None,
        "Match": "failed",
        "Bounded": False,
        "Sonic": False,
        "PF used": int(row.get("PF Pressure (psi)", 0)),
        "Current ken": None,
        "Current kth": None,
        "Current kdi": None,
        "Cal ken": None,
        "Cal kth": None,
        "Cal kdi": None,
        "Δ ken": None,
        "Δ kth": None,
        "Δ kdi": None,
        "Prod hydro": None,
        "Prod fric": None,
        "Prod ρ̄": None,
        "Prod grad": None,
        "PF hydro": None,
        "PF fric": None,
        "Pump Δp": None,
        "Starts": 0,
        "Iters": None,
        "Status": f"failed: {reason}",
    }




def calibrate_one(row_dict: dict, chars: dict | None, pump: dict | None,
                  vogel_row: dict | None = None) -> dict:
    """One well's friction calibration. Module-level and picklable: pool worker.

    Lifted from the tab's render loop so the fan-out has something to call.
    Two fixes from that loop are preserved deliberately:

    * P1-29 - the pump is the one resolved AT TABLE-BUILD time (installed at
      the BHP test's date), passed in, NOT a fresh get_current_pump(today).
      Re-resolving would swap the geometry out from under the row the
      engineer reviewed.
    * P1-35 - WHP comes from the SAME test row as the BHP target, which the
      table already paired, not an independently fetched "latest WHP".

    Failures return a ``_failed_row`` so one bad well cannot kill the batch.
    """
    from woffl.gui.fric_calibration import calibrate_friction_coefs

    wn = row_dict["Well"]
    if not chars:
        return _failed_row(wn, row_dict, "no jp_chars row")
    if not pump:
        return _failed_row(wn, row_dict, "no pump info for well")

    whp_val = row_dict.get("WHP (psi)")
    well_surf = float(whp_val) if pd.notna(whp_val) else 210.0
    try:
        wc = _common.build_well_config(wn, {wn: chars}, vogel_row, surf_pres=well_surf)
        wellbore, well_profile, inflow, res_mix, prop_pf = _common.create_well_objects(wc)
    except Exception as exc:  # noqa: BLE001
        return _failed_row(wn, row_dict, f"setup error: {exc}")

    cur_fric = _common.friction_coefs_from_chars(chars)
    ken_fixed = float(cur_fric.get("ken", 0.03))
    try:
        cal = calibrate_friction_coefs(
            well_name=wn,
            target_bhp=float(row_dict["BHP (psi)"]),
            pwh=well_surf,
            tsu=wc.form_temp,
            ppf_surf=float(row_dict["PF Pressure (psi)"]),
            nozzle=pump["nozzle"],
            throat=pump["throat"],
            knz=0.01,
            ken=ken_fixed,
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=wc.jpump_direction,  # live-detected (EVID-F22)
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_row(wn, row_dict, f"calibration error: {exc}")

    conv = bool(cal.converged)
    return {
        "Well": wn,
        "Pad": row_dict.get("Pad"),
        "Pump": row_dict.get("Pump"),
        "Pump changed": bool(row_dict.get("Pump changed", False)),
        "IPR": "well tests" if vogel_row else "defaults",
        "TVD": int(round(wc.jpump_tvd)) if wc.jpump_tvd else None,
        "Actual BHP": int(round(cal.target_bhp)),
        "Modeled BHP": int(round(cal.best_modeled_bhp)) if conv else None,
        "BHP err": int(round(cal.bhp_error)) if conv else None,
        "Match": cal.match_quality,
        "Bounded": cal.bounded,
        "Sonic": cal.sonic,
        "PF used": int(row_dict["PF Pressure (psi)"]),
        "Current ken": cur_fric.get("ken"),
        "Current kth": cur_fric.get("kth"),
        "Current kdi": cur_fric.get("kdi"),
        "Cal ken": round(cal.best_ken, 4) if conv else None,
        "Cal kth": round(cal.best_kth, 4) if conv else None,
        "Cal kdi": round(cal.best_kdi, 4) if conv else None,
        "d ken": round(cal.best_ken - cur_fric["ken"], 4) if conv and "ken" in cur_fric else None,
        "d kth": round(cal.best_kth - cur_fric["kth"], 4) if conv and "kth" in cur_fric else None,
        "d kdi": round(cal.best_kdi - cur_fric["kdi"], 4) if conv and "kdi" in cur_fric else None,
        "Starts": cal.starts_tried,
        "Iters": cal.iterations,
        "Status": "converged" if conv else "did_not_converge",
    }
