"""JP Friction Calibration tab.

Bulk friction-coef calibration for pushing back to Databricks. Calibrates
ken/kth/kdi per well so modeled BHP matches the latest measured BHP.
"""

import math
import os

import pandas as pd
import streamlit as st

from woffl.assembly.jp_history import get_current_pump
from woffl.gui.explainers import render_kcoef_explainer
from woffl.gui.utils import (
    PAD_PF_DEFAULTS,
    PAD_PF_FALLBACK,
    default_pad_pf as _default_pad_pf,
    load_well_characteristics,
)

from ._common import (
    build_well_config,
    casing_dims_from_chars,
    create_well_objects,
    friction_coefs_from_chars,
    get_latest_bhp_per_well,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    pad_from_mp_name,
)

# PAD_PF_DEFAULTS / PAD_PF_FALLBACK / _default_pad_pf are now imported above
# from woffl.gui.utils — single source of truth shared with sidebar.py.


def _denormalize_for_db(well_name: str) -> str:
    """MPB-28 → B-028 (Databricks vw_well_header format)."""
    from woffl.assembly.well_test_client import _denormalize_well_name
    return _denormalize_well_name(well_name)


def _has_databricks_casing(chars: dict | None) -> bool:
    """True iff casing_out_dia and casing_inn_dia are populated in chars."""
    if not chars:
        return False
    out_dia = chars.get("casing_out_dia")
    inn_dia = chars.get("casing_inn_dia")
    if out_dia is None or inn_dia is None:
        return False
    try:
        if pd.isna(out_dia) or pd.isna(inn_dia):
            return False
        return float(out_dia) > float(inn_dia) > 0
    except (TypeError, ValueError):
        return False


def _build_calibration_input_table(months_back: int) -> pd.DataFrame | None:
    """Build the per-well input table for the calibration tab.

    Only includes wells that:
      - Have a measured BHP within the lookback window
      - Have a current pump in the JP history (i.e., are jet-pump wells —
        non-JP wells like ESP get filtered out)

    Each row's initial PF Pressure is set from ``PAD_PF_DEFAULTS`` based on
    the well's pad. The user can broadcast a per-pad value via the pad
    inputs in the tab UI, or override individual rows in the table.

    Tubing/casing geometry columns are also surfaced so the user can spot
    wells where casing dims came from the fallback (6.875"/0.5") instead
    of Databricks — those wells will have wrong annulus area and therefore
    wrong PF friction.

    Returns a DataFrame with columns: Well, Pad, Pump, BHP (psi), WHP (psi),
    PF Pressure (psi), Tube OD, Tube ID, Case OD, Case ID, Case src,
    Ann area, Include. Returns None when no eligible wells are found.
    """
    bhp_map = get_latest_bhp_per_well(months_back)
    if not bhp_map:
        return None
    whp_map = get_latest_whp_per_well(months_back)

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return None

    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

    rows = []
    for wn, bhp in bhp_map.items():
        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            continue  # non-JP well — skip
        pump_str = f"{pump['nozzle_no']}{pump['throat_ratio']}"
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
        case_src = "DB" if _has_databricks_casing(chars) else "fallback"
        # Annulus cross-section (in²) — what the PF friction calc actually uses
        ann_area_in2 = (math.pi / 4) * (case_id**2 - tube_od**2)

        rows.append(
            {
                "Well": wn,
                "Pad": pad,
                "Pump": pump_str,
                "BHP (psi)": int(round(bhp)),
                "WHP (psi)": int(round(whp_map[wn])) if wn in whp_map else None,
                "PF Pressure (psi)": _default_pad_pf(pad),
                "Tube OD": round(tube_od, 3),
                "Tube ID": round(tube_id, 3),
                "Case OD": round(case_od, 3),
                "Case ID": round(case_id, 3),
                "Case src": case_src,
                "Ann area": round(ann_area_in2, 2),
                "Include": True,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.sort_values(["Pad", "Well"]).reset_index(drop=True)


def _format_sql_preview(results_df: pd.DataFrame, target_table: str) -> str:
    """Render a copy-paste-ready SQL UPDATE block for the data team to review.

    Updates jpfric_entry, jpfric_throat, and jpfric_diffuser per well using
    CASE expressions. Well names are denormalized to Databricks format
    (MPB-28 → B-028). knz (jpfric_nozzle) is NOT touched (held fixed at
    0.01 throughout calibration).
    """
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
        "",
        f"UPDATE mpu.wells.{target_table}",
        "SET jpfric_entry = CASE well_name",
    ]
    for wn_db, ken in zip(db_names, converged["Cal ken"]):
        lines.append(f"        WHEN '{wn_db}' THEN {ken:.4f}")
    lines.append("        ELSE jpfric_entry")
    lines.append("    END,")
    lines.append("    jpfric_throat = CASE well_name")
    for wn_db, kth in zip(db_names, converged["Cal kth"]):
        lines.append(f"        WHEN '{wn_db}' THEN {kth:.4f}")
    lines.append("        ELSE jpfric_throat")
    lines.append("    END,")
    lines.append("    jpfric_diffuser = CASE well_name")
    for wn_db, kdi in zip(db_names, converged["Cal kdi"]):
        lines.append(f"        WHEN '{wn_db}' THEN {kdi:.4f}")
    lines.append("        ELSE jpfric_diffuser")
    lines.append("    END")
    well_list = ", ".join(f"'{w}'" for w in db_names)
    lines.append(f"WHERE well_name IN ({well_list});")
    return "\n".join(lines)


def render_tab() -> None:
    """Bulk friction-coef calibration tab for pushing back to Databricks."""
    from woffl.gui.fric_calibration import (
        calibrate_friction_coefs,
        compute_bhp_decomposition,
    )

    st.header("JP Friction Calibration")
    st.markdown(
        "Calibrate `ken`, `kth`, and `kdi` per well so modeled BHP matches "
        "the latest measured BHP. Set a PF pressure per pad, then optionally "
        "fine-tune individual wells in the table. Non-JP wells (no pump in "
        "JP history) are excluded automatically. "
        "`knz` is held fixed at 0.01 — varying it trades off PF rate match."
    )

    render_kcoef_explainer()

    # ── settings ──────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])
    with col_a:
        months_back = st.number_input(
            "Test lookback (months)",
            min_value=1, max_value=24, value=6, step=1,
            key="jp_cal_months",
            help="How far back to pull tests when finding latest measured BHP.",
        )
    with col_b:
        st.write("")
        st.write("")
        load_clicked = st.button(
            "Load JP wells with measured BHP",
            key="jp_cal_load",
        )

    if load_clicked:
        with st.spinner("Loading wells from Databricks..."):
            input_df = _build_calibration_input_table(months_back)
        if input_df is None or input_df.empty:
            st.warning(
                "No JP wells with measured BHP found. Confirm JP history is "
                "loaded and wells have a recent test with BHP."
            )
            return
        st.session_state["jp_cal_input_df"] = input_df
        st.session_state.pop("jp_cal_results_df", None)
        st.session_state.pop("jp_cal_editor", None)

    input_df = st.session_state.get("jp_cal_input_df")
    if input_df is None:
        st.info("Click 'Load JP wells with measured BHP' to start.")
        return

    # ── per-pad PF inputs ──────────────────────────────────────────────
    pads = sorted(input_df["Pad"].unique())
    st.markdown("**Pad-level PF pressure (psi)**")
    st.caption(
        "Set PF pressure per pad and click 'Apply pad PFs to table' to "
        "broadcast. Individual rows below can still be overridden after."
    )

    n_cols = min(6, len(pads))
    pad_cols = st.columns(n_cols)
    pad_pfs: dict[str, int] = {}
    for i, pad in enumerate(pads):
        with pad_cols[i % n_cols]:
            pad_pfs[pad] = st.number_input(
                f"Pad {pad}",
                min_value=1000, max_value=5000,
                value=int(st.session_state.get(
                    f"jp_cal_pad_pf_{pad}", _default_pad_pf(pad)
                )),
                step=50,
                key=f"jp_cal_pad_pf_{pad}",
            )

    apply_pads = st.button(
        "Apply pad PFs to table",
        key="jp_cal_apply_pad_pfs",
        help="Broadcasts each pad's PF to every well on that pad. Wipes "
        "individual row edits — re-edit afterward if needed.",
    )
    if apply_pads:
        df = input_df.copy()
        for pad, pf in pad_pfs.items():
            df.loc[df["Pad"] == pad, "PF Pressure (psi)"] = int(pf)
        st.session_state["jp_cal_input_df"] = df
        # Drop the data_editor delta so the new pad-level values render
        # cleanly without stale individual-row overrides.
        st.session_state.pop("jp_cal_editor", None)
        st.rerun()

    st.caption(
        f"{len(input_df)} wells loaded. Override individual rows below if "
        "needed; uncheck `Include` to skip a well."
    )

    # Warn when casing dims defaulted to 6.875"/0.5" — wrong annulus area
    # → wrong PF friction → BHP calibration absorbs the error into kth/kdi.
    fallback_wells = input_df.loc[input_df["Case src"] == "fallback", "Well"].tolist()
    if fallback_wells:
        st.warning(
            f"{len(fallback_wells)} well(s) are using the default casing geometry "
            f"(6.875\" OD × 0.5\" wall) because vw_prop_mech has no casing data: "
            f"{', '.join(fallback_wells[:10])}"
            + (f" (+{len(fallback_wells) - 10} more)" if len(fallback_wells) > 10 else "")
            + ". If their actual casing is different (e.g., 9-5/8\"), the "
            "annulus area is wrong and PF friction will be miscomputed. Have "
            "the data team populate casing_out_dia / casing_inn_dia for these."
        )

    edited_df = st.data_editor(
        input_df,
        key="jp_cal_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", disabled=True, pinned="left"),
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Pump": st.column_config.TextColumn("Pump", disabled=True),
            "BHP (psi)": st.column_config.NumberColumn(
                "Latest BHP", format="%d", disabled=True,
                help="Most recent measured BHP — calibration target",
            ),
            "WHP (psi)": st.column_config.NumberColumn(
                "Latest WHP", format="%d", disabled=True,
            ),
            "PF Pressure (psi)": st.column_config.NumberColumn(
                "PF Pressure (psi)", format="%d",
                min_value=1000, max_value=5000, step=50,
                help="PF pressure to calibrate at. Overrides the pad-level value.",
            ),
            "Tube OD": st.column_config.NumberColumn(
                "Tube OD", format="%.3f", disabled=True,
                help="Tubing outer diameter (in). From vw_prop_mech.tubing_out_dia, "
                "or 4.5 fallback when missing.",
            ),
            "Tube ID": st.column_config.NumberColumn(
                "Tube ID", format="%.3f", disabled=True,
                help="Tubing inner diameter (in). Computed as Tube OD − 2×wall.",
            ),
            "Case OD": st.column_config.NumberColumn(
                "Case OD", format="%.3f", disabled=True,
                help="Casing outer diameter (in). From vw_prop_mech.casing_out_dia, "
                "or 6.875 fallback when missing.",
            ),
            "Case ID": st.column_config.NumberColumn(
                "Case ID", format="%.3f", disabled=True,
                help="Casing inner diameter (in). Drives annulus flow area together "
                "with Tube OD.",
            ),
            "Case src": st.column_config.TextColumn(
                "Case src", disabled=True,
                help="'DB' = pulled from vw_prop_mech, 'fallback' = used 6.875\"/0.5\" "
                "default (likely wrong if the well's actual casing isn't 7-5/8\").",
            ),
            "Ann area": st.column_config.NumberColumn(
                "Ann area (in²)", format="%.2f", disabled=True,
                help="Annulus cross-section: π/4 × (Case ID² − Tube OD²). "
                "This is what the PF-friction calc uses. Wrong inputs → wrong friction.",
            ),
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Uncheck to skip this well in the calibration run.",
            ),
        },
    )

    run_clicked = st.button(
        "Run calibration on selected wells",
        type="primary",
        key="jp_cal_run",
        use_container_width=True,
    )

    if not run_clicked:
        results_df = st.session_state.get("jp_cal_results_df")
        if results_df is not None:
            _render_results(results_df)
        return

    # ── run analysis ───────────────────────────────────────────────────
    selected = edited_df[edited_df["Include"]].copy()
    if selected.empty:
        st.warning("No wells selected. Check at least one 'Include' box.")
        return

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.error("JP history not loaded — cannot determine current pumps.")
        return

    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")
    whp_map = get_latest_whp_per_well(months_back)
    # Per-well Vogel IPR from recent tests. Calibrating against the generic
    # defaults (WC=0.5, GOR=250, qwf=750) made the friction coefficients
    # absorb the IPR/fluid error — and these coefs get pushed to Databricks.
    vogel_map = get_vogel_for_wells(selected["Well"].tolist(), months_back)

    results = []
    progress = st.progress(0, text="Starting...")
    n = len(selected)
    for i, (_, row) in enumerate(selected.iterrows()):
        wn = row["Well"]
        progress.progress(i / max(n, 1), text=f"Calibrating {wn} ({i+1}/{n})...")

        chars = jp_chars_dict.get(wn)
        if not chars:
            results.append(_failed_row(wn, row, "no jp_chars row"))
            continue

        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            results.append(_failed_row(wn, row, "no current pump in JP history"))
            continue

        well_surf = float(whp_map.get(wn, 210.0))
        vogel_row = vogel_map.get(wn)
        try:
            wc = build_well_config(wn, jp_chars_dict, vogel_row, surf_pres=well_surf)
            well_objs = create_well_objects(wc)
        except Exception as e:
            results.append(_failed_row(wn, row, f"setup error: {e}"))
            continue

        wellbore, well_profile, inflow, res_mix, prop_pf = well_objs
        cur_fric = friction_coefs_from_chars(chars)
        ken_fixed = float(cur_fric.get("ken", 0.03))

        try:
            cal = calibrate_friction_coefs(
                well_name=wn,
                target_bhp=float(row["BHP (psi)"]),
                pwh=well_surf,
                tsu=wc.form_temp,
                ppf_surf=float(row["PF Pressure (psi)"]),
                nozzle=pump["nozzle_no"],
                throat=pump["throat_ratio"],
                knz=0.01,
                ken=ken_fixed,
                wellbore=wellbore,
                wellprof=well_profile,
                ipr_su=inflow,
                prop_su=res_mix,
                prop_pf=prop_pf,
                jpump_direction="reverse",
            )
        except Exception as e:
            results.append(_failed_row(wn, row, f"calibration error: {e}"))
            continue

        # BHP decomposition for diagnosing where the residual error lives
        # (PVT / hydrostatic vs friction). None when calibration didn't converge.
        try:
            decomp = compute_bhp_decomposition(
                cal,
                pwh=well_surf,
                tsu=wc.form_temp,
                ppf_surf=float(row["PF Pressure (psi)"]),
                wellbore=wellbore,
                wellprof=well_profile,
                prop_su=res_mix,
                prop_pf=prop_pf,
                jpump_direction="reverse",
            )
        except Exception:
            decomp = None

        results.append(
            {
                "Well": wn,
                "Pad": row["Pad"],
                "Pump": row["Pump"],
                "IPR": "well tests" if vogel_row else "defaults",
                "TVD": int(round(wc.jpump_tvd)) if wc.jpump_tvd else None,
                "Actual BHP": int(round(cal.target_bhp)),
                "Modeled BHP": int(round(cal.best_modeled_bhp)) if cal.converged else None,
                "BHP err": int(round(cal.bhp_error)) if cal.converged else None,
                "Match": cal.match_quality,
                "Bounded": cal.bounded,
                "Sonic": cal.sonic,
                "PF used": int(row["PF Pressure (psi)"]),
                "Current ken": cur_fric.get("ken"),
                "Current kth": cur_fric.get("kth"),
                "Current kdi": cur_fric.get("kdi"),
                "Cal ken": round(cal.best_ken, 4) if cal.converged else None,
                "Cal kth": round(cal.best_kth, 4) if cal.converged else None,
                "Cal kdi": round(cal.best_kdi, 4) if cal.converged else None,
                "Δ ken": (
                    round(cal.best_ken - cur_fric["ken"], 4)
                    if cal.converged and "ken" in cur_fric else None
                ),
                "Δ kth": (
                    round(cal.best_kth - cur_fric["kth"], 4)
                    if cal.converged and "kth" in cur_fric else None
                ),
                "Δ kdi": (
                    round(cal.best_kdi - cur_fric["kdi"], 4)
                    if cal.converged and "kdi" in cur_fric else None
                ),
                # BHP decomposition (psi). Helps localize where modeled BHP
                # comes from for poor-match deep wells.
                "Prod hydro": (
                    int(round(decomp["prod_hydrostatic"])) if decomp else None
                ),
                "Prod fric": (
                    int(round(decomp["prod_friction"])) if decomp else None
                ),
                "Prod ρ̄": (
                    round(decomp["rho_prod_avg"], 1) if decomp else None
                ),
                "Prod grad": (
                    round(decomp["prod_grad_psi_per_ft"], 3)
                    if decomp and decomp["prod_grad_psi_per_ft"] is not None else None
                ),
                "PF hydro": (
                    int(round(decomp["pf_hydrostatic"])) if decomp else None
                ),
                "PF fric": (
                    int(round(decomp["pf_friction"])) if decomp else None
                ),
                "Pump Δp": (
                    int(round(decomp["pump_dp"])) if decomp else None
                ),
                "Starts": cal.starts_tried,
                "Iters": cal.iterations,
                "Status": "converged" if cal.converged else "did_not_converge",
            }
        )

    progress.progress(1.0, text="Done.")
    results_df = pd.DataFrame(results)
    st.session_state["jp_cal_results_df"] = results_df
    _render_results(results_df)


def _failed_row(wn: str, row, reason: str) -> dict:
    return {
        "Well": wn,
        "Pad": row.get("Pad"),
        "Pump": row.get("Pump"),
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


def _render_results(results_df: pd.DataFrame) -> None:
    """Render calibration results, CSV download, and SQL preview."""
    n_total = len(results_df)
    n_good = (results_df["Match"] == "good").sum()
    n_fair = (results_df["Match"] == "fair").sum()
    n_poor = (results_df["Match"] == "poor").sum()
    n_fail = (results_df["Match"] == "failed").sum()
    n_bounded = int(results_df["Bounded"].fillna(False).sum())
    n_sonic = int(results_df["Sonic"].fillna(False).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", n_total)
    c2.metric("Good (≤25)", n_good, help="|BHP error| ≤ 25 psi")
    c3.metric("Fair (≤75)", n_fair, help="|BHP error| ≤ 75 psi")
    c4.metric("Poor (>75)", n_poor, help="|BHP error| > 75 psi — see Bounded/Sonic flags")
    c5.metric("Failed", n_fail, help="Solver could not produce a valid result")

    if n_bounded or n_sonic:
        st.caption(
            f"Note: {n_bounded} bounded (an optimal coef is at a search edge — "
            f"likely an IPR or geometry mismatch the calibration cannot resolve), "
            f"{n_sonic} sonic (BHP choke-pinned by throat geometry; friction "
            f"coefs cannot bring it down further)."
        )

    fmt = {
        "Current ken": "{:.3f}",
        "Current kth": "{:.3f}",
        "Current kdi": "{:.3f}",
        "Cal ken": "{:.3f}",
        "Cal kth": "{:.3f}",
        "Cal kdi": "{:.3f}",
        "Δ ken": "{:+.3f}",
        "Δ kth": "{:+.3f}",
        "Δ kdi": "{:+.3f}",
        "Prod ρ̄": "{:.1f}",
        "Prod grad": "{:.3f}",
    }
    st.dataframe(
        results_df.style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "TVD": st.column_config.NumberColumn(
                "TVD (ft)", format="%d",
                help="Pump true vertical depth — context for hydrostatic gradient",
            ),
            "Match": st.column_config.TextColumn(
                "Match",
                help="good ≤ 25 psi, fair ≤ 75, poor > 75, failed = solver error",
            ),
            "Bounded": st.column_config.CheckboxColumn(
                "Bounded",
                help="A coef is at the search bound (optimum likely outside [0.05, 1.0])",
            ),
            "Sonic": st.column_config.CheckboxColumn(
                "Sonic",
                help="Sonic flow at throat — BHP choke-pinned by geometry, not coefs",
            ),
            "Prod hydro": st.column_config.NumberColumn(
                "Prod hydro (psi)", format="%d",
                help="Hydrostatic component of the production-column dp (depth-avg "
                "mixture density × jetpump_vd). Watch for outliers vs TVD — high "
                "values relative to TVD suggest PVT density is too high.",
            ),
            "Prod fric": st.column_config.NumberColumn(
                "Prod fric (psi)", format="%d",
                help="Friction component of the production-column dp (Beggs total "
                "minus the hydrostatic estimate above).",
            ),
            "Prod ρ̄": st.column_config.NumberColumn(
                "Prod ρ̄ (lbm/ft³)", format="%.1f",
                help="Depth-averaged production-fluid density. Pure water ~62.4, "
                "Schrader oil ~50, Kuparuk oil ~55. Anomalies here flag PVT issues.",
            ),
            "Prod grad": st.column_config.NumberColumn(
                "Prod grad (psi/ft)", format="%.3f",
                help="Production hydrostatic gradient (Prod hydro / TVD). Lets you "
                "compare deep and shallow wells on the same axis. Water is ~0.433.",
            ),
            "PF hydro": st.column_config.NumberColumn(
                "PF hydro (psi)", format="%d",
                help="Hydrostatic gain of the PF column going down (single-phase "
                "water at 60°F surface — does NOT account for column heating).",
            ),
            "PF fric": st.column_config.NumberColumn(
                "PF fric (psi)", format="%d",
                help="Friction loss in the PF column.",
            ),
            "Pump Δp": st.column_config.NumberColumn(
                "Pump Δp (psi)", format="%d",
                help="Pump pressure rise (discharge minus suction at the optimum).",
            ),
            "Starts": st.column_config.NumberColumn(
                "Starts", format="%d",
                help="Seed points tried (multi-start kicks in when first pass > 50 psi)",
            ),
            "Iters": st.column_config.NumberColumn(
                "Iters", format="%d", help="Total Nelder-Mead iterations",
            ),
        },
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="jp_friction_calibration.csv",
        mime="text/csv",
        key="jp_cal_csv_download",
    )

    with st.expander("SQL preview (push to Databricks)", expanded=False):
        st.caption(
            "Replace `<target_table>` with the writable underlying table behind "
            "`vw_prop_mech` (your data team will know which one). Review carefully "
            "before running. Only converged calibrations are included."
        )
        target_table = st.text_input(
            "Target table",
            value="<target_table>",
            key="jp_cal_target_table",
            help="The actual writable table behind vw_prop_mech.",
        )
        sql = _format_sql_preview(results_df, target_table)
        st.code(sql, language="sql")

        # Direct write button (placeholder — wire up after data team confirms
        # target table and writes are gated behind ALLOW_DATABRICKS_WRITES env).
        write_enabled = os.environ.get("ALLOW_DATABRICKS_WRITES", "").lower() in (
            "1", "true", "yes"
        )
        if write_enabled:
            st.warning(
                "Direct Databricks writes are not yet implemented. "
                "Once the data team confirms the target table, this button "
                "will execute the SQL above via the SQL warehouse."
            )
            st.button(
                "Push to Databricks",
                disabled=True,
                key="jp_cal_push_db",
                help="Disabled — awaiting data team confirmation of target table.",
            )
        else:
            st.caption(
                "To enable a direct write button, set "
                "`ALLOW_DATABRICKS_WRITES=true` in the environment "
                "(after the data team confirms the target table)."
            )
