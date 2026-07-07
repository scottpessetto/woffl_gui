"""JP Friction Calibration tab.

Bulk friction-coef calibration for pushing back to Databricks. Calibrates
ken/kth/kdi per well so modeled BHP matches the latest measured BHP.
"""

import math
import os

import pandas as pd
import streamlit as st

from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from woffl.gui.explainers import render_kcoef_explainer
from woffl.gui.utils import PAD_PF_DEFAULTS, PAD_PF_FALLBACK
from woffl.gui.utils import default_pad_pf as _default_pad_pf
from woffl.gui.utils import live_pf_for_seed, load_well_characteristics

from ._common import (
    build_well_config,
    casing_dims_from_chars,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_vogel_for_wells,
    pad_from_mp_name,
)

# PAD_PF_DEFAULTS / PAD_PF_FALLBACK / _default_pad_pf are now imported above
# from woffl.gui.utils — single source of truth shared with sidebar.py.


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

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return None

    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

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
        case_src = "DB" if _has_databricks_casing(chars) else "fallback"
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
    lines += [
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
            min_value=1,
            max_value=24,
            value=6,
            step=1,
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
            built = _build_calibration_input_table(months_back)
        if built is None or built[0].empty:
            st.warning(
                "No JP wells with measured BHP found. Confirm JP history is "
                "loaded and wells have a recent test with BHP."
            )
            return
        input_df, pump_info_map = built
        st.session_state["jp_cal_input_df"] = input_df
        st.session_state["jp_cal_pump_info"] = pump_info_map
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
                min_value=1000,
                max_value=5000,
                value=int(
                    st.session_state.get(f"jp_cal_pad_pf_{pad}", _default_pad_pf(pad))
                ),
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
            f'(6.875" OD × 0.5" wall) because vw_prop_mech has no casing data: '
            f"{', '.join(fallback_wells[:10])}"
            + (
                f" (+{len(fallback_wells) - 10} more)"
                if len(fallback_wells) > 10
                else ""
            )
            + '. If their actual casing is different (e.g., 9-5/8"), the '
            "annulus area is wrong and PF friction will be miscomputed. Have "
            "the data team populate casing_out_dia / casing_inn_dia for these."
        )

    # [P1-29 guard] The pump modeled here is the one installed AT the BHP
    # test's date (not today's) — but pushing a fit made against a pump
    # that's since been changed out to today's `vw_prop_mech` row is
    # questionable (nozzle/throat areas differ). Flag it and default those
    # rows' Include unchecked (see _build_calibration_input_table).
    pump_changed_wells = input_df.loc[input_df["Pump changed"], "Well"].tolist()
    if pump_changed_wells:
        st.warning(
            f"{len(pump_changed_wells)} well(s) had their pump changed out "
            f"AFTER the BHP test being calibrated against: "
            f"{', '.join(pump_changed_wells[:10])}"
            + (
                f" (+{len(pump_changed_wells) - 10} more)"
                if len(pump_changed_wells) > 10
                else ""
            )
            + ". These are modeled with the pump that was actually in the "
            "hole at test time (see 'Pump changed' column), but the fitted "
            "coefficients may not carry over cleanly to the pump in the "
            "hole today — `Include` defaults OFF for these; re-check them "
            "only after confirming the geometry difference is acceptable."
        )

    edited_df = st.data_editor(
        input_df,
        key="jp_cal_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", disabled=True, pinned="left"),
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Pump": st.column_config.TextColumn(
                "Pump",
                disabled=True,
                help="Pump installed AT the BHP test's date (get_pump_at_date), "
                "not necessarily today's current pump — see 'Pump changed'.",
            ),
            "Test Date": st.column_config.DateColumn(
                "Test Date",
                disabled=True,
                format="YYYY-MM-DD",
                help="Date of the BHP test this row calibrates against — the "
                "same row's WHP is used as the surface-pressure boundary.",
            ),
            "Pump changed": st.column_config.CheckboxColumn(
                "Pump changed",
                disabled=True,
                help="True when the pump installed at the test date differs "
                "from the well's CURRENT pump — a JPCO happened since this "
                "test. Include defaults OFF for these rows.",
            ),
            "BHP (psi)": st.column_config.NumberColumn(
                "Latest BHP",
                format="%d",
                disabled=True,
                help="Most recent measured BHP — calibration target",
            ),
            "WHP (psi)": st.column_config.NumberColumn(
                "Latest WHP",
                format="%d",
                disabled=True,
            ),
            "PF Pressure (psi)": st.column_config.NumberColumn(
                "PF Pressure (psi)",
                format="%d",
                min_value=1000,
                max_value=5000,
                step=50,
                help="PF pressure to calibrate at. Overrides the pad-level value.",
            ),
            "Tube OD": st.column_config.NumberColumn(
                "Tube OD",
                format="%.3f",
                disabled=True,
                help="Tubing outer diameter (in). From vw_prop_mech.tubing_out_dia, "
                "or 4.5 fallback when missing.",
            ),
            "Tube ID": st.column_config.NumberColumn(
                "Tube ID",
                format="%.3f",
                disabled=True,
                help="Tubing inner diameter (in). Computed as Tube OD − 2×wall.",
            ),
            "Case OD": st.column_config.NumberColumn(
                "Case OD",
                format="%.3f",
                disabled=True,
                help="Casing outer diameter (in). From vw_prop_mech.casing_out_dia, "
                "or 6.875 fallback when missing.",
            ),
            "Case ID": st.column_config.NumberColumn(
                "Case ID",
                format="%.3f",
                disabled=True,
                help="Casing inner diameter (in). Drives annulus flow area together "
                "with Tube OD.",
            ),
            "Case src": st.column_config.TextColumn(
                "Case src",
                disabled=True,
                help="'DB' = pulled from vw_prop_mech, 'fallback' = used 6.875\"/0.5\" "
                "default (likely wrong if the well's actual casing isn't 7-5/8\").",
            ),
            "Ann area": st.column_config.NumberColumn(
                "Ann area (in²)",
                format="%.2f",
                disabled=True,
                help="Annulus cross-section: π/4 × (Case ID² − Tube OD²). "
                "This is what the PF-friction calc uses. Wrong inputs → wrong friction.",
            ),
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Uncheck to skip this well in the calibration run.",
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

    pump_info_map = st.session_state.get("jp_cal_pump_info")
    if not pump_info_map:
        st.error(
            "Pump info not loaded — click 'Load JP wells with measured BHP' again."
        )
        return

    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")
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

        # [P1-29 fix] Use the SAME pump resolved for this row at table-build
        # time (pump installed AT the BHP test's date) — not a fresh
        # get_current_pump(today) lookup, which would silently swap the
        # geometry out from under the row the user reviewed/approved.
        pump = pump_info_map.get(wn)
        if pump is None:
            results.append(_failed_row(wn, row, "no pump info for well"))
            continue

        # [P1-35 fix] WHP comes from the SAME test row as the BHP target
        # (paired at table-build time), not an independently-fetched
        # "latest WHP" that could be a different day's test.
        whp_val = row.get("WHP (psi)")
        well_surf = float(whp_val) if pd.notna(whp_val) else 210.0
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
                nozzle=pump["nozzle"],
                throat=pump["throat"],
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
                "Pump changed": bool(row.get("Pump changed", False)),
                "IPR": "well tests" if vogel_row else "defaults",
                "TVD": int(round(wc.jpump_tvd)) if wc.jpump_tvd else None,
                "Actual BHP": int(round(cal.target_bhp)),
                "Modeled BHP": (
                    int(round(cal.best_modeled_bhp)) if cal.converged else None
                ),
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
                    if cal.converged and "ken" in cur_fric
                    else None
                ),
                "Δ kth": (
                    round(cal.best_kth - cur_fric["kth"], 4)
                    if cal.converged and "kth" in cur_fric
                    else None
                ),
                "Δ kdi": (
                    round(cal.best_kdi - cur_fric["kdi"], 4)
                    if cal.converged and "kdi" in cur_fric
                    else None
                ),
                # BHP decomposition (psi). Helps localize where modeled BHP
                # comes from for poor-match deep wells.
                "Prod hydro": (
                    int(round(decomp["prod_hydrostatic"])) if decomp else None
                ),
                "Prod fric": (int(round(decomp["prod_friction"])) if decomp else None),
                "Prod ρ̄": (round(decomp["rho_prod_avg"], 1) if decomp else None),
                "Prod grad": (
                    round(decomp["prod_grad_psi_per_ft"], 3)
                    if decomp and decomp["prod_grad_psi_per_ft"] is not None
                    else None
                ),
                "PF hydro": (int(round(decomp["pf_hydrostatic"])) if decomp else None),
                "PF fric": (int(round(decomp["pf_friction"])) if decomp else None),
                "Pump Δp": (int(round(decomp["pump_dp"])) if decomp else None),
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
    c4.metric(
        "Poor (>75)", n_poor, help="|BHP error| > 75 psi — see Bounded/Sonic flags"
    )
    c5.metric("Failed", n_fail, help="Solver could not produce a valid result")

    if n_bounded or n_sonic:
        st.caption(
            f"Note: {n_bounded} bounded (an optimal coef is at a search edge — "
            f"likely an IPR or geometry mismatch the calibration cannot resolve), "
            f"{n_sonic} sonic (BHP choke-pinned by throat geometry; friction "
            f"coefs cannot bring it down further)."
        )

    n_pump_changed = int(
        results_df.get("Pump changed", pd.Series(dtype=bool)).fillna(False).sum()
    )
    if n_pump_changed:
        st.caption(
            f"{n_pump_changed} well(s) below have 'Pump changed' set — the "
            "fitted coefficients are excluded from the SQL preview below "
            "regardless of convergence (see the SQL comment for the list)."
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
                "TVD (ft)",
                format="%d",
                help="Pump true vertical depth — context for hydrostatic gradient",
            ),
            "Match": st.column_config.TextColumn(
                "Match",
                help="good ≤ 25 psi, fair ≤ 75, poor > 75, failed = solver error",
            ),
            "Pump changed": st.column_config.CheckboxColumn(
                "Pump changed",
                help="Pump was changed out after the calibrated test — "
                "excluded from the SQL preview regardless of convergence.",
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
                "Prod hydro (psi)",
                format="%d",
                help="Hydrostatic component of the production-column dp (depth-avg "
                "mixture density × jetpump_vd). Watch for outliers vs TVD — high "
                "values relative to TVD suggest PVT density is too high.",
            ),
            "Prod fric": st.column_config.NumberColumn(
                "Prod fric (psi)",
                format="%d",
                help="Friction component of the production-column dp (Beggs total "
                "minus the hydrostatic estimate above).",
            ),
            "Prod ρ̄": st.column_config.NumberColumn(
                "Prod ρ̄ (lbm/ft³)",
                format="%.1f",
                help="Depth-averaged production-fluid density. Pure water ~62.4, "
                "Schrader oil ~50, Kuparuk oil ~55. Anomalies here flag PVT issues.",
            ),
            "Prod grad": st.column_config.NumberColumn(
                "Prod grad (psi/ft)",
                format="%.3f",
                help="Production hydrostatic gradient (Prod hydro / TVD). Lets you "
                "compare deep and shallow wells on the same axis. Water is ~0.433.",
            ),
            "PF hydro": st.column_config.NumberColumn(
                "PF hydro (psi)",
                format="%d",
                help="Hydrostatic gain of the PF column going down (single-phase "
                "water at 60°F surface — does NOT account for column heating).",
            ),
            "PF fric": st.column_config.NumberColumn(
                "PF fric (psi)",
                format="%d",
                help="Friction loss in the PF column.",
            ),
            "Pump Δp": st.column_config.NumberColumn(
                "Pump Δp (psi)",
                format="%d",
                help="Pump pressure rise (discharge minus suction at the optimum).",
            ),
            "Starts": st.column_config.NumberColumn(
                "Starts",
                format="%d",
                help="Seed points tried (multi-start kicks in when first pass > 50 psi)",
            ),
            "Iters": st.column_config.NumberColumn(
                "Iters",
                format="%d",
                help="Total Nelder-Mead iterations",
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
            "1",
            "true",
            "yes",
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
