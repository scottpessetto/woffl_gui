"""JP Wash-Out Detection tab.

For every JP producer with a recent well test that measures lift_wat, find
the power-fluid surface pressure required for the modeled nozzle rate to
match the observed lift_wat at the well's current friction coefficients.
Wells needing more than the threshold (default 3400 psi — surface PF
infrastructure cap) are flagged as likely washed-out jet pumps that need
changeout.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from woffl.gui.pf_calibration import calibrate_pf_for_lift
from woffl.gui.utils import PAD_PF_FALLBACK, default_pad_pf, load_well_characteristics

from ._common import (
    build_well_config,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_vogel_for_wells,
    pad_from_mp_name,
    worker_ceiling,
)


def _build_scan_input(months_back: int) -> pd.DataFrame | None:
    """Latest lift_wat-bearing test per current-JP well.

    Returns one row per well with the most-recent test that has a non-zero
    lift_wat measurement, joined with the well's current pump from
    ``vw_jet_pump_history`` and characteristics from
    ``load_well_characteristics()``. Non-JP wells (no current pump) and
    wells missing from jp_chars are dropped.
    """
    raw = fetch_well_tests_raw(months_back)
    if raw is None or raw.empty or "lift_wat" not in raw.columns:
        return None
    valid = raw.dropna(subset=["lift_wat"]).copy()
    valid = valid[valid["lift_wat"] > 0]
    if valid.empty:
        return None
    latest = valid.sort_values("WtDate").groupby("well").tail(1).copy()

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return None

    jp_chars_df = load_well_characteristics()
    if jp_chars_df is None or jp_chars_df.empty:
        return None
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

    # Vogel IPR per well — without it the calibration ran against the
    # generic WC=0.5/GOR=250/qwf=750 defaults and the PpfRequired numbers
    # absorbed the IPR error.
    vogel_map = get_vogel_for_wells(latest["well"].unique().tolist(), months_back)

    rows = []
    for _, t in latest.iterrows():
        wn = t["well"]
        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            continue
        chars = jp_chars_dict.get(wn)
        if not chars:
            continue
        # The lift_wat was measured through the pump installed AT the test —
        # use it (fall back to the current pump when no record covers the date).
        pump_at = get_pump_at_date(jp_hist, wn, t["WtDate"])
        if not (pump_at and pump_at.get("nozzle_no") and pump_at.get("throat_ratio")):
            pump_at = pump
        pump_changed = str(pump_at["nozzle_no"]) != str(pump["nozzle_no"]) or str(
            pump_at["throat_ratio"]
        ) != str(pump["throat_ratio"])
        whp = float(t["whp"]) if pd.notna(t.get("whp")) else 210.0
        rows.append(
            {
                "Well": wn,
                "Pad": pad_from_mp_name(wn),
                "Pump": f"{pump_at['nozzle_no']}{pump_at['throat_ratio']}",
                "Nozzle": str(pump_at["nozzle_no"]),
                "Throat": str(pump_at["throat_ratio"]),
                "PumpChangedSinceTest": pump_changed,
                "_vogel": vogel_map.get(wn),
                "WtDate": t["WtDate"],
                "Oil": float(t.get("WtOilVol") or 0.0),
                "Water": float(t.get("WtWaterVol") or 0.0),
                "Gas": float(t.get("WtGasVol") or 0.0),
                "LiftWat": float(t["lift_wat"]),
                "WHP": whp,
                "BHP": float(t["BHP"]) if pd.notna(t.get("BHP")) else None,
                "_chars": chars,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["Pad", "Well"]).reset_index(drop=True)


def _calibrate_one(row_dict: dict) -> dict:
    """Module-level pickleable worker for one well's PF calibration.

    Takes a plain dict (not a Series — pandas Series can carry extra state
    that pickles awkwardly) and returns the full result row. Errors are
    caught here so a single bad well doesn't kill the whole scan.
    """
    wn = row_dict["Well"]
    chars = row_dict["_chars"]
    # P1-32: carry the pump-changed flag computed in _build_scan_input
    # through into the result row (both success and error branches) — it
    # was previously computed there and never propagated, so a well whose
    # pump had already been changed out since the scanned test still got
    # flagged for a changeout that already happened.
    pump_changed = bool(row_dict.get("PumpChangedSinceTest", False))
    try:
        wc = build_well_config(
            wn, {wn: chars}, row_dict.get("_vogel"), surf_pres=row_dict["WHP"]
        )
        wellbore, well_profile, inflow, res_mix, prop_pf = create_well_objects(wc)
        cur = friction_coefs_from_chars(chars)
        result = calibrate_pf_for_lift(
            well_name=wn,
            target_lift=row_dict["LiftWat"],
            pwh=row_dict["WHP"],
            tsu=wc.form_temp,
            nozzle=row_dict["Nozzle"],
            throat=row_dict["Throat"],
            knz=cur.get("knz", 0.01),
            ken=cur.get("ken", 0.03),
            kth=cur.get("kth", 0.30),
            kdi=cur.get("kdi", 0.30),
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
        )
        return {
            "Well": wn,
            "Pad": row_dict["Pad"],
            "Pump": row_dict["Pump"],
            "PumpChangedSinceTest": pump_changed,
            "WtDate": row_dict["WtDate"],
            "Oil": row_dict["Oil"],
            "Water": row_dict["Water"],
            "Gas": row_dict["Gas"],
            "LiftWat": row_dict["LiftWat"],
            "BHP": row_dict["BHP"],
            "WHP": row_dict["WHP"],
            "PpfRequired": result.ppf_surf,
            "ModeledQnz": result.modeled_qnz,
            "LiftResidual": result.lift_residual,
            "Converged": result.converged,
            "Bounded": result.bounded,
            "Sonic": result.sonic,
            "Iterations": result.iterations,
            "Status": "ok",
            "Error": "",
        }
    except Exception as e:
        return {
            "Well": wn,
            "Pad": row_dict["Pad"],
            "Pump": row_dict["Pump"],
            "PumpChangedSinceTest": pump_changed,
            "WtDate": row_dict["WtDate"],
            "Oil": row_dict["Oil"],
            "Water": row_dict["Water"],
            "Gas": row_dict["Gas"],
            "LiftWat": row_dict["LiftWat"],
            "BHP": row_dict["BHP"],
            "WHP": row_dict["WHP"],
            "PpfRequired": float("nan"),
            "ModeledQnz": float("nan"),
            "LiftResidual": float("nan"),
            "Converged": False,
            "Bounded": False,
            "Sonic": False,
            "Iterations": 0,
            "Status": "error",
            "Error": str(e)[:200],
        }


def _run_scan(input_df: pd.DataFrame, workers: int = 1) -> pd.DataFrame:
    """Iterate the scan input and calibrate each well.

    Workers > 1 dispatches via ProcessPoolExecutor (one task per well).
    Workers == 1 stays sequential with no pool overhead. Streamlit
    progress is updated in the main thread off of ``as_completed``.
    """
    task_dicts = [row.to_dict() for _, row in input_df.iterrows()]
    rows: list[dict] = []
    n = len(task_dicts)
    progress = st.progress(
        0.0,
        text=f"Starting scan ({workers} worker{'s' if workers > 1 else ''})",
    )
    done = 0
    if workers == 1:
        for rd in task_dicts:
            rows.append(_calibrate_one(rd))
            done += 1
            progress.progress(
                done / n,
                text=f"{rd['Well']} done ({done}/{n})",
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_calibrate_one, rd): rd["Well"] for rd in task_dicts}
            for fut in as_completed(futs):
                wn = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:  # pragma: no cover — defensive
                    rows.append(
                        {
                            "Well": wn,
                            "Status": "error",
                            "Error": f"worker crashed: {str(e)[:160]}",
                            "PpfRequired": float("nan"),
                            "ModeledQnz": float("nan"),
                            "LiftResidual": float("nan"),
                            "Converged": False,
                            "Bounded": False,
                            "Sonic": False,
                            "Iterations": 0,
                        }
                    )
                done += 1
                progress.progress(
                    done / n,
                    text=f"{wn} done ({done}/{n})",
                )
    progress.empty()
    return pd.DataFrame(rows)


def render_tab() -> None:
    st.header("JP Wash-Out Detection")
    st.caption(
        "For each JP producer's latest test with measured lift_wat, "
        "binary-search the power-fluid surface pressure (ppf_surf) needed "
        "for the modeled nozzle rate to match the observed lift_wat at the "
        "well's current friction coefficients. Wells needing more than the "
        "threshold are candidates for jet-pump changeout."
    )

    c1, c2, c3 = st.columns([1.3, 1.5, 1.5])
    with c1:
        months_back = st.slider(
            "Test lookback (months)",
            1,
            24,
            6,
            key="jp_washout_months",
            help="How far back to look for a recent lift-wat test per well.",
        )
    with c2:
        ceiling = worker_ceiling()
        if ceiling > 1:
            workers = st.slider(
                "Parallel workers",
                1,
                ceiling,
                ceiling,
                key="jp_washout_workers",
                help=f"ProcessPool workers. Capped by WOFFL_MAX_WORKERS "
                f"(current={ceiling}). Defaults to the cap; drag down to "
                "throttle.",
            )
        else:
            workers = 1
            st.caption("Workers: 1 (set WOFFL_MAX_WORKERS to enable)")
    with c3:
        st.write("")
        if st.button("Refresh inputs", help="Clear well-test cache and reload"):
            try:
                fetch_well_tests_raw.clear()
            except Exception:
                pass
            st.session_state.pop("jp_washout_results", None)
            st.session_state.pop("_jp_washout_scan_input", None)
            st.rerun()

    # Session-memoized scan input: st.tabs runs this tool's body on every
    # rerun of the Scott's Tools page, and _build_scan_input does per-well
    # pump lookups + the Vogel IPR fits each time. Refresh (above) and a
    # lookback change invalidate it.
    try:
        memo = st.session_state.get("_jp_washout_scan_input")
        if memo is not None and memo.get("months") == months_back:
            input_df = memo["df"]
        else:
            input_df = _build_scan_input(months_back)
            # Memoize SUCCESS only — a None (jp_history not loaded yet /
            # transient fetch failure) must keep recomputing so the tool
            # self-heals once the data appears.
            if input_df is not None and not input_df.empty:
                st.session_state["_jp_washout_scan_input"] = {
                    "months": months_back,
                    "df": input_df,
                }
    except Exception as e:
        st.error(f"Could not load well tests: {e}")
        return

    if input_df is None or input_df.empty:
        st.warning(
            "No JP wells with measured lift_wat in the lookback window. "
            "Increase the lookback months or check that jp_history is loaded."
        )
        return

    st.caption(f"{len(input_df)} JP wells with lift_wat ready to scan.")

    # Per-pad wash-out thresholds — mirrors the pad PF inputs on JP Friction
    # Calibration. Each pad's threshold defaults to PAD_PF_DEFAULTS so a B pad
    # well isn't held to the C pad's 3400-psi bar.
    st.markdown("**Pad wash-out thresholds (psi)**")
    st.caption(
        "Each pad's threshold defaults to the pad's operating PF. Edits "
        "update the wash-out flag column below without re-scanning."
    )
    pads_in_scope = sorted(input_df["Pad"].dropna().unique().tolist())
    n_cols = min(6, max(1, len(pads_in_scope)))
    pad_cols = st.columns(n_cols)
    pad_thresholds: dict[str, int] = {}
    for i, pad in enumerate(pads_in_scope):
        with pad_cols[i % n_cols]:
            pad_thresholds[pad] = st.number_input(
                f"Pad {pad}",
                min_value=1000,
                max_value=5000,
                value=int(
                    st.session_state.get(
                        f"jp_washout_pad_th_{pad}", default_pad_pf(pad)
                    )
                ),
                step=50,
                key=f"jp_washout_pad_th_{pad}",
            )

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_clicked = st.button("Run wash-out scan", type="primary")
    with col_clear:
        if (
            "jp_washout_results" in st.session_state
            and not st.session_state["jp_washout_results"].empty
        ):
            if st.button("Clear results"):
                st.session_state.pop("jp_washout_results", None)
                st.rerun()

    if run_clicked:
        results = _run_scan(input_df, workers=workers)
        st.session_state["jp_washout_results"] = results

    results = st.session_state.get("jp_washout_results")
    if results is None or results.empty:
        st.info(
            "Click 'Run wash-out scan' to start. Each well takes ~6 seconds — "
            "expect ~10 minutes for ~100 wells."
        )
        return

    # Reactive flag: PpfRequired > per-pad threshold. Bounded-high case
    # has ppf=ppf_hi=5000, which exceeds any reasonable pad threshold and
    # gets flagged. Bounded-low has ppf=1000, below any threshold, stays
    # unflagged.
    disp = results.copy()
    # Defensive: a cached jp_washout_results from before P1-32 wired this
    # column through won't have it yet.
    if "PumpChangedSinceTest" not in disp.columns:
        disp["PumpChangedSinceTest"] = False
    disp["PumpChangedSinceTest"] = (
        disp["PumpChangedSinceTest"].fillna(False).astype(bool)
    )

    disp["PadThreshold"] = (
        disp["Pad"].map(pad_thresholds).fillna(PAD_PF_FALLBACK).astype(int)
    )
    over_threshold = disp["PpfRequired"] > disp["PadThreshold"]
    # P1-32: PumpChangedSinceTest was computed in _build_scan_input but
    # never propagated or used — a well whose current pump already differs
    # from the pump modeled at the scanned test has ALREADY been changed
    # out, so recommending a changeout for it is stale advice. Exclude
    # those rows from the actionable flag; they're still visible (and
    # explained) via the "Pump Changed" column + the metric/caption below.
    disp["FlagWashOut"] = over_threshold & ~disp["PumpChangedSinceTest"]
    disp = disp.sort_values(
        "PpfRequired", ascending=False, na_position="last"
    ).reset_index(drop=True)

    n_flagged = int(disp["FlagWashOut"].sum())
    n_bounded = int(disp["Bounded"].sum())
    n_sonic = int(disp["Sonic"].sum())
    n_error = int((disp["Status"] == "error").sum())
    n_stale = int((over_threshold & disp["PumpChangedSinceTest"]).sum())
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Wells scanned", len(disp))
    m2.metric("Flagged wash-out", n_flagged)
    m3.metric("Bounded", n_bounded, help="Binary search hit ppf_lo or ppf_hi")
    m4.metric("Sonic", n_sonic)
    m5.metric("Errored", n_error)
    m6.metric(
        "Pump already changed",
        n_stale,
        help="Over the pad threshold, but the pump modeled (installed at "
        "the scanned test) has already been changed out since — excluded "
        "from 'Flagged wash-out' since that recommendation is already "
        "moot. See the 'Pump Changed' column.",
    )
    if n_stale:
        st.caption(
            f"{n_stale} well(s) exceeded their pad's PF threshold but the "
            "pump has already been changed out since the scanned test — "
            "not counted in 'Flagged wash-out'."
        )

    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Pump": st.column_config.TextColumn("Pump"),
            "PumpChangedSinceTest": st.column_config.CheckboxColumn(
                "Pump Changed",
                help="Current pump differs from the pump modeled (installed "
                "at the scanned test's date) — a changeout has already "
                "happened, so 'Wash-out?' is suppressed for this row.",
            ),
            "WtDate": st.column_config.DatetimeColumn(
                "Test Date",
                format="YYYY-MM-DD",
            ),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            "Water": st.column_config.NumberColumn("Water (BWPD)", format="%.0f"),
            "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
            "LiftWat": st.column_config.NumberColumn(
                "Lift Wat (BWPD)",
                format="%.0f",
                help="Observed power-fluid rate from well test",
            ),
            "BHP": st.column_config.NumberColumn("BHP (psi)", format="%.0f"),
            "WHP": st.column_config.NumberColumn("WHP (psi)", format="%.0f"),
            "PpfRequired": st.column_config.NumberColumn(
                "PF Required (psi)",
                format="%.0f",
                help="ppf_surf that makes modeled qnz match observed lift_wat",
            ),
            "PadThreshold": st.column_config.NumberColumn(
                "Pad Threshold (psi)",
                format="%.0f",
                help="Wash-out threshold for this well's pad (see inputs above)",
            ),
            "FlagWashOut": st.column_config.CheckboxColumn(
                "Wash-out?",
                help="PF Required exceeds the pad's threshold AND the pump "
                "hasn't already been changed out since the scanned test "
                "(see 'Pump Changed')",
            ),
            "Bounded": st.column_config.CheckboxColumn(
                "Bounded",
                help="Binary search hit ppf_lo (1000) or ppf_hi (5000)",
            ),
            "Sonic": st.column_config.CheckboxColumn(
                "Sonic",
                help="Throat at sonic velocity at the recovered ppf_surf",
            ),
            "ModeledQnz": st.column_config.NumberColumn(
                "Modeled Qnz (BWPD)",
                format="%.0f",
            ),
            "LiftResidual": st.column_config.NumberColumn(
                "Residual (BWPD)",
                format="%.1f",
                help="Modeled qnz - target lift_wat",
            ),
            "Iterations": st.column_config.NumberColumn("Iters", format="%.0f"),
            "Status": st.column_config.TextColumn("Status"),
            "Error": st.column_config.TextColumn(
                "Error",
                help="Error message if scan failed",
            ),
        },
    )

    st.download_button(
        "Download CSV",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name=f"jp_washout_scan_{pd.Timestamp.now():%Y_%m_%d}.csv",
        mime="text/csv",
        key="jp_washout_dl",
    )
