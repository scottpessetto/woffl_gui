"""Step 2.5: Pre-Calibrate Selected Wells (optional).

For each well config carried over from Step 2, fetch the most recent well
test that has both ``lift_wat`` and ``BHP`` measured, then:

1. **PF calibration** — binary-search ``ppf_surf`` such that the modeled
   nozzle rate matches the observed ``lift_wat`` at the well's current
   friction coefficients (from ``vw_prop_mech``).
2. **Friction-coef calibration** — Nelder-Mead refit of ``ken/kth/kdi``
   so the modeled BHP matches the observed BHP at that ``ppf_surf``.

Accepting the results writes the calibrated values into each WellConfig's
new ``ppf_surf_well`` / ``ken_well`` / ``kth_well`` / ``kdi_well`` fields,
which the optimizer then reads in Step 3 instead of library defaults.

Skip this step to keep the existing behavior (library-default coefs +
field-level PF pressure).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.network_optimizer import WellConfig
from woffl.gui.fric_calibration import calibrate_friction_coefs
from woffl.gui.pf_calibration import calibrate_pf_for_lift
from woffl.gui.scotts_tools._common import (
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    worker_ceiling,
)
from woffl.gui.utils import load_well_characteristics
from woffl.gui.workflow_page import _clear_downstream

DEFAULT_TEST_LOOKBACK_MONTHS = 12


def _calibrate_one(well_config: WellConfig, test_row: dict, chars: dict) -> dict:
    """Pickleable per-well worker — same shape as jp_fric_trend._calibrate_well.

    Runs PF cal then coef cal at the recovered ppf_surf. Returns a single
    result dict; catches exceptions and tags them as ``status='error'`` so
    one bad well doesn't kill the pool.
    """
    wn = well_config.well_name
    try:
        wellbore, wellprof, inflow, res_mix, prop_pf = create_well_objects(well_config)
        seed = friction_coefs_from_chars(chars)
        nozzle = str(test_row["nozzle"])
        throat = str(test_row["throat"])
        knz = float(seed.get("knz", 0.01))
        ken_seed = float(seed.get("ken", 0.03))
        kth_seed = float(seed.get("kth", 0.30))
        kdi_seed = float(seed.get("kdi", 0.30))
        whp_raw = test_row.get("whp")
        pwh = float(whp_raw) if whp_raw is not None and not pd.isna(whp_raw) else float(well_config.surf_pres)

        pf = calibrate_pf_for_lift(
            well_name=wn,
            target_lift=float(test_row["lift_wat"]),
            pwh=pwh, tsu=well_config.form_temp,
            nozzle=nozzle, throat=throat,
            knz=knz, ken=ken_seed, kth=kth_seed, kdi=kdi_seed,
            wellbore=wellbore, wellprof=wellprof,
            ipr_su=inflow, prop_su=res_mix, prop_pf=prop_pf,
        )
        coef = calibrate_friction_coefs(
            well_name=wn,
            target_bhp=float(test_row["BHP"]),
            pwh=pwh, tsu=well_config.form_temp,
            ppf_surf=float(pf.ppf_surf),
            nozzle=nozzle, throat=throat,
            knz=knz, ken=ken_seed,
            wellbore=wellbore, wellprof=wellprof,
            ipr_su=inflow, prop_su=res_mix, prop_pf=prop_pf,
        )
        # NaN-safe: `float('nan') or 0.0` returns nan (nan is truthy), so a test
        # with no oil volume showed NaN instead of 0 in the Obs Oil column.
        _oil = test_row.get("WtOilVol")
        obs_oil = float(_oil) if _oil is not None and not pd.isna(_oil) else 0.0
        return {
            "well_name": wn,
            "status": "ok",
            "test_date": test_row.get("WtDate"),
            "observed_lift_wat": float(test_row["lift_wat"]),
            "observed_bhp": float(test_row["BHP"]),
            "observed_oil": obs_oil,
            "pump": f"{nozzle}{throat}",
            "ppf_surf": float(pf.ppf_surf),
            "lift_residual": float(pf.lift_residual),
            "pf_bounded": bool(pf.bounded),
            "pf_sonic": bool(pf.sonic),
            "knz": knz,
            "ken": float(coef.best_ken),
            "kth": float(coef.best_kth),
            "kdi": float(coef.best_kdi),
            "bhp_error": float(coef.bhp_error),
            "match_quality": str(coef.match_quality),
            "coef_bounded": bool(coef.bounded),
            "coef_sonic": bool(coef.sonic),
        }
    except Exception as e:  # pragma: no cover — per-well safety net
        return {"well_name": wn, "status": "error", "error": str(e)[:200]}


def _build_tasks(
    well_configs: list[WellConfig],
    months_back: int,
    jp_history: pd.DataFrame,
    chars_dict: dict,
) -> tuple[list[tuple[WellConfig, dict, dict]], list[tuple[str, str]]]:
    """Pair each WellConfig with its latest qualifying test + current pump.

    A well is included in tasks only if:
      - it has at least one test with both lift_wat>0 and BHP populated in the
        lookback window
      - jp_history has a current pump install with both nozzle_no and throat_ratio
      - the well is present in load_well_characteristics output

    Returns (tasks, skipped). ``skipped`` is a list of (well_name, reason)
    so the UI can show why some wells didn't run.
    """
    raw = fetch_well_tests_raw(months_back)
    if raw is None or raw.empty:
        return [], [(wc.well_name, "No test data in lookback") for wc in well_configs]

    valid = raw.dropna(subset=["lift_wat", "BHP"]).copy()
    valid = valid[valid["lift_wat"] > 0]
    if valid.empty:
        return [], [
            (wc.well_name, "No test with both lift_wat>0 and BHP")
            for wc in well_configs
        ]
    latest = (
        valid.sort_values("WtDate").groupby("well").tail(1).set_index("well")
    )

    tasks: list[tuple[WellConfig, dict, dict]] = []
    skipped: list[tuple[str, str]] = []
    for wc in well_configs:
        if wc.well_name not in latest.index:
            skipped.append((wc.well_name, "No qualifying test (need lift_wat + BHP)"))
            continue
        pump = get_current_pump(jp_history, wc.well_name)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            skipped.append((wc.well_name, "No current JP pump in history"))
            continue
        chars = chars_dict.get(wc.well_name)
        if not chars:
            skipped.append((wc.well_name, "Missing from jp_chars / vw_prop_mech"))
            continue
        test = latest.loc[wc.well_name].to_dict()
        test["nozzle"] = pump["nozzle_no"]
        test["throat"] = pump["throat_ratio"]
        tasks.append((wc, test, dict(chars)))
    return tasks, skipped


def render_step2_5() -> None:
    st.subheader("Step 2.5: Pre-Calibrate (optional)")
    st.caption(
        "For each well, find the PF surface pressure needed to match the "
        "observed lift_wat in the latest test, then refit ken/kth/kdi to "
        "match the observed BHP at that PF. Calibrated values feed Step 3's "
        "optimizer instead of library defaults. Skip to keep current behavior."
    )

    well_configs: list[WellConfig] | None = st.session_state.get("uw_well_configs")
    if not well_configs:
        st.error("No well configs from Step 2. Go back and complete the IPR step.")
        return

    c1, c2, c3 = st.columns([1.4, 1.2, 2])
    with c1:
        months_back = st.slider(
            "Test lookback (months)",
            1, 24, DEFAULT_TEST_LOOKBACK_MONTHS,
            key="uw_precal_months",
        )
    with c2:
        ceiling = worker_ceiling()
        if ceiling > 1:
            workers = st.slider(
                "Workers", 1, ceiling, ceiling,
                key="uw_precal_workers",
                help=f"ProcessPool workers (cap from WOFFL_MAX_WORKERS = {ceiling}).",
            )
        else:
            workers = 1
            st.caption("Workers: 1 (set WOFFL_MAX_WORKERS to enable)")
    with c3:
        st.write("")
        st.caption(f"**{len(well_configs)}** wells from Step 2.")

    col_run, col_skip = st.columns([1.2, 1.5])
    with col_run:
        run_clicked = st.button("Run pre-calibration", type="primary")
    with col_skip:
        if st.button("Skip — use library defaults"):
            # Actually deliver "library defaults": a previous Accept mutated
            # the shared WellConfigs in place, so without this reset the old
            # calibrated overrides silently survived a later Skip.
            for wc in well_configs:
                wc.ppf_surf_well = None
                wc.knz_well = None
                wc.ken_well = None
                wc.kth_well = None
                wc.kdi_well = None
            _clear_downstream(3)
            st.session_state["uw_current_step"] = 3
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 2.5), 3
            )
            st.rerun()

    if run_clicked:
        jp_history = st.session_state.get("jp_history_df")
        if jp_history is None:
            st.error("jp_history not loaded — visit the main app first.")
            return
        chars_df = load_well_characteristics()
        if chars_df is None or chars_df.empty:
            st.error("Could not load well characteristics.")
            return
        chars_dict = chars_df.set_index("Well").to_dict("index")

        tasks, skipped = _build_tasks(well_configs, months_back, jp_history, chars_dict)
        st.session_state["uw_precal_skipped"] = skipped

        if not tasks:
            st.warning(
                "No wells have a qualifying recent test. Use Skip to advance "
                "to Step 3 with library-default coefs."
            )
            st.session_state["uw_precal_results"] = []
        else:
            n = len(tasks)
            progress = st.progress(
                0.0,
                text=f"Calibrating {n} wells "
                f"({workers} worker{'s' if workers > 1 else ''})",
            )
            results: list[dict] = []
            if workers == 1:
                for i, (wc, test, chars) in enumerate(tasks):
                    results.append(_calibrate_one(wc, test, chars))
                    progress.progress(
                        (i + 1) / n,
                        text=f"{wc.well_name} done ({i + 1}/{n})",
                    )
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futs = {
                        pool.submit(_calibrate_one, wc, test, chars): wc.well_name
                        for wc, test, chars in tasks
                    }
                    done = 0
                    for fut in as_completed(futs):
                        wn = futs[fut]
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            results.append({
                                "well_name": wn, "status": "error",
                                "error": str(e)[:200],
                            })
                        done += 1
                        progress.progress(
                            done / n, text=f"{wn} done ({done}/{n})",
                        )
            progress.empty()
            st.session_state["uw_precal_results"] = results
            st.success(f"Calibrated {sum(1 for r in results if r['status']=='ok')} of {n} wells.")

    skipped = st.session_state.get("uw_precal_skipped", [])
    results = st.session_state.get("uw_precal_results")

    if skipped:
        with st.expander(f"{len(skipped)} wells skipped (no qualifying test)"):
            st.dataframe(
                pd.DataFrame(skipped, columns=["Well", "Reason"]),
                hide_index=True, use_container_width=True,
            )

    if not results:
        return

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]

    if ok:
        disp = pd.DataFrame(ok)
        st.dataframe(
            disp, hide_index=True, use_container_width=True,
            column_config={
                "well_name": st.column_config.TextColumn("Well", pinned="left"),
                "pump": st.column_config.TextColumn("Pump"),
                "test_date": st.column_config.DatetimeColumn(
                    "Test", format="YYYY-MM-DD",
                ),
                "observed_oil": st.column_config.NumberColumn(
                    "Obs Oil (BOPD)", format="%.0f",
                    help="Observed oil rate from the latest qualifying well test.",
                ),
                "observed_lift_wat": st.column_config.NumberColumn(
                    "Obs PF (BWPD)", format="%.0f",
                    help=(
                        "Observed power-fluid (lift) water rate from the "
                        "latest qualifying well test."
                    ),
                ),
                "observed_bhp": st.column_config.NumberColumn(
                    "Obs BHP (psi)", format="%.0f",
                    help=(
                        "Observed bottom-hole pressure from the latest "
                        "qualifying well test."
                    ),
                ),
                "ppf_surf": st.column_config.NumberColumn(
                    "PF Cal (psi)", format="%.0f",
                    help=(
                        "Calibrated power-fluid surface pressure — drives "
                        "the modeled nozzle rate to match Obs PF."
                    ),
                ),
                "lift_residual": st.column_config.NumberColumn(
                    "PF Err (BWPD)", format="%+.1f",
                    help=(
                        "modeled PF rate − Obs PF after calibration. "
                        "Positive = over-prediction, negative = under. "
                        "Should be near zero on convergence."
                    ),
                ),
                "knz": st.column_config.NumberColumn("knz", format="%.4f"),
                "ken": st.column_config.NumberColumn("ken", format="%.4f"),
                "kth": st.column_config.NumberColumn("kth", format="%.4f"),
                "kdi": st.column_config.NumberColumn("kdi", format="%.4f"),
                "bhp_error": st.column_config.NumberColumn(
                    "BHP Err (psi)", format="%+.1f",
                    help=(
                        "modeled BHP − Obs BHP after friction-coef "
                        "calibration. Positive = over-prediction."
                    ),
                ),
                "match_quality": st.column_config.TextColumn("Match"),
                "pf_bounded": st.column_config.CheckboxColumn("PF Bnd"),
                "pf_sonic": st.column_config.CheckboxColumn("PF Son"),
                "coef_bounded": st.column_config.CheckboxColumn("Coef Bnd"),
                "coef_sonic": st.column_config.CheckboxColumn("Coef Son"),
            },
        )

    if err:
        with st.expander(f"{len(err)} wells errored"):
            st.dataframe(
                pd.DataFrame(err), hide_index=True, use_container_width=True,
            )

    if ok:
        col_accept, col_back = st.columns([1.2, 1])
        with col_accept:
            if st.button(
                f"Accept calibrated values → Step 3 ({len(ok)} wells)",
                type="primary",
            ):
                # Mutate WellConfigs in place — they live in session_state and
                # Step 3 reads them directly.
                configs_by_name = {wc.well_name: wc for wc in well_configs}
                for r in ok:
                    wc = configs_by_name.get(r["well_name"])
                    if wc is None:
                        continue
                    wc.ppf_surf_well = float(r["ppf_surf"])
                    wc.knz_well = float(r["knz"])
                    wc.ken_well = float(r["ken"])
                    wc.kth_well = float(r["kth"])
                    wc.kdi_well = float(r["kdi"])
                # The configs just changed — any existing optimization
                # results were computed WITHOUT these overrides; clear them
                # so Step 3's "View Results" can't show stale numbers.
                _clear_downstream(3)
                st.session_state["uw_current_step"] = 3
                st.session_state["uw_max_step_reached"] = max(
                    st.session_state.get("uw_max_step_reached", 2.5), 3
                )
                st.rerun()
        with col_back:
            st.caption(
                "Accept writes ppf_surf_well + ken/kth/kdi_well into each "
                "WellConfig. Skipped wells keep library defaults."
            )
