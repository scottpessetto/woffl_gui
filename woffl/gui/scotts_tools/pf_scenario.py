"""PF Scenario Analysis tab.

Compares oil rates and PF rates at two different PF pressures per well.
Three-tier IPR resolution: (1) Vogel from BHP gauge data, (2) JP back-calc
BHP for gaugeless wells, (3) jp_chars defaults as last resort.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.network_optimizer import WellConfig
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.gui.utils import create_pvt_components, load_well_characteristics
from woffl.pvt.resmix import ResMix

from ._common import (
    build_well_config,
    casing_dims_from_chars,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_latest_bhp_per_well,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    normalize_short_name,
    pad_from_mp_name,
)


# ── CSV parsing ────────────────────────────────────────────────────────────


def _parse_scenario_csv(uploaded_file) -> pd.DataFrame | None:
    """Parse scenario CSV.

    Column A: well names  (e.g. B-30, I-15)
    Column B: Scenario A PF pressures (psi)
    Column C: Scenario B PF pressures (psi)

    Returns DataFrame with columns: short_name, well_name, pad, pf_pres_a, pf_pres_b
    """
    df = pd.read_csv(uploaded_file, header=None)
    if len(df.columns) < 3:
        st.error("CSV must have 3 columns: well name, Scenario A pressure, Scenario B pressure.")
        return None

    records = []
    for _, row in df.iterrows():
        raw = str(row.iloc[0]).strip()
        if not raw or raw.lower() == "nan":
            continue
        try:
            pf_a = float(row.iloc[1])
            pf_b = float(row.iloc[2])
        except (ValueError, TypeError):
            st.warning(f"Skipping '{raw}' — non-numeric PF pressure.")
            continue
        mp = normalize_short_name(raw)
        records.append(
            {
                "short_name": raw,
                "well_name": mp,
                "pad": pad_from_mp_name(mp),
                "pf_pres_a": pf_a,
                "pf_pres_b": pf_b,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


# ── gaugeless BHP estimation ──────────────────────────────────────────────


def _discharge_residual_fixed_rate(
    psu, qoil_std, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, prop_su, prop_pf,
    jpump_direction="reverse",
):
    """Jet pump discharge residual with a FIXED oil rate (no IPR).

    Mirrors solopump.discharge_residual but replaces the IPR lookup with
    a known oil rate so we can back-calculate suction pressure.
    """
    import woffl.assembly.solopump as so
    from woffl.flow import jetflow as jf
    from woffl.flow import outflow as of
    from woffl.flow import singlephase as sp
    from woffl.flow.jetplot import JetBook

    prod_path = "tubing" if jpump_direction == "reverse" else "annulus"
    pf_path = "annulus" if jpump_direction == "reverse" else "tubing"
    ate = jpump.ath - jpump.anz

    # Throat entry with fixed oil rate (replaces throat_entry_zero_tde)
    prop_c = prop_su.condition(psu, tsu)
    qtot = sum(prop_c.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)
    te_book = JetBook(psu, vte, prop_c.rho_mix(), prop_c.cmix(), jf.enterance_ke(jpump.ken, vte))

    pdec = 25
    while (te_book.tde_ray[-1] > 0) and (te_book.prs_ray[-1] > 50):
        pte_step = te_book.prs_ray[-1] - pdec
        prop_c = prop_c.condition(pte_step, tsu)
        qtot = sum(prop_c.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)
        te_book.append(pte_step, vte, prop_c.rho_mix(), prop_c.cmix(), jf.enterance_ke(jpump.ken, vte))

    pte, vte, rho_te, mach_te = te_book.dete_zero()

    # Power fluid iteration (identical to solopump.discharge_residual)
    dp_stat = sp.diff_press_static(prop_pf.density, -1 * wellprof.jetpump_vd)
    qpf_list = [2000.0, 3000.0]
    res_pf = []
    for qpf in qpf_list:
        r, vnz, pni = so.powerfluid_residual(qpf, pte, ppf_surf, tsu, dp_stat, jpump, wellbore, wellprof, prop_pf, pf_path)
        res_pf.append(r)
    from woffl.flow.errors import ConvergenceError

    n = 0
    while abs(res_pf[-1]) > 5:
        qpf = so.qpf_secant(qpf_list[-2], qpf_list[-1], res_pf[-2], res_pf[-1])
        r, vnz, pni = so.powerfluid_residual(qpf, pte, ppf_surf, tsu, dp_stat, jpump, wellbore, wellprof, prop_pf, pf_path)
        qpf_list.append(qpf)
        res_pf.append(r)
        n += 1
        if n == 20:  # uncapped, this loop could hang the whole app process
            raise ConvergenceError("power fluid rate did not converge")

    qnz = qpf_list[-1]
    wc_tm, _ = jf.throat_wc(qoil_std, prop_su.wc, qnz)

    prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)
    ptm = jf.throat_discharge(pte, tsu, jpump.kth, vnz, jpump.anz, prop_pf.density, vte, ate, rho_te, prop_tm)
    _, pdi_jp = jf.diffuser_discharge(ptm, tsu, jpump.kdi, jpump.ath, wellbore.inn_pipe.inn_area, qoil_std, prop_tm)

    # Outflow
    _, prs_ray, _ = of.production_top_down_press(pwh, tsu, qoil_std, prop_tm, wellbore, wellprof, prod_path)
    pdi_of = prs_ray[-1]

    return pdi_jp - pdi_of


def _estimate_bhp(
    qoil_std, wc, fgor, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, prop_pf,
    field_model="Schrader", psu_max=1800.0,
):
    """Estimate suction pressure (BHP) from known production + jet pump conditions.

    Binary-searches suction pressure until the jet pump discharge matches
    the outflow requirement at the known oil rate.
    """
    oil, water, gas = create_pvt_components(field_model)
    prop_su = ResMix(wc=wc, fgor=fgor, oil=oil, wat=water, gas=gas)

    psu_lo, psu_hi = 100.0, psu_max

    def _safe_residual(psu):
        try:
            return _discharge_residual_fixed_rate(
                psu, qoil_std, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, prop_su, prop_pf,
            )
        except Exception:
            return None

    res_lo = _safe_residual(psu_lo)
    res_hi = _safe_residual(psu_hi)
    if res_lo is None or res_hi is None:
        return None
    # Both negative → pump can't lift this rate at any psu
    if res_lo < 0 and res_hi < 0:
        return None
    # Both positive → well is sonic, return low bound
    if res_lo > 0 and res_hi > 0:
        return psu_lo

    for _ in range(30):
        psu_mid = (psu_lo + psu_hi) / 2
        if abs(psu_hi - psu_lo) < 5:
            return psu_mid
        res_mid = _safe_residual(psu_mid)
        if res_mid is None:
            psu_hi = psu_mid
            continue
        if res_mid > 0:
            psu_lo = psu_mid
        else:
            psu_hi = psu_mid

    return (psu_lo + psu_hi) / 2


def _estimate_gaugeless_ipr(
    missing_wells, months_back, test_pf_pres, jp_hist, jp_chars_dict,
    whp_map: dict[str, float] | None = None,
    pf_map: dict[str, float] | None = None,
):
    """For wells without BHP gauges, estimate BHP from production + jet pump physics.

    Returns dict of {well_name: synthetic_vogel_row} compatible with the
    Vogel dict used elsewhere.

    ``pf_map`` optionally supplies a per-well power-fluid pressure (psi); a well
    absent from it falls back to the scalar ``test_pf_pres``. The Header Pressure
    Impact tab passes per-pad PF here so the back-calc matches each pad's actual
    power-fluid pressure instead of one global value.
    """
    from woffl.assembly.network_optimizer import _load_well_profile

    try:
        raw_df = fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw_df is None or raw_df.empty:
        return {}

    whp_map = whp_map or {}
    results = {}
    for wn in missing_wells:
        well_tests = raw_df[raw_df["well"] == wn].copy()
        # Keep only rows that have production but NO BHP
        well_tests = well_tests[well_tests["BHP"].isna() & well_tests["WtTotalFluid"].notna()]
        if well_tests.empty:
            continue

        # Most recent test
        latest = well_tests.sort_values("WtDate", ascending=False).iloc[0]
        oil_rate = latest.get("WtOilVol", 0)
        total_fluid = latest["WtTotalFluid"]
        if total_fluid <= 0 or pd.isna(oil_rate) or oil_rate <= 0:
            continue

        wc = latest.get("form_wc", np.nan)
        if pd.isna(wc) and total_fluid > 0:
            water = latest.get("WtWaterVol", 0) or 0
            wc = water / total_fluid if total_fluid > 0 else 0.5
        fgor = latest.get("fgor", 250)
        if pd.isna(fgor):
            fgor = 250

        # Get pump config
        pump = get_current_pump(jp_hist, wn)
        if pump is None or pump["nozzle_no"] is None:
            continue

        # Get well geometry
        chars = jp_chars_dict.get(wn)
        if not chars:
            continue

        # Per-well friction coefficients (vw_prop_mech), fall back to JetPump defaults
        fric_coefs = friction_coefs_from_chars(chars)
        jpump = JetPump(pump["nozzle_no"], pump["throat_ratio"], **fric_coefs)

        is_sch = chars.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        fm = "Schrader" if is_sch else "Kuparuk"
        tsu = float(chars.get("form_temp", 75 if is_sch else 170))
        jpump_md = float(chars.get("JP_MD", chars.get("JP_TVD", 4000)))

        tube = Pipe(out_dia=float(chars.get("out_dia", 4.5)), thick=float(chars.get("thick", 0.271)))
        casing_od, casing_thk = casing_dims_from_chars(chars)
        case = Pipe(out_dia=casing_od, thick=casing_thk)
        wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
        well_profile = _load_well_profile(wn, jpump_md, fm)
        _, water_obj, _ = create_pvt_components(fm)
        prop_pf = water_obj.condition(0, 60)

        # Per-well WHP: use the test row's whp first (most consistent with the
        # BHP we're back-calculating), then map fallback, then 210
        test_whp = latest.get("whp")
        if test_whp is None or pd.isna(test_whp):
            pwh = float(whp_map.get(wn, 210.0))
        else:
            pwh = float(test_whp)

        # Use oil rate (not total fluid) as qoil_std for the solver
        psu_max = float(chars.get("res_pres", 1800))
        pf_for_well = float((pf_map or {}).get(wn, test_pf_pres))
        bhp_est = _estimate_bhp(
            oil_rate, wc, fgor, pwh, tsu, pf_for_well,
            jpump, wellbore, well_profile, prop_pf,
            field_model=fm, psu_max=psu_max,
        )
        if bhp_est is None:
            continue

        # Estimate reservoir pressure as BHP + cushion (simple heuristic)
        res_pres = max(bhp_est + 200, psu_max)

        results[wn] = {
            "ResP": res_pres,
            "qwf": total_fluid,
            "pwf": bhp_est,
            "form_wc": round(wc, 3),
            "fgor": fgor,
            "QMax_recent": 0,
            "QMax_lowest_bhp": 0,
            "QMax_median": 0,
            "num_tests": len(well_tests),
            "R2": 0,
            "_bhp_estimated": True,
        }

    return results


# ── solver ────────────────────────────────────────────────────────────────


def _solve_at_pf(
    wc: WellConfig,
    well_objects: tuple,
    nozzle: str,
    throat: str,
    pf_pres: float,
    fric_coefs: dict | None = None,
) -> dict:
    """Run solver for one well / one pump / one PF pressure.

    Args:
        fric_coefs: Optional {knz, ken, kth, kdi} from vw_prop_mech.
            Missing keys fall through to JetPump class defaults.
    """
    wellbore, well_profile, inflow, res_mix, prop_pf = well_objects
    jp = JetPump(nozzle, throat, **(fric_coefs or {}))
    batch = BatchPump(
        pwh=wc.surf_pres,
        tsu=wc.form_temp,
        ppf_surf=pf_pres,
        wellbore=wellbore,
        wellprof=well_profile,
        ipr_su=inflow,
        prop_su=res_mix,
        prop_pf=prop_pf,
        wellname=wc.well_name,
    )
    result_df = batch.batch_run([jp])
    if result_df.empty:
        return dict(oil=np.nan, pf_rate=np.nan, psu=np.nan, sonic=False, mach=np.nan)
    r = result_df.iloc[0]
    return dict(
        oil=float(r["qoil_std"]) if pd.notna(r["qoil_std"]) else np.nan,
        pf_rate=float(r["lift_wat"]) if pd.notna(r["lift_wat"]) else np.nan,
        psu=float(r["psu_solv"]) if pd.notna(r["psu_solv"]) else np.nan,
        sonic=bool(r["sonic_status"]) if pd.notna(r.get("sonic_status")) else False,
        mach=float(r["mach_te"]) if pd.notna(r.get("mach_te")) else np.nan,
    )


# ── main tab ───────────────────────────────────────────────────────────────


def render_tab():
    st.header("Power Fluid Scenario Comparison")
    st.markdown(
        "Upload a CSV — **Col A**: well name (e.g. `B-30`), "
        "**Col B**: Scenario A PF pressure (psi), "
        "**Col C**: Scenario B PF pressure (psi)."
    )

    # Sample CSV download
    sample = "B-30,3400,3200\nI-15,2700,3400\nI-22,2700,3400\nC-14,3400,3200\nJ-27,3400,3200"
    st.download_button(
        "Download sample CSV",
        data=sample,
        file_name="pf_scenario_sample.csv",
        mime="text/csv",
    )

    csv_file = st.file_uploader("Scenario CSV", type=["csv"], key="pf_scenario_csv")
    if csv_file is None:
        return

    scenario_df = _parse_scenario_csv(csv_file)
    if scenario_df is None or scenario_df.empty:
        st.error("Could not parse the CSV. Check format: 3 rows — wells, PF-A, PF-B.")
        return

    # Show parsed input
    st.subheader("Parsed Input")
    display_input = scenario_df[["well_name", "pad", "pf_pres_a", "pf_pres_b"]].copy()
    display_input.columns = ["Well", "Pad", "Scenario A PF (psi)", "Scenario B PF (psi)"]
    st.dataframe(display_input, use_container_width=True, hide_index=True)

    # Settings
    col_a, col_b = st.columns(2)
    with col_a:
        months_back = st.number_input(
            "Well test lookback (months)",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
            key="pf_scenario_months",
            help="How far back to pull well tests for IPR fitting.",
        )
    with col_b:
        test_pf_pres = st.number_input(
            "PF pressure at test time (psi)",
            min_value=1000,
            max_value=5000,
            value=3400,
            step=100,
            key="pf_scenario_test_pf",
            help="PF pressure when well tests were run. Used to estimate BHP for wells without gauges.",
        )

    # Friction-coefficient mode: per-well from Databricks (default), manual
    # override sliders applied to all wells, or auto-calibrate per well to
    # actual BHP from the most recent measured test.
    fric_mode = st.radio(
        "Friction-coef source",
        options=["databricks", "manual", "calibrate"],
        format_func=lambda m: {
            "databricks": "Databricks defaults (per-well jpfric_*)",
            "manual": "Manual override (sliders, applied to all wells)",
            "calibrate": "Auto-calibrate per well to actual BHP",
        }[m],
        key="pf_scenario_fric_mode",
        horizontal=False,
    )

    fric_override_values: dict | None = None

    if fric_mode == "manual":
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            ken_ov = st.slider(
                "ken (entry)", min_value=0.0, max_value=0.20, value=0.03, step=0.005,
                key="pf_scenario_ken",
                help="Higher ken → higher modeled BHP, lower oil. JetPump default 0.03.",
            )
        with fc2:
            kth_ov = st.slider(
                "kth (throat)", min_value=0.05, max_value=1.0, value=0.30, step=0.01,
                key="pf_scenario_kth",
                help="Higher kth → higher modeled BHP, lower oil. JetPump default 0.30.",
            )
        with fc3:
            kdi_ov = st.slider(
                "kdi (diffuser)", min_value=0.05, max_value=1.0, value=0.30, step=0.01,
                key="pf_scenario_kdi",
                help="Higher kdi → higher modeled BHP, lower oil. JetPump default 0.30.",
            )
        fric_override_values = {"ken": ken_ov, "kth": kth_ov, "kdi": kdi_ov}

    elif fric_mode == "calibrate":
        st.caption(
            "Per-well Nelder-Mead calibration sweeps kth and kdi at the test "
            "PF pressure (set above) to drive modeled BHP toward each well's "
            "most recent measured BHP. Wells without measured BHP keep their "
            "Databricks defaults. ~15 solver evals per well."
        )

    # Prerequisites
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.error("JP history not loaded. Check Databricks connection or upload JP history Excel in the sidebar.")
        return

    if not st.button("Run Scenario Analysis", type="primary", use_container_width=True):
        # Re-render the last run's results — without this, ANY rerun after
        # the run (including clicking the results CSV download button)
        # erased the multi-minute analysis from the screen.
        prev = st.session_state.get("pf_scenario_results")
        if prev is not None and not prev.empty:
            st.caption(
                "Showing results from the last run this session — "
                "click **Run Scenario Analysis** to recompute."
            )
            _render_results(prev)
        return

    # ── run analysis ───────────────────────────────────────────────────
    from woffl.gui.fric_calibration import calibrate_friction_coefs

    well_names = scenario_df["well_name"].tolist()

    # Load jp_chars lookup
    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

    # Pull latest test WHP per well for surface pressure
    with st.spinner("Pulling latest WHP per well from well tests..."):
        whp_map = get_latest_whp_per_well(months_back)

    # Pull latest test BHP per well — only used when calibrating
    bhp_map: dict[str, float] = {}
    if fric_mode == "calibrate":
        with st.spinner("Pulling latest measured BHP per well..."):
            bhp_map = get_latest_bhp_per_well(months_back)

    # Compute Vogel IPR from well test data (wells with BHP gauges)
    with st.spinner(f"Fetching well tests ({months_back} months) and computing IPR..."):
        vogel_dict = get_vogel_for_wells(well_names, months_back)

    # Estimate BHP for gaugeless wells using jet pump back-calculation
    missing = [w for w in well_names if w not in vogel_dict]
    if missing:
        with st.spinner(f"Estimating BHP for {len(missing)} gaugeless wells..."):
            gaugeless = _estimate_gaugeless_ipr(
                missing, months_back, test_pf_pres, jp_hist, jp_chars_dict,
                whp_map=whp_map,
            )
            vogel_dict.update(gaugeless)

    # Report IPR sources
    gauged = [w for w in well_names if w in vogel_dict and not vogel_dict[w].get("_bhp_estimated")]
    estimated = [w for w in well_names if w in vogel_dict and vogel_dict[w].get("_bhp_estimated")]
    fallback = [w for w in well_names if w not in vogel_dict]
    if gauged:
        st.caption(f"IPR from BHP gauge: {', '.join(gauged)}")
    if estimated:
        st.caption(f"IPR from JP back-calc (estimated BHP): {', '.join(estimated)}")
    if fallback:
        st.caption(f"Using jp_chars defaults (no test data): {', '.join(fallback)}")
    no_whp = [w for w in well_names if w not in whp_map]
    if no_whp:
        st.caption(f"Surface pressure fallback (210 psi, no WHP in tests): {', '.join(no_whp)}")
    if fric_mode == "calibrate":
        no_bhp = [w for w in well_names if w not in bhp_map]
        if no_bhp:
            st.caption(
                f"No measured BHP — using Databricks defaults (no calibration): "
                f"{', '.join(no_bhp)}"
            )

    # ── Phase 1: build per-well configs ────────────────────────────────
    configured: dict[str, dict] = {}
    errors: list[str] = []
    for _, row in scenario_df.iterrows():
        wn = row["well_name"]
        pump = get_current_pump(jp_hist, wn)
        if pump is None or pump["nozzle_no"] is None:
            errors.append(f"{wn}: no current pump in JP history")
            continue
        nozzle = pump["nozzle_no"]
        throat = pump["throat_ratio"]
        chars = jp_chars_dict.get(wn)
        well_surf_pres = float(whp_map.get(wn, 210.0))
        try:
            wc = build_well_config(
                wn, jp_chars_dict, vogel_dict.get(wn), surf_pres=well_surf_pres
            )
            well_objs = create_well_objects(wc)
        except (ValueError, KeyError, Exception) as e:
            errors.append(f"{wn}: setup error — {e}")
            continue
        configured[wn] = dict(
            wc=wc,
            well_objs=well_objs,
            nozzle=nozzle,
            throat=throat,
            chars=chars,
            row=row,
            well_surf_pres=well_surf_pres,
        )

    # ── Phase 2: calibrate (only when mode == calibrate) ──────────────
    cal_results: dict[str, object] = {}
    if fric_mode == "calibrate":
        eligible = [w for w in configured if w in bhp_map]
        if eligible:
            cal_prog = st.progress(0, text="Calibration: starting...")
            for i, wn in enumerate(eligible):
                cfg = configured[wn]
                cal_prog.progress(
                    i / len(eligible),
                    text=f"Calibrating {wn} ({i+1}/{len(eligible)})...",
                )
                wellbore, well_profile, inflow, res_mix, prop_pf = cfg["well_objs"]
                # ken from Databricks (or default), knz fixed at 0.01
                fc = friction_coefs_from_chars(cfg["chars"])
                ken_fixed = float(fc.get("ken", 0.03))
                try:
                    result = calibrate_friction_coefs(
                        well_name=wn,
                        target_bhp=float(bhp_map[wn]),
                        pwh=cfg["wc"].surf_pres,
                        tsu=cfg["wc"].form_temp,
                        ppf_surf=float(test_pf_pres),
                        nozzle=cfg["nozzle"],
                        throat=cfg["throat"],
                        knz=0.01,
                        ken=ken_fixed,
                        wellbore=wellbore,
                        wellprof=well_profile,
                        ipr_su=inflow,
                        prop_su=res_mix,
                        prop_pf=prop_pf,
                        jpump_direction="reverse",
                    )
                    cal_results[wn] = result
                except Exception as e:
                    errors.append(f"{wn}: calibration error — {e}")
                    cal_results[wn] = None
            cal_prog.progress(1.0, text="Calibration done.")
        st.session_state["pf_scenario_calibration"] = cal_results

    # ── Phase 3: solve each well at both scenarios ────────────────────
    results = []
    progress = st.progress(0, text="Starting...")
    total = len(configured)

    for i, (wn, cfg) in enumerate(configured.items()):
        progress.progress(i / max(total, 1), text=f"Solving {wn}...")
        nozzle = cfg["nozzle"]
        throat = cfg["throat"]
        wc = cfg["wc"]
        well_objs = cfg["well_objs"]
        chars = cfg["chars"]
        well_surf_pres = cfg["well_surf_pres"]
        row = cfg["row"]

        # Resolve friction coefs by mode
        fric_source = "databricks"
        cal_target_bhp = None
        cal_modeled_bhp = None
        cal_kth = None
        cal_kdi = None

        if fric_override_values is not None:
            fric_coefs = dict(fric_override_values)
            fric_source = "manual"
        elif fric_mode == "calibrate" and cal_results.get(wn) is not None and cal_results[wn].converged:
            cal = cal_results[wn]
            fric_coefs = {
                "knz": cal.knz,
                "ken": cal.best_ken,
                "kth": cal.best_kth,
                "kdi": cal.best_kdi,
            }
            fric_source = "calibrated"
            cal_target_bhp = cal.target_bhp
            cal_modeled_bhp = cal.best_modeled_bhp
            cal_kth = cal.best_kth
            cal_kdi = cal.best_kdi
        else:
            fric_coefs = friction_coefs_from_chars(chars)

        try:
            res_a = _solve_at_pf(wc, well_objs, nozzle, throat, row["pf_pres_a"], fric_coefs)
            res_b = _solve_at_pf(wc, well_objs, nozzle, throat, row["pf_pres_b"], fric_coefs)
        except Exception as e:
            errors.append(f"{wn}: solver error — {e}")
            continue

        results.append(
            {
                "Well": wn,
                "Pad": row["pad"],
                "Nozzle": nozzle,
                "Throat": throat,
                "WHP (psi)": int(round(well_surf_pres)),
                "Fric Source": fric_source,
                "kth used": fric_coefs.get("kth", np.nan),
                "kdi used": fric_coefs.get("kdi", np.nan),
                "BHP actual": cal_target_bhp,
                "BHP cal": cal_modeled_bhp,
                "BHP cal err": (
                    cal_modeled_bhp - cal_target_bhp
                    if cal_modeled_bhp is not None and cal_target_bhp is not None
                    else None
                ),
                "PF Pres A": int(row["pf_pres_a"]),
                "Oil A (BOPD)": res_a["oil"],
                "PF Rate A (BWPD)": res_a["pf_rate"],
                "BHP A (psi)": res_a["psu"],
                "Sonic A": res_a["sonic"],
                "Mach A": res_a["mach"],
                "PF Pres B": int(row["pf_pres_b"]),
                "Oil B (BOPD)": res_b["oil"],
                "PF Rate B (BWPD)": res_b["pf_rate"],
                "BHP B (psi)": res_b["psu"],
                "Sonic B": res_b["sonic"],
                "Mach B": res_b["mach"],
                "Delta Oil": res_b["oil"] - res_a["oil"],
                "Delta PF Rate": res_b["pf_rate"] - res_a["pf_rate"],
            }
        )

    progress.progress(1.0, text="Done.")

    if errors:
        with st.expander(f"{len(errors)} wells skipped", expanded=False):
            for e in errors:
                st.write(f"- {e}")

    if not results:
        st.error("No wells could be solved. Check errors above.")
        return

    results_df = pd.DataFrame(results)
    st.session_state["pf_scenario_results"] = results_df

    _render_results(results_df)


# ── results display ───────────────────────────────────────────────────────


def _render_results(results_df: pd.DataFrame):
    """Display scenario comparison results."""
    # Summary metrics
    st.subheader("Field Summary")
    total_oil_a = results_df["Oil A (BOPD)"].sum()
    total_oil_b = results_df["Oil B (BOPD)"].sum()
    total_pf_a = results_df["PF Rate A (BWPD)"].sum()
    total_pf_b = results_df["PF Rate B (BWPD)"].sum()
    delta_oil = total_oil_b - total_oil_a
    delta_pf = total_pf_b - total_pf_a

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenario A Oil", f"{total_oil_a:,.0f} BOPD")
    c2.metric("Scenario B Oil", f"{total_oil_b:,.0f} BOPD", f"{delta_oil:+,.0f}")
    c3.metric("Scenario A PF", f"{total_pf_a:,.0f} BWPD")
    c4.metric("Scenario B PF", f"{total_pf_b:,.0f} BWPD", f"{delta_pf:+,.0f}")

    # Pad subtotals
    st.subheader("By Pad")
    pad_agg = (
        results_df.groupby("Pad")
        .agg(
            {
                "Oil A (BOPD)": "sum",
                "Oil B (BOPD)": "sum",
                "PF Rate A (BWPD)": "sum",
                "PF Rate B (BWPD)": "sum",
                "Delta Oil": "sum",
                "Delta PF Rate": "sum",
            }
        )
        .reset_index()
    )
    st.dataframe(
        pad_agg.style.format(
            {
                "Oil A (BOPD)": "{:,.0f}",
                "Oil B (BOPD)": "{:,.0f}",
                "PF Rate A (BWPD)": "{:,.0f}",
                "PF Rate B (BWPD)": "{:,.0f}",
                "Delta Oil": "{:+,.0f}",
                "Delta PF Rate": "{:+,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Well details table
    st.subheader("Well Details")
    fmt = {
        "Oil A (BOPD)": "{:,.0f}",
        "Oil B (BOPD)": "{:,.0f}",
        "PF Rate A (BWPD)": "{:,.0f}",
        "PF Rate B (BWPD)": "{:,.0f}",
        "BHP A (psi)": "{:,.0f}",
        "BHP B (psi)": "{:,.0f}",
        "Mach A": "{:.2f}",
        "Mach B": "{:.2f}",
        "BHP actual": "{:,.0f}",
        "BHP cal": "{:,.0f}",
        "BHP cal err": "{:+,.0f}",
        "kth used": "{:.3f}",
        "kdi used": "{:.3f}",
        "Delta Oil": "{:+,.0f}",
        "Delta PF Rate": "{:+,.0f}",
    }
    st.dataframe(results_df.style.format(fmt), use_container_width=True, hide_index=True)

    # Bar chart — delta oil by well, colored by pad
    st.subheader("Oil Rate Change by Well (Scenario B vs A)")
    pad_colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for i, pad in enumerate(sorted(results_df["Pad"].unique())):
        pad_colors[pad] = palette[i % len(palette)]

    fig = go.Figure()
    for pad in sorted(results_df["Pad"].unique()):
        mask = results_df["Pad"] == pad
        sub = results_df[mask]
        fig.add_trace(
            go.Bar(
                x=sub["Well"],
                y=sub["Delta Oil"],
                name=f"Pad {pad}",
                marker_color=pad_colors[pad],
                hovertemplate="%{x}<br>Delta Oil: %{y:,.0f} BOPD<extra></extra>",
            )
        )
    fig.update_layout(
        yaxis_title="Delta Oil (BOPD)",
        xaxis_title="Well",
        barmode="group",
        height=400,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # CSV download
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_bytes,
        file_name="pf_scenario_results.csv",
        mime="text/csv",
    )
