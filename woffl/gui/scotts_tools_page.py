"""Scott's Tools — hidden utilities that run on the Databricks server.

Unlocked via the easter egg input in the sidebar.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.ipr_analyzer import compute_vogel_coefficients, estimate_reservoir_pressure
from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.network_optimizer import WellConfig, _load_well_profile
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.gui.utils import create_pvt_components, load_well_characteristics
from woffl.pvt.resmix import ResMix


def run_scotts_tools_page():
    """Render the Scott's Tools page."""
    st.title("Scott's Tools")
    st.caption("You found the secret menu.")

    tab_labels = ["PF Scenario Analysis", "Tool 2", "Tool 3"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_pf_scenario_tab()

    with tabs[1]:
        st.header("Tool 2")
        st.info("Placeholder.")

    with tabs[2]:
        st.header("Tool 3")
        st.info("Placeholder.")


# ── name helpers ───────────────────────────────────────────────────────────


def _normalize_short_name(name: str) -> str:
    """Convert shorthand well names to MP format.

    B-30 -> MPB-30, I-15 -> MPI-15.  Already-prefixed names pass through.
    """
    name = name.strip()
    if name.upper().startswith("MP"):
        return name
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    if m:
        return f"MP{m.group(1).upper()}-{int(m.group(2))}"
    return name


def _pad_from_mp_name(mp_name: str) -> str:
    """MPB-30 -> B, MPI-15 -> I."""
    return mp_name.replace("MP", "").split("-")[0]


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
        mp = _normalize_short_name(raw)
        records.append(
            {
                "short_name": raw,
                "well_name": mp,
                "pad": _pad_from_mp_name(mp),
                "pf_pres_a": pf_a,
                "pf_pres_b": pf_b,
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


# ── IPR from well tests ───────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_well_tests(months_back: int):
    """Fetch well tests with BHP filter. Cached 24h per months_back value."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    df, _ = fetch_milne_well_tests(start_date, end_date)
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_well_tests_raw(months_back: int):
    """Fetch well tests WITHOUT dropping gaugeless rows. Cached 24h."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import (
        WELL_TEST_QUERY,
        _normalize_well_name,
        get_mpu_well_names,
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    well_names = get_mpu_well_names()
    if not well_names:
        return pd.DataFrame()
    well_list = ", ".join(f"'{w}'" for w in well_names)
    df = execute_query(WELL_TEST_QUERY.format(well_list=well_list, start_date=start_date, end_date=end_date))
    if df.empty:
        return df

    rename = {"well_name": "well", "wt_date": "WtDate", "bhp": "BHP", "oil_rate": "WtOilVol", "fwat_rate": "WtWaterVol", "fgas_rate": "WtGasVol"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "WtOilVol" in df.columns and "WtWaterVol" in df.columns:
        df["WtTotalFluid"] = df["WtOilVol"] + df["WtWaterVol"]
    if "well" in df.columns:
        df["well"] = df["well"].apply(_normalize_well_name)
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], utc=True).dt.tz_localize(None)
    for col in ["BHP", "WtOilVol", "WtWaterVol", "WtTotalFluid", "WtGasVol", "lift_wat", "whp", "fgor", "form_wc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only require well + date + fluid rate (NOT BHP)
    req = [c for c in ["well", "WtDate", "WtTotalFluid"] if c in df.columns]
    df = df.dropna(subset=req)
    return df.sort_values(by=["well", "WtDate"])


def _get_vogel_for_wells(well_names: list[str], months_back: int = 3) -> dict:
    """Return {well_name: vogel_row_dict} for wells with BHP gauge data."""
    try:
        wt_df = _fetch_well_tests(months_back)
    except Exception:
        wt_df = st.session_state.get("all_well_tests_df")

    if wt_df is None or wt_df.empty:
        return {}

    jp_chars_df = load_well_characteristics()
    filtered = wt_df[wt_df["well"].isin(well_names)].copy()
    if filtered.empty:
        return {}

    try:
        merged = estimate_reservoir_pressure(filtered, jp_chars=jp_chars_df)
        vogel = compute_vogel_coefficients(merged)
        return vogel.set_index("Well").to_dict("index")
    except Exception:
        return {}


# ── gaugeless BHP estimation ─────────────────────────────────────────────


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
    while abs(res_pf[-1]) > 5:
        qpf = so.qpf_secant(qpf_list[-2], qpf_list[-1], res_pf[-2], res_pf[-1])
        r, vnz, pni = so.powerfluid_residual(qpf, pte, ppf_surf, tsu, dp_stat, jpump, wellbore, wellprof, prop_pf, pf_path)
        qpf_list.append(qpf)
        res_pf.append(r)

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
):
    """For wells without BHP gauges, estimate BHP from production + jet pump physics.

    Returns dict of {well_name: synthetic_vogel_row} compatible with the
    Vogel dict used elsewhere.
    """
    try:
        raw_df = _fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw_df is None or raw_df.empty:
        return {}

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
        jpump = JetPump(pump["nozzle_no"], pump["throat_ratio"])

        # Get well geometry
        chars = jp_chars_dict.get(wn)
        if not chars:
            continue
        is_sch = chars.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        fm = "Schrader" if is_sch else "Kuparuk"
        tsu = float(chars.get("form_temp", 75 if is_sch else 170))
        jpump_md = float(chars.get("JP_MD", chars.get("JP_TVD", 4000)))
        jpump_tvd = float(chars.get("JP_TVD", 4000))

        tube = Pipe(out_dia=float(chars.get("out_dia", 4.5)), thick=float(chars.get("thick", 0.271)))
        case = Pipe(out_dia=6.875, thick=0.5)
        wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
        well_profile = _load_well_profile(wn, jpump_md, fm)
        _, water_obj, _ = create_pvt_components(fm)
        prop_pf = water_obj.condition(0, 60)

        # Use oil rate (not total fluid) as qoil_std for the solver
        psu_max = float(chars.get("res_pres", 1800))
        bhp_est = _estimate_bhp(
            oil_rate, wc, fgor, 210.0, tsu, test_pf_pres,
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


# ── well config + solver ──────────────────────────────────────────────────


def _build_well_config(
    well_name: str, jp_chars_dict: dict, vogel_row: dict | None = None
) -> WellConfig:
    """Build WellConfig from jp_chars, optionally overriding IPR with Vogel data."""
    chars = jp_chars_dict.get(well_name)
    if not chars:
        raise ValueError(f"{well_name} not in jp_chars database")

    is_sch = chars.get("is_sch", True)
    if isinstance(is_sch, str):
        is_sch = is_sch.lower() in ("true", "1", "yes")
    fm = "Schrader" if is_sch else "Kuparuk"

    params = dict(
        well_name=well_name,
        res_pres=float(chars.get("res_pres", 1800)),
        form_temp=float(chars.get("form_temp", 75 if is_sch else 170)),
        jpump_tvd=float(chars["JP_TVD"]),
        jpump_md=float(chars.get("JP_MD", chars["JP_TVD"])),
        tubing_od=float(chars.get("out_dia", 4.5)),
        tubing_thickness=float(chars.get("thick", 0.271)),
        field_model=fm,
        surf_pres=210.0,
        form_wc=0.5,
        form_gor=250.0,
        qwf=750.0,
        pwf=500.0,
    )

    # Override with well-test-derived IPR when available
    if vogel_row:
        params["res_pres"] = float(vogel_row["ResP"])
        params["form_wc"] = float(vogel_row.get("form_wc", 0.5))
        params["form_gor"] = float(vogel_row.get("fgor", 250))
        params["qwf"] = float(vogel_row["qwf"])
        params["pwf"] = float(vogel_row["pwf"])

    return WellConfig(**params)


def _create_well_objects(wc: WellConfig):
    """Create simulation objects (mirrors NetworkOptimizer._create_well_objects)."""
    tube = Pipe(out_dia=wc.tubing_od, thick=wc.tubing_thickness)
    case = Pipe(out_dia=wc.casing_od, thick=wc.casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)

    jpump_md = wc.jpump_md if wc.jpump_md else wc.jpump_tvd
    well_profile = _load_well_profile(wc.well_name, jpump_md, wc.field_model)

    oil_qwf = wc.qwf * (1 - wc.form_wc)
    inflow = InFlow(qwf=oil_qwf, pwf=wc.pwf, pres=wc.res_pres)

    oil, water, gas = create_pvt_components(wc.field_model)
    res_mix = ResMix(wc=wc.form_wc, fgor=wc.form_gor, oil=oil, wat=water, gas=gas)
    prop_pf = water.condition(0, 60)

    return wellbore, well_profile, inflow, res_mix, prop_pf


def _solve_at_pf(wc: WellConfig, well_objects: tuple, nozzle: str, throat: str, pf_pres: float) -> dict:
    """Run solver for one well / one pump / one PF pressure."""
    wellbore, well_profile, inflow, res_mix, prop_pf = well_objects
    jp = JetPump(nozzle, throat)
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


def _render_pf_scenario_tab():
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

    # Prerequisites
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.error("JP history not loaded. Check Databricks connection or upload JP history Excel in the sidebar.")
        return

    if not st.button("Run Scenario Analysis", type="primary", use_container_width=True):
        return

    # ── run analysis ───────────────────────────────────────────────────
    well_names = scenario_df["well_name"].tolist()

    # Load jp_chars lookup
    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

    # Compute Vogel IPR from well test data (wells with BHP gauges)
    with st.spinner(f"Fetching well tests ({months_back} months) and computing IPR..."):
        vogel_dict = _get_vogel_for_wells(well_names, months_back)

    # Estimate BHP for gaugeless wells using jet pump back-calculation
    missing = [w for w in well_names if w not in vogel_dict]
    if missing:
        with st.spinner(f"Estimating BHP for {len(missing)} gaugeless wells..."):
            gaugeless = _estimate_gaugeless_ipr(
                missing, months_back, test_pf_pres, jp_hist, jp_chars_dict,
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

    # Solve each well at both scenarios
    results = []
    errors = []
    progress = st.progress(0, text="Starting...")
    total = len(scenario_df)

    for i, (_, row) in enumerate(scenario_df.iterrows()):
        wn = row["well_name"]
        progress.progress(i / total, text=f"Solving {wn}...")

        # Current pump from JP history
        pump = get_current_pump(jp_hist, wn)
        if pump is None or pump["nozzle_no"] is None:
            errors.append(f"{wn}: no current pump in JP history")
            continue

        nozzle = pump["nozzle_no"]
        throat = pump["throat_ratio"]

        # Build WellConfig
        try:
            wc = _build_well_config(wn, jp_chars_dict, vogel_dict.get(wn))
        except (ValueError, KeyError) as e:
            errors.append(f"{wn}: {e}")
            continue

        # Create well objects once, solve at both PF pressures
        try:
            well_objs = _create_well_objects(wc)
            res_a = _solve_at_pf(wc, well_objs, nozzle, throat, row["pf_pres_a"])
            res_b = _solve_at_pf(wc, well_objs, nozzle, throat, row["pf_pres_b"])
        except Exception as e:
            errors.append(f"{wn}: solver error — {e}")
            continue

        results.append(
            {
                "Well": wn,
                "Pad": row["pad"],
                "Nozzle": nozzle,
                "Throat": throat,
                "PF Pres A": int(row["pf_pres_a"]),
                "Oil A (BOPD)": res_a["oil"],
                "PF Rate A (BWPD)": res_a["pf_rate"],
                "PF Pres B": int(row["pf_pres_b"]),
                "Oil B (BOPD)": res_b["oil"],
                "PF Rate B (BWPD)": res_b["pf_rate"],
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
