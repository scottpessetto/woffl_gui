"""Scott's Tools — hidden utilities that run on the Databricks server.

Unlocked via the easter egg input in the sidebar.
"""

import os
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

    tab_labels = [
        "PF Scenario Analysis",
        "JP Friction Calibration",
        "Well Sort",
        "Pad Water Cut",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_pf_scenario_tab()

    with tabs[1]:
        _render_jp_calibration_tab()

    with tabs[2]:
        _render_well_sort_tab()

    with tabs[3]:
        _render_pad_watercut_tab()


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


def _get_latest_whp_per_well(months_back: int) -> dict[str, float]:
    """Return {well_name: latest_whp} from raw well tests within the lookback window.

    Pulls from the unfiltered (BHP-not-required) raw test set so wells without
    BHP gauges still get a real WHP value when one is available.
    """
    try:
        raw = _fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw is None or raw.empty or "whp" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["whp"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").last()
    return {w: float(latest.loc[w, "whp"]) for w in latest.index}


def _get_latest_bhp_per_well(months_back: int) -> dict[str, float]:
    """Return {well_name: latest_bhp} for wells with measured BHP in the window.

    Used as the calibration target when "Auto-calibrate per well" is selected.
    Wells absent from the returned dict have no measured BHP and cannot be
    friction-coef-calibrated.
    """
    try:
        raw = _fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw is None or raw.empty or "BHP" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["BHP"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").last()
    return {w: float(latest.loc[w, "BHP"]) for w in latest.index}


def _friction_coefs_from_chars(chars: dict | None) -> dict:
    """Extract jet-pump friction coefficients from a jp_chars row.

    Reads the Databricks vw_prop_mech columns (jpfric_*); missing/NaN values
    are omitted so JetPump falls back to its class defaults (knz=0.01,
    ken=0.03, kth=0.3, kdi=0.3).
    """
    if not chars:
        return {}
    mapping = {
        "knz": "jpfric_nozzle",
        "ken": "jpfric_entry",
        "kth": "jpfric_throat",
        "kdi": "jpfric_diffuser",
    }
    out: dict = {}
    for kw, col in mapping.items():
        v = chars.get(col)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if pd.isna(fv):
            continue
        out[kw] = fv
    return out


def _casing_dims_from_chars(chars: dict | None) -> tuple[float, float]:
    """Return (casing_od, casing_thickness) from chars; fallback 6.875 / 0.5."""
    if chars:
        out_dia = chars.get("casing_out_dia")
        inn_dia = chars.get("casing_inn_dia")
        try:
            if out_dia is not None and inn_dia is not None:
                od_f = float(out_dia)
                id_f = float(inn_dia)
                if not pd.isna(od_f) and not pd.isna(id_f) and od_f > id_f > 0:
                    return od_f, (od_f - id_f) / 2.0
        except (TypeError, ValueError):
            pass
    return 6.875, 0.5


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
    whp_map: dict[str, float] | None = None,
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
        fric_coefs = _friction_coefs_from_chars(chars)
        jpump = JetPump(pump["nozzle_no"], pump["throat_ratio"], **fric_coefs)

        is_sch = chars.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        fm = "Schrader" if is_sch else "Kuparuk"
        tsu = float(chars.get("form_temp", 75 if is_sch else 170))
        jpump_md = float(chars.get("JP_MD", chars.get("JP_TVD", 4000)))
        jpump_tvd = float(chars.get("JP_TVD", 4000))

        tube = Pipe(out_dia=float(chars.get("out_dia", 4.5)), thick=float(chars.get("thick", 0.271)))
        casing_od, casing_thk = _casing_dims_from_chars(chars)
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
        bhp_est = _estimate_bhp(
            oil_rate, wc, fgor, pwh, tsu, test_pf_pres,
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
    well_name: str,
    jp_chars_dict: dict,
    vogel_row: dict | None = None,
    surf_pres: float = 210.0,
) -> WellConfig:
    """Build WellConfig from jp_chars, optionally overriding IPR with Vogel data.

    Args:
        well_name: Well identifier (e.g. "MPB-30")
        jp_chars_dict: Dict from load_well_characteristics().set_index("Well")
        vogel_row: Optional Vogel coefficient row to override IPR
        surf_pres: Surface (wellhead) pressure in psi (typically pulled from
            latest well test by the caller)
    """
    chars = jp_chars_dict.get(well_name)
    if not chars:
        raise ValueError(f"{well_name} not in jp_chars database")

    is_sch = chars.get("is_sch", True)
    if isinstance(is_sch, str):
        is_sch = is_sch.lower() in ("true", "1", "yes")
    fm = "Schrader" if is_sch else "Kuparuk"

    casing_od, casing_thk = _casing_dims_from_chars(chars)

    params = dict(
        well_name=well_name,
        res_pres=float(chars.get("res_pres", 1800)),
        form_temp=float(chars.get("form_temp", 75 if is_sch else 170)),
        jpump_tvd=float(chars["JP_TVD"]),
        jpump_md=float(chars.get("JP_MD", chars["JP_TVD"])),
        tubing_od=float(chars.get("out_dia", 4.5)),
        tubing_thickness=float(chars.get("thick", 0.271)),
        casing_od=casing_od,
        casing_thickness=casing_thk,
        field_model=fm,
        surf_pres=float(surf_pres),
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
        return

    # ── run analysis ───────────────────────────────────────────────────
    from woffl.gui.fric_calibration import calibrate_friction_coefs

    well_names = scenario_df["well_name"].tolist()

    # Load jp_chars lookup
    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")

    # Pull latest test WHP per well for surface pressure
    with st.spinner("Pulling latest WHP per well from well tests..."):
        whp_map = _get_latest_whp_per_well(months_back)

    # Pull latest test BHP per well — only used when calibrating
    bhp_map: dict[str, float] = {}
    if fric_mode == "calibrate":
        with st.spinner("Pulling latest measured BHP per well..."):
            bhp_map = _get_latest_bhp_per_well(months_back)

    # Compute Vogel IPR from well test data (wells with BHP gauges)
    with st.spinner(f"Fetching well tests ({months_back} months) and computing IPR..."):
        vogel_dict = _get_vogel_for_wells(well_names, months_back)

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
            wc = _build_well_config(
                wn, jp_chars_dict, vogel_dict.get(wn), surf_pres=well_surf_pres
            )
            well_objs = _create_well_objects(wc)
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
                fc = _friction_coefs_from_chars(cfg["chars"])
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
            fric_coefs = _friction_coefs_from_chars(chars)

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


# ── JP Friction Calibration tab ──────────────────────────────────────────


# Pad-level default PF pressures (psi). These reflect known pad operating
# conditions: C/E/H/I/M/S run at 3400, B/G/J at 2200 (booster pads), F at
# 2800. Pad K has no jet pumps so it won't appear in the calibration list.
PAD_PF_DEFAULTS: dict[str, int] = {
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
PAD_PF_FALLBACK = 3400


def _default_pad_pf(pad: str) -> int:
    """Default PF pressure (psi) for a given pad."""
    return PAD_PF_DEFAULTS.get(pad, PAD_PF_FALLBACK)


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
    bhp_map = _get_latest_bhp_per_well(months_back)
    if not bhp_map:
        return None
    whp_map = _get_latest_whp_per_well(months_back)

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
        pad = _pad_from_mp_name(wn)

        # Geometry being passed into the solver — surface for verification
        chars = jp_chars_dict.get(wn, {})
        tube_od = float(chars.get("out_dia") or 4.5)
        tube_thk = float(chars.get("thick") or 0.271)
        tube_id = tube_od - 2 * tube_thk
        case_od, case_thk = _casing_dims_from_chars(chars)
        case_id = case_od - 2 * case_thk
        case_src = "DB" if _has_databricks_casing(chars) else "fallback"
        # Annulus cross-section (in²) — what the PF friction calc actually uses
        import math
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


def _render_jp_calibration_tab() -> None:
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

    with st.expander("What do these coefficients represent?", expanded=False):
        st.markdown(
            """
The friction coefficients are **dimensionless energy-loss factors** in
the four pressure-drop stages of the jet pump. Each captures the fraction
of dynamic head lost to friction/turbulence in its section — higher value
means a less efficient (more lossy) component.

**The four coefficients:**

- **`knz` (nozzle)** — held fixed at 0.01. Loss as power fluid accelerates
  through the nozzle. Primarily affects PF flow rate and nozzle exit
  velocity. Default 0.01 is good when measured PF rates match the model.
- **`ken` (entrance / suction)** — calibrated, range [0.005, 0.20]. Loss
  as formation fluid enters the throat from the suction side. Higher
  `ken` means it's harder for produced fluid to flow into the throat →
  pump can't pull suction pressure as far down → **higher modeled BHP**.
  Affects drawdown directly.
- **`kth` (throat / mixing)** — calibrated, range [0.05, 1.0]. Loss
  during mixing of high-velocity power fluid with low-velocity formation
  fluid in the throat (constant-area mixing chamber). The biggest
  dissipative section in a jet pump. Higher `kth` means worse momentum
  transfer → less pressure built up downstream → pump needs higher
  suction (BHP). Affects both BHP and PF rate.
- **`kdi` (diffuser)** — calibrated, range [0.05, 1.0]. Loss as the
  mixed stream decelerates in the diverging diffuser, converting kinetic
  energy back into static pressure. Higher `kdi` means less pressure
  recovery → lower discharge pressure → pump needs more suction (BHP) to
  lift fluid out. Primarily affects BHP.

**Why these values change in practice:**

The defaults come from idealized Cunningham-style jet pump theory. Real
pumps deviate because of:

- **Wear / erosion** — sand or solids enlarging or roughening the
  throat/diffuser surfaces
- **Scale / deposits** — restricting flow areas, increasing turbulence
- **Manufacturing tolerances** — actual nozzle/throat geometry differs
  slightly from catalog
- **Fluid-property assumptions** — viscosity, density, or two-phase
  effects not captured by single-phase loss correlations
- **Geometry simplifications** — the model uses a 1D approximation; real
  flow has 3D structure

Calibrating per-well fits a one-number-per-component "wear / efficiency
factor" so the model matches actual measured BHP. The coefficients absorb
whatever the pump physics + simplified model couldn't predict from
spec-sheet geometry alone.
"""
        )

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
            _render_jp_cal_results(results_df)
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
    whp_map = _get_latest_whp_per_well(months_back)

    results = []
    progress = st.progress(0, text="Starting...")
    n = len(selected)
    for i, (_, row) in enumerate(selected.iterrows()):
        wn = row["Well"]
        progress.progress(i / max(n, 1), text=f"Calibrating {wn} ({i+1}/{n})...")

        chars = jp_chars_dict.get(wn)
        if not chars:
            results.append(_jp_cal_failed_row(wn, row, "no jp_chars row"))
            continue

        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            results.append(_jp_cal_failed_row(wn, row, "no current pump in JP history"))
            continue

        well_surf = float(whp_map.get(wn, 210.0))
        try:
            wc = _build_well_config(wn, jp_chars_dict, None, surf_pres=well_surf)
            well_objs = _create_well_objects(wc)
        except Exception as e:
            results.append(_jp_cal_failed_row(wn, row, f"setup error: {e}"))
            continue

        wellbore, well_profile, inflow, res_mix, prop_pf = well_objs
        cur_fric = _friction_coefs_from_chars(chars)
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
            results.append(_jp_cal_failed_row(wn, row, f"calibration error: {e}"))
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
    _render_jp_cal_results(results_df)


def _jp_cal_failed_row(wn: str, row, reason: str) -> dict:
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


def _render_jp_cal_results(results_df: pd.DataFrame) -> None:
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


# ── Well Sort tab ──────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Loading shut-in history from Databricks...")
def _cached_shut_in_history() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_current_shut_in_history
    return fetch_current_shut_in_history()


@st.cache_data(ttl=3600, show_spinner="Loading well tests from Databricks...")
def _cached_recent_tests(days: int) -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_recent_tests
    return fetch_recent_tests(days=days)


@st.cache_data(ttl=3600, show_spinner="Loading producer list...")
def _cached_producers() -> list[str]:
    from woffl.assembly.well_sort_client import fetch_mpu_producers
    return fetch_mpu_producers()


@st.cache_data(ttl=3600, show_spinner="Loading producer catalog...")
def _cached_producer_catalog() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_producer_catalog
    return fetch_producer_catalog()


@st.cache_data(ttl=3600, show_spinner="Loading all-time last tests...")
def _cached_last_tests_ever() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_last_tests_ever
    return fetch_last_tests_ever()


@st.cache_data(ttl=300, show_spinner="Loading safety-valve status...")
def _cached_xv_status() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_xv_status
    return fetch_xv_status()


def _render_well_sort_tab() -> None:
    from woffl.assembly.well_sort_client import (
        apply_pops_pad,
        build_online_table,
        build_shut_in_table,
        classify_wells,
        export_bench_xlsx,
        split_offline_ltsi,
    )

    st.header("Well Sort")
    st.caption(
        "Online = not in vw_shut_in, OR in vw_shut_in but live ProdXV = open "
        "(daily log lags restarts up to 24 h). Wells absent from the log stay "
        "online regardless of XV — handles flowback edge cases like H-31. "
        "Outlier = |test − 2-mo avg| > 25% on oil or water."
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.8, 1.5, 1.0])
    with ctrl1:
        if st.button("Refresh data", help="Clear cache and re-query Databricks"):
            _cached_shut_in_history.clear()
            _cached_recent_tests.clear()
            _cached_producers.clear()
            _cached_producer_catalog.clear()
            _cached_last_tests_ever.clear()
            _cached_xv_status.clear()
            st.rerun()
    with ctrl2:
        mode_label = st.radio(
            "Display test",
            ["Most recent allocated", "Most recent (any)"],
            horizontal=True,
            key="well_sort_mode",
        )
    with ctrl3:
        stale_days = st.slider(
            "Stale-test threshold (days)", 14, 180, 60,
            key="well_sort_stale_days",
            help="Flag wells whose most-recent test is older than this.",
        )
    with ctrl4:
        st.write("")
        st.write("")
        st.caption("Tests window: 180 d")

    mode = "allocated" if mode_label.startswith("Most recent allocated") else "any"

    shut_in_hist = _cached_shut_in_history()
    tests = _cached_recent_tests(180)
    producers = _cached_producers()
    catalog = _cached_producer_catalog()
    last_tests = _cached_last_tests_ever()
    xv = _cached_xv_status()

    if not producers:
        st.error("No producers returned from vw_well_header.")
        return

    # On-pad separation settings (persist across interactions)
    all_pads = sorted(catalog["well_pad"].dropna().unique().tolist()) if not catalog.empty else []
    pops_pads = st.multiselect(
        "Pads with on-pad production separation",
        options=all_pads,
        default=st.session_state.get(
            "well_sort_pops_pads", ["E", "F", "H", "I", "M", "S"]
        ),
        key="well_sort_pops_pads",
        help="Wells on these pads get PopsPad=True. Per-well overrides apply after.",
    )
    # Per-well PopsPad=True overrides (wells that get True even if their pad
    # doesn't have separation — e.g. MPS-08 in the Apr-20 bench sheet).
    force_true_wells = st.multiselect(
        "Per-well PopsPad=True overrides",
        options=sorted(producers),
        default=st.session_state.get("well_sort_pops_force_true", []),
        key="well_sort_pops_force_true",
        help="These wells are treated as having on-pad separation regardless "
        "of the pad-level setting above.",
    )
    overrides = {w: True for w in force_true_wells}

    online_set, shut_set = classify_wells(
        producers, shut_in_hist, xv_df=xv, trust_xv=True
    )
    online_df = build_online_table(
        tests, shut_in_hist, producers, mode=mode,
        stale_days=stale_days, xv_df=xv, online_wells=online_set,
        catalog_df=catalog,
    )
    shut_df = build_shut_in_table(
        shut_in_hist, tests, xv_df=xv, shut_in_wells=shut_set,
        catalog_df=catalog, last_tests_df=last_tests,
    )
    online_df = apply_pops_pad(online_df, set(pops_pads), overrides)
    shut_df = apply_pops_pad(shut_df, set(pops_pads), overrides)
    offline_df, ltsi_df = split_offline_ltsi(shut_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Online Wells", len(online_df))
    m2.metric("Shut-In Wells", len(shut_df))
    m3.metric(
        "Outliers flagged",
        int(online_df["FlagOutlier"].sum()) if not online_df.empty else 0,
        help="|test − 2-month avg| > 25% on oil or water",
    )
    just_restarted = int(online_df["JustRestarted"].sum()) if not online_df.empty else 0
    m4.metric(
        "Just restarted",
        just_restarted,
        help="Wells in vw_shut_in but ProdXV shows open — rescued to online by "
        "live XV. Reflects the 12-24 h lag in the daily shut-in log.",
    )

    # Bench export (matches MPU_Well_Bench_YYYY_MM_DD.xlsx layout)
    from datetime import date
    xlsx_bytes = export_bench_xlsx(online_df, offline_df, ltsi_df)
    st.download_button(
        f"Download bench xlsx  ({len(online_df)} online, {len(offline_df)} offline, "
        f"{len(ltsi_df)} ltsi)",
        data=xlsx_bytes,
        file_name=f"MPU_Well_Bench_{date.today():%Y_%m_%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="well_sort_dl_bench",
    )

    sub_online, sub_off, sub_ltsi = st.tabs(
        [f"Online ({len(online_df)})",
         f"Offline ({len(offline_df)})",
         f"LTSI ({len(ltsi_df)})"]
    )

    with sub_online:
        if online_df.empty:
            st.info("No online wells with recent tests.")
        else:
            display = online_df.copy()
            display["OilDev"] = display["OilDev"] * 100
            display["WatDev"] = display["WatDev"] * 100
            display["AllocVsInfoOilPct"] = display["AllocVsInfoOilPct"] * 100
            display["WC"] = display["WC"] * 100
            if "TotalWC" in display.columns:
                display["TotalWC"] = display["TotalWC"] * 100

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Well": st.column_config.TextColumn("Well", pinned="left"),
                    "Pad": st.column_config.TextColumn("Pad"),
                    "Reservoir": st.column_config.TextColumn("Reservoir"),
                    "LiftType": st.column_config.TextColumn("Lift Type"),
                    "PopsPad": st.column_config.CheckboxColumn(
                        "OnPad Sep?",
                        help="Pad has on-pad production separation",
                    ),
                    "TestDate": st.column_config.DatetimeColumn(
                        "Test Date", format="YYYY-MM-DD"
                    ),
                    "DaysSinceTest": st.column_config.NumberColumn(
                        "Days since", format="%.0f",
                        help="Days between today and the displayed test date",
                    ),
                    "StaleTest": st.column_config.CheckboxColumn(
                        "Stale?",
                        help=f"Displayed test older than {stale_days} days",
                    ),
                    "ProdXV": st.column_config.NumberColumn(
                        "Prod XV", format="%.0f",
                        help="Production safety valve: 1=open, 0=closed",
                    ),
                    "PFXV": st.column_config.NumberColumn(
                        "PF XV", format="%.0f",
                        help="Power-fluid safety valve: 1=open, 0=closed",
                    ),
                    "XVTime": st.column_config.DatetimeColumn(
                        "XV Time", format="MM-DD HH:mm",
                        help="Timestamp of most recent XV reading",
                    ),
                    "JustRestarted": st.column_config.CheckboxColumn(
                        "Just restarted?",
                        help="XV shows flowing but vw_shut_in still has it as "
                        "shut-in. The daily log hasn't caught up yet.",
                    ),
                    "Allocated": st.column_config.CheckboxColumn("Alloc."),
                    "FallbackUsed": st.column_config.CheckboxColumn(
                        "Fallback",
                        help="No allocated test exists; displayed row is info-only "
                        "(or there are no tests at all for this well)",
                    ),
                    "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
                    "Water": st.column_config.NumberColumn("Water (BWPD)", format="%.0f"),
                    "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
                    "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
                    "TotalWC": st.column_config.NumberColumn("Total WC (%)", format="%.1f"),
                    "GOR": st.column_config.NumberColumn("GOR (scf/bbl)", format="%.0f"),
                    "TotalGOR": st.column_config.NumberColumn("Total GOR", format="%.0f"),
                    "BHP": st.column_config.NumberColumn("BHP (psi)", format="%.0f"),
                    "WHP": st.column_config.NumberColumn("WHP (psi)", format="%.0f"),
                    "LiftWater": st.column_config.NumberColumn(
                        "Lift Water (BWPD)", format="%.0f"
                    ),
                    "LiftGas": st.column_config.NumberColumn(
                        "Lift Gas (MCFD)", format="%.0f"
                    ),
                    "TotalWater": st.column_config.NumberColumn(
                        "Total Water (BWPD)", format="%.0f"
                    ),
                    "TotalGas": st.column_config.NumberColumn(
                        "Total Gas (MCFD)", format="%.0f"
                    ),
                    "EspHz": st.column_config.NumberColumn(
                        "ESP Hz", format="%.1f",
                        help="ESP frequency from displayed test (blank for non-ESP wells)",
                    ),
                    "EspAmps": st.column_config.NumberColumn(
                        "ESP Amps", format="%.0f",
                        help="ESP motor amps from displayed test (blank for non-ESP wells)",
                    ),
                    "Oil_2moAvg": st.column_config.NumberColumn(
                        "Oil 2mo avg", format="%.0f"
                    ),
                    "Wat_2moAvg": st.column_config.NumberColumn(
                        "Wat 2mo avg", format="%.0f"
                    ),
                    "OilDev": st.column_config.NumberColumn(
                        "Oil Δ vs 2mo (%)", format="%.0f"
                    ),
                    "WatDev": st.column_config.NumberColumn(
                        "Wat Δ vs 2mo (%)", format="%.0f"
                    ),
                    "FlagOutlier": st.column_config.CheckboxColumn(
                        "Outlier?",
                        help="|Δ| > 25% on oil or water vs trailing 2-month average",
                    ),
                    "AllocVsInfoOilPct": st.column_config.NumberColumn(
                        "Info vs Alloc Oil Δ (%)",
                        help="Latest info-only oil rate vs latest allocated. "
                        "Large values = allocation drift.",
                        format="%.0f",
                    ),
                    "LatestAllocDate": st.column_config.DatetimeColumn(
                        "Latest Alloc Date", format="YYYY-MM-DD"
                    ),
                    "LatestInfoDate": st.column_config.DatetimeColumn(
                        "Latest Info Date", format="YYYY-MM-DD"
                    ),
                },
            )

            csv = online_df.to_csv(index=False, float_format="%.2f").encode("utf-8")
            st.download_button(
                "Download Online CSV",
                data=csv,
                file_name="well_sort_online.csv",
                mime="text/csv",
                key="well_sort_dl_online",
            )

    def _shut_columns_config():
        return {
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Reservoir": st.column_config.TextColumn("Reservoir"),
            "LiftType": st.column_config.TextColumn("Lift Type"),
            "PopsPad": st.column_config.CheckboxColumn("OnPad Sep?"),
            "ShutInSince": st.column_config.DateColumn(
                "Shut-In Since",
                help="Start of current consecutive full-day shut-in streak",
            ),
            "CurrentCode": st.column_config.TextColumn("Code"),
            "CurrentReason": st.column_config.TextColumn("Reason"),
            "Notes": st.column_config.TextColumn("Notes"),
            "DownHours": st.column_config.NumberColumn("Down hrs", format="%.1f"),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            "Water": st.column_config.NumberColumn("Water (BWPD)", format="%.0f"),
            "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
            "LiftWater": st.column_config.NumberColumn("Lift Wat (BWPD)", format="%.0f"),
            "LiftGas": st.column_config.NumberColumn("Lift Gas (MCFD)", format="%.0f"),
            "TotalWater": st.column_config.NumberColumn("Total Wat (BWPD)", format="%.0f"),
            "TotalGas": st.column_config.NumberColumn("Total Gas (MCFD)", format="%.0f"),
            "EspHz": st.column_config.NumberColumn(
                "ESP Hz", format="%.1f",
                help="ESP frequency from last test (blank for non-ESP wells)",
            ),
            "EspAmps": st.column_config.NumberColumn(
                "ESP Amps", format="%.0f",
                help="ESP motor amps from last test (blank for non-ESP wells)",
            ),
            "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
            "TotalWC": st.column_config.NumberColumn("Total WC (%)", format="%.1f"),
            "GOR": st.column_config.NumberColumn("GOR", format="%.0f"),
            "TotalGOR": st.column_config.NumberColumn("Total GOR", format="%.0f"),
            "LastOnlineDate": st.column_config.DateColumn("Last Online"),
            "LastTestDate": st.column_config.DatetimeColumn(
                "Last Test", format="YYYY-MM-DD",
                help="Absolute-latest test date, any age (not bounded by 180d window)",
            ),
            "NearAvgOil": st.column_config.NumberColumn(
                "Near Avg Oil", format="%.0f",
                help="Avg oil rate over tests within 90 days of last test",
            ),
            "NearAvgWater": st.column_config.NumberColumn("Near Avg Wat", format="%.0f"),
            "NearAvgGas": st.column_config.NumberColumn("Near Avg Gas", format="%.0f"),
            "NTestsNear": st.column_config.NumberColumn(
                "# Near Tests", format="%.0f",
                help="How many tests in the 90-day near-last window",
            ),
            "ProdXV": st.column_config.NumberColumn("Prod XV", format="%.0f"),
            "PFXV": st.column_config.NumberColumn("PF XV", format="%.0f"),
            "XVTime": st.column_config.DatetimeColumn("XV Time", format="MM-DD HH:mm"),
        }

    def _render_shut_section(df: pd.DataFrame, label: str, key: str):
        if df.empty:
            st.info(f"No {label} wells.")
            return
        disp = df.copy()
        for c in ("WC", "TotalWC"):
            if c in disp.columns:
                disp[c] = disp[c] * 100
        st.dataframe(
            disp, use_container_width=True, hide_index=True,
            column_config=_shut_columns_config(),
        )
        st.download_button(
            f"Download {label} CSV",
            data=df.to_csv(index=False, float_format="%.2f").encode("utf-8"),
            file_name=f"well_sort_{key}.csv",
            mime="text/csv",
            key=f"well_sort_dl_{key}",
        )

    with sub_off:
        _render_shut_section(offline_df, "Offline", "offline")

    with sub_ltsi:
        _render_shut_section(ltsi_df, "LTSI", "ltsi")


# ── Pad Water Cut tab ──────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Building pad water-cut series...")
def _cached_pad_watercut(start_date: str, end_date: str) -> pd.DataFrame:
    from woffl.assembly.pad_watercut_client import fetch_pad_watercut
    return fetch_pad_watercut(start_date, end_date)


def _render_pad_watercut_tab() -> None:
    import datetime as _dt

    st.header("Pad Water Cut")
    st.caption(
        "Daily pad-level WC over time for pads G, H, I, J. Each well's last "
        "allocated test is forward-filled; well-days with >6 h shut-in are "
        "excluded. H and I are treated as on-pad PF recycle (lift water "
        "stays), G and J ship lift water back to the plant."
    )

    today = _dt.date.today()
    default_start = today - _dt.timedelta(days=365 * 3)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 2.5])
    with c1:
        start_date = st.date_input("Start", value=default_start, key="pad_wc_start")
    with c2:
        end_date = st.date_input("End", value=today, key="pad_wc_end")
    with c3:
        if st.button("Refresh", key="pad_wc_refresh", help="Clear cache and re-query"):
            _cached_pad_watercut.clear()
            st.rerun()
    with c4:
        show_series = st.multiselect(
            "Series",
            options=["G", "H", "I", "J", "All"],
            default=["G", "H", "I", "J", "All"],
            key="pad_wc_series",
        )

    if start_date >= end_date:
        st.warning("Start date must be before end date.")
        return

    df = _cached_pad_watercut(start_date.isoformat(), end_date.isoformat())
    if df.empty:
        st.info("No data returned for the selected date range.")
        return

    pad_colors = {"G": "#1f77b4", "H": "#2ca02c", "I": "#ff7f0e", "J": "#d62728", "All": "#555555"}

    fig = go.Figure()
    for pad in ["G", "H", "I", "J", "All"]:
        if pad not in show_series:
            continue
        sub = df[df["pad"] == pad]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["wc"] * 100,
                mode="lines",
                name=pad,
                line=dict(
                    color=pad_colors.get(pad, "#888"),
                    width=3 if pad == "All" else 1.8,
                    dash="dash" if pad == "All" else "solid",
                ),
                hovertemplate=f"<b>{pad}</b><br>%{{x|%Y-%m-%d}}<br>WC: %{{y:.0f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Cut (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        height=520,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Secondary: pad oil + water rates
    with st.expander("Pad oil & water rates"):
        fig2 = go.Figure()
        for pad in ["G", "H", "I", "J", "All"]:
            if pad not in show_series:
                continue
            sub = df[df["pad"] == pad]
            if sub.empty:
                continue
            color = pad_colors.get(pad, "#888")
            fig2.add_trace(
                go.Scatter(
                    x=sub["date"], y=sub["oil"], mode="lines", name=f"{pad} oil",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{pad} oil</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.0f}} BOPD<extra></extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=sub["date"], y=sub["water"], mode="lines", name=f"{pad} water",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"<b>{pad} water</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.0f}} BWPD<extra></extra>",
                )
            )
        fig2.update_layout(
            xaxis_title="Date", yaxis_title="Rate (BPD)", hovermode="x unified",
            height=460, margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"pad_watercut_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="pad_wc_dl",
    )
