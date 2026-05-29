"""Header Pressure Impact tab.

Estimates the oil & BHP impact of changing a pad/header operating pressure,
well by well. Sibling to the PF Scenario tab: that one sweeps power-fluid
pressure (``ppf_surf``) holding wellhead pressure fixed; this one sweeps the
wellhead pressure (``pwh`` — the header lever) holding power-fluid pressure
fixed.

Physics: for each jet-pump well on the selected pad(s), the WOFFL ``BatchPump``
solver runs at the current wellhead pressure and again at ``WHP + Δ``. The
change in suction pressure (BHP) and oil rate fall straight out of the solve.
IPR resolution is the same three tiers as PF Scenario: Vogel from BHP gauge →
JP back-calc for gaugeless wells → jp_chars defaults.

Empirical comparison (optional): for each well with historian tags, fit the
within-day BHP-vs-header/WHP slope from hourly trends (see ``header_trend``)
and show it next to the physics-implied slope, so genuine field behavior can be
compared to the model. The empirical *override* and analog-donor handling for
gaugeless non-JP wells (ESP) build on this.

────────────────────────────────────────────────────────────────────────────
⚠  POWER-FLUID PRESSURE DEPENDENCY  (grep: PF-PRESSURE-DEPENDENCY)
This tool holds power-fluid pressure FIXED while it sweeps wellhead pressure,
so per-well PF pressure is a first-class input to the modeled baseline. There
is no per-well PF pressure in Databricks yet (``vw_power_fluid_*`` carries no
pressure field), so PF is seeded from pad-level defaults (``PAD_PF_DEFAULTS``)
and shown/editable per row in the UI. Replace with actuals when the data lands.
See the ``power-fluid-pressure-source`` project memory.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.network_optimizer import WellConfig
from woffl.geometry.jetpump import JetPump
from woffl.gui.utils import default_pad_pf, load_well_characteristics

from . import header_trend as ht
from ._common import (
    build_well_config,
    create_well_objects,
    friction_coefs_from_chars,
    get_latest_bhp_per_well,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    pad_from_mp_name,
)
from .pf_scenario import _estimate_gaugeless_ipr


# ── solver (mirror of pf_scenario._solve_at_pf, but sweep pwh) ──────────────


def _solve_at_whp(
    wc: WellConfig,
    well_objects: tuple,
    nozzle: str,
    throat: str,
    pwh: float,
    ppf_surf: float,
    fric_coefs: dict | None = None,
) -> dict:
    """Solve one well at one wellhead pressure.

    Mirrors ``pf_scenario._solve_at_pf`` but the swept variable is the
    wellhead pressure ``pwh`` (the header lever); power fluid ``ppf_surf`` is
    held fixed. Returns oil rate, PF (lift water) rate, suction pressure (BHP),
    and sonic/Mach flags.
    """
    wellbore, well_profile, inflow, res_mix, prop_pf = well_objects
    jp = JetPump(nozzle, throat, **(fric_coefs or {}))
    batch = BatchPump(
        pwh=pwh,
        tsu=wc.form_temp,
        ppf_surf=ppf_surf,
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


# ── producer universe + lift-type classification ─────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_well_overview(months_back: int) -> pd.DataFrame:
    """Per-producer latest-test snapshot for ALL producers (not just JP).

    Columns: well (normalized), well_pad, oil, esp_amps, lift_gas, lift_wat, whp,
    resvr_press. The universe + lift-type inputs + generic-IPR inputs (oil rate,
    reservoir pressure) for the all-lift-types flow.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    days = int(months_back * 31)
    q = f"""
    WITH latest AS (
        SELECT vwt.enthid, vwt.well_name, vwt.form_oil AS oil, vwt.esp_amps,
               vwt.lift_gas, vwt.lift_wat, vwt.whp,
               ROW_NUMBER() OVER (PARTITION BY vwt.enthid ORDER BY vwt.wt_date DESC) AS rn
        FROM mpu.wells.vw_well_test vwt
        WHERE vwt.wt_date >= DATE_SUB(current_date(), {days}) AND vwt.allocated = True
    )
    SELECT l.well_name, l.oil, l.esp_amps, l.lift_gas, l.lift_wat, l.whp,
           r.resvr_press, h.well_pad
    FROM latest l
    LEFT JOIN mpu.wells.vw_prop_resvr r ON l.enthid = r.enthid
    LEFT JOIN mpu.wells.vw_well_header h ON l.enthid = h.enthid
    WHERE l.rn = 1 AND h.well_type = 'prod'
    """
    df = execute_query(q)
    if df is None or df.empty:
        return pd.DataFrame()
    df["well"] = df["well_name"].apply(_normalize_well_name)
    for c in ("oil", "esp_amps", "lift_gas", "lift_wat", "whp", "resvr_press"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _classify_lift(well: str, jp_hist, ov_row) -> str:
    """JP (current pump) / ESP (esp_amps) / gas-lift (lift_gas) / flowing."""
    pump = get_current_pump(jp_hist, well)
    if pump is not None and pump.get("nozzle_no") and pump.get("throat_ratio"):
        return "JP"
    if ov_row is not None:
        if pd.notna(ov_row.get("esp_amps")) and float(ov_row.get("esp_amps") or 0) > 0:
            return "ESP"
        if pd.notna(ov_row.get("lift_gas")) and float(ov_row.get("lift_gas") or 0) > 0:
            return "gas-lift"
    return "flowing"


# ── input table ─────────────────────────────────────────────────────────────


def _build_input_table(pads: list[str], months_back: int) -> pd.DataFrame | None:
    """Per-well input table for the selected pads — ALL producers, by lift type.

    JP wells (current pump) take the physics path; ESP / gas-lift / flowing wells
    take the empirical/analog path. WHP from the latest test; ResP from
    vw_prop_resvr / jp_chars where available, else assumed 1800 (ESPs aren't
    characterized in vw_prop_resvr). PF held + Pump apply to JP wells only.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return None
    overview = fetch_well_overview(months_back)
    if overview is None or overview.empty:
        return None
    jp_chars_df = load_well_characteristics()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")
    whp_map = get_latest_whp_per_well(months_back)

    rows = []
    for _, ov in overview.iterrows():
        wn = ov["well"]
        pad = ov.get("well_pad") or pad_from_mp_name(wn)
        if pad not in pads:
            continue
        lift = _classify_lift(wn, jp_hist, ov)
        pump = get_current_pump(jp_hist, wn)
        is_jp = lift == "JP" and pump is not None and pump.get("nozzle_no") and pump.get("throat_ratio")
        chars = jp_chars_dict.get(wn, {})

        res_pres = chars.get("res_pres")
        if res_pres is None or pd.isna(res_pres):
            res_pres = ov.get("resvr_press")
        if res_pres is None or pd.isna(res_pres):
            res_pres = 1800.0  # ESPs aren't in vw_prop_resvr — assumed
        whp_now = ov.get("whp")
        if pd.isna(whp_now):
            whp_now = whp_map.get(wn)
        oil = ov.get("oil")

        rows.append(
            {
                "Well": wn,
                "Pad": pad,
                "Lift": lift,
                "Pump": f"{pump['nozzle_no']}{pump['throat_ratio']}" if is_jp else "",
                "Oil (BOPD)": round(float(oil), 0) if pd.notna(oil) else None,
                # PF-PRESSURE-DEPENDENCY: pad default, JP only, no per-well source yet
                "PF held (psi)": default_pad_pf(pad) if is_jp else None,
                "WHP now (psi)": int(round(whp_now)) if pd.notna(whp_now) else None,
                "ResP (psi)": int(round(float(res_pres))),
                "Include": True,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["Pad", "Lift", "Well"]).reset_index(drop=True)


# ── empirical comparison ─────────────────────────────────────────────────────


def _fetch_empirical_fits(well_names: list[str], months_back: int):
    """Pull hourly historian trends and fit within-day slopes per well.

    Returns (well_dfs, fits_by_well, missing_wells): well_dfs maps a well to its
    raw hourly trend DataFrame, fits_by_well to the dict from header_trend.fit_well,
    and missing_wells lists wells with no historian tag (empirical N/A).
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - relativedelta(months=int(months_back))).strftime("%Y-%m-%d")
    well_dfs, missing = ht.fetch_header_trends(tuple(sorted(well_names)), start, end)
    fits = {wn: ht.fit_well(tdf) for wn, tdf in well_dfs.items()}
    return well_dfs, fits, missing


def _empirical_columns(
    well_fits: dict | None,
    inflow,
    res_pres: float,
    bhp_now: float,
    delta_p: float,
) -> dict:
    """Build the empirical comparison columns for one well.

    Uses the within-day BHP~HeaderP slope (the direct header lever) to predict
    ΔBHP for the header move, then reads ΔOil off the same Vogel IPR the physics
    solve uses. Reports the BHP~WHP slope too, which is directly comparable to
    the physics-implied dBHP/dWHP.
    """
    blank = {
        "Emp class": "no tag",
        "Emp days": 0,
        "Emp dBHP/dWHP": np.nan,
        "Emp ΔOil (BOPD)": np.nan,
    }
    if not well_fits:
        return blank

    # BHP~WHP is the clean, physics-comparable coupling: WHP varies enough
    # intraday for good fits, whereas the direct BHP~HeaderP fit is data-starved
    # (the pad header is steadier intraday). delta_p is applied as the WHP change,
    # matching the physics 1:1 ΔWHP = ΔHeader convention.
    f_wp = well_fits.get("BHP~WHP")
    emp_class = ht.classify_response(f_wp) if f_wp else "no data"

    emp_doil = np.nan
    if (
        f_wp is not None
        and emp_class == "responsive"
        and not np.isnan(f_wp.mean_slope)
        and pd.notna(bhp_now)
    ):
        emp_dbhp = f_wp.mean_slope * delta_p
        try:
            b0 = float(np.clip(bhp_now, 0.0, res_pres))
            b1 = float(np.clip(bhp_now + emp_dbhp, 0.0, res_pres))
            emp_doil = inflow.oil_flow(b1, "vogel") - inflow.oil_flow(b0, "vogel")
        except Exception:
            emp_doil = np.nan

    return {
        "Emp class": emp_class,
        "Emp days": f_wp.n_days if f_wp is not None else 0,
        "Emp dBHP/dWHP": f_wp.mean_slope if f_wp is not None else np.nan,
        "Emp ΔOil (BOPD)": emp_doil,
    }


def _verdict(
    sonic_now: bool,
    sonic_scen: bool,
    delta_oil: float,
    emp_class: str | None,
    compare_emp: bool,
    resp_thresh: float = 5.0,
) -> str:
    """Combine the physics sonic flag + ΔOil with the empirical class into an
    actionable per-well label (Scott's sonic-decoupling insight made explicit).

    A jet pump already sonic at the current WHP is choked — a header move can't
    propagate past the critical throat, so it won't respond regardless of what
    the IPR suggests. Non-sonic wells that the physics says respond are then
    confirmed or contradicted by the empirical slope.
    """
    if sonic_now:
        return "sonic-decoupled"          # already choked — header won't help
    if sonic_scen:
        return "chokes when lowered"       # responds, then sonic-limits
    responds = pd.notna(delta_oil) and abs(delta_oil) >= resp_thresh
    if not responds:
        return "no response"
    if not compare_emp or emp_class in (None, "no tag", "no data", "insufficient"):
        return "responsive (physics)"
    if emp_class == "responsive":
        return "responsive ✓"
    if emp_class == "slugging":
        return "disagree — check"
    return "responsive (physics)"


def _solve_nonjp_row(wn, row, emp_fits: dict, emp_well_dfs: dict, delta_p: float) -> dict:
    """Empirical-only result row for a non-JP well (ESP / gas-lift / flowing).

    No jet-pump physics: ``pwf`` = the well's own recent measured BHP; a generic
    Vogel IPR from the latest test oil rate + assumed reservoir pressure gives
    ΔOil for ΔBHP = (empirical BHP~WHP slope) × Δ. Gaugeless wells (no BHP trend)
    are flagged to use the Analog estimate instead.
    """
    from woffl.flow.inflow import InFlow

    fit = (emp_fits.get(wn) or {}).get("BHP~WHP")
    emp_class = ht.classify_response(fit) if fit is not None else "no data"
    emp_slope = fit.mean_slope if fit is not None else np.nan

    trend = emp_well_dfs.get(wn)
    bhp_now = np.nan
    if trend is not None and "BHP" in getattr(trend, "columns", []):
        bhp_s = trend["BHP"].dropna()
        if not bhp_s.empty:
            bhp_now = float(bhp_s.iloc[-1])

    oil = row.get("Oil (BOPD)")
    res_pres = float(row.get("ResP (psi)") or 1800.0)
    whp_now = row.get("WHP now (psi)")

    emp_dbhp = np.nan
    emp_doil = np.nan
    if pd.isna(bhp_now):
        verdict = "gaugeless — use Analog"
    elif fit is None or emp_class != "responsive":
        verdict = emp_class if emp_class in ("slugging", "insufficient") else "slugging"
        emp_dbhp = 0.0
        emp_doil = 0.0  # an unresponsive well won't move with the header
    elif oil is None or pd.isna(oil) or float(oil) <= 0:
        verdict = "no test oil rate"
    else:
        pres = max(res_pres, bhp_now + 100.0)
        ipr = InFlow(qwf=float(oil), pwf=bhp_now, pres=pres)
        emp_dbhp = emp_slope * delta_p
        b0 = float(np.clip(bhp_now, 1.0, pres))
        b1 = float(np.clip(bhp_now + emp_dbhp, 1.0, pres))
        emp_doil = ipr.oil_flow(b1, "vogel") - ipr.oil_flow(b0, "vogel")
        verdict = "responsive (empirical)"

    return {
        "Well": wn,
        "Pad": row["Pad"],
        "Lift": row.get("Lift", ""),
        "Pump": "",
        "IPR src": "empirical (test + assumed ResP)",
        "Fric src": "—",
        "BHP cal err": None,
        "PF held (psi)": None,
        "WHP now (psi)": int(round(whp_now)) if pd.notna(whp_now) else None,
        "WHP scen (psi)": int(round(whp_now + delta_p)) if pd.notna(whp_now) else None,
        "BHP now (psi)": bhp_now,
        "BHP scen (psi)": (bhp_now + emp_dbhp) if (pd.notna(emp_dbhp) and pd.notna(bhp_now)) else np.nan,
        "ΔBHP (psi)": emp_dbhp,
        "Oil now (BOPD)": float(oil) if pd.notna(oil) else np.nan,
        "Oil scen (BOPD)": np.nan,
        "ΔOil (BOPD)": np.nan,  # no physics estimate for non-JP
        "Phys dBHP/dWHP": np.nan,
        "PF rate now (BWPD)": np.nan,
        "PF rate scen (BWPD)": np.nan,
        "Sonic now": False,
        "Sonic scen": False,
        "Mach scen": np.nan,
        "Emp class": emp_class,
        "Emp days": fit.n_days if fit is not None else 0,
        "Emp dBHP/dWHP": emp_slope,
        "Emp ΔOil (BOPD)": emp_doil,
        "Verdict": verdict,
    }


# ── main tab ─────────────────────────────────────────────────────────────────


def render_tab() -> None:
    st.header("Header Pressure Impact")
    st.markdown(
        "Estimate the oil & BHP impact of changing a pad/header operating "
        "pressure. Each jet-pump well is solved at its current wellhead "
        "pressure and again at **WHP + Δ**, holding power-fluid pressure "
        "fixed — so ΔBHP and ΔOil come straight from the jet-pump physics."
    )
    st.caption(
        "Sibling to PF Scenario (which sweeps power-fluid pressure). Optionally "
        "compares the physics-implied response to an empirical within-day slope "
        "fit from historian trends."
    )

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.error(
            "JP history not loaded. Check the Databricks connection or upload "
            "JP history in the sidebar."
        )
        return

    jp_chars_df = load_well_characteristics()
    all_pads = sorted({pad_from_mp_name(w) for w in jp_chars_df["Well"]})

    # ── settings ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pads = st.multiselect(
            "Pad(s) / header",
            options=all_pads,
            default=all_pads[:1],
            key="hpi_pads",
            help="Wells on a shared header move together — analyze them as a group.",
        )
    with c2:
        months_back = st.number_input(
            "Test/trend lookback (months)",
            min_value=1, max_value=24, value=6, step=1,
            key="hpi_months",
            help="Window for well tests (IPR) and historian trends (empirical slope).",
        )
    with c3:
        test_pf_pres = st.number_input(
            "PF at test time (psi)",
            min_value=1000, max_value=5000, value=3400, step=100,
            key="hpi_test_pf",
            help="Used only to back-calculate BHP for gaugeless jet-pump wells.",
        )

    delta_p = st.number_input(
        "Header pressure change Δ (psi)",
        min_value=-500, max_value=500, value=-50, step=10,
        key="hpi_delta",
        help="Applied to each well's wellhead pressure (assumes ΔWHP = ΔHeader, "
        "1:1). Negative = header reduction (expect oil uplift).",
    )

    compare_emp = st.checkbox(
        "Compare against empirical historian trends",
        value=True,
        key="hpi_compare_emp",
        help="Fit the within-day BHP-vs-header slope from hourly historian data "
        "and show it next to the physics-implied slope. Tags come from "
        "vw_bhp_tags; wells with no live gauge show 'no data'.",
    )

    # Friction-coef source. Calibrating to measured BHP before the impact solve
    # reins in default-coef optimism (e.g. an over-eager modeled response).
    fric_mode = st.radio(
        "Friction-coef source",
        options=["calibrate", "databricks", "manual"],
        format_func=lambda m: {
            "calibrate": "Auto-calibrate per well to measured BHP (recommended — trustworthy ΔOil)",
            "databricks": "Databricks defaults (per-well jpfric_*)",
            "manual": "Manual override (sliders, applied to all wells)",
        }[m],
        key="hpi_fric_mode",
    )
    fric_override_values: dict | None = None
    if fric_mode == "manual":
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            ken_ov = st.slider("ken (entry)", 0.0, 0.20, 0.03, 0.005, key="hpi_ken")
        with fc2:
            kth_ov = st.slider("kth (throat)", 0.05, 1.0, 0.30, 0.01, key="hpi_kth")
        with fc3:
            kdi_ov = st.slider("kdi (diffuser)", 0.05, 1.0, 0.30, 0.01, key="hpi_kdi")
        fric_override_values = {"ken": ken_ov, "kth": kth_ov, "kdi": kdi_ov}
    elif fric_mode == "calibrate":
        st.caption(
            "Per-well Nelder-Mead sweep of kth/kdi at the well's WHP + PF held to "
            "match its latest measured BHP, *before* the header-impact solve. "
            "~15 solver evals/well; wells without measured BHP keep Databricks "
            "defaults."
        )

    if not pads:
        st.info("Select at least one pad to begin.")
        return

    if st.button("Load wells", key="hpi_load"):
        with st.spinner("Loading producers on the selected pad(s)..."):
            input_df = _build_input_table(pads, months_back)
        if input_df is None or input_df.empty:
            st.warning("No producers with recent allocated tests on the selected pad(s).")
            return
        st.session_state["hpi_input_df"] = input_df
        st.session_state.pop("hpi_results_df", None)
        st.session_state.pop("hpi_editor", None)

    input_df = st.session_state.get("hpi_input_df")
    if input_df is None:
        st.info("Click **Load wells** to start.")
        return

    st.caption(
        "All producers on the pad: **JP** → physics, **ESP / gas-lift / flowing** → "
        "empirical (no jet-pump model). **PF held** (JP only) is a pad default — no "
        "per-well source yet. **ResP** is from vw_prop_resvr where available, else "
        "assumed 1800 (ESPs aren't characterized there) — edit it. **WHP now** is the "
        "latest test WHP."
    )

    edited_df = st.data_editor(
        input_df,
        key="hpi_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", disabled=True, pinned="left"),
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Pump": st.column_config.TextColumn("Pump", disabled=True),
            "PF held (psi)": st.column_config.NumberColumn(
                "PF held (psi)", format="%d",
                min_value=1000, max_value=5000, step=50,
                help="Power-fluid pressure held fixed during the WHP sweep. "
                "Pad default — no per-well source yet.",
            ),
            "WHP now (psi)": st.column_config.NumberColumn(
                "WHP now (psi)", format="%d",
                min_value=0, max_value=2000, step=10,
                help="Baseline wellhead pressure (latest test). Edit if the "
                "current header differs.",
            ),
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Uncheck to skip this well in the run.",
            ),
        },
    )

    _render_analog(months_back, delta_p)

    run_clicked = st.button(
        "Run header impact", type="primary", use_container_width=True, key="hpi_run"
    )
    if not run_clicked:
        results_df = st.session_state.get("hpi_results_df")
        if results_df is not None:
            _render_results(results_df, st.session_state.get("hpi_delta_used", delta_p))
            _render_diagnostics()
        return

    selected = edited_df[edited_df["Include"]].copy()
    if selected.empty:
        st.warning("No wells selected. Check at least one 'Include' box.")
        return

    well_names = selected["Well"].tolist()
    jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")
    whp_map = get_latest_whp_per_well(months_back)

    # Calibration prerequisites: the calibrator + latest measured BHP per well.
    bhp_map: dict[str, float] = {}
    if fric_mode == "calibrate":
        from woffl.gui.fric_calibration import calibrate_friction_coefs
        with st.spinner("Pulling latest measured BHP per well..."):
            bhp_map = get_latest_bhp_per_well(months_back)
        no_bhp = [w for w in well_names if w not in bhp_map]
        if no_bhp:
            st.caption(f"No measured BHP — Databricks coefs (no calibration): {', '.join(no_bhp)}")

    # ── IPR: three tiers (gauge → gaugeless back-calc → jp_chars default) ──
    with st.spinner(f"Fetching well tests ({months_back} mo) and computing IPR..."):
        vogel_dict = get_vogel_for_wells(well_names, months_back)
    missing = [w for w in well_names if w not in vogel_dict]
    if missing:
        with st.spinner(f"Estimating BHP for {len(missing)} gaugeless wells..."):
            vogel_dict.update(
                _estimate_gaugeless_ipr(
                    missing, months_back, test_pf_pres, jp_hist, jp_chars_dict,
                    whp_map=whp_map,
                )
            )

    gauged = [w for w in well_names if w in vogel_dict and not vogel_dict[w].get("_bhp_estimated")]
    estimated = [w for w in well_names if w in vogel_dict and vogel_dict[w].get("_bhp_estimated")]
    fallback = [w for w in well_names if w not in vogel_dict]
    if gauged:
        st.caption(f"IPR from BHP gauge: {', '.join(gauged)}")
    if estimated:
        st.caption(f"IPR from JP back-calc (estimated BHP): {', '.join(estimated)}")
    if fallback:
        st.caption(f"IPR from jp_chars defaults (no test data): {', '.join(fallback)}")

    # ── empirical: hourly trends + within-day slope fits ──────────────
    # Empirical is REQUIRED for non-JP wells (their only estimate), so fetch it
    # whenever a non-JP well is selected, even if the JP comparison box is off.
    emp_fits: dict = {}
    emp_well_dfs: dict = {}
    need_emp = bool(compare_emp or (selected["Lift"] != "JP").any())
    if need_emp:
        with st.spinner("Pulling hourly historian trends and fitting empirical slopes..."):
            try:
                emp_well_dfs, emp_fits, emp_missing = _fetch_empirical_fits(well_names, months_back)
            except Exception as e:
                emp_well_dfs, emp_fits, emp_missing = {}, {}, []
                st.warning(
                    f"Empirical trends unavailable ({type(e).__name__}: {e}). "
                    "Showing physics only. If this looks like a column/SQL error, "
                    "the historian query in header_trend.py needs adjusting."
                )
        # Stash raw trends + fits so the per-well diagnostic plot survives reruns.
        st.session_state["hpi_emp_well_dfs"] = emp_well_dfs
        st.session_state["hpi_emp_fits"] = emp_fits
        if emp_missing:
            st.caption(f"No historian tags (empirical N/A): {', '.join(emp_missing)}")
    else:
        st.session_state.pop("hpi_emp_well_dfs", None)
        st.session_state.pop("hpi_emp_fits", None)

    # ── solve each well at baseline WHP and at WHP + Δ ─────────────────
    results = []
    errors: list[str] = []
    progress = st.progress(0, text="Starting...")
    n = len(selected)
    for i, (_, row) in enumerate(selected.iterrows()):
        wn = row["Well"]
        progress.progress(i / max(n, 1), text=f"Solving {wn} ({i+1}/{n})...")

        # Non-JP wells (ESP / gas-lift / flowing) take the empirical-only path.
        if str(row.get("Lift", "JP")) != "JP":
            results.append(_solve_nonjp_row(wn, row, emp_fits, emp_well_dfs, delta_p))
            continue

        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            # Classified JP but no current pump now → fall back to empirical-only.
            results.append(_solve_nonjp_row(wn, row, emp_fits, emp_well_dfs, delta_p))
            continue
        chars = jp_chars_dict.get(wn)

        whp_now = row["WHP now (psi)"]
        if whp_now is None or pd.isna(whp_now):
            whp_now = float(whp_map.get(wn, 210.0))
        whp_now = float(whp_now)
        pf_held = float(row["PF held (psi)"])
        ipr_src = "gauge" if wn in gauged else ("back-calc" if wn in estimated else "jp_chars")

        try:
            wc = build_well_config(wn, jp_chars_dict, vogel_dict.get(wn), surf_pres=whp_now)
            well_objs = create_well_objects(wc)
            wellbore, well_profile, inflow, res_mix, prop_pf = well_objs

            # Friction coefs by mode: Databricks defaults, manual override, or
            # per-well calibration to measured BHP (tames default-coef optimism
            # before the impact solve).
            fric = friction_coefs_from_chars(chars)
            fric_source = "databricks"
            cal_err = None
            if fric_override_values is not None:
                fric = dict(fric_override_values)
                fric_source = "manual"
            elif fric_mode == "calibrate" and wn in bhp_map:
                ken_fixed = float(friction_coefs_from_chars(chars).get("ken", 0.03))
                try:
                    cal = calibrate_friction_coefs(
                        well_name=wn, target_bhp=float(bhp_map[wn]),
                        pwh=whp_now, tsu=wc.form_temp, ppf_surf=pf_held,
                        nozzle=pump["nozzle_no"], throat=pump["throat_ratio"],
                        knz=0.01, ken=ken_fixed,
                        wellbore=wellbore, wellprof=well_profile,
                        ipr_su=inflow, prop_su=res_mix, prop_pf=prop_pf,
                        jpump_direction="reverse",
                    )
                    if cal.converged:
                        fric = {"knz": cal.knz, "ken": cal.best_ken,
                                "kth": cal.best_kth, "kdi": cal.best_kdi}
                        fric_source = "calibrated"
                        cal_err = cal.best_modeled_bhp - cal.target_bhp
                except Exception as ce:
                    errors.append(f"{wn}: calibration error — {ce}")

            res_now = _solve_at_whp(
                wc, well_objs, pump["nozzle_no"], pump["throat_ratio"], whp_now, pf_held, fric
            )
            res_scen = _solve_at_whp(
                wc, well_objs, pump["nozzle_no"], pump["throat_ratio"],
                whp_now + delta_p, pf_held, fric,
            )
        except Exception as e:
            errors.append(f"{wn}: solver error — {e}")
            continue

        # Physics-implied dBHP/dWHP (the secant of the two solves)
        if delta_p != 0 and pd.notna(res_scen["psu"]) and pd.notna(res_now["psu"]):
            phys_slope = (res_scen["psu"] - res_now["psu"]) / delta_p
        else:
            phys_slope = np.nan

        row_d = {
            "Well": wn,
            "Pad": row["Pad"],
            "Lift": "JP",
            "Pump": row["Pump"],
            "IPR src": ipr_src,
            "Fric src": fric_source,
            "BHP cal err": cal_err,
            "PF held (psi)": int(pf_held),
            "WHP now (psi)": int(round(whp_now)),
            "WHP scen (psi)": int(round(whp_now + delta_p)),
            "BHP now (psi)": res_now["psu"],
            "BHP scen (psi)": res_scen["psu"],
            "ΔBHP (psi)": res_scen["psu"] - res_now["psu"],
            "Oil now (BOPD)": res_now["oil"],
            "Oil scen (BOPD)": res_scen["oil"],
            "ΔOil (BOPD)": res_scen["oil"] - res_now["oil"],
            "Phys dBHP/dWHP": phys_slope,
            "PF rate now (BWPD)": res_now["pf_rate"],
            "PF rate scen (BWPD)": res_scen["pf_rate"],
            "Sonic now": res_now["sonic"],
            "Sonic scen": res_scen["sonic"],
            "Mach scen": res_scen["mach"],
        }
        emp_class = None
        if compare_emp:
            ecols = _empirical_columns(
                emp_fits.get(wn), well_objs[2], wc.res_pres, res_now["psu"], delta_p
            )
            row_d.update(ecols)
            emp_class = ecols.get("Emp class")
        row_d["Verdict"] = _verdict(
            res_now["sonic"], res_scen["sonic"], row_d["ΔOil (BOPD)"], emp_class, compare_emp
        )
        results.append(row_d)

    progress.progress(1.0, text="Done.")

    if errors:
        with st.expander(f"{len(errors)} wells skipped", expanded=False):
            for e in errors:
                st.write(f"- {e}")

    if not results:
        st.error("No wells could be solved. Check the skipped list above.")
        return

    results_df = pd.DataFrame(results)
    st.session_state["hpi_results_df"] = results_df
    st.session_state["hpi_delta_used"] = delta_p
    _render_results(results_df, delta_p)
    _render_diagnostics()


# ── results display ───────────────────────────────────────────────────────


def _chosen_method(phys_doil, emp_doil, method: str):
    """Pick the ΔOil to trust for a well + a label, per the global method.

    auto = keep physics unless the empirical ΔOil is available AND materially
    disagrees (Scott's "override physics when reality differs"); empirical = use
    the field value wherever a responsive empirical slope produced one.
    """
    phys = float(phys_doil) if pd.notna(phys_doil) else np.nan
    emp = float(emp_doil) if pd.notna(emp_doil) else np.nan
    if pd.isna(phys):  # non-JP well (no physics) — empirical is the only estimate
        return (emp, "empirical") if pd.notna(emp) else (np.nan, "—")
    if method == "physics":
        return phys, "physics"
    if method == "empirical":
        return (emp, "empirical") if pd.notna(emp) else (phys, "physics (emp N/A)")
    # auto
    if pd.notna(emp) and pd.notna(phys) and abs(phys - emp) >= max(20.0, 0.3 * abs(phys)):
        return emp, "empirical (override)"
    return phys, "physics"


def _render_results(results_df: pd.DataFrame, delta_p: float) -> None:
    """Field summary, pad subtotals, well detail, ΔOil chart, physics-vs-empirical."""
    direction = "reduction" if delta_p < 0 else "increase"
    has_emp = "Emp class" in results_df.columns

    # ── method override: which ΔOil drives the group total / ranking ──
    results_df = results_df.copy()
    if has_emp:
        method = st.radio(
            "Primary method (drives the group total & ranking)",
            options=["auto", "physics", "empirical"],
            horizontal=True,
            key="hpi_method",
            format_func=lambda m: {
                "auto": "Auto — empirical where it disagrees with physics",
                "physics": "Physics",
                "empirical": "Empirical where available",
            }[m],
        )
        chosen = [
            _chosen_method(r["ΔOil (BOPD)"], r.get("Emp ΔOil (BOPD)", np.nan), method)
            for _, r in results_df.iterrows()
        ]
        results_df["Chosen ΔOil (BOPD)"] = [c[0] for c in chosen]
        results_df["Method used"] = [c[1] for c in chosen]
    else:
        method = "physics"
        results_df["Chosen ΔOil (BOPD)"] = results_df["ΔOil (BOPD)"]
        results_df["Method used"] = "physics"

    st.subheader("Field Summary")
    oil_now = results_df["Oil now (BOPD)"].sum()
    d_oil = results_df["Chosen ΔOil (BOPD)"].sum()
    oil_scen = oil_now + d_oil
    n_sonic = int(results_df["Sonic scen"].fillna(False).sum())
    n_override = int(results_df["Method used"].astype(str).str.startswith("empirical").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wells", len(results_df))
    c2.metric("Oil now (modeled)", f"{oil_now:,.0f} BOPD")
    c3.metric(f"Oil at {int(delta_p):+d} psi", f"{oil_scen:,.0f} BOPD", f"{d_oil:+,.0f}")
    c4.metric(
        "Sonic in scenario", n_sonic,
        help="Wells that hit sonic flow at the new WHP — result unreliable / "
        "the header can't be dropped that far on these.",
    )
    st.caption(
        f"Header {direction} of {abs(int(delta_p))} psi applied to each well's "
        f"wellhead pressure (ΔWHP = ΔHeader, 1:1). Group ΔOil uses the "
        f"**{method}** method"
        + (f" — {n_override} well(s) on empirical." if n_override else ".")
    )
    if has_emp:
        phys_tot = float(results_df["ΔOil (BOPD)"].sum())
        emp_tot = float(sum(
            _chosen_method(r["ΔOil (BOPD)"], r.get("Emp ΔOil (BOPD)", np.nan), "empirical")[0]
            for _, r in results_df.iterrows()
        ))
        auto_tot = float(sum(
            _chosen_method(r["ΔOil (BOPD)"], r.get("Emp ΔOil (BOPD)", np.nan), "auto")[0]
            for _, r in results_df.iterrows()
        ))
        n_emp = int(results_df["Emp ΔOil (BOPD)"].notna().sum()) if "Emp ΔOil (BOPD)" in results_df.columns else 0
        st.caption(
            f"Group ΔOil by method — Physics **{phys_tot:+,.0f}** · Empirical **{emp_tot:+,.0f}** · "
            f"Auto **{auto_tot:+,.0f}** BOPD. Only **{n_emp} of {len(results_df)}** wells have a "
            "usable empirical estimate (the rest are sonic/slugging/no-gauge and fall back to "
            "physics), so the totals move only as much as those few wells disagree."
        )

    # Verdict breakdown — how many wells fall in each response category.
    if "Verdict" in results_df.columns:
        vc = results_df["Verdict"].value_counts()
        st.caption("Verdict mix — " + "  ·  ".join(f"**{k}**: {v}" for k, v in vc.items()))

    # Pad subtotals
    st.subheader("By Pad")
    pad_agg = (
        results_df.groupby("Pad")
        .agg(**{
            "Oil now (BOPD)": ("Oil now (BOPD)", "sum"),
            "ΔOil (BOPD)": ("Chosen ΔOil (BOPD)", "sum"),
        })
        .reset_index()
    )
    pad_agg["Oil after Δ (BOPD)"] = pad_agg["Oil now (BOPD)"] + pad_agg["ΔOil (BOPD)"]
    st.dataframe(
        pad_agg.style.format(
            {
                "Oil now (BOPD)": "{:,.0f}",
                "ΔOil (BOPD)": "{:+,.0f}",
                "Oil after Δ (BOPD)": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Well details — Verdict + provenance up front.
    st.subheader("Well Details")
    front = [c for c in ["Well", "Pad", "Lift", "Verdict", "Method used",
                          "Chosen ΔOil (BOPD)", "Pump", "IPR src", "Fric src"]
             if c in results_df.columns]
    disp_df = results_df[front + [c for c in results_df.columns if c not in front]]
    fmt = {
        "BHP cal err": "{:+,.0f}",
        "BHP now (psi)": "{:,.0f}",
        "BHP scen (psi)": "{:,.0f}",
        "ΔBHP (psi)": "{:+,.0f}",
        "Oil now (BOPD)": "{:,.0f}",
        "Oil scen (BOPD)": "{:,.0f}",
        "ΔOil (BOPD)": "{:+,.0f}",
        "Chosen ΔOil (BOPD)": "{:+,.0f}",
        "Phys dBHP/dWHP": "{:.2f}",
        "PF rate now (BWPD)": "{:,.0f}",
        "PF rate scen (BWPD)": "{:,.0f}",
        "Mach scen": "{:.2f}",
    }
    if has_emp:
        fmt["Emp dBHP/dWHP"] = "{:.2f}"
        fmt["Emp ΔOil (BOPD)"] = "{:+,.0f}"
    st.dataframe(
        disp_df.style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Verdict": st.column_config.TextColumn(
                "Verdict",
                help="sonic-decoupled = pump choked at current WHP, header won't help; "
                "chokes when lowered = responds then sonic-limits; responsive ✓ = "
                "physics + field agree; disagree — check = physics says respond but "
                "field is flat (calibrate / investigate); responsive (physics) = no "
                "empirical to confirm.",
            ),
            "Method used": st.column_config.TextColumn(
                "Method used",
                help="Which ΔOil feeds the group total for this well, per the "
                "Primary method selector above.",
            ),
            "Chosen ΔOil (BOPD)": st.column_config.NumberColumn(
                "Chosen ΔOil (BOPD)", format="%+,d",
                help="The ΔOil used in the group total — physics or empirical per "
                "Method used.",
            ),
            "Fric src": st.column_config.TextColumn(
                "Fric src",
                help="calibrated = kth/kdi tuned to measured BHP; databricks = "
                "vw_prop_mech defaults; manual = slider override.",
            ),
            "BHP cal err": st.column_config.NumberColumn(
                "BHP cal err", format="%+d",
                help="Modeled minus measured BHP after calibration (psi). Near 0 = "
                "good match; large = model couldn't hit the gauge (see Sonic).",
            ),
            "IPR src": st.column_config.TextColumn(
                "IPR src",
                help="gauge = Vogel from measured BHP; back-calc = BHP estimated "
                "from JP physics; jp_chars = Databricks defaults (least accurate).",
            ),
            "Phys dBHP/dWHP": st.column_config.NumberColumn(
                "Phys dBHP/dWHP", format="%.2f",
                help="Physics-implied BHP sensitivity to wellhead pressure "
                "(ΔBHP / Δ from the two solves).",
            ),
            "Emp dBHP/dWHP": st.column_config.NumberColumn(
                "Emp dBHP/dWHP", format="%.2f",
                help="Empirical within-day BHP-vs-WHP slope from historian trends "
                "— directly comparable to Phys dBHP/dWHP.",
            ),
            "Emp class": st.column_config.TextColumn(
                "Emp class",
                help="responsive = BHP tracks surface pressure; slugging = it "
                "doesn't (header move won't help); insufficient/no tag = can't fit.",
            ),
            "Sonic now": st.column_config.CheckboxColumn("Sonic now"),
            "Sonic scen": st.column_config.CheckboxColumn(
                "Sonic scen",
                help="Sonic at the new WHP — solve unreliable; header can't drop "
                "this far here.",
            ),
        },
    )

    # ── physics vs empirical comparison ────────────────────────────────
    if has_emp:
        st.subheader("Physics vs Empirical (dBHP/dWHP)")
        comp = results_df.dropna(subset=["Phys dBHP/dWHP", "Emp dBHP/dWHP"]).copy()
        if comp.empty:
            st.caption(
                "No wells had both a physics solve and a usable empirical slope "
                "to compare. Check the 'Emp class' column — many 'insufficient' "
                "means the headers didn't move enough intraday over the window."
            )
        else:
            class_color = {"responsive": "#2ca02c", "slugging": "#d62728"}
            fig = go.Figure()
            for cls, grp in comp.groupby("Emp class"):
                fig.add_trace(
                    go.Scatter(
                        x=grp["Phys dBHP/dWHP"],
                        y=grp["Emp dBHP/dWHP"],
                        mode="markers+text",
                        text=grp["Well"],
                        textposition="top center",
                        name=str(cls),
                        marker=dict(size=10, color=class_color.get(str(cls), "#7f7f7f")),
                        hovertemplate="%{text}<br>Phys: %{x:.2f}<br>Emp: %{y:.2f}<extra></extra>",
                    )
                )
            lo = float(np.nanmin([comp["Phys dBHP/dWHP"].min(), comp["Emp dBHP/dWHP"].min(), 0]))
            hi = float(np.nanmax([comp["Phys dBHP/dWHP"].max(), comp["Emp dBHP/dWHP"].max(), 1]))
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines", name="agree (y=x)",
                    line=dict(dash="dash", color="gray"), hoverinfo="skip",
                )
            )
            fig.update_layout(
                xaxis_title="Physics dBHP/dWHP",
                yaxis_title="Empirical dBHP/dWHP",
                height=460,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Points near the dashed line mean physics and field data agree on "
                "how strongly BHP tracks wellhead pressure. Big gaps are where the "
                "model and reality diverge — candidates for the empirical override."
            )

    # ── chosen ΔOil by well, colored by pad ────────────────────────────
    st.subheader("Oil Change by Well (chosen method)")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    pad_colors = {
        pad: palette[i % len(palette)]
        for i, pad in enumerate(sorted(results_df["Pad"].unique()))
    }
    fig = go.Figure()
    for pad in sorted(results_df["Pad"].unique()):
        sub = results_df[results_df["Pad"] == pad].sort_values("Chosen ΔOil (BOPD)", ascending=False)
        fig.add_trace(
            go.Bar(
                x=sub["Well"],
                y=sub["Chosen ΔOil (BOPD)"],
                name=f"Pad {pad}",
                marker_color=pad_colors[pad],
                hovertemplate="%{x}<br>ΔOil: %{y:,.0f} BOPD<extra></extra>",
            )
        )
    fig.update_layout(
        yaxis_title="Chosen ΔOil (BOPD)",
        xaxis_title="Well",
        barmode="group",
        height=400,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # ── candidate ranking — best header-move opportunities ─────────────
    st.subheader("Candidate Ranking")
    rank = results_df.copy()
    rank["BOPD per 100 psi"] = (
        rank["Chosen ΔOil (BOPD)"] / abs(delta_p) * 100 if delta_p != 0 else np.nan
    )
    # Keep wells that actually move; sonic-decoupled / no-response fall out.
    rank = rank[rank["Chosen ΔOil (BOPD)"].abs() >= 1.0].sort_values(
        "Chosen ΔOil (BOPD)", ascending=(delta_p > 0)
    )
    if rank.empty:
        st.caption("No wells show a material response to this header move.")
    else:
        st.caption(
            f"Ranked by ΔOil for the {int(delta_p):+d} psi move (≈ BOPD per 100 psi of "
            "header change). Sonic-decoupled / no-response wells drop out — they're the "
            "ones a header move won't help."
        )
        rcols = [c for c in ["Well", "Pad", "Verdict", "Method used",
                             "Chosen ΔOil (BOPD)", "BOPD per 100 psi", "Sonic now"]
                 if c in rank.columns]
        st.dataframe(
            rank[rcols].style.format(
                {"Chosen ΔOil (BOPD)": "{:+,.0f}", "BOPD per 100 psi": "{:+,.1f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # CSV
    st.download_button(
        "Download Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="header_impact_results.csv",
        mime="text/csv",
        key="hpi_csv_download",
    )


# ── per-well empirical fit diagnostics ────────────────────────────────────


def _plot_well_fits(well: str, trend_df: pd.DataFrame, well_fits: dict):
    """Scatter of BHP vs WHP and BHP vs HeaderP, colored by date, with each
    day's within-day fit drawn over that day's points.

    Returns a Plotly figure, or None if no usable driver columns are present.
    """
    from plotly.subplots import make_subplots

    panels = [(drv, f"BHP~{drv}") for drv in ("WHP", "HeaderP")]
    avail = [
        (drv, key) for drv, key in panels
        if drv in trend_df.columns and "BHP" in trend_df.columns
    ]
    if not avail:
        return None

    base_min = pd.DatetimeIndex(trend_df.index).normalize().min()
    titles = []
    for drv, key in avail:
        f = well_fits.get(key)
        if f is not None and not np.isnan(f.mean_slope):
            titles.append(
                f"BHP vs {drv} — slope {f.mean_slope:.2f}, "
                f"{ht.classify_response(f)} ({f.n_days}d, {f.n_good_days} good)"
            )
        else:
            titles.append(f"BHP vs {drv} — no fit")

    fig = make_subplots(rows=1, cols=len(avail), subplot_titles=titles)
    for i, (drv, key) in enumerate(avail, start=1):
        d = trend_df[[drv, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d["BHP"] > 50]
        ords = (pd.DatetimeIndex(d.index).normalize() - base_min).days
        fig.add_trace(
            go.Scatter(
                x=d[drv], y=d["BHP"], mode="markers",
                marker=dict(
                    size=4, color=ords, colorscale="Viridis",
                    showscale=(i == 1),
                    colorbar=dict(title="Days") if i == 1 else None,
                ),
                name="hourly", showlegend=False,
                hovertemplate=f"{drv}: %{{x:.0f}} psi<br>BHP: %{{y:.0f}} psi<extra></extra>",
            ),
            row=1, col=i,
        )
        f = well_fits.get(key)
        if f is not None and not f.daily.empty:
            # Only the "good" days (clean r² + slope in band) — the bad/negative
            # daily fits are dropped, matching the original tool's clean plot.
            gd = f.daily[f.daily["good"]] if "good" in f.daily.columns else f.daily
            xseg: list = []
            yseg: list = []
            for _, dr in gd.iterrows():
                if pd.isna(dr["slope"]):
                    continue
                x0, x1 = dr["x_min"], dr["x_max"]
                xseg += [x0, x1, None]
                yseg += [dr["slope"] * x0 + dr["intercept"], dr["slope"] * x1 + dr["intercept"], None]
            if xseg:
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=yseg, mode="lines",
                        line=dict(color="crimson", width=1), opacity=0.6,
                        name="good daily fits", showlegend=(i == 1), hoverinfo="skip",
                    ),
                    row=1, col=i,
                )
        fig.update_xaxes(title_text=f"{drv} (psi)", row=1, col=i)
        fig.update_yaxes(title_text="BHP (psi)", row=1, col=i)

    fig.update_layout(
        height=480, title=f"{well} — within-day fits (hourly points colored by date)"
    )
    return fig


def _grid_fig(well_dfs: dict, fits: dict, driver: str = "WHP"):
    """Plotly subplot grid — one panel per well, BHP vs ``driver``, hourly points
    colored by date with the good daily fits overlaid in red. Replaces the
    matplotlib grid: interactive in-app (pan/zoom, hover) and crisp PNG export
    via kaleido (no rasterization blur).

    Returns a ``plotly.graph_objects.Figure``, or None if no well has usable data.
    """
    from plotly.subplots import make_subplots

    key = f"BHP~{driver}"
    wells = sorted(
        w for w, d in well_dfs.items()
        if driver in d.columns and "BHP" in d.columns
        and not d[[driver, "BHP"]].dropna().empty
    )
    if not wells:
        return None
    n = len(wells)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    titles = []
    for w in wells:
        f = fits.get(w, {}).get(key)
        cls = ht.classify_response(f) if f is not None else "n/a"
        slope = f.mean_slope if f is not None else float("nan")
        titles.append(f"{w}  m={slope:.2f} · {cls}")

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        vertical_spacing=0.09, horizontal_spacing=0.06,
        x_title=f"{driver} (psi)", y_title="BHP (psi)",
    )
    for i, w in enumerate(wells):
        r = i // cols + 1
        c = i % cols + 1
        d = well_dfs[w][[driver, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d["BHP"] > 50]
        if d.empty:
            continue
        idx = pd.DatetimeIndex(d.index)
        ords = (idx.normalize() - idx.normalize().min()).days
        fig.add_trace(
            go.Scatter(
                x=d[driver], y=d["BHP"], mode="markers",
                marker=dict(size=4, color=ords, colorscale="Viridis", showscale=False),
                hovertemplate=f"{driver}: %{{x:.0f}}<br>BHP: %{{y:.0f}}<extra>{w}</extra>",
                showlegend=False, name=w,
            ),
            row=r, col=c,
        )
        f_fit = fits.get(w, {}).get(key)
        if f_fit is not None and not f_fit.daily.empty and "good" in f_fit.daily.columns:
            xseg: list = []
            yseg: list = []
            for _, dr in f_fit.daily[f_fit.daily["good"]].iterrows():
                x0, x1 = dr["x_min"], dr["x_max"]
                xseg += [x0, x1, None]
                yseg += [
                    dr["slope"] * x0 + dr["intercept"],
                    dr["slope"] * x1 + dr["intercept"],
                    None,
                ]
            if xseg:
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=yseg, mode="lines",
                        line=dict(color="crimson", width=1), opacity=0.6,
                        hoverinfo="skip", showlegend=False,
                    ),
                    row=r, col=c,
                )
        # Clip axes to the bulk (1–99th pct) so sensor spikes don't compress the cloud.
        xlo, xhi = np.nanpercentile(d[driver], [1, 99])
        ylo, yhi = np.nanpercentile(d["BHP"], [1, 99])
        xpad = max((xhi - xlo) * 0.05, 1.0)
        ypad = max((yhi - ylo) * 0.05, 1.0)
        fig.update_xaxes(range=[xlo - xpad, xhi + xpad], row=r, col=c)
        fig.update_yaxes(range=[ylo - ypad, yhi + ypad], row=r, col=c)

    fig.update_layout(
        height=max(360, rows * 280),
        title_text=f"BHP vs {driver} — good daily fits in red",
        showlegend=False,
        margin=dict(l=60, r=20, t=70, b=60),
    )
    # tighter title font across the grid (subplot titles + shared axis titles)
    fig.update_annotations(font_size=10)
    return fig


def _render_diagnostics() -> None:
    """Per-well empirical fit diagnostics with a PNG download.

    Reads the raw trends + fits stashed during the run, so the well selector
    survives Streamlit reruns.
    """
    well_dfs = st.session_state.get("hpi_emp_well_dfs")
    fits = st.session_state.get("hpi_emp_fits")
    if not well_dfs or not fits:
        return

    st.subheader("Per-Well Empirical Fit Diagnostics")
    st.caption(
        "Hourly BHP vs surface pressure, colored by date, with each day's "
        "within-day fit drawn over that day's points — the scatter the mean "
        "slope is built from. Use it to judge whether the fit is real."
    )
    wells = sorted(well_dfs.keys())
    well = st.selectbox("Well", options=wells, key="hpi_diag_well")
    fig = _plot_well_fits(well, well_dfs[well], fits.get(well, {}))
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        try:
            png = fig.to_image(format="png", width=1200, height=480, scale=2)
            st.download_button(
                "Download this plot (PNG)", data=png,
                file_name=f"header_fit_{well}.png", mime="image/png",
                key="hpi_diag_png",
            )
        except Exception as e:
            st.caption(f"PNG export unavailable ({e}). Use the chart's camera icon to save instead.")
    else:
        st.info("No usable trend data for this well.")

    # ── grid plot (all wells) — the original tool's deliverable ──
    st.markdown("---")
    st.markdown(
        "**Grid — all wells** (BHP vs driver, hourly points colored by date, "
        "good daily fits in red)"
    )
    gc1, _ = st.columns([1, 3])
    with gc1:
        grid_driver = st.selectbox("Grid driver", ["WHP", "HeaderP"], key="hpi_grid_driver")
    if st.button("Build grid", key="hpi_grid_btn"):
        with st.spinner("Rendering grid..."):
            gfig = _grid_fig(well_dfs, fits, grid_driver)
        if gfig is None:
            st.info("No wells with usable data for that driver.")
            st.session_state.pop("hpi_grid_fig", None)
        else:
            st.session_state["hpi_grid_fig"] = gfig
            st.session_state["hpi_grid_driver_used"] = grid_driver
    gfig = st.session_state.get("hpi_grid_fig")
    if gfig is not None:
        used = st.session_state.get("hpi_grid_driver_used", "WHP")
        st.plotly_chart(gfig, use_container_width=True)
        try:
            png = gfig.to_image(format="png", scale=2)
            st.download_button(
                "Download grid PNG", data=png,
                file_name=f"header_fit_grid_{used}.png", mime="image/png",
                key="hpi_grid_dl",
            )
        except Exception as e:
            st.caption(
                f"PNG export unavailable ({e}). Use the camera icon on the chart to save instead."
            )


# ── analog-donor estimate (gaugeless / non-JP wells) ──────────────────────────


def _analog_doil(
    donor_slope: float,
    donor_pwf: float,
    donor_res_pres: float,
    target_qoil: float,
    target_res_pres: float,
    delta_p: float,
) -> tuple[float, float, float]:
    """Gaugeless ΔOil via an analog donor (borrow-donor-drawdown method).

    The donor's drawdown fraction (pwf ÷ ResP) anchors the target's flowing BHP;
    the donor's empirical dBHP/dWHP slope drives ΔBHP for the header move; ΔOil is
    read off a Vogel IPR built from the target's test oil rate + assumed ResP.
    Returns (target_pwf, delta_bhp, delta_oil).
    """
    from woffl.flow.inflow import InFlow

    frac = donor_pwf / donor_res_pres if donor_res_pres else 0.5
    frac = float(np.clip(frac, 0.05, 0.95))
    target_pwf = frac * target_res_pres
    ipr = InFlow(qwf=target_qoil, pwf=target_pwf, pres=target_res_pres)
    delta_bhp = donor_slope * delta_p
    b0 = float(np.clip(target_pwf, 1.0, target_res_pres))
    b1 = float(np.clip(target_pwf + delta_bhp, 1.0, target_res_pres))
    delta_oil = ipr.oil_flow(b1, "vogel") - ipr.oil_flow(b0, "vogel")
    return target_pwf, delta_bhp, delta_oil


def _render_analog(months_back: int, delta_p: float) -> None:
    """Analog-donor estimate for gaugeless / non-JP wells (e.g. ESP)."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from ._common import normalize_short_name

    with st.expander("Analog estimate — gaugeless / non-JP wells (e.g. ESP)", expanded=False):
        st.caption(
            "No jet pump and no BHP gauge? Pick a similar **gauged, responsive** well "
            "as a donor. The donor supplies the empirical dBHP/dWHP slope and its "
            "drawdown fraction (pwf ÷ ResP); the target supplies its latest test oil "
            "rate and assumed reservoir pressure. ΔOil is read off the target's Vogel "
            f"IPR for the same {int(delta_p)} psi header move. (Estimates shown here "
            "are separate from the jet-pump group total above.)"
        )

        # Seed defaults + donor options from the last header-impact run, so the
        # user just picks a donor instead of typing names.
        results_df = st.session_state.get("hpi_results_df")
        input_df = st.session_state.get("hpi_input_df")
        donor_options: list[str] = []
        gaugeless_seed: list[dict] = []
        if results_df is not None and "Verdict" in results_df.columns:
            resp_map: dict = {}
            if input_df is not None and "ResP (psi)" in input_df.columns:
                resp_map = {row["Well"]: row.get("ResP (psi)") for _, row in input_df.iterrows()}
            flagged = results_df[results_df["Verdict"] == "gaugeless — use Analog"]
            for _, r in flagged.iterrows():
                wn = r["Well"]
                gaugeless_seed.append({
                    "Target": wn,
                    "Donor": "",
                    "Target oil (BOPD)": float(r.get("Oil now (BOPD)") or 0.0),
                    "Target ResP (psi)": float(resp_map.get(wn) or 1800.0),
                })
            if "Emp class" in results_df.columns:
                donor_options = sorted(
                    results_df[results_df["Emp class"] == "responsive"]["Well"].dropna().unique().tolist()
                )

        if gaugeless_seed:
            st.caption(
                f"**{len(gaugeless_seed)}** gaugeless well(s) flagged in the last run "
                f"are pre-filled below; pick a donor for each. "
                f"**{len(donor_options)}** responsive donor(s) available."
            )
            if st.button("Reset to flagged gaugeless wells", key="hpi_analog_reset"):
                st.session_state["hpi_analog_df"] = pd.DataFrame(gaugeless_seed)

        default = pd.DataFrame(
            gaugeless_seed
            if gaugeless_seed
            else [{"Target": "", "Donor": "", "Target oil (BOPD)": 0.0, "Target ResP (psi)": 1800.0}]
        )
        donor_col = (
            st.column_config.SelectboxColumn(
                "Donor (responsive)", options=donor_options,
                help=f"Pick from {len(donor_options)} responsive wells in the last run.",
            )
            if donor_options
            else st.column_config.TextColumn(
                "Donor (gauged, responsive)",
                help="Type a similar gauged well; run the scenario first to get a dropdown of responsive donors.",
            )
        )

        ed = st.data_editor(
            st.session_state.get("hpi_analog_df", default),
            num_rows="dynamic", hide_index=True, use_container_width=True,
            key="hpi_analog_editor",
            column_config={
                "Target": st.column_config.TextColumn(
                    "Target (gaugeless)",
                    help="Well to estimate — pre-filled from the last run's flagged wells.",
                ),
                "Donor": donor_col,
                "Target oil (BOPD)": st.column_config.NumberColumn("Target oil (BOPD)", format="%.0f"),
                "Target ResP (psi)": st.column_config.NumberColumn(
                    "Target ResP (psi)", format="%.0f", min_value=500, max_value=5000, step=50,
                ),
            },
        )
        if not st.button("Compute analog", key="hpi_analog_btn"):
            return
        st.session_state["hpi_analog_df"] = ed

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - relativedelta(months=int(months_back))).strftime("%Y-%m-%d")
        rows = []
        for _, r in ed.iterrows():
            tgt = normalize_short_name(str(r.get("Target", "")).strip())
            donor = normalize_short_name(str(r.get("Donor", "")).strip())
            if not tgt or tgt.upper() == "MP" or not donor:
                continue
            try:
                tgt_oil = float(r["Target oil (BOPD)"])
                tgt_resp = float(r["Target ResP (psi)"])
            except (TypeError, ValueError):
                continue
            if tgt_oil <= 0 or tgt_resp <= 0:
                rows.append({"Target": tgt, "Donor": donor, "Note": "need oil > 0 and ResP > 0"})
                continue
            try:
                dwd, _ = ht.fetch_header_trends((donor,), start, end)
            except Exception as e:
                rows.append({"Target": tgt, "Donor": donor, "Note": f"donor trend error: {e}"})
                continue
            dfit = ht.fit_well(dwd[donor]).get("BHP~WHP") if donor in dwd else None
            if dfit is None or ht.classify_response(dfit) != "responsive":
                rows.append({"Target": tgt, "Donor": donor, "Note": "donor not responsive / no gauge"})
                continue
            dvogel = get_vogel_for_wells([donor], months_back).get(donor)
            if not dvogel or not dvogel.get("ResP"):
                rows.append({"Target": tgt, "Donor": donor, "Note": "donor has no Vogel IPR (needs gauge tests)"})
                continue
            pwf, dbhp, doil = _analog_doil(
                dfit.mean_slope, float(dvogel["pwf"]), float(dvogel["ResP"]),
                tgt_oil, tgt_resp, delta_p,
            )
            rows.append({
                "Target": tgt, "Donor": donor,
                "Donor slope": round(dfit.mean_slope, 2),
                "Donor drawdown": round(float(dvogel["pwf"]) / float(dvogel["ResP"]), 2),
                "Target pwf (psi)": round(pwf, 0),
                "ΔBHP (psi)": round(dbhp, 0),
                "ΔOil (BOPD)": round(doil, 0),
                "Note": "ok",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Enter at least one Target + Donor with oil > 0 and ResP > 0.")
