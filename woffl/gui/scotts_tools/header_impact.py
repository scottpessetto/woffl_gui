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
from woffl.gui.utils import PAD_PF_FALLBACK, default_pad_pf, load_well_characteristics

from . import header_trend as ht
from ._common import (
    build_well_config,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_latest_bhp_per_well,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    pad_from_mp_name,
)
from .header_engine import (
    DONOR_GROUP_TOKENS,
    GROUP_LIFT,
    GROUP_PADFORMATION,
    OWN_TOKEN,
    _chosen_method,
    _verdict,
    aggregate_response_curve,
    FORMATION_PR_MAX,
    average_slope,
    average_vogel_rows,
    backpressure_consistency,
    backtest_anchors,
    bias_by_pad,
    corr_display_plan,
    describe_donor,
    donor_member_wells,
    donor_tokens,
    pad_updown_lever,
    predict_dbhp_from_curve,
    pf_map_from_selected,
    pr_hi_for_formation,
    resolve_pad_pf,
    depletion_signature,
    estimate_header_impacts,
    fit_well_ipr,
    group_correlation_stats,
    sense_check_response,
    sense_check_table,
    summarize_sensitivity,
    vogel_oil,
)
from .pf_scenario import _estimate_gaugeless_ipr

# Header-change grid for the per-pad response curves (psi, relative to each well's
# current WHP). Symmetric — show the oil response equally for header drops and rises.
_SWEEP_DELTAS = (-150, -100, -50, 0, 50, 100, 150)
_WHP_FLOOR = 30.0  # don't solve below this absolute WHP (avoid nonsense)


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


# ── per-pad power-fluid pressure ─────────────────────────────────────────────


def _render_pad_pf_editor(pads: list[str]) -> dict[str, int]:
    """Editable per-pad power-fluid pressure table — the modeled PF baseline.

    Each pad runs its jet pumps at a different PF pressure; this makes that
    explicit (e.g. G/H/I/J distinct) instead of one global value. Values seed
    from PAD_PF_DEFAULTS, are editable, and persist across reruns in session
    state. Jet-pump wells inherit their pad's value as the "PF held" default in
    the well table below, overridable per well.
    """
    # STABLE editor base, rebuilt only when the pad set changes (CLAUDE.md
    # data_editor gotcha — feeding the editor its own prior output via
    # hpi_pad_pf makes a feedback loop). Edits live in the widget state and
    # are read from the return value into hpi_pad_pf below.
    tok = tuple(pads)
    grid = st.session_state.get("hpi_pad_pf_base")
    if grid is None or st.session_state.get("hpi_pad_pf_base_tok") != tok:
        resolved = resolve_pad_pf(pads, st.session_state.get("hpi_pad_pf"), default_pad_pf)
        grid = pd.DataFrame(
            [
                {
                    "Pad": p,
                    "PF (psi)": resolved[p],
                    "Source": "default" if resolved[p] == int(default_pad_pf(p)) else "edited",
                }
                for p in pads
            ]
        )
        st.session_state["hpi_pad_pf_base"] = grid
        st.session_state["hpi_pad_pf_base_tok"] = tok
        st.session_state.pop("hpi_pad_pf_editor", None)
    st.caption(
        "**Power-fluid pressure per pad** — held fixed while the header (WHP) is "
        "swept, so it sets the modeled baseline. Seeded from pad defaults; edit to "
        "match each pad's actual PF. JP wells inherit their pad's value below "
        "(overridable per well)."
    )
    edited = st.data_editor(
        grid,
        key="hpi_pad_pf_editor",
        hide_index=True,
        column_config={
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "PF (psi)": st.column_config.NumberColumn(
                "PF (psi)", format="%d", min_value=1000, max_value=5000, step=50,
                help="Power-fluid surface pressure for this pad's jet pumps.",
            ),
            "Source": st.column_config.TextColumn(
                "Source", disabled=True,
                help="'default' = pad default; 'edited' = you changed it.",
            ),
        },
    )
    out: dict[str, int] = {}
    for _, r in edited.iterrows():
        try:
            out[str(r["Pad"])] = int(round(float(r["PF (psi)"])))
        except (TypeError, ValueError):
            out[str(r["Pad"])] = int(default_pad_pf(str(r["Pad"])))
    st.session_state["hpi_pad_pf"] = out
    return out


def _render_pad_header_baseline(
    pads: list[str], delta_p: float, months_back: int
) -> dict[str, float]:
    """Per-pad current header (manifold) pressure with the Δ target — the visible
    anchor the static Δ is applied from.

    The operating level matters: +100 psi from 500 is a different oil lever than
    +100 from 0, so the Δ alone is ambiguous without a baseline. This seeds
    'Header now' from the latest live header tag (editable, to model a different
    level), shows target = header + Δ, and persists to ``hpi_pad_header`` so the
    **Apply pad header → WHP** button under the well table can push it onto each
    well's WHP baseline (which is what the solve actually sweeps).
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - relativedelta(months=int(months_back))).strftime("%Y-%m-%d")
    try:
        live = ht.fetch_latest_pad_header(tuple(sorted(pads)), start, end)
    except Exception:
        live = {}
    st.session_state["hpi_pad_header_live"] = live

    saved = st.session_state.get("hpi_pad_header", {}) or {}

    def _base(p):
        return saved.get(p, live.get(p))

    def _source(p):
        if p in saved and (p not in live or abs(saved[p] - live[p]) > 0.5):
            return "edited"
        return "live" if p in live else "no tag"

    # Today's ACTUAL wellhead pressure the solve sweeps from = pad-average of the
    # loaded wells' 'WHP now' (reflects any Apply-pad-header re-baseline). WHP sits
    # above the header by the flowline drop, so showing both makes the offset clear.
    idf = st.session_state.get("hpi_input_df")
    whp_avg: dict[str, float] = {}
    if idf is not None and "WHP now (psi)" in idf.columns and "Pad" in idf.columns:
        w = pd.to_numeric(idf["WHP now (psi)"], errors="coerce")
        whp_avg = {
            str(p): float(v) for p, v in w.groupby(idf["Pad"]).mean().items() if pd.notna(v)
        }

    # STABLE editor base, rebuilt only when its non-editable context changes
    # (pads / Δ / live header / WHP averages) — never from its own output
    # (hpi_pad_header), which would make the data_editor feedback loop. User
    # edits to "Header now" survive a rebuild because _base() reads `saved`.
    tok = (
        tuple(pads),
        int(delta_p),
        tuple(sorted((str(k), round(float(v), 1)) for k, v in live.items())),
        tuple(sorted((str(k), round(float(v), 1)) for k, v in whp_avg.items())),
    )
    grid = st.session_state.get("hpi_pad_hdr_base")
    if grid is None or st.session_state.get("hpi_pad_hdr_base_tok") != tok:
        grid = pd.DataFrame(
            [
                {
                    "Pad": p,
                    "Header now (psi)": int(round(_base(p))) if _base(p) is not None else None,
                    "WHP now (avg, psi)": int(round(whp_avg[p])) if p in whp_avg else None,
                    "→ WHP after Δ (psi)": (
                        int(round(whp_avg[p] + delta_p)) if p in whp_avg else None
                    ),
                    "Source": _source(p),
                }
                for p in pads
            ]
        )
        st.session_state["hpi_pad_hdr_base"] = grid
        st.session_state["hpi_pad_hdr_base_tok"] = tok
        st.session_state.pop("hpi_pad_header_editor", None)
    st.caption(
        f"**Header pressure vs actual WHP per pad**, and where the **Δ {int(delta_p):+d} "
        "psi** lands. *Header now* = latest live manifold reading (editable to model a "
        "different level). *WHP now (avg)* = today's actual wellhead the solve sweeps "
        "from (pad average; sits above the header by the flowline drop). The same Δ "
        "moves oil differently at 500 vs 0 psi — **Apply pad header → WHP** (under the "
        "well table) re-baselines the wells, preserving each well's offset."
    )
    edited = st.data_editor(
        grid, key="hpi_pad_header_editor", hide_index=True, use_container_width=True,
        column_config={
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Header now (psi)": st.column_config.NumberColumn(
                "Header now (psi)", format="%d", min_value=0, max_value=2000, step=10,
                help="Latest live pad header (manifold) pressure; edit to model a "
                "different level, then Apply under the well table.",
            ),
            "WHP now (avg, psi)": st.column_config.NumberColumn(
                "WHP now (avg, psi)", format="%d", disabled=True,
                help="Today's actual wellhead pressure the solve sweeps from "
                "(pad average of the loaded wells' WHP now). Blank until wells load.",
            ),
            "→ WHP after Δ (psi)": st.column_config.NumberColumn(
                "→ WHP after Δ (psi)", format="%d", disabled=True,
                help="WHP now (avg) + Δ — the target wellhead pressure (updates on Δ).",
            ),
            "Source": st.column_config.TextColumn(
                "Source", disabled=True,
                help="Header now: 'live' = queried historian · 'edited' = you changed "
                "it · 'no tag' = no header tag/data for this pad.",
            ),
        },
    )
    out: dict[str, float] = {}
    for _, r in edited.iterrows():
        v = r["Header now (psi)"]
        if pd.notna(v):
            out[str(r["Pad"])] = float(v)
    st.session_state["hpi_pad_header"] = out
    return out


# ── input table ─────────────────────────────────────────────────────────────


def _build_input_table(
    pads: list[str], months_back: int, pad_pf: dict[str, int] | None = None
) -> pd.DataFrame | None:
    """Per-well input table for the selected pads — ALL producers, by lift type.

    JP wells (current pump) take the physics path; ESP / gas-lift / flowing wells
    take the empirical/analog path. WHP from the latest test; ResP from
    vw_prop_resvr / jp_chars where available, else assumed 1800 (ESPs aren't
    characterized in vw_prop_resvr). PF held + Pump apply to JP wells only.

    ``pad_pf`` is the per-pad power-fluid pressure map (from the per-pad PF
    editor); each JP well's "PF held" seeds from its pad's value, falling back to
    ``default_pad_pf`` when a pad is absent. Overridable per well in the editor.
    """
    pad_pf = pad_pf or {}
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

        is_sch = chars.get("is_sch", True)
        if isinstance(is_sch, str):
            is_sch = is_sch.lower() in ("true", "1", "yes")
        formation = "Schrader" if is_sch else "Kuparuk"

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
                # Per-pad PF (from the per-pad PF editor); JP only, overridable per well.
                "PF held (psi)": (
                    int(pad_pf.get(pad, default_pad_pf(pad))) if is_jp else None
                ),
                "WHP now (psi)": int(round(whp_now)) if pd.notna(whp_now) else None,
                "ResP (psi)": int(round(float(res_pres))),
                "Formation": formation,
                # Like-wells donors (G3): blank = use the well's own IPR/correlation.
                "IPR donor": OWN_TOKEN,
                "Corr donor": OWN_TOKEN,
                "Include": True,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["Pad", "Lift", "Well"]).reset_index(drop=True)
    # Fill a missing 'WHP now' with the average of the other wells on the same pad —
    # wells on a shared header sit near the same WHP, so the pad mean is a sensible
    # stand-in (a NaN well doesn't contribute to its own pad mean). Editable per row.
    if "WHP now (psi)" in df.columns:
        whp = pd.to_numeric(df["WHP now (psi)"], errors="coerce")
        if whp.isna().any():
            pad_mean = whp.groupby(df["Pad"]).transform("mean")
            whp = whp.fillna(pad_mean)
        df["WHP now (psi)"] = [int(round(v)) if pd.notna(v) else None for v in whp]
    return df


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


def _synthetic_fit(slope: float):
    """A WithinDayFit carrying just an averaged BHP~WHP slope, classified
    responsive, so a donor / group-average correlation flows through
    ``_empirical_columns`` and ``_solve_nonjp_row`` unchanged."""
    return ht.WithinDayFit(
        y_name="BHP", x_name="WHP",
        mean_slope=float(slope), median_slope=float(slope), slope_std=0.0,
        n_days=30, n_good_days=20, mean_r2=0.8, daily=pd.DataFrame(),
    )


def _resolve_corr_fits(well: str, token: str, rows_meta: dict, emp_fits: dict):
    """Resolve the {key: WithinDayFit} dict to use for a well's empirical / non-JP
    estimate, per its Corr-donor token. Returns ``(fit_dict, provenance)``."""
    members = donor_member_wells(well, token, rows_meta)
    if members is None:
        return emp_fits.get(well), describe_donor(token)
    prov = describe_donor(token, len(members))
    if len(members) == 1:
        return (emp_fits.get(members[0]) or emp_fits.get(well)), prov
    slopes = [
        emp_fits[m]["BHP~WHP"].mean_slope
        for m in members
        # RESPONSIVE members only — a slugging/insufficient well still carries
        # a numeric mean_slope whenever it has >=1 good day, and _synthetic_fit
        # hard-codes a responsive classification, so without this filter
        # rejected wells' slopes get laundered into a "responsive" donor.
        # Mirrors group_correlation_stats, which only pools responsive wells.
        if m in emp_fits
        and emp_fits[m].get("BHP~WHP") is not None
        and ht.classify_response(emp_fits[m]["BHP~WHP"]) == "responsive"
        and pd.notna(emp_fits[m]["BHP~WHP"].mean_slope)
    ]
    avg = average_slope(slopes)
    if avg is None:
        return emp_fits.get(well), prov + " (no slope → own)"
    return {"BHP~WHP": _synthetic_fit(avg)}, prov


def _solve_nonjp_row(
    wn, row, emp_fits: dict, emp_well_dfs: dict, delta_p: float,
    fit_override: dict | None = None, corr_prov: str = "own",
    ipr_sink: dict | None = None, res_pres_override: float | None = None,
) -> dict:
    """Empirical-only result row for a non-JP well (ESP / gas-lift / flowing).

    No jet-pump physics: ``pwf`` = the well's own recent measured BHP; a generic
    Vogel IPR from the latest test oil rate + assumed reservoir pressure gives
    ΔOil for ΔBHP = (empirical BHP~WHP slope) × Δ. Gaugeless wells (no BHP trend)
    are flagged to use the Analog estimate instead. ``fit_override`` lets a Corr
    donor (a like well or a group average) supply the BHP~WHP slope.
    ``res_pres_override`` (Standing) swaps the assumed reservoir pressure for the
    well's fitted pseudo-Pr.
    """
    from woffl.flow.inflow import InFlow

    fit = (fit_override or emp_fits.get(wn) or {}).get("BHP~WHP")
    emp_class = ht.classify_response(fit) if fit is not None else "no data"
    emp_slope = fit.mean_slope if fit is not None else np.nan

    trend = emp_well_dfs.get(wn)
    bhp_now = np.nan
    if trend is not None and "BHP" in getattr(trend, "columns", []):
        bhp_s = trend["BHP"].dropna()
        if not bhp_s.empty:
            bhp_now = float(bhp_s.iloc[-1])

    oil = row.get("Oil (BOPD)")
    pseudo = res_pres_override is not None and not pd.isna(res_pres_override)
    res_pres = float(res_pres_override) if pseudo else float(row.get("ResP (psi)") or 1800.0)
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

    # Stash the generic Vogel IPR (anchored at the measured BHP + test oil +
    # assumed ResP) so the review shows an IPR curve for ESP / non-JP wells too.
    if (
        ipr_sink is not None
        and pd.notna(bhp_now)
        and oil is not None
        and pd.notna(oil)
        and float(oil) > 0
    ):
        ipr_sink[wn] = {
            "res_pres": float(max(res_pres, bhp_now + 100.0)),
            "qwf": float(oil),
            "pwf": float(bhp_now),
            "form_wc": 0.0,  # qwf is already the oil rate for non-JP
        }
    oil_scen = (float(oil) + emp_doil) if (pd.notna(oil) and pd.notna(emp_doil)) else np.nan

    return {
        "Well": wn,
        "Pad": row["Pad"],
        "Lift": row.get("Lift", ""),
        "Formation": row.get("Formation"),
        "Pump": "",
        "IPR src": "empirical (pseudo-Pr)" if pseudo else "empirical (test + assumed ResP)",
        "Fric src": "—",
        "BHP cal err": None,
        "PF held (psi)": None,
        "WHP now (psi)": int(round(whp_now)) if pd.notna(whp_now) else None,
        "WHP scen (psi)": int(round(whp_now + delta_p)) if pd.notna(whp_now) else None,
        "BHP now (psi)": bhp_now,
        "BHP scen (psi)": (bhp_now + emp_dbhp) if (pd.notna(emp_dbhp) and pd.notna(bhp_now)) else np.nan,
        "ΔBHP (psi)": emp_dbhp,
        "Oil now (BOPD)": float(oil) if pd.notna(oil) else np.nan,
        "Oil scen (BOPD)": oil_scen,
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
        "IPR donor": "—",
        "Corr donor": corr_prov,
        "Verdict": verdict,
    }


def _add_nonjp_curve(wn, rrow, ipr_rows, curve_wells, deltas) -> None:
    """Add a non-JP well's empirical oil-vs-header curve to ``curve_wells`` so the
    per-pad / field response curve includes ESP / non-JP producers, not just the
    swept JP wells. Oil at each Δ = the well's Vogel IPR sampled at BHP + slope×Δ.
    """
    from woffl.flow.inflow import InFlow

    ir = ipr_rows.get(wn)
    slope = rrow.get("Emp dBHP/dWHP")
    if ir is None or pd.isna(slope) or rrow.get("Emp class") != "responsive":
        return
    try:
        rp = float(ir["res_pres"])
        inflow = InFlow(qwf=float(ir["qwf"]), pwf=float(ir["pwf"]), pres=rp)
        whp_now = rrow.get("WHP now (psi)")
        oils, bhps, whps = [], [], []
        for d in deltas:
            b = float(np.clip(float(ir["pwf"]) + float(slope) * d, 1.0, rp))
            oils.append(float(inflow.oil_flow(b, "vogel")))
            bhps.append(b)
            whps.append((float(whp_now) + d) if pd.notna(whp_now) else np.nan)
        curve_wells[wn] = {"pad": rrow["Pad"], "oil": oils, "bhp": bhps, "whp": whps}
    except Exception:
        pass


# ── saved scenarios (G6) ─────────────────────────────────────────────────────


def _current_scenario() -> dict:
    """Snapshot the configured inputs + overrides (not results) for reuse."""
    df = st.session_state.get("hpi_input_df")
    return {
        "version": 1,
        "pads": st.session_state.get("hpi_pads", []),
        "months_back": int(st.session_state.get("hpi_months", 6)),
        "delta_p": int(st.session_state.get("hpi_delta", -50)),
        "build_curves": bool(st.session_state.get("hpi_build_curves", False)),
        "fric_mode": st.session_state.get("hpi_fric_mode", "calibrate"),
        "pad_pf": st.session_state.get("hpi_pad_pf", {}),
        "input_table": df.to_dict("records") if df is not None else [],
    }


def _apply_scenario(blob: dict, all_pads: list[str]) -> None:
    """Restore a scenario into session state. Widget keys are set before the
    widgets render (the supported Streamlit pattern); editor deltas + stale
    results are cleared so the restored logical state wins."""
    pads = [p for p in (blob.get("pads") or []) if p in all_pads]
    if pads:
        st.session_state["hpi_pads"] = pads
    if blob.get("months_back") is not None:
        st.session_state["hpi_months"] = int(blob["months_back"])
    if blob.get("delta_p") is not None:
        st.session_state["hpi_delta"] = int(blob["delta_p"])
    if blob.get("build_curves") is not None:
        st.session_state["hpi_build_curves"] = bool(blob["build_curves"])
    if blob.get("fric_mode") in ("calibrate", "databricks", "manual"):
        st.session_state["hpi_fric_mode"] = blob["fric_mode"]
    if blob.get("pad_pf"):
        st.session_state["hpi_pad_pf"] = {str(k): int(v) for k, v in blob["pad_pf"].items()}
    if blob.get("input_table"):
        restored = pd.DataFrame(blob["input_table"])
        st.session_state["hpi_input_df"] = restored
        st.session_state["hpi_input_base"] = restored.copy()
        # Re-baseline the measured-WHP snapshot from the restored table —
        # without it, "Apply pad header → WHP" silently no-ops (or uses a
        # stale snapshot from a different pad set) after a scenario load.
        if {"Well", "WHP now (psi)"}.issubset(restored.columns):
            st.session_state["hpi_whp_measured"] = dict(
                zip(restored["Well"], pd.to_numeric(restored["WHP now (psi)"], errors="coerce"))
            )
        else:
            st.session_state.pop("hpi_whp_measured", None)
        # Stamp the loaded-pads guard from the restored table itself.
        if "Pad" in restored.columns:
            st.session_state["hpi_loaded_pads"] = sorted(
                {str(p) for p in restored["Pad"].dropna()}
            )
        elif pads:
            st.session_state["hpi_loaded_pads"] = pads
    for k in (
        "hpi_editor", "hpi_pad_pf_editor", "hpi_pad_pf_base", "hpi_pad_hdr_base",
        "hpi_pad_header_editor", "hpi_results_df", "hpi_pdf_bytes", "hpi_curve",
    ):
        st.session_state.pop(k, None)


def _render_scenario_io(all_pads: list[str]) -> None:
    """Expander to save / load a full scenario (inputs + overrides) so a run is
    reproducible. Results aren't saved — reload restores inputs and you re-run."""
    import json

    with st.expander("💾 Saved scenario — save / load inputs + overrides", expanded=False):
        up = st.file_uploader("Load scenario (.json)", type=["json"], key="hpi_scn_up")
        if up is not None:
            sig = f"{up.name}:{up.size}"
            if st.session_state.get("hpi_scn_sig") != sig:
                try:
                    _apply_scenario(json.load(up), all_pads)
                    st.session_state["hpi_scn_sig"] = sig
                    st.success("Scenario loaded — press **Run header impact** to compute.")
                    st.rerun()
                except Exception as e:
                    st.warning(f"Could not load scenario ({type(e).__name__}: {e}).")
        st.download_button(
            "Download current scenario (.json)",
            data=json.dumps(_current_scenario(), indent=2, default=str).encode("utf-8"),
            file_name="header_impact_scenario.json", mime="application/json",
            key="hpi_scn_dl",
        )


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

    _render_scenario_io(all_pads)

    # ── settings ───────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
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

    # Per-pad power-fluid pressure (the modeled baseline; each pad differs). This
    # also drives the gaugeless BHP back-calc — no single global "PF at test time".
    pad_pf: dict[str, int] = {}
    if pads:
        pad_pf = _render_pad_pf_editor(pads)

    delta_p = st.number_input(
        "Header pressure change Δ (psi)",
        min_value=-500, max_value=500, value=-50, step=10,
        key="hpi_delta",
        help="Applied to each well's wellhead pressure (assumes ΔWHP = ΔHeader, "
        "1:1). Negative = header reduction (expect oil uplift).",
    )

    # The Δ is applied from each well's WHP baseline — surface the current per-pad
    # header level (queried) and the resulting target so the static Δ isn't a
    # level-free abstraction (+100 from 500 psi ≠ +100 from 0 psi).
    if pads:
        _render_pad_header_baseline(pads, delta_p, months_back)

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
            input_df = _build_input_table(pads, months_back, pad_pf)
        if input_df is None or input_df.empty:
            st.warning("No producers with recent allocated tests on the selected pad(s).")
            return
        st.session_state["hpi_input_df"] = input_df
        # STABLE frame for the data_editor (see CLAUDE.md data_editor gotcha):
        # the editor's data= must not be its own prior output, so programmatic
        # replacements set this base and pop the editor's widget key.
        st.session_state["hpi_input_base"] = input_df.copy()
        # Stamp which pads this table was loaded for, so changing the pad
        # multiselect can't silently run the OLD pads' wells.
        st.session_state["hpi_loaded_pads"] = list(pads)
        # Immutable snapshot of each well's measured WHP — the stable reference for
        # the offset-preserving "Apply pad header" re-baseline (so re-applying or
        # editing the header recomputes cleanly instead of compounding).
        st.session_state["hpi_whp_measured"] = dict(
            zip(input_df["Well"], pd.to_numeric(input_df["WHP now (psi)"], errors="coerce"))
        )
        st.session_state.pop("hpi_results_df", None)
        st.session_state.pop("hpi_editor", None)

    input_df = st.session_state.get("hpi_input_df")
    if input_df is None:
        st.info("Click **Load wells** to start.")
        return

    # Stale-pad guard: the well table, results, and Run all operate on the
    # loaded table — if the pad selection changed since Load, say so and block
    # the run instead of silently producing results for the old pads.
    loaded_pads = st.session_state.get("hpi_loaded_pads")
    pads_stale = loaded_pads is not None and set(loaded_pads) != set(pads)
    if pads_stale:
        st.warning(
            f"Pad selection changed (loaded: {', '.join(sorted(loaded_pads))} → "
            f"selected: {', '.join(sorted(pads))}) — click **Load wells** to "
            "refresh the table before running."
        )

    st.caption(
        "All producers on the pad: **JP** → physics, **ESP / gas-lift / flowing** → "
        "empirical (no jet-pump model). **PF held** (JP only) seeds from the per-pad "
        "PF above — edit a cell to override a single well. **ResP** is from "
        "vw_prop_resvr where available, else assumed 1800 (ESPs aren't characterized "
        "there) — edit it. **WHP now** is the latest test WHP (pad average where a "
        "well's own is missing). After a run, the "
        "**IPR & Correlation Coverage** panel (in the results) shows which wells are "
        "missing one and lets you assign a like-well / group-average donor, then re-run."
    )

    # Feed the editor the STABLE base frame, never its own prior output —
    # writing the edited frame back as next render's data= creates a feedback
    # loop that drops/duplicates cells during mass edits (CLAUDE.md gotcha;
    # same pattern as _render_coverage_panel). The widget replays edits onto
    # the stable base each rerun; the merged result lives in hpi_input_df.
    base_df = st.session_state.get("hpi_input_base")
    if base_df is None:
        base_df = input_df.copy()
        st.session_state["hpi_input_base"] = base_df

    config_cols = [c for c in [
        "Well", "Pad", "Lift", "Pump", "Oil (BOPD)", "PF held (psi)",
        "WHP now (psi)", "ResP (psi)", "Formation", "Include",
    ] if c in base_df.columns]
    edited_cfg = st.data_editor(
        base_df[config_cols],
        key="hpi_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", disabled=True, pinned="left"),
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Pump": st.column_config.TextColumn("Pump", disabled=True),
            "Formation": st.column_config.TextColumn("Formation", disabled=True),
            "PF held (psi)": st.column_config.NumberColumn(
                "PF held (psi)", format="%d",
                min_value=1000, max_value=5000, step=50,
                help="Power-fluid pressure held fixed during the WHP sweep. "
                "Seeds from the per-pad PF above; edit to override this well.",
            ),
            "WHP now (psi)": st.column_config.NumberColumn(
                "WHP now (psi)", format="%d",
                min_value=0, max_value=2000, step=10,
                help="Baseline wellhead pressure (latest test; the pad average where "
                "a well's own test WHP is missing). Edit if the current header differs.",
            ),
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Uncheck to skip this well in the run.",
            ),
        },
    )

    # Donors are assigned in the post-run "IPR & Correlation Coverage" panel (where
    # you can see which wells need one) and carried forward to the run from here.
    edited_df = edited_cfg.copy()
    edited_df["IPR donor"] = (
        input_df["IPR donor"].values if "IPR donor" in input_df.columns else OWN_TOKEN
    )
    edited_df["Corr donor"] = (
        input_df["Corr donor"].values if "Corr donor" in input_df.columns else OWN_TOKEN
    )

    # Persist the merged (base + edits) frame as the canonical input the run
    # reads. hpi_input_base stays untouched — only programmatic replacements
    # (Load wells, the two Apply buttons, scenario load) reset it.
    st.session_state["hpi_input_df"] = edited_df.copy()

    bpf, bhdr = st.columns(2)
    if bpf.button(
        "↻ Re-apply pad PF to wells",
        key="hpi_reapply_pf", use_container_width=True,
        help="Reset each JP well's 'PF held' to its pad's current value above "
        "(discards per-well PF overrides; keeps Include / WHP edits).",
    ):
        df2 = edited_df.copy()
        jp_mask = df2["Lift"] == "JP"
        df2.loc[jp_mask, "PF held (psi)"] = df2.loc[jp_mask, "Pad"].map(
            lambda p: int(pad_pf.get(p, default_pad_pf(p)))
        )
        st.session_state["hpi_input_df"] = df2
        st.session_state["hpi_input_base"] = df2.copy()
        st.session_state.pop("hpi_editor", None)  # clear stale editor delta
        st.rerun()
    if bhdr.button(
        "⤓ Apply pad header → wells' WHP now",
        key="hpi_apply_header", use_container_width=True,
        help="Re-baseline each well's 'WHP now' to the pad header above, PRESERVING "
        "its flowline drop: WHP = edited header + (measured WHP − live header). "
        "Editing a pad header to a hypothetical level shifts its wells together by "
        "that change (keeps their spread); idempotent. Overwrites manual WHP edits. "
        "Re-run to solve from the new baseline.",
    ):
        hdr_edit = st.session_state.get("hpi_pad_header", {}) or {}
        hdr_live = st.session_state.get("hpi_pad_header_live", {}) or {}
        whp_meas = st.session_state.get("hpi_whp_measured", {}) or {}
        df2 = edited_df.copy()
        new_whp = []
        for p, well, w in zip(df2["Pad"], df2["Well"], df2["WHP now (psi)"]):
            m = whp_meas.get(well)
            if p in hdr_edit and p in hdr_live and m is not None and pd.notna(m):
                # Flowline/choke drop (WHP sits above the manifold); clamp ≥ 0 so
                # noisy WHP < header doesn't invert it.
                offset = max(0.0, float(m) - float(hdr_live[p]))
                new_whp.append(int(round(float(hdr_edit[p]) + offset)))
            else:
                new_whp.append(w)  # no live header / no measured WHP → leave as-is
        df2["WHP now (psi)"] = new_whp
        st.session_state["hpi_input_df"] = df2
        st.session_state["hpi_input_base"] = df2.copy()
        st.session_state.pop("hpi_editor", None)  # clear stale editor delta
        st.rerun()

    _render_analog(months_back, delta_p)

    build_curves = st.checkbox(
        "Build per-pad oil-vs-header response curves (recommended; adds a sweep per well)",
        value=st.session_state.get("hpi_build_curves", True),
        key="hpi_build_curves",
        help="Solves each well across a range of header changes so the Sensitivity "
        "section shows the oil-vs-WHP lever per pad plus a field total. Untick for a "
        "faster run if you only need the single-Δ numbers.",
    )

    solve_use_pseudo = st.checkbox(
        "⚠ Experimental: solve with fitted pseudo-Pr as the reservoir pressure",
        value=st.session_state.get("hpi_solve_use_pseudo_pr", False),
        key="hpi_solve_use_pseudo_pr",
        help="Replace each well's database/assumed reservoir pressure with the pseudo-Pr "
        "from a single Vogel fit to its tests (formation-capped, Schrader 2200 / Kuparuk "
        "4200). NOTE: the field probe showed production tests rarely have enough BHP spread "
        "to pin a reservoir pressure — the fit often slams to the cap and WORSENS the ΔOil "
        "prediction (~2.3× the error). Off by default; kept for experimentation. Needs a run.",
    )

    run_clicked = st.button(
        "Run header impact", type="primary", use_container_width=True, key="hpi_run",
        disabled=pads_stale,
    ) or st.session_state.pop("hpi_trigger_run", False)
    if pads_stale:
        run_clicked = False  # a queued coverage-panel rerun can't bypass the guard
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
        # Per-well PF (from each well's "PF held" = its pad's PF unless overridden)
        # drives the gaugeless BHP back-calc — replaces the old global test_pf_pres.
        pf_map_sel = pf_map_from_selected(selected)
        with st.spinner(f"Estimating BHP for {len(missing)} gaugeless wells..."):
            vogel_dict.update(
                _estimate_gaugeless_ipr(
                    missing, months_back, PAD_PF_FALLBACK, jp_hist, jp_chars_dict,
                    whp_map=whp_map, pf_map=pf_map_sel,
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

    # ── Standing back-test correction (optional): replace each well's reservoir
    # pressure with the pseudo-Pr from a SINGLE Vogel fit to all its tests (formation-
    # capped). NOTE: the field probe showed this often pins to the cap and worsens the
    # ΔOil prediction, so it's off by default — kept for experimentation only.
    pseudo_pr_map: dict[str, float] = {}
    if solve_use_pseudo:
        form_map = dict(zip(selected["Well"], selected.get("Formation", [])))
        try:
            traw_pp = fetch_well_tests_raw(months_back)
        except Exception:
            traw_pp = None
        if traw_pp is not None and not traw_pp.empty:
            for wn in well_names:
                tw = traw_pp[traw_pp["well"] == wn]
                fit = fit_well_ipr(tw, pr_hi=pr_hi_for_formation(form_map.get(wn))) \
                    if not tw.empty else None
                if fit:
                    pseudo_pr_map[wn] = float(fit["pr"])
        if pseudo_pr_map:
            st.caption(
                f"Solving with fitted pseudo-Pr for {len(pseudo_pr_map)} well(s): "
                + ", ".join(f"{w}≈{int(p)}" for w, p in pseudo_pr_map.items())
            )

    # ── solve each well at baseline WHP and at WHP + Δ ─────────────────
    results = []
    errors: list[str] = []
    ipr_rows: dict = {}     # exact IPR (ResP/qwf/pwf/wc) used per JP well, for the review
    curve_wells: dict = {}  # per-well swept oil for the response curves (if requested)
    # Like-wells donor metadata (G3): pad / lift / formation per selected well.
    rows_meta = {
        str(r["Well"]): {
            "pad": r["Pad"], "lift": r.get("Lift"), "formation": r.get("Formation"),
        }
        for _, r in selected.iterrows()
    }
    progress = st.progress(0, text="Starting...")
    n = len(selected)
    for i, (_, row) in enumerate(selected.iterrows()):
        wn = row["Well"]
        progress.progress(i / max(n, 1), text=f"Solving {wn} ({i+1}/{n})...")
        ipr_token = str(row.get("IPR donor", OWN_TOKEN))
        corr_token = str(row.get("Corr donor", OWN_TOKEN))

        # Non-JP wells (ESP / gas-lift / flowing) take the empirical-only path.
        if str(row.get("Lift", "JP")) != "JP":
            fit_ov, corr_prov = _resolve_corr_fits(wn, corr_token, rows_meta, emp_fits)
            nrow = _solve_nonjp_row(
                wn, row, emp_fits, emp_well_dfs, delta_p, fit_ov, corr_prov,
                ipr_sink=ipr_rows, res_pres_override=pseudo_pr_map.get(wn),
            )
            results.append(nrow)
            if build_curves:
                _add_nonjp_curve(wn, nrow, ipr_rows, curve_wells, _SWEEP_DELTAS)
            continue

        pump = get_current_pump(jp_hist, wn)
        if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
            # Classified JP but no current pump now → fall back to empirical-only.
            fit_ov, corr_prov = _resolve_corr_fits(wn, corr_token, rows_meta, emp_fits)
            nrow = _solve_nonjp_row(
                wn, row, emp_fits, emp_well_dfs, delta_p, fit_ov, corr_prov,
                ipr_sink=ipr_rows, res_pres_override=pseudo_pr_map.get(wn),
            )
            results.append(nrow)
            if build_curves:
                _add_nonjp_curve(wn, nrow, ipr_rows, curve_wells, _SWEEP_DELTAS)
            continue
        chars = jp_chars_dict.get(wn)

        whp_now = row["WHP now (psi)"]
        if whp_now is None or pd.isna(whp_now):
            whp_now = float(whp_map.get(wn, 210.0))
        whp_now = float(whp_now)
        pf_held = float(row["PF held (psi)"])
        ipr_src = "gauge" if wn in gauged else ("back-calc" if wn in estimated else "jp_chars")

        try:
            # IPR donor (G3): own, a specific well, or a like-wells group average.
            ipr_members = donor_member_wells(wn, ipr_token, rows_meta)
            if ipr_members is None:
                donor_vogel = vogel_dict.get(wn)
            else:
                donor_vogel = average_vogel_rows(
                    [vogel_dict[m] for m in ipr_members if m in vogel_dict]
                ) or vogel_dict.get(wn)
            # Standing correction: swap in the fitted pseudo-Pr as reservoir pressure
            # (keeps the operating point qwf/pwf; only the IPR's Pr changes). Skip if
            # it would invert the IPR (pr must exceed the operating pwf).
            if wn in pseudo_pr_map and donor_vogel is not None:
                ppr = pseudo_pr_map[wn]
                if ppr > float(donor_vogel.get("pwf", 0) or 0):
                    donor_vogel = {**donor_vogel, "ResP": ppr}
                    ipr_src = f"{ipr_src}+psPr"
            wc = build_well_config(wn, jp_chars_dict, donor_vogel, surf_pres=whp_now)
            well_objs = create_well_objects(wc)
            wellbore, well_profile, inflow, res_mix, prop_pf = well_objs
            ipr_rows[wn] = {
                "res_pres": float(wc.res_pres), "qwf": float(wc.qwf),
                "pwf": float(wc.pwf), "form_wc": float(wc.form_wc),
            }

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
            # Clamp the scenario WHP to the sweep floor — delta_p reaches -500
            # while typical WHPs run 100-400 psi, and the solver gets garbage
            # (or dies into the skipped list) below the floor. The sweep
            # already applies the same clamp via _WHP_FLOOR.
            whp_scen = whp_now + delta_p
            if whp_scen < _WHP_FLOOR:
                errors.append(
                    f"{wn}: note — scenario WHP {whp_scen:.0f} psi is below the "
                    f"{_WHP_FLOOR:.0f}-psi floor; solved at the floor instead"
                )
                whp_scen = _WHP_FLOOR
            res_scen = _solve_at_whp(
                wc, well_objs, pump["nozzle_no"], pump["throat_ratio"],
                whp_scen, pf_held, fric,
            )
            if build_curves:
                # Capture model BHP (psu) + WHP at each swept point too — the
                # model's BHP-vs-WHP curve, overlaid on the historian scatter in
                # the per-pad sense-check ("did we see what the model expects?").
                oils, bhps, whps = [], [], []
                for d in _SWEEP_DELTAS:
                    whp_d = whp_now + d
                    whps.append(whp_d)
                    if whp_d < _WHP_FLOOR:
                        oils.append(np.nan)
                        bhps.append(np.nan)
                    elif d == 0:
                        oils.append(res_now["oil"])
                        bhps.append(res_now["psu"])
                    else:
                        rd = _solve_at_whp(
                            wc, well_objs, pump["nozzle_no"], pump["throat_ratio"],
                            whp_d, pf_held, fric,
                        )
                        oils.append(rd["oil"])
                        bhps.append(rd["psu"])
                curve_wells[wn] = {"pad": row["Pad"], "oil": oils, "bhp": bhps, "whp": whps}
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
            "Formation": row.get("Formation"),
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
        corr_fits, corr_prov = _resolve_corr_fits(wn, corr_token, rows_meta, emp_fits)
        emp_class = None
        if compare_emp:
            ecols = _empirical_columns(
                corr_fits, well_objs[2], wc.res_pres, res_now["psu"], delta_p
            )
            row_d.update(ecols)
            emp_class = ecols.get("Emp class")
        row_d["IPR donor"] = describe_donor(ipr_token, len(ipr_members or []))
        row_d["Corr donor"] = corr_prov
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
    st.session_state["hpi_run_token"] = st.session_state.get("hpi_run_token", 0) + 1
    # Review inputs: exact IPR rows (for the IPR panel), measured test oil (for
    # the sense check), and any response-curve points.
    st.session_state["hpi_ipr_rows"] = ipr_rows
    # Well-test points for the IPR scatter overlay (oil/water/BHP/date/... per test).
    try:
        _traw = fetch_well_tests_raw(months_back)
        st.session_state["hpi_test_df"] = (
            _traw[_traw["well"].isin(well_names)].copy()
            if _traw is not None and not _traw.empty else None
        )
    except Exception:
        st.session_state["hpi_test_df"] = None
    st.session_state["hpi_test_oil"] = {
        str(r["Well"]): r.get("Oil (BOPD)") for _, r in selected.iterrows()
    }
    if build_curves and curve_wells:
        st.session_state["hpi_curve"] = {"deltas": list(_SWEEP_DELTAS), "wells": curve_wells}
    else:
        st.session_state.pop("hpi_curve", None)
    _render_results(results_df, delta_p)
    _render_diagnostics()


# ── results display ───────────────────────────────────────────────────────


def _render_header_estimate(results_df: pd.DataFrame, delta_p: float) -> None:
    """PRIMARY header-impact estimate via Scott's deterministic chain, per well:
    ΔWHP → ① correlation (own / sonic→0 / group) → ΔBHP → ② liquid IPR (own / pad+
    formation donor) → ΔLiquid → ③ ×(1−WC) → ΔOil — then the field rollup with a
    confidence split. Sonic JPs (physics critical flow) contribute a confident 0.
    """
    test_df = st.session_state.get("hpi_test_df")
    emp_fits = st.session_state.get("hpi_emp_fits", {}) or {}
    fit_map = _pad_single_fits(results_df, test_df)
    input_df = st.session_state.get("hpi_input_df")

    # Water cut per well: form_wc → measured water fraction → pad average →
    # 0.0 (flagged). A silent 0.0 default counted the entire liquid delta as
    # oil for any well missing form_wc, inflating the field ΔOil rollup.
    wc_map: dict = {}
    if test_df is not None and {"well", "WtDate"}.issubset(getattr(test_df, "columns", [])):
        cols = set(test_df.columns)
        for w, g in test_df.groupby("well"):
            g = g.sort_values("WtDate")
            if "form_wc" in cols:
                s = pd.to_numeric(g["form_wc"], errors="coerce").dropna()
                if not s.empty:
                    wc_map[w] = float(s.iloc[-1])
                    continue
            if {"WtWaterVol", "WtTotalFluid"}.issubset(cols):
                wat = pd.to_numeric(g["WtWaterVol"], errors="coerce")
                tot = pd.to_numeric(g["WtTotalFluid"], errors="coerce")
                frac = (wat / tot).where(tot > 0).dropna()
                if not frac.empty:
                    wc_map[w] = float(min(max(frac.iloc[-1], 0.0), 1.0))

    pad_of = dict(
        zip(
            results_df["Well"],
            results_df["Pad"] if "Pad" in results_df.columns else [""] * len(results_df),
        )
    )
    _pad_wcs: dict = {}
    for w, v in wc_map.items():
        _pad_wcs.setdefault(pad_of.get(w), []).append(v)
    pad_avg_wc = {p: sum(v) / len(v) for p, v in _pad_wcs.items() if v}
    wc_defaulted: list[str] = []

    def _wc_for(w: str) -> float:
        if w in wc_map:
            return wc_map[w]
        pa = pad_avg_wc.get(pad_of.get(w))
        if pa is not None:
            return pa
        wc_defaulted.append(w)
        return 0.0

    # Manual SPECIFIC-WELL overrides from the coverage panel: only a real well token
    # counts (own / group tokens fall through to the auto own→group resolution).
    wellset = set(results_df["Well"])
    not_specific = {OWN_TOKEN, "", "nan", "None", *DONOR_GROUP_TOKENS}
    corr_tok = dict(zip(input_df["Well"], input_df["Corr donor"])) if (
        input_df is not None and "Corr donor" in input_df.columns) else {}
    ipr_tok = dict(zip(input_df["Well"], input_df["IPR donor"])) if (
        input_df is not None and "IPR donor" in input_df.columns) else {}

    def _specific(tok):
        t = str(tok)
        return t if (t not in not_specific and t in wellset) else None

    wells: dict = {}
    for _, r in results_df.iterrows():
        w = r["Well"]
        cf = (emp_fits.get(w) or {}).get("BHP~WHP")
        own_corr = None
        if cf is not None and ht.classify_response(cf) == "responsive" and pd.notna(cf.mean_slope):
            own_corr = float(cf.mean_slope)
        fit = (fit_map.get(w) or {}).get("fit") or {}
        wells[w] = {
            "own_corr": own_corr, "sonic": bool(r.get("Sonic now", False)),
            "lift": r.get("Lift", "?"), "qmax": fit.get("qmax"), "pr": fit.get("pr"),
            "pad": r.get("Pad"), "formation": r.get("Formation"),
            "bhp_now": r.get("BHP now (psi)"), "wc": _wc_for(w),
            "corr_donor": _specific(corr_tok.get(w)), "ipr_donor": _specific(ipr_tok.get(w)),
        }
    est = estimate_header_impacts(wells, delta_p)
    # Field confidence interval: vary the GROUP correlation by ±1σ (the dominant shared
    # uncertainty, since most wells borrow it). Own-correlation wells stay fixed.
    gstats = group_correlation_stats(wells)
    lo_ovr = {k: max(0.0, m - s) for k, (m, s) in gstats.items()}
    hi_ovr = {k: m + s for k, (m, s) in gstats.items()}
    est_lo = estimate_header_impacts(wells, delta_p, lo_ovr)
    est_hi = estimate_header_impacts(wells, delta_p, hi_ovr)
    field_lo = sum(e["doil"] for e in est_lo.values() if pd.notna(e["doil"]))
    field_hi = sum(e["doil"] for e in est_hi.values() if pd.notna(e["doil"]))
    df = pd.DataFrame([
        {
            "Well": w, "Pad": wells[w]["pad"], "Lift": wells[w]["lift"],
            "dBHP/dWHP": e["corr"], "corr src": e["corr_src"], "ΔBHP (psi)": e["dbhp"],
            "ΔLiquid (BPD)": e["dliquid"], "WC": wells[w]["wc"], "ΔOil (BOPD)": e["doil"],
            "IPR src": e["ipr_src"], "Confidence": e["conf"],
        }
        for w, e in est.items()
    ])
    if df.empty:
        return

    st.subheader("Header-Impact Estimate — correlation → IPR → water cut")
    if wc_defaulted:
        shown = ", ".join(sorted(set(wc_defaulted))[:8])
        more = "…" if len(set(wc_defaulted)) > 8 else ""
        st.caption(
            f"⚠️ No water cut found for {len(set(wc_defaulted))} well(s) "
            f"({shown}{more}) — their ΔOil is the full liquid delta (WC=0)."
        )
    valid = df["ΔOil (BOPD)"].notna()
    field = float(df.loc[valid, "ΔOil (BOPD)"].sum())
    by_conf = df.loc[valid].groupby("Confidence")["ΔOil (BOPD)"].agg(["sum", "count"])
    band_lo, band_hi = sorted([field_lo, field_hi])   # CI on the field total
    direction = "rise" if delta_p > 0 else "drop"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Field ΔOil · {int(delta_p):+d} psi header {direction}",
              f"{field:+,.0f} BOPD", f"CI [{band_lo:+,.0f}, {band_hi:+,.0f}]",
              delta_color="off",
              help="Sum over wells with a resolvable chain. CI = field total with the "
              "borrowed group correlation varied ±1σ (the dominant shared uncertainty).")
    n_sonic = int((df["Confidence"] == "sonic").sum())
    c2.metric("Sonic (no lever)", n_sonic,
              help="JP wells in physics critical flow — header can't reach BHP, ΔOil=0.")
    n_high = int((df["Confidence"] == "high").sum())
    c3.metric("Own corr + own IPR", f"{n_high}/{len(df)}",
              help="High-confidence wells (own data both steps); the rest borrow donors.")
    parts = " · ".join(
        f"**{idx}** {row['sum']:+,.0f} BOPD ({int(row['count'])}w)"
        for idx, row in by_conf.iterrows()
    )
    st.caption(
        f"Per-well **ΔWHP → ΔBHP** (correlation: own if responsive, **0 if sonic**, else a "
        f"like-lift group average — override a single well in the Coverage panel) **→ ΔLiquid** "
        f"(the well's liquid IPR or a pad+formation donor) **→ ΔOil** = ΔLiquid×(1−WC). Field "
        f"total **{field:+,.0f} BOPD**, 1σ range **[{band_lo:+,.0f}, {band_hi:+,.0f}]** (group "
        f"correlation ±σ). By confidence — {parts}. Wells with no resolvable correlation/IPR are dropped."
    )

    # ── Per-pad rollup (ΔOil + CI + confidence mix) ─────────────────────────
    norm = 100.0 / abs(delta_p) if delta_p else np.nan
    roll = []
    for pad in sorted({wells[w]["pad"] for w in est}, key=lambda p: str(p)):
        ws = [w for w in est if wells[w]["pad"] == pad]
        d = sum(est[w]["doil"] for w in ws if pd.notna(est[w]["doil"]))
        lo = sum(est_lo[w]["doil"] for w in ws if pd.notna(est_lo[w]["doil"]))
        hi = sum(est_hi[w]["doil"] for w in ws if pd.notna(est_hi[w]["doil"]))
        plo, phi = sorted([lo, hi])
        roll.append({
            "Pad": pad, "Wells": len(ws),
            "ΔOil (BOPD)": d, "CI low": plo, "CI high": phi,
            "BOPD/100psi": d * norm if pd.notna(norm) else np.nan,
            "Sonic": sum(1 for w in ws if est[w]["conf"] == "sonic"),
            "Own corr+IPR": sum(1 for w in ws if est[w]["conf"] == "high"),
        })
    roll.append({
        "Pad": "ALL", "Wells": len(est), "ΔOil (BOPD)": field,
        "CI low": band_lo, "CI high": band_hi,
        "BOPD/100psi": field * norm if pd.notna(norm) else np.nan,
        "Sonic": n_sonic, "Own corr+IPR": n_high,
    })
    st.markdown("**Per-pad rollup**")
    st.dataframe(
        pd.DataFrame(roll).style.format(
            {
                "ΔOil (BOPD)": "{:+,.0f}", "CI low": "{:+,.0f}", "CI high": "{:+,.0f}",
                "BOPD/100psi": "{:+,.1f}",
            }, na_rep="—",
        ),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Per-well chain detail", expanded=False):
        st.dataframe(
            df.style.format(
                {
                    "dBHP/dWHP": "{:.2f}", "ΔBHP (psi)": "{:+,.0f}", "ΔLiquid (BPD)": "{:+,.0f}",
                    "WC": "{:.2f}", "ΔOil (BOPD)": "{:+,.0f}",
                }, na_rep="—",
            ),
            use_container_width=True, hide_index=True,
        )


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

    # PRIMARY estimate: the deterministic correlation→IPR→WC chain per well + field total.
    _render_header_estimate(results_df, delta_p)

    # Coverage: which wells lack a real IPR / correlation, with inline donor
    # assignment (so you see the gap and fix it without scrolling back up).
    _render_coverage_panel(results_df)

    # Per-pad + overall sensitivity (response-class breakdown + BOPD/100psi) and,
    # if response curves were built, the oil-vs-header lever per pad / field.
    _render_field_sensitivity(results_df, delta_p)

    # Per-pad review — IPR curves, WHP→BHP correlations, sensitivity, drill-down.
    _render_pad_review(results_df, delta_p)

    # Full back-test result dump (JSON) so the whole result set is reviewable
    # outside the UI — the shared substrate for iterating on the back-test.
    _render_backtest_export(results_df, delta_p)

    # Well details — Verdict + provenance up front.
    st.subheader("Well Details")
    front = [c for c in ["Well", "Pad", "Lift", "WHP now (psi)", "WHP scen (psi)",
                          "Verdict", "Method used",
                          "Chosen ΔOil (BOPD)", "Pump", "IPR src", "IPR donor",
                          "Corr donor", "Fric src"]
             if c in results_df.columns]
    disp_df = results_df[front + [c for c in results_df.columns if c not in front]]
    fmt = {
        "BHP cal err": "{:+,.0f}",
        "WHP now (psi)": "{:,.0f}",
        "WHP scen (psi)": "{:,.0f}",
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

    # ── modeled vs reality (oil residual + optimism bias) ──────────────
    # The slope half of the sense check is the Physics-vs-Empirical scatter below.
    _render_sense_check(results_df)

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

    # Printouts & export — per-section CSVs, correlation summary, PDF report.
    _render_exports(results_df, delta_p)


# ── v2 review surface: sensitivity, per-pad expanders, IPR, sense check ──────


def _ipr_grid_fig(pad_df: pd.DataFrame, ipr_rows: dict, test_df: pd.DataFrame | None = None):
    """Grid of Vogel IPR curves for a pad's wells (JP and ESP/non-JP alike), with:
    the operating point (oil now @ BHP now, ●), the scenario point (oil scen @ BHP
    scen, ✕), and the well's actual well-test points overlaid (hover shows the full
    test detail) so the header lever and the data behind the IPR are both visible.
    Returns a Figure or None.
    """
    from plotly.subplots import make_subplots

    from woffl.flow.inflow import InFlow

    wells = [w for w in pad_df["Well"].tolist() if w in ipr_rows]
    if not wells:
        return None
    cols = int(np.ceil(np.sqrt(len(wells))))
    rows = int(np.ceil(len(wells) / cols))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=wells,
        vertical_spacing=0.13, horizontal_spacing=0.08,
        x_title="Oil (BOPD)", y_title="BHP (psi)",
    )
    by_well = pad_df.set_index("Well")
    for i, w in enumerate(wells):
        r, c = i // cols + 1, i % cols + 1
        ir = ipr_rows[w]
        res_pres = float(ir.get("res_pres", 1800.0))
        form_wc = float(ir.get("form_wc", 0.5))
        try:
            inflow = InFlow(
                qwf=float(ir["qwf"]) * (1.0 - form_wc), pwf=float(ir["pwf"]), pres=res_pres
            )
            bhp_grid = np.linspace(0.0, res_pres, 40)
            oil_grid = [float(inflow.oil_flow(float(b), "vogel")) for b in bhp_grid]
        except Exception:
            continue
        fig.add_trace(
            go.Scatter(
                x=oil_grid, y=bhp_grid, mode="lines",
                line=dict(color="#1f77b4", width=2), showlegend=False,
                hovertemplate="Oil %{x:.0f} BOPD<br>BHP %{y:.0f} psi<extra></extra>",
            ),
            row=r, col=c,
        )
        wr = by_well.loc[w] if w in by_well.index else None
        if wr is not None:
            o0, b0 = wr.get("Oil now (BOPD)"), wr.get("BHP now (psi)")
            o1, b1 = wr.get("Oil scen (BOPD)"), wr.get("BHP scen (psi)")
            if pd.notna(o0) and pd.notna(b0):
                fig.add_trace(
                    go.Scatter(
                        x=[o0], y=[b0], mode="markers",
                        marker=dict(color="black", size=9, symbol="circle"), showlegend=False,
                        hovertemplate="now: %{x:.0f} BOPD @ %{y:.0f} psi<extra></extra>",
                    ),
                    row=r, col=c,
                )
            if pd.notna(o1) and pd.notna(b1):
                fig.add_trace(
                    go.Scatter(
                        x=[o1], y=[b1], mode="markers",
                        marker=dict(color="crimson", size=10, symbol="x"), showlegend=False,
                        hovertemplate="scenario: %{x:.0f} BOPD @ %{y:.0f} psi<extra></extra>",
                    ),
                    row=r, col=c,
                )
        # Actual well-test points (oil vs BHP) with full test detail on hover.
        if test_df is not None and not getattr(test_df, "empty", True) and "well" in test_df.columns:
            wt = test_df[test_df["well"] == w]
            if "BHP" in wt.columns and "WtOilVol" in wt.columns:
                wt = wt.dropna(subset=["BHP", "WtOilVol"])
            else:
                wt = wt.iloc[0:0]
            if not wt.empty:
                hover = []
                for _, t in wt.iterrows():
                    parts = [str(w)]
                    if pd.notna(t.get("WtDate")):
                        parts.append(f"Date: {pd.Timestamp(t['WtDate']).date()}")
                    if pd.notna(t.get("WtOilVol")):
                        parts.append(f"Oil: {t['WtOilVol']:.0f} BOPD")
                    if pd.notna(t.get("WtWaterVol")):
                        parts.append(f"Water: {t['WtWaterVol']:.0f} BWPD")
                    if pd.notna(t.get("WtTotalFluid")):
                        parts.append(f"Fluid: {t['WtTotalFluid']:.0f} BPD")
                    if pd.notna(t.get("fgor")):
                        parts.append(f"GOR: {t['fgor']:.0f} scf/bbl")
                    if pd.notna(t.get("whp")):
                        parts.append(f"WHP: {t['whp']:.0f} psi")
                    if pd.notna(t.get("BHP")):
                        parts.append(f"BHP: {t['BHP']:.0f} psi")
                    hover.append("<br>".join(parts))
                fig.add_trace(
                    go.Scatter(
                        x=wt["WtOilVol"], y=wt["BHP"], mode="markers",
                        marker=dict(size=6, color="#777", symbol="circle-open",
                                    line=dict(width=1, color="#777")),
                        text=hover, hoverinfo="text", showlegend=False, name=f"{w} tests",
                    ),
                    row=r, col=c,
                )
    fig.update_layout(
        height=max(280, rows * 250), showlegend=False,
        title_text="Vogel IPR — ● now → ✕ scenario",
        margin=dict(l=60, r=20, t=60, b=55),
    )
    fig.update_annotations(font_size=10)
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    return fig


def _response_curve_fig(agg: dict, pads_to_show: list[str] | None = None, title: str = ""):
    """Line chart of ΔOil vs header change per pad (and the field total)."""
    if not agg or "deltas" not in agg:
        return None
    deltas = agg["deltas"]
    pads = agg.get("pads", {})
    keys = pads_to_show if pads_to_show is not None else sorted(pads.keys())
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    fig = go.Figure()
    for i, pad in enumerate(keys):
        if pad not in pads:
            continue
        fig.add_trace(
            go.Scatter(
                x=deltas, y=pads[pad], mode="lines+markers", name=f"Pad {pad}",
                line=dict(color=palette[i % len(palette)]),
                hovertemplate="Δ%{x:+d} psi<br>ΔOil %{y:+.0f} BOPD<extra></extra>",
            )
        )
    if pads_to_show is None and "ALL" in agg:
        fig.add_trace(
            go.Scatter(
                x=deltas, y=agg["ALL"], mode="lines+markers", name="Field (ALL)",
                line=dict(color="black", width=3, dash="dot"),
                hovertemplate="Δ%{x:+d} psi<br>ΔOil %{y:+.0f} BOPD<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title or "Oil response vs header change",
        xaxis_title="Header change Δ (psi)", yaxis_title="ΔOil vs current (BOPD)",
        height=420,
    )
    return fig


def _render_field_sensitivity(results_df: pd.DataFrame, delta_p: float) -> None:
    """Overall + per-pad sensitivity table, plus the field response curve if built."""
    st.subheader("Sensitivity")
    per_pad, overall = summarize_sensitivity(results_df, delta_p)
    if per_pad.empty:
        return
    table = pd.concat([per_pad, pd.DataFrame([overall])], ignore_index=True)
    st.dataframe(
        table.style.format(
            {
                "Oil now (BOPD)": "{:,.0f}", "ΔOil (BOPD)": "{:+,.0f}",
                "Oil after Δ (BOPD)": "{:,.0f}", "BOPD per 100 psi": "{:+,.1f}",
            },
            na_rep="—",
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"ΔOil at the {int(delta_p):+d} psi run, normalized to BOPD per 100 psi of header "
        "change, with each pad split by response class. The **ALL** row is the field total."
    )
    agg = aggregate_response_curve(st.session_state.get("hpi_curve"))
    if agg:
        pads_in = sorted(str(p) for p in results_df["Pad"].dropna().unique())
        pad_lbl = ", ".join(pads_in) if pads_in else "Selected"
        title = f"Pad{'s' if len(pads_in) != 1 else ''} {pad_lbl} — Response vs Header Change"
        fig = _response_curve_fig(agg, title=title)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key="hpi_field_curve")
            st.caption(
                "Each line is a pad's total oil change as the header moves; the dotted black "
                "line is the total across all selected pads. Flattening = wells choking sonic "
                "(the lever stops working)."
            )
    else:
        st.caption(
            "Tick **Build per-pad response curves** before running to see the oil-vs-header "
            "lever (a curve per pad) here and in each pad below."
        )


def _corr_review_fig(pad_df: pd.DataFrame, well_dfs: dict, fits: dict, driver: str = "WHP"):
    """Per-pad WHP→BHP correlation grid for the review — one panel for EVERY well
    used in the impact: its own trend, the donor's trend, or (for group-average
    correlations / no data) a labeled note in the panel title. Returns a Figure
    or None."""
    from plotly.subplots import make_subplots

    key = f"BHP~{driver}"
    wells = list(pad_df["Well"])
    if not wells:
        return None
    plan = corr_display_plan(pad_df, list((well_dfs or {}).keys()))
    n = len(wells)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    titles = []
    for w in wells:
        p = plan.get(w, {})
        source = p.get("source")
        note = p.get("note", "")
        f = (fits.get(source) or {}).get(key) if source else None
        if f is not None:
            titles.append(
                f"{w}  m={f.mean_slope:.2f}·{ht.classify_response(f)}"
                + (f"<br>{note}" if note else "")
            )
        else:
            titles.append(f"{w}<br>{note}" if note else f"{w} · no corr data")
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        vertical_spacing=0.10, horizontal_spacing=0.06,
        x_title=f"{driver} (psi)", y_title="BHP (psi)",
    )
    for i, w in enumerate(wells):
        r, c = i // cols + 1, i % cols + 1
        source = plan.get(w, {}).get("source")
        src_df = (well_dfs or {}).get(source) if source else None
        d = None
        if src_df is not None and driver in src_df.columns and "BHP" in src_df.columns:
            d = src_df[[driver, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
            d = d[d["BHP"] > 50]
        if d is None or d.empty:
            continue
        idx = pd.DatetimeIndex(d.index)
        ords = (idx.normalize() - idx.normalize().min()).days
        fig.add_trace(
            go.Scatter(
                x=d[driver], y=d["BHP"], mode="markers",
                marker=dict(size=4, color=ords, colorscale="Viridis", showscale=False),
                hovertemplate=f"{driver}: %{{x:.0f}}<br>BHP: %{{y:.0f}}<extra>{w}</extra>",
                showlegend=False,
            ),
            row=r, col=c,
        )
        f_fit = (fits.get(source) or {}).get(key)
        if f_fit is not None and getattr(f_fit, "daily", None) is not None \
                and not f_fit.daily.empty and "good" in f_fit.daily.columns:
            xseg: list = []
            yseg: list = []
            for _, dr in f_fit.daily[f_fit.daily["good"]].iterrows():
                x0, x1 = dr["x_min"], dr["x_max"]
                xseg += [x0, x1, None]
                yseg += [dr["slope"] * x0 + dr["intercept"], dr["slope"] * x1 + dr["intercept"], None]
            if xseg:
                fig.add_trace(
                    go.Scatter(x=xseg, y=yseg, mode="lines", line=dict(color="crimson", width=1),
                               opacity=0.6, hoverinfo="skip", showlegend=False),
                    row=r, col=c,
                )
        try:
            xlo, xhi = np.nanpercentile(d[driver], [1, 99])
            ylo, yhi = np.nanpercentile(d["BHP"], [1, 99])
            xpad = max((xhi - xlo) * 0.05, 1.0)
            ypad = max((yhi - ylo) * 0.05, 1.0)
            fig.update_xaxes(range=[xlo - xpad, xhi + xpad], row=r, col=c)
            fig.update_yaxes(range=[ylo - ypad, yhi + ypad], row=r, col=c)
        except Exception:
            pass
    fig.update_layout(
        height=max(360, rows * 280), showlegend=False,
        title_text=f"BHP vs {driver} — good daily fits in red",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_annotations(font_size=9)
    return fig


def _model_vs_obs_fig(pad_df, well_dfs: dict, curve: dict | None, sense_map: dict):
    """The sense-check picture: the MODEL's BHP-vs-WHP curve (blue) drawn over the
    ACTUAL historian (WHP, BHP) cloud (grey), with the operating point (● now) per
    well. If the cloud hugs the blue line, the model's header lever is trustworthy
    here; if the cloud is flatter, the field didn't move the way the model expects.

    One panel per well that has either a model curve or a historian trend. Returns
    a Plotly Figure or None.
    """
    from plotly.subplots import make_subplots

    cwells = (curve or {}).get("wells", {}) or {}
    wells = [
        w for w in pad_df["Well"]
        if w in cwells or w in (well_dfs or {})
    ]
    if not wells:
        return None
    n = len(wells)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    whp_now = dict(zip(pad_df["Well"], pad_df.get("WHP now (psi)", [])))
    bhp_now = dict(zip(pad_df["Well"], pad_df.get("BHP now (psi)", [])))
    titles = [f"{w} · {sense_map.get(w, '')}".rstrip(" ·") for w in wells]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        vertical_spacing=0.12, horizontal_spacing=0.07,
        x_title="WHP (psi)", y_title="BHP (psi)",
    )
    for i, w in enumerate(wells):
        r, c = i // cols + 1, i % cols + 1
        # actual historian cloud
        tdf = (well_dfs or {}).get(w)
        if tdf is not None and "WHP" in getattr(tdf, "columns", []) and "BHP" in tdf.columns:
            d = tdf[["WHP", "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
            d = d[d["BHP"] > 50]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=d["WHP"], y=d["BHP"], mode="markers",
                        marker=dict(size=3, color="lightslategray", opacity=0.45),
                        hovertemplate="WHP %{x:.0f}<br>BHP %{y:.0f}<extra>observed</extra>",
                        showlegend=False,
                    ),
                    row=r, col=c,
                )
        # model BHP-vs-WHP curve from the sweep
        cw = cwells.get(w, {})
        mw, mb = cw.get("whp"), cw.get("bhp")
        if mw and mb:
            pts = sorted(
                (float(x), float(y)) for x, y in zip(mw, mb)
                if x is not None and y is not None and not pd.isna(x) and not pd.isna(y)
            )
            if pts:
                xs, ys = zip(*pts)
                fig.add_trace(
                    go.Scatter(
                        x=list(xs), y=list(ys), mode="lines",
                        line=dict(color="royalblue", width=2),
                        hovertemplate="WHP %{x:.0f}<br>model BHP %{y:.0f}<extra>model</extra>",
                        showlegend=False,
                    ),
                    row=r, col=c,
                )
        # operating point (today)
        xn, yn = whp_now.get(w), bhp_now.get(w)
        if pd.notna(xn) and pd.notna(yn):
            fig.add_trace(
                go.Scatter(
                    x=[float(xn)], y=[float(yn)], mode="markers",
                    marker=dict(size=9, color="crimson", symbol="circle"),
                    hovertemplate="now: WHP %{x:.0f}<br>BHP %{y:.0f}<extra></extra>",
                    showlegend=False,
                ),
                row=r, col=c,
            )
    fig.update_layout(
        height=max(340, rows * 300), showlegend=False,
        title_text="Model BHP-vs-WHP (blue) over observed historian (grey) — ● = today",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_annotations(font_size=9)
    return fig


def _ipr_walk_fig(pad_df, ipr_rows: dict, test_df):
    """The IPR-walk: each well's modeled Vogel IPR (oil vs BHP) with its ACTUAL
    well-test operating points overlaid, COLORED BY WHP (back pressure) and joined
    in date order.

    Petroleum reading: back pressure slides the operating point along a fixed IPR —
    WHP↑ ⇒ BHP↑ ⇒ oil↓ ⇒ the marker climbs up-left. A clean WHP color gradient
    marching along the blue curve = the lever is real and the IPR shape is right.
    Older/low-WHP points sitting OFF (high-rate side of) today's IPR hint the curve
    has since shifted down — i.e. reservoir depletion, not back pressure. ● now ✕ scen.
    """
    from plotly.subplots import make_subplots

    from woffl.flow.inflow import InFlow

    wells = [w for w in pad_df["Well"].tolist() if w in ipr_rows]
    if not wells:
        return None
    cols = int(np.ceil(np.sqrt(len(wells))))
    rows = int(np.ceil(len(wells) / cols))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=wells,
        vertical_spacing=0.13, horizontal_spacing=0.08,
        x_title="Oil (BOPD)", y_title="BHP (psi)",
    )
    by_well = pad_df.set_index("Well")
    tdf = test_df if (test_df is not None and not getattr(test_df, "empty", True)) else None
    showed_scale = False
    for i, w in enumerate(wells):
        r, c = i // cols + 1, i % cols + 1
        ir = ipr_rows[w]
        res_pres = float(ir.get("res_pres", 1800.0))
        form_wc = float(ir.get("form_wc", 0.5))
        try:
            inflow = InFlow(
                qwf=float(ir["qwf"]) * (1.0 - form_wc), pwf=float(ir["pwf"]), pres=res_pres
            )
            bhp_grid = np.linspace(0.0, res_pres, 40)
            oil_grid = [float(inflow.oil_flow(float(b), "vogel")) for b in bhp_grid]
        except Exception:
            continue
        fig.add_trace(
            go.Scatter(
                x=oil_grid, y=bhp_grid, mode="lines",
                line=dict(color="#1f77b4", width=2), showlegend=False,
                hovertemplate="model IPR<br>Oil %{x:.0f}<br>BHP %{y:.0f}<extra></extra>",
            ),
            row=r, col=c,
        )
        # the actual walk — test operating points, colored by WHP, joined by date
        if tdf is not None and "well" in tdf.columns:
            wt = tdf[tdf["well"] == w]
            need = {"BHP", "WtOilVol", "whp", "WtDate"}
            wt = wt.dropna(subset=["BHP", "WtOilVol"]).sort_values("WtDate") \
                if need.issubset(wt.columns) else wt.iloc[0:0]
            if not wt.empty:
                hov = [
                    f"{w}<br>{pd.Timestamp(t.WtDate).date()}<br>Oil {t.WtOilVol:.0f} BOPD<br>"
                    f"BHP {t.BHP:.0f} psi<br>WHP {t.whp:.0f} psi" for t in wt.itertuples()
                ]
                fig.add_trace(
                    go.Scatter(
                        x=wt["WtOilVol"], y=wt["BHP"], mode="lines+markers",
                        line=dict(color="rgba(120,120,120,0.4)", width=1),
                        marker=dict(
                            size=7, color=wt["whp"], colorscale="Turbo",
                            showscale=not showed_scale,
                            colorbar=dict(title="WHP psi") if not showed_scale else None,
                            line=dict(width=0.5, color="white"),
                        ),
                        text=hov, hovertemplate="%{text}<extra></extra>", showlegend=False,
                    ),
                    row=r, col=c,
                )
                showed_scale = True
        wr = by_well.loc[w] if w in by_well.index else None
        if wr is not None:
            o0, b0 = wr.get("Oil now (BOPD)"), wr.get("BHP now (psi)")
            o1, b1 = wr.get("Oil scen (BOPD)"), wr.get("BHP scen (psi)")
            if pd.notna(o0) and pd.notna(b0):
                fig.add_trace(
                    go.Scatter(
                        x=[o0], y=[b0], mode="markers",
                        marker=dict(color="black", size=11, symbol="circle"), showlegend=False,
                        hovertemplate="now %{x:.0f} BOPD @ %{y:.0f} psi<extra></extra>",
                    ),
                    row=r, col=c,
                )
            if pd.notna(o1) and pd.notna(b1):
                fig.add_trace(
                    go.Scatter(
                        x=[o1], y=[b1], mode="markers",
                        marker=dict(color="crimson", size=12, symbol="x"), showlegend=False,
                        hovertemplate="scenario %{x:.0f} BOPD @ %{y:.0f} psi<extra></extra>",
                    ),
                    row=r, col=c,
                )
    fig.update_layout(
        height=max(360, rows * 320), showlegend=False,
        title_text="IPR walk — model IPR (blue) + actual tests colored by WHP; ● now · ✕ scenario",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_annotations(font_size=9)
    return fig


def _predict_doil(bhp_then, bhp_now, ir, fit, use_pseudo: bool, wc: float = 0.0):
    """Predicted oil change for the actual BHP move, off an IPR. ``use_pseudo`` → the
    single-fit blue IPR (on TOTAL LIQUID → ΔLiquid × (1−WC) = ΔOil); else the model
    IPR (already oil, qwf×(1−WC)). NaN if the inputs aren't there."""
    if pd.isna(bhp_then) or pd.isna(bhp_now):
        return np.nan
    if use_pseudo and fit:
        dliq = vogel_oil(bhp_now, fit["qmax"], fit["pr"]) - vogel_oil(bhp_then, fit["qmax"], fit["pr"])
        return dliq * (1.0 - float(wc))   # liquid IPR → oil via water cut
    if ir:
        try:
            from woffl.flow.inflow import InFlow
            fwc = float(ir.get("form_wc", 0.5))
            inflow = InFlow(qwf=float(ir["qwf"]) * (1.0 - fwc), pwf=float(ir["pwf"]),
                            pres=float(ir.get("res_pres", 1800.0)))
            return float(inflow.oil_flow(float(bhp_now), "vogel")
                         - inflow.oil_flow(float(bhp_then), "vogel"))
        except Exception:
            return np.nan
    return np.nan


def _backtest_table(
    pad_df, test_df, curve, pump_changed: set | None = None, fit_map: dict | None = None,
    ipr_rows: dict | None = None, use_pseudo: bool = False,
) -> pd.DataFrame:
    """Per-well long-horizon back-test for a pad: the REAL header move that happened
    over the test window vs what the model would have predicted, plus the total-liquid
    corroboration, an IPR-predicted ΔOil (off the database Pr or the single-fit
    pseudo-Pr), the single-fit pseudo-Pr, and the geometric depletion read (sign of
    the BHP↔oil correlation). Wells with <2 tests are skipped.
    """
    pump_changed = pump_changed or set()
    fit_map = fit_map or {}
    ipr_rows = ipr_rows or {}
    cwells = (curve or {}).get("wells", {}) or {}
    sonic = dict(zip(pad_df["Well"], pad_df.get("Sonic now", [])))
    has_well = test_df is not None and "well" in getattr(test_df, "columns", [])
    rows = []
    for w in pad_df["Well"]:
        tw = test_df[test_df["well"] == w] if has_well else None
        a = backtest_anchors(tw) if tw is not None else {}
        if not a:
            continue
        dbhp_pred, extr = predict_dbhp_from_curve(cwells.get(w), a["whp_then"], a["whp_now"])
        diagnosis = backpressure_consistency(a["d_whp"], a["d_bhp"], a["d_liquid"])
        fm = fit_map.get(w) or {}
        fit, depl = fm.get("fit"), fm.get("depl") or {}
        wc = 0.0
        if tw is not None and "form_wc" in getattr(tw, "columns", []):
            s = pd.to_numeric(tw.sort_values("WtDate")["form_wc"], errors="coerce").dropna()
            if not s.empty:
                wc = float(s.iloc[-1])
        doil_pred = _predict_doil(a["bhp_then"], a["bhp_now"], ipr_rows.get(w), fit, use_pseudo, wc)
        # Single-fit pseudo-Pr (* = pinned at the formation cap → soft) + the
        # geometric depletion verdict (corr sign), which is the robust read.
        pr_fit = (f"{fit['pr']:.0f}{'*' if fit.get('pr_at_bound') else ''}") if fit else "—"
        corr = depl.get("corr", np.nan)
        flags = []
        if w in pump_changed:
            flags.append("⚠ pump chg")
        if bool(sonic.get(w, False)):
            flags.append("⚠ sonic")
        if extr:
            flags.append("extrap")
        rows.append(
            {
                "Well": w, "n tests": a["n_tests"],
                "ΔWHP (psi)": a["d_whp"], "ΔBHP actual (psi)": a["d_bhp"],
                "ΔBHP pred (psi)": dbhp_pred, "ΔOil actual (BOPD)": a["d_oil"],
                "ΔOil pred (BOPD)": doil_pred, "ΔLiquid (BPD)": a["d_liquid"],
                "Pr fit": pr_fit, "BHP↔liq corr": corr, "Depletion": depl.get("verdict", "—"),
                "Diagnosis": diagnosis, "Flags": " ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def _backtest_parity_fig(bt: pd.DataFrame, use_pseudo: bool):
    """Back-test parity plot: predicted vs actual for ΔBHP and ΔOil, one point per
    well, against the y=x line. On the line = the model reproduced what happened;
    above/below = it over/under-predicted. The headline 'did the back-test pass?'
    picture. Returns a Figure or None."""
    from plotly.subplots import make_subplots

    if bt is None or bt.empty:
        return None
    panels = [
        ("ΔBHP", "ΔBHP actual (psi)", "ΔBHP pred (psi)"),
        ("ΔOil", "ΔOil actual (BOPD)", "ΔOil pred (BOPD)"),
    ]
    avail = [(t, xa, ya) for t, xa, ya in panels if xa in bt.columns and ya in bt.columns]
    if not avail:
        return None
    fig = make_subplots(rows=1, cols=len(avail), subplot_titles=[t for t, _, _ in avail])
    for i, (t, xa, ya) in enumerate(avail, start=1):
        x = pd.to_numeric(bt[xa], errors="coerce")
        y = pd.to_numeric(bt[ya], errors="coerce")
        m = x.notna() & y.notna()
        if not m.any():
            continue
        lo = float(min(x[m].min(), y[m].min()))
        hi = float(max(x[m].max(), y[m].max()))
        pad = max((hi - lo) * 0.1, 5.0)
        lo, hi = lo - pad, hi + pad
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color="lightgray", dash="dash", width=1),
                       hoverinfo="skip", showlegend=False),
            row=1, col=i,
        )
        unit = "psi" if t == "ΔBHP" else "BOPD"
        fig.add_trace(
            go.Scatter(
                x=x[m], y=y[m], mode="markers+text", text=bt["Well"][m],
                textposition="top center", textfont=dict(size=8),
                marker=dict(size=9, color="#1f77b4"),
                hovertemplate=f"%{{text}}<br>actual %{{x:.0f}} {unit}<br>pred %{{y:.0f}} {unit}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text=f"actual ({unit})", row=1, col=i)
        fig.update_yaxes(title_text=f"predicted ({unit})", row=1, col=i)
    src = "fitted pseudo-Pr" if use_pseudo else "database/assumed Pr"
    fig.update_layout(
        height=380, showlegend=False, margin=dict(l=60, r=20, t=60, b=50),
        title_text=f"Back-test parity — predicted vs actual (ΔOil IPR: {src}); on the dashed line = model matched reality",
    )
    fig.update_annotations(font_size=11)
    return fig


def _pump_changed_in_window(results_df: pd.DataFrame) -> set:
    """Wells whose CURRENT jet pump was installed inside the test/trend lookback —
    a bigger pump shifts BHP independent of the header, so the back-test row is
    suspect. Uses jp_history install dates vs the months-back window."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return set()
    months_back = int(st.session_state.get("hpi_months", 6))
    window_start = pd.Timestamp(datetime.now() - relativedelta(months=months_back))
    changed = set()
    for w in results_df["Well"]:
        try:
            p = get_current_pump(jp_hist, w)
        except Exception:
            p = None
        ds = p.get("date_set") if p else None
        if ds is not None and pd.notna(ds) and pd.Timestamp(ds) > window_start:
            changed.add(w)
    return changed


def _json_safe(obj):
    """Recursively coerce numpy / Timestamp / NaN into JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return None if pd.isna(obj) else float(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        return None if pd.isna(obj) else obj
    except (TypeError, ValueError):
        return str(obj)


def _build_backtest_export(results_df, delta_p) -> dict:
    """Assemble the FULL back-test (all pads) + windowed pseudo-Pr fits + config into
    one JSON-safe dict — the whole result set, reviewable outside the UI so we can
    iterate on it together."""
    from datetime import datetime

    test_df = st.session_state.get("hpi_test_df")
    curve = st.session_state.get("hpi_curve")
    ipr_rows = st.session_state.get("hpi_ipr_rows", {}) or {}
    pump_changed = _pump_changed_in_window(results_df)
    use_pseudo = bool(st.session_state.get("hpi_bt_use_pseudo_pr", False))
    solve_pseudo = bool(st.session_state.get("hpi_solve_use_pseudo_pr", False))
    pads = sorted(str(p) for p in results_df["Pad"].dropna().unique())
    rows, fits = [], {}
    for pad in pads:
        pad_df = results_df[results_df["Pad"] == pad]
        fm = _pad_single_fits(pad_df, test_df)
        bt = _backtest_table(pad_df, test_df, curve, pump_changed, fm, ipr_rows, use_pseudo)
        for rec in bt.to_dict("records"):
            rec["Pad"] = pad
            rows.append(rec)
        fits.update(fm)
    return _json_safe(
        {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "delta_p": int(delta_p),
                "months_back": int(st.session_state.get("hpi_months", 6)),
                "fric_mode": st.session_state.get("hpi_fric_mode"),
                "backtest_pred_uses_pseudo_pr": use_pseudo,
                "solve_uses_pseudo_pr": solve_pseudo,
                "pads": pads,
                "formation_pr_cap": dict(FORMATION_PR_MAX),
            },
            "n_wells_backtested": len(rows),
            "backtest_rows": rows,
            "single_fits": fits,
        }
    )


def _render_backtest_export(results_df, delta_p) -> None:
    """Write the full back-test result set to a JSON file (for offline review +
    iterating together) and offer it as a download. Best-effort — never breaks the
    UI if the filesystem is read-only (e.g. on Databricks)."""
    import json
    from pathlib import Path

    try:
        payload = _build_backtest_export(results_df, delta_p)
    except Exception as e:
        st.caption(f"Back-test dump unavailable ({type(e).__name__}: {e}).")
        return
    blob = json.dumps(payload, indent=2)
    wrote = None
    try:
        out = Path(__file__).resolve().parents[3] / "temp" / "hpi_backtest_latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")
        wrote = str(out)
    except Exception:
        wrote = None
    with st.expander("🧪 Back-test result dump (for review / iteration)", expanded=False):
        st.caption(
            f"Captured **{payload['n_wells_backtested']}** back-tested well(s)"
            + (f" → wrote `{wrote}`" if wrote else " (file write unavailable here — use Download)")
            + ". This JSON has every well's ΔWHP/ΔBHP/ΔOil actual-vs-pred, the windowed "
            "pseudo-Pr fits, and the run config — the shared substrate for tuning the back-test."
        )
        st.download_button(
            "Download back-test JSON", data=blob.encode("utf-8"),
            file_name="hpi_backtest_latest.json", mime="application/json",
            key="hpi_bt_dump_dl", use_container_width=True,
        )


def _pad_single_fits(pad_df, test_df) -> dict:
    """Per-well SINGLE Vogel fit (all points) + geometric depletion read for a pad,
    computed once and reused by the back-test table, the IPR figure, and the export.
    Returns ``{well: {"fit": <fit|None>, "depl": <signature>}}``. The fit's pr cap is
    the well's formation ceiling (Schrader 2200 / Kuparuk 4200)."""
    if test_df is None or "well" not in getattr(test_df, "columns", []):
        return {}
    form = dict(zip(pad_df["Well"], pad_df.get("Formation", [])))
    out = {}
    for w in pad_df["Well"]:
        tw = test_df[test_df["well"] == w]
        if tw.empty:
            continue
        fit = fit_well_ipr(tw, pr_hi=pr_hi_for_formation(form.get(w)))
        depl = depletion_signature(tw)
        if fit or depl.get("verdict") != "insufficient":
            out[w] = {"fit": fit, "depl": depl}
    return out


def _vogel_curve(qmax: float, pr: float, npts: int = 40):
    """(oil, pwf) arrays for a Vogel IPR with the given qmax & pr — drawn directly
    from the fitted params (no InFlow needed)."""
    pwf = np.linspace(0.0, pr, npts)
    r = pwf / pr
    return qmax * (1.0 - 0.2 * r - 0.8 * r * r), pwf


def _ipr_fit_fig(pad_df, test_df, fit_map: dict):
    """Per well: ONE Vogel IPR fitted to all the well's tests (blue), on TOTAL LIQUID,
    with the actual test points colored by TIME (early→late), and ● now ✕ scen.

    Read the depletion the geometric way (Scott): on a fixed IPR liquid & BHP are
    inverse (the points hug the blue curve). If the cloud instead forms a flat line
    with the late (warm) points down-and-left — BHP & liquid falling together — the IPR
    is depleting. The verdict (from the BHP↔liquid correlation) is in each panel title.
    Operating-point markers are oil/(1−WC) → liquid so they sit on the liquid axis.
    """
    from plotly.subplots import make_subplots

    wells = [w for w in pad_df["Well"].tolist() if fit_map.get(w)]
    if not wells:
        return None
    cols = int(np.ceil(np.sqrt(len(wells))))
    rows = int(np.ceil(len(wells) / cols))
    titles = []
    for w in wells:
        fm = fit_map[w]
        fit, depl = fm.get("fit"), fm.get("depl") or {}
        pr_txt = f"Pr {fit['pr']:.0f}{'*' if fit.get('pr_at_bound') else ''}" if fit else "no fit"
        titles.append(f"{w} · {pr_txt} · {depl.get('verdict', '—')}")
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        vertical_spacing=0.14, horizontal_spacing=0.08,
        x_title="Total liquid (BPD)", y_title="BHP (psi)",
    )
    by_well = pad_df.set_index("Well")
    has_well = test_df is not None and "well" in getattr(test_df, "columns", [])
    showed_scale = False
    for i, w in enumerate(wells):
        r, c = i // cols + 1, i % cols + 1
        fit = fit_map[w].get("fit")
        if fit:
            liq, pwf = _vogel_curve(fit["qmax"], fit["pr"])
            fig.add_trace(
                go.Scatter(x=liq, y=pwf, mode="lines", line=dict(color="#1f77b4", width=2),
                           showlegend=False,
                           hovertemplate=f"fit IPR · Pr {fit['pr']:.0f}<br>Liquid %{{x:.0f}}<br>BHP %{{y:.0f}}<extra></extra>"),
                row=r, col=c,
            )
        wt = test_df[test_df["well"] == w] if has_well else None
        wc_w = 0.0
        if wt is not None and {"BHP", "WtTotalFluid", "WtDate"}.issubset(getattr(wt, "columns", [])):
            wt = wt.dropna(subset=["BHP", "WtTotalFluid", "WtDate"]).sort_values("WtDate")
            if "form_wc" in wt.columns:
                wcs = pd.to_numeric(wt["form_wc"], errors="coerce").dropna()
                wc_w = float(wcs.iloc[-1]) if not wcs.empty else 0.0
            if not wt.empty:
                base = pd.DatetimeIndex(wt["WtDate"]).normalize().min()
                ords = (pd.DatetimeIndex(wt["WtDate"]).normalize() - base).days
                fig.add_trace(
                    go.Scatter(
                        x=wt["WtTotalFluid"], y=wt["BHP"], mode="markers",
                        marker=dict(size=6, color=ords, colorscale="Turbo",
                                    showscale=not showed_scale,
                                    colorbar=dict(title="days") if not showed_scale else None,
                                    line=dict(width=0.4, color="white")),
                        showlegend=False,
                        hovertemplate="Liquid %{x:.0f}<br>BHP %{y:.0f}<extra></extra>",
                    ),
                    row=r, col=c,
                )
                showed_scale = True
        wr = by_well.loc[w] if w in by_well.index else None
        if wr is not None:
            denom = max(1.0 - wc_w, 0.05)   # oil → liquid for the operating-point markers
            o0, b0 = wr.get("Oil now (BOPD)"), wr.get("BHP now (psi)")
            o1, b1 = wr.get("Oil scen (BOPD)"), wr.get("BHP scen (psi)")
            if pd.notna(o0) and pd.notna(b0):
                fig.add_trace(go.Scatter(x=[o0 / denom], y=[b0], mode="markers",
                    marker=dict(color="black", size=11, symbol="circle"), showlegend=False,
                    hovertemplate="now %{x:.0f} BLPD @ %{y:.0f} psi<extra></extra>"), row=r, col=c)
            if pd.notna(o1) and pd.notna(b1):
                fig.add_trace(go.Scatter(x=[o1 / denom], y=[b1], mode="markers",
                    marker=dict(color="crimson", size=12, symbol="x"), showlegend=False,
                    hovertemplate="scenario %{x:.0f} BLPD @ %{y:.0f} psi<extra></extra>"), row=r, col=c)
    fig.update_layout(
        height=max(360, rows * 320), showlegend=False,
        title_text="IPR — one Vogel fit per well on TOTAL LIQUID (blue) + tests colored by time · ● now ✕ scenario",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig.update_annotations(font_size=9)
    return fig


def _render_pad_review(results_df: pd.DataFrame, delta_p: float) -> None:
    """Per-pad expanders: where the pad sits today, the up/down oil lever, the
    long-horizon back-test (a real header move vs the model), the windowed-IPR
    depletion picture (Standing), then correlations, the response curve, and a
    drill-down. The intraday BHP↔WHP coupling check sits in a collapsed sub-expander."""
    st.subheader("Per-Pad Review — Where We Sit, Sensitivity & Back-Test")
    ipr_rows = st.session_state.get("hpi_ipr_rows", {}) or {}
    test_df = st.session_state.get("hpi_test_df")
    well_dfs = st.session_state.get("hpi_emp_well_dfs", {}) or {}
    fits = st.session_state.get("hpi_emp_fits", {}) or {}
    curve = st.session_state.get("hpi_curve")
    per_pad, _ = summarize_sensitivity(results_df, delta_p)
    bias = bias_by_pad(results_df)
    agg = aggregate_response_curve(curve)
    sense_all = sense_check_response(results_df)
    pump_changed = _pump_changed_in_window(results_df)
    use_pseudo = st.checkbox(
        "Back-test ΔOil with fitted pseudo-Pr (else database / assumed reservoir pressure)",
        value=st.session_state.get("hpi_bt_use_pseudo_pr", False),
        key="hpi_bt_use_pseudo_pr",
        help="Which IPR predicts ΔOil from the observed ΔBHP in the back-test. ON = the "
        "Vogel IPR fitted to each well's recent tests (data-driven pseudo reservoir "
        "pressure, Schrader capped ~2200 psi); OFF = the model's database/assumed Pr. "
        "Toggling re-renders the back-test only — no full re-run.",
    )
    pads = sorted(results_df["Pad"].unique())
    for pi, pad in enumerate(pads):
        pad_df = results_df[results_df["Pad"] == pad].copy()
        prow = per_pad[per_pad["Pad"] == pad]
        doil = float(prow["ΔOil (BOPD)"].iloc[0]) if not prow.empty else float("nan")
        bopd100 = float(prow["BOPD per 100 psi"].iloc[0]) if not prow.empty else float("nan")
        title = f"Pad {pad}  ·  ΔOil {doil:+,.0f} BOPD  ·  {bopd100:+,.1f}/100psi  ·  {len(pad_df)} wells"
        with st.expander(title, expanded=(pi == 0)):
            # ── Where we sit today ──────────────────────────────────────────
            whp_avg = pd.to_numeric(pad_df.get("WHP now (psi)"), errors="coerce").mean()
            bhp_avg = pd.to_numeric(pad_df.get("BHP now (psi)"), errors="coerce").mean()
            oil_now = pd.to_numeric(pad_df.get("Oil now (BOPD)"), errors="coerce").sum()
            down, up = pad_updown_lever(curve, pad, ref=100)
            m1, m2, m3 = st.columns(3)
            m1.metric("WHP now (avg)", f"{whp_avg:,.0f} psi" if pd.notna(whp_avg) else "—")
            m2.metric("BHP now (avg)", f"{bhp_avg:,.0f} psi" if pd.notna(bhp_avg) else "—")
            m3.metric("Oil now", f"{oil_now:,.0f} BOPD" if pd.notna(oil_now) else "—")
            # ── Up/down lever from today ────────────────────────────────────
            d1, d2 = st.columns(2)
            d1.metric(
                "↓ Header −100 psi", f"{down:+,.0f} BOPD" if pd.notna(down) else "—",
                help="Oil change for a 100-psi header DROP from today (summed over "
                "the pad). Flattens if wells choke sonic on the way down.",
            )
            d2.metric(
                "↑ Header +100 psi", f"{up:+,.0f} BOPD" if pd.notna(up) else "—",
                help="Oil change for a 100-psi header RISE from today. Asymmetry vs "
                "the drop = the lever is nonlinear at this operating level.",
            )
            if pd.isna(down) and pd.isna(up):
                st.caption(
                    "Up/down lever needs the **±100 psi** sweep points — tick *Build "
                    "per-pad response curves* before the run."
                )
            if not bias.empty:
                brow = bias[bias["Pad"] == pad]
                if not brow.empty:
                    st.caption(
                        f"Modeled-vs-field bias (responsive wells): "
                        f"**{brow['Emp/Phys bias'].iloc[0]:.2f}** — {brow['Flag'].iloc[0]}"
                    )
            # ── Long-horizon back-test: the real header move vs the model ────
            fit_map = _pad_single_fits(pad_df, test_df)
            bt = _backtest_table(pad_df, test_df, curve, pump_changed, fit_map,
                                 ipr_rows, use_pseudo)
            if not bt.empty:
                src = "fitted pseudo-Pr" if use_pseudo else "database/assumed Pr"
                st.markdown(
                    "**Back-test — did the real header move play out?** Over the test "
                    "window the header actually moved; here's the model's predicted ΔBHP "
                    f"and ΔOil (IPR: **{src}**) vs what actually happened, the single-fit "
                    "pseudo-Pr, and the geometric depletion read."
                )
                st.dataframe(
                    bt.style.format(
                        {
                            "ΔWHP (psi)": "{:+,.0f}", "ΔBHP actual (psi)": "{:+,.0f}",
                            "ΔBHP pred (psi)": "{:+,.0f}", "ΔOil actual (BOPD)": "{:+,.0f}",
                            "ΔOil pred (BOPD)": "{:+,.0f}", "ΔLiquid (BPD)": "{:+,.0f}",
                            "BHP↔liq corr": "{:+.2f}",
                        },
                        na_rep="—",
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "**ΔBHP / ΔOil actual vs pred** = did the well move as the model expects "
                    f"for the real ΔWHP (ΔOil predicted off the **{src}** IPR — toggle above). "
                    "**Pr fit** = single Vogel fit to all the well's tests (`*` = pinned at the "
                    "formation cap → soft; trust the shape, not the value). **Depletion** (the "
                    "robust read) = sign of **BHP↔liquid corr**: *back-pressure* (corr<0, riding "
                    "the IPR) · *depleting* (corr>0, BHP & liquid falling together) · *flat/mixed*. "
                    "**Diagnosis** is the ΔWHP/ΔBHP/Δliquid sign check. ⚠ pump-chg / sonic = confounded."
                )
                par = _backtest_parity_fig(bt, use_pseudo)
                if par is not None:
                    st.plotly_chart(par, use_container_width=True, key=f"hpi_parity_{pad}")
            dep = _ipr_fit_fig(pad_df, test_df, fit_map)
            if dep is not None:
                st.markdown("**IPR fit + depletion read** — does the cloud ride the curve (back pressure), or form a flat both-falling line (depletion)?")
                st.plotly_chart(dep, use_container_width=True, key=f"hpi_iprfit_{pad}")
                st.caption(
                    "Blue = ONE Vogel IPR fit to all the well's tests, on **total liquid**; dots = "
                    "those tests colored by time (early→late); ● now · ✕ scenario. Points hugging "
                    "the blue curve (liquid↑ when BHP↓) = **back pressure** on a fixed IPR. Late "
                    "(warm) points drifting **down-and-left** in a flat line (BHP & liquid falling "
                    "together) = **depletion**. `Pr*` in a title = fit pinned at the formation cap (soft)."
                )
            # ── Intraday BHP↔WHP coupling (advanced, collapsed) ─────────────
            sense_pad = sense_all[sense_all["Pad"] == pad] if not sense_all.empty else pd.DataFrame()
            if not sense_pad.empty:
                with st.expander("Intraday BHP↔WHP coupling (does it respond at all?)", expanded=False):
                    st.dataframe(
                        sense_pad[[
                            "Well", "WHP now (psi)", "BHP now (psi)", "Model dBHP/dWHP",
                            "Obs dBHP/dWHP", "Obs/Model", "Sense-check", "Note",
                        ]].style.format(
                            {
                                "WHP now (psi)": "{:,.0f}", "BHP now (psi)": "{:,.0f}",
                                "Model dBHP/dWHP": "{:.2f}", "Obs dBHP/dWHP": "{:.2f}",
                                "Obs/Model": "{:.2f}",
                            },
                            na_rep="—",
                        ),
                        use_container_width=True, hide_index=True,
                    )
                    sense_map = dict(zip(sense_pad["Well"], sense_pad["Sense-check"]))
                    mvo = _model_vs_obs_fig(pad_df, well_dfs, curve, sense_map)
                    if mvo is not None:
                        st.plotly_chart(mvo, use_container_width=True, key=f"hpi_mvo_{pad}")
                        st.caption(
                            "Intraday: blue = model BHP across WHP; grey = historian; ● today. "
                            "Hourly co-movement — tests whether BHP responds to WHP *at all*, "
                            "complementing the long-horizon back-test above."
                        )
            wcols = [
                c for c in ["Well", "Lift", "Verdict", "Method used", "Chosen ΔOil (BOPD)",
                            "BHP now (psi)", "BHP scen (psi)", "Oil now (BOPD)", "Oil scen (BOPD)",
                            "Emp class"]
                if c in pad_df.columns
            ]
            st.dataframe(
                pad_df[wcols].style.format(
                    {
                        "Chosen ΔOil (BOPD)": "{:+,.0f}", "BHP now (psi)": "{:,.0f}",
                        "BHP scen (psi)": "{:,.0f}", "Oil now (BOPD)": "{:,.0f}",
                        "Oil scen (BOPD)": "{:,.0f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True, hide_index=True,
            )
            if agg and pad in agg.get("pads", {}):
                f = _response_curve_fig(agg, pads_to_show=[pad], title=f"Pad {pad} — oil vs header change")
                if f is not None:
                    st.plotly_chart(f, use_container_width=True, key=f"hpi_padcurve_{pad}")
            pad_wells = pad_df["Well"].tolist()
            sub_dfs = {w: well_dfs[w] for w in pad_wells if w in well_dfs}
            sub_fits = {w: fits.get(w, {}) for w in pad_wells if w in well_dfs}
            gfig = _corr_review_fig(pad_df, well_dfs, fits, "WHP")
            if gfig is not None:
                st.markdown(
                    "**WHP→BHP correlations** — every well used; donor/group correlations "
                    "noted in the panel title (good daily fits in red)"
                )
                st.plotly_chart(gfig, use_container_width=True, key=f"hpi_corr_{pad}")
            if sub_dfs:
                drill = st.selectbox(
                    f"Drill-down well (Pad {pad})",
                    options=["—"] + sorted(sub_dfs.keys()), key=f"hpi_drill_{pad}",
                )
                if drill and drill != "—":
                    wf = _plot_well_fits(drill, sub_dfs[drill], sub_fits.get(drill, {}))
                    if wf is not None:
                        st.plotly_chart(wf, use_container_width=True, key=f"hpi_drillfig_{pad}")


def _render_sense_check(results_df: pd.DataFrame) -> None:
    """Modeled-vs-reality: per-well Phys/Emp/Chosen ΔOil + modeled vs measured
    Oil-now, plus the per-pad optimism bias."""
    st.subheader("Modeled vs Reality")
    test_oil = st.session_state.get("hpi_test_oil", {}) or {}
    tbl = sense_check_table(results_df, test_oil)
    if tbl.empty:
        return
    st.dataframe(
        tbl.style.format(
            {
                "Phys ΔOil (BOPD)": "{:+,.0f}", "Emp ΔOil (BOPD)": "{:+,.0f}",
                "Chosen ΔOil (BOPD)": "{:+,.0f}", "Oil now modeled (BOPD)": "{:,.0f}",
                "Oil now measured (BOPD)": "{:,.0f}", "Oil residual (BOPD)": "{:+,.0f}",
                "Oil residual %": "{:+,.0f}%",
            },
            na_rep="—",
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Oil residual** = modeled Oil-now − the well's measured test oil. Large residuals mean "
        "the IPR/friction baseline is off for that well (calibrate, or assign a better IPR in M3). "
        "Phys vs Emp ΔOil shows where the field disagrees with the model."
    )
    bias = bias_by_pad(results_df)
    if not bias.empty:
        st.markdown("**Per-pad optimism bias** — Emp ÷ Phys ΔOil over responsive wells")
        st.dataframe(
            bias.style.format({"Emp/Phys bias": "{:.2f}"}, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "A soft flag, not a gate: <0.8 the physics looks optimistic vs the field; "
            ">1.2 conservative."
        )


def _autodownload(data: bytes, filename: str, dom_id: str, mime: str = "application/pdf") -> None:
    """Trigger an immediate browser download of ``data`` — single-click, no second
    button. ALL Generate-PDF buttons use this so generating = downloading."""
    import base64

    import streamlit.components.v1 as components

    b64 = base64.b64encode(data).decode()
    components.html(
        f'<a id="{dom_id}" download="{filename}" href="data:{mime};base64,{b64}"></a>'
        f'<script>document.getElementById("{dom_id}").click()</script>',
        height=0,
    )


def _render_coverage_panel(results_df: pd.DataFrame) -> None:
    """Post-run coverage of IPR + WHP→BHP correlation per well, with inline donor
    assignment. This is where you SEE which wells lack a real IPR / correlation
    (e.g. a well with no within-day fit shows ⚠ for Corr); assign a like-well or
    group-average donor in the last two columns and press Run again to apply.
    """
    input_df = st.session_state.get("hpi_input_df")
    if input_df is None or "Well" not in results_df.columns:
        return

    cur_ipr = dict(zip(input_df.get("Well", []), input_df.get("IPR donor", [])))
    cur_corr = dict(zip(input_df.get("Well", []), input_df.get("Corr donor", [])))

    def _ipr_status(r) -> str:
        donor = str(r.get("IPR donor", "own"))
        if donor not in ("own", "—", "nan", "None"):
            return f"✎ {donor}"
        src = str(r.get("IPR src", ""))
        if src == "jp_chars":
            return "⚠ fallback (no test data)"
        if src in ("gauge", "back-calc"):
            return f"✓ {src}"
        if str(r.get("Lift", "")) != "JP":
            return "✓ generic (ESP test)"
        return src or "—"

    def _corr_status(r) -> str:
        donor = str(r.get("Corr donor", "own"))
        if donor not in ("own", "—", "nan", "None"):
            return f"✎ {donor}"
        cls = str(r.get("Emp class", ""))
        if cls == "responsive":
            return "✓ responsive"
        if cls in ("no data", "no tag", "insufficient", "slugging", ""):
            return f"⚠ {cls or 'no fit'}"
        return cls

    def _ipr_missing(r) -> bool:
        """Well has no IPR test data of its own — the solve fell back to jp_chars
        defaults. Donor-independent (stays true even once a group IPR is borrowed),
        so it drives the persistent 🟥 flag, the count, and the auto-default."""
        return str(r.get("IPR src", "")) == "jp_chars"

    def _corr_missing(r) -> bool:
        """No usable WHP→BHP coupling of the well's own — anything but a clean
        'responsive' fit (no-data / no-tag / insufficient / slugging)."""
        return str(r.get("Emp class", "")) not in ("responsive",)

    # Build the coverage table ONCE per run (keyed to the run token) and reuse it
    # across reruns — feeding the data_editor a STABLE frame avoids the flicker /
    # "refresh on every change" from rebuilding + writing back every render.
    token = st.session_state.get("hpi_run_token", 0)
    cov = st.session_state.get("hpi_cov_df")
    if cov is None or st.session_state.get("hpi_cov_token") != token \
            or list(cov.get("Well", [])) != list(results_df["Well"]):
        # Auto-default a group donor for wells lacking their own data — but only
        # the FIRST time a well appears in coverage (tracked in hpi_cov_seen), so
        # an explicit user clear-to-(own) on a later re-run isn't fought:
        #   no IPR test data → pad+formation analogs
        #   no usable corr   → all like-lift wells (the widest analog pool)
        # The chosen donor is persisted into hpi_input_df below, so the next
        # Re-run actually applies it.
        seen = st.session_state.setdefault("hpi_cov_seen", set())
        rows = []
        for _, r in results_df.iterrows():
            wn = r["Well"]
            ipr_miss, corr_miss = _ipr_missing(r), _corr_missing(r)
            ipr_donor = cur_ipr.get(wn, OWN_TOKEN)
            corr_donor = cur_corr.get(wn, OWN_TOKEN)
            first_time = wn not in seen
            if first_time and ipr_miss and str(ipr_donor) in (OWN_TOKEN, "", "nan", "None"):
                ipr_donor = GROUP_PADFORMATION
            if first_time and corr_miss and str(corr_donor) in (OWN_TOKEN, "", "nan", "None"):
                corr_donor = GROUP_LIFT
            rows.append(
                {
                    "Flag": ("🟥" if ipr_miss else "") + ("🟧" if corr_miss else ""),
                    "Well": wn, "Pad": r.get("Pad"), "Lift": r.get("Lift"),
                    "IPR status": _ipr_status(r), "Corr status": _corr_status(r),
                    "IPR donor": ipr_donor, "Corr donor": corr_donor,
                }
            )
        cov = pd.DataFrame(rows)
        seen.update(cov["Well"].tolist())
        st.session_state["hpi_cov_df"] = cov
        st.session_state["hpi_cov_token"] = token
        st.session_state.pop("hpi_coverage_editor", None)

    base = st.session_state["hpi_cov_df"]
    n_ipr = int(base["Flag"].str.contains("🟥", regex=False).sum())
    n_corr = int(base["Flag"].str.contains("🟧", regex=False).sum())

    st.subheader("IPR & Correlation Coverage")
    st.caption(
        f"🟥 **{n_ipr}** well(s) have no IPR test data — auto-set to a **pad+formation** "
        f"group IPR. 🟧 **{n_corr}** have no usable WHP→BHP correlation — auto-set to a "
        "**like-lift** group correlation. Review the **IPR donor** / **Corr donor** below "
        "(clear to `(own)` to keep a well on its own fallback, or pick a specific well / "
        "other group), then **Re-run**. ✓ real · ⚠ weak/missing · ✎ donor assigned."
    )
    donor_opts = donor_tokens(list(results_df["Well"]))
    edited = st.data_editor(
        base, key="hpi_coverage_editor", hide_index=True, use_container_width=True,
        column_config={
            "Flag": st.column_config.TextColumn(
                "⚑", disabled=True, pinned="left", width="small",
                help="🟥 no IPR test data (borrowing a pad+formation group IPR) · "
                "🟧 no usable WHP→BHP correlation (borrowing a like-lift group).",
            ),
            "Well": st.column_config.TextColumn("Well", disabled=True, pinned="left"),
            "Pad": st.column_config.TextColumn("Pad", disabled=True),
            "Lift": st.column_config.TextColumn("Lift", disabled=True),
            "IPR status": st.column_config.TextColumn("IPR status", disabled=True),
            "Corr status": st.column_config.TextColumn("Corr status", disabled=True),
            "IPR donor": st.column_config.SelectboxColumn(
                "IPR donor", options=donor_opts, required=False,
                help="Borrow an IPR (JP wells): a specific well or a like-wells group average.",
            ),
            "Corr donor": st.column_config.SelectboxColumn(
                "Corr donor", options=donor_opts, required=False,
                help="Borrow a WHP→BHP correlation: a specific well or a group average.",
            ),
        },
    )
    # Persist donor edits onto the input table so the (re-)run applies them. Do NOT
    # write `edited` back into hpi_cov_df — feeding a data_editor its own output makes
    # a feedback loop that drops/duplicates cells during mass edits. The widget replays
    # its edits onto the STABLE frame each rerun, so the source must stay fixed.
    new_ipr = dict(zip(edited["Well"], edited["IPR donor"]))
    new_corr = dict(zip(edited["Well"], edited["Corr donor"]))
    idf = input_df.copy()
    idf["IPR donor"] = idf["Well"].map(lambda w: new_ipr.get(w, cur_ipr.get(w, OWN_TOKEN)))
    idf["Corr donor"] = idf["Well"].map(lambda w: new_corr.get(w, cur_corr.get(w, OWN_TOKEN)))
    st.session_state["hpi_input_df"] = idf

    b1, b2 = st.columns(2)
    if b1.button("🔄 Re-run impact with these donor assignments", key="hpi_cov_rerun",
                 type="primary", use_container_width=True):
        st.session_state["hpi_trigger_run"] = True
        st.rerun()
    if b2.button("📄 Download per-well review PDF (IPR + correlation)",
                 key="hpi_cov_wellpdf", use_container_width=True):
        from datetime import datetime

        from .header_report import build_per_well_pdf
        with st.spinner("Building per-well PDF..."):
            try:
                wpdf = build_per_well_pdf(
                    results_df,
                    ipr_rows=st.session_state.get("hpi_ipr_rows", {}) or {},
                    well_dfs=st.session_state.get("hpi_emp_well_dfs", {}) or {},
                    fits=st.session_state.get("hpi_emp_fits", {}) or {},
                    test_df=st.session_state.get("hpi_test_df"),
                    stamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                )
                _autodownload(wpdf, "header_per_well.pdf", "hpiwpdf")
            except Exception as e:
                st.warning(f"Per-well PDF failed ({type(e).__name__}: {e}).")


# ── printouts & export (G4) ──────────────────────────────────────────────────


def _correlation_table(results_df: pd.DataFrame, emp_fits: dict) -> pd.DataFrame:
    """Per-well WHP→BHP correlation summary from the within-day fits — the slopes
    that were previously trapped inside the scatter/grid plots."""
    if "Well" not in results_df.columns:
        return pd.DataFrame()
    pad_map = dict(zip(results_df["Well"], results_df["Pad"]))
    rows = []
    for wn in results_df["Well"]:
        fitd = (emp_fits or {}).get(wn) or {}
        fwp = fitd.get("BHP~WHP")
        fhp = fitd.get("BHP~HeaderP")
        rows.append(
            {
                "Well": wn,
                "Pad": pad_map.get(wn, "—"),
                "BHP~WHP slope": fwp.mean_slope if fwp is not None else np.nan,
                "n days": fwp.n_days if fwp is not None else 0,
                "n good": fwp.n_good_days if fwp is not None else 0,
                "r²": fwp.mean_r2 if fwp is not None else np.nan,
                "class": ht.classify_response(fwp) if fwp is not None else "no data",
                "BHP~HeaderP slope": fhp.mean_slope if fhp is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _csv_dl(container, label: str, df: pd.DataFrame, fname: str, key: str) -> None:
    if df is None or df.empty:
        return
    container.download_button(
        label, data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname, mime="text/csv", key=key, use_container_width=True,
    )


def _render_exports(results_df: pd.DataFrame, delta_p: float) -> None:
    """G4 printouts: per-section CSVs, the correlation summary, and a one-click
    PDF report assembling all four artifacts (IPR, correlations, per-pad
    sensitivity, overall) plus the IPR curves."""
    from datetime import datetime

    st.subheader("Printouts & Export")
    emp_fits = st.session_state.get("hpi_emp_fits", {}) or {}
    well_dfs = st.session_state.get("hpi_emp_well_dfs", {}) or {}
    ipr_rows = st.session_state.get("hpi_ipr_rows", {}) or {}
    test_oil = st.session_state.get("hpi_test_oil", {}) or {}
    test_df = st.session_state.get("hpi_test_df")
    curve = st.session_state.get("hpi_curve")

    per_pad, overall = summarize_sensitivity(results_df, delta_p)
    sens_tbl = (
        pd.concat([per_pad, pd.DataFrame([overall])], ignore_index=True)
        if not per_pad.empty else pd.DataFrame()
    )
    sense_tbl = sense_check_table(results_df, test_oil)
    corr_tbl = _correlation_table(results_df, emp_fits)

    if not corr_tbl.empty:
        with st.expander("WHP→BHP correlation summary (per well)", expanded=False):
            st.dataframe(
                corr_tbl.style.format(
                    {"BHP~WHP slope": "{:.2f}", "r²": "{:.2f}", "BHP~HeaderP slope": "{:.2f}"},
                    na_rep="—",
                ),
                use_container_width=True, hide_index=True,
            )

    cols = st.columns(4)
    _csv_dl(cols[0], "Results CSV", results_df, "header_impact_results.csv", "hpi_dl_results")
    _csv_dl(cols[1], "Sensitivity CSV", sens_tbl, "header_impact_sensitivity.csv", "hpi_dl_sens")
    _csv_dl(cols[2], "Sense-check CSV", sense_tbl, "header_impact_sensecheck.csv", "hpi_dl_sense")
    _csv_dl(cols[3], "Correlations CSV", corr_tbl, "header_impact_correlations.csv", "hpi_dl_corr")

    if st.button("Generate PDF report (downloads)", key="hpi_pdf_btn"):
        from .header_report import build_report_pdf

        with st.spinner("Building PDF report (rendering figures)..."):
            try:
                pdf = build_report_pdf(
                    results_df, delta_p, ipr_rows=ipr_rows, test_oil=test_oil,
                    curve=curve, corr_table=corr_tbl, test_df=test_df,
                    well_dfs=well_dfs, fits=emp_fits,
                    stamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                )
                _autodownload(pdf, "header_impact_report.pdf", "hpipdf")
            except Exception as e:
                st.warning(f"PDF generation failed ({type(e).__name__}: {e}).")
    st.caption(
        "Full report = field/pad summary + sensitivity + sense-check + per-pad IPR & "
        "correlation pages. The per-well flip-through PDF is in the **Coverage panel** above."
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
    matplotlib grid: interactive in-app (pan/zoom, hover). PNG / PDF export is
    rendered separately in matplotlib (see ``header_report.corr_grid_mpl``) —
    kaleido-free, so it works on Databricks Apps.

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
        from .header_report import fig_to_png_bytes, well_fit_mpl
        try:
            mf = well_fit_mpl(well, well_dfs[well], fits.get(well, {}))
            if mf is not None:
                st.download_button(
                    "Download this plot (PNG)", data=fig_to_png_bytes(mf),
                    file_name=f"header_fit_{well}.png", mime="image/png",
                    key="hpi_diag_png",
                )
        except Exception as e:
            st.caption(
                f"PNG export unavailable ({type(e).__name__}). Use the chart's camera "
                "icon to save instead."
            )
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
        from .header_report import corr_grid_mpl, fig_to_png_bytes
        try:
            mf = corr_grid_mpl(well_dfs, fits, sorted(well_dfs.keys()), used)
            if mf is not None:
                st.download_button(
                    "Download grid PNG", data=fig_to_png_bytes(mf),
                    file_name=f"header_fit_grid_{used}.png", mime="image/png",
                    key="hpi_grid_dl",
                )
        except Exception as e:
            st.caption(f"PNG export unavailable ({type(e).__name__}).")


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
                # Drop the editor's widget state — otherwise it replays the
                # user's prior edits (incl. added rows) onto the reset frame
                # and the reset visibly doesn't take.
                st.session_state.pop("hpi_analog_editor", None)
                st.rerun()

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
        # The persisted frame becomes the next render's data= — drop the
        # widget state so it doesn't replay the same edits onto it (dynamic
        # rows would duplicate).
        st.session_state.pop("hpi_analog_editor", None)

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
