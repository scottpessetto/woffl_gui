"""Header Pressure Impact - the oil/BHP effect of moving a pad header.

Port of woffl/gui/scotts_tools/header_impact.py (engine only; the 353
Streamlit calls are replaced by the React page).

Sibling of PF Scenario: that one sweeps power-fluid pressure holding wellhead
pressure fixed; this sweeps WELLHEAD pressure (the header lever) holding PF
fixed. JP wells go through WOFFL physics (BatchPump swept on pwh); ESP /
gas-lift / flowing wells use an empirical slope plus a generic Vogel IPR
anchored at each well's own measured BHP.

Sonic decoupling is why "unresponsive" JPs exist: a choked pump cannot
propagate a downstream pressure change past the throat, so BHP does not move
with the header. Those wells come back tagged, not silently flat.

PF-PRESSURE-DEPENDENCY: PF pressure is a first-class input to the modeled
baseline here and there is still no per-well PF pressure in Databricks, so it
is seeded from pad defaults and stays editable per row.
"""


import numpy as np
import pandas as pd

from server.services import datasources

from server import config
from server.cache import ttl_cache

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.network_optimizer import WellConfig
from woffl.geometry.jetpump import JetPump
from server.services.tools import _common
from server.services.tools._common import (
    PAD_PF_DEFAULTS,
    PAD_PF_FALLBACK,
    default_pad_pf,
    live_pf_for_seed,
    load_well_characteristics,
)

from server.services.tools import header_trend as ht
from server.services.tools._common import (
    build_well_config,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_latest_bhp_per_well,
    get_latest_whp_per_well,
    get_vogel_for_wells,
    pad_from_mp_name,
)
from server.services.tools.header_engine import (
    DONOR_GROUP_TOKENS,
    FORMATION_PR_MAX,
    GROUP_LIFT,
    GROUP_PADFORMATION,
    MODEL_FAILED_PREFIX,
    OWN_TOKEN,
    _chosen_method,
    _verdict,
    aggregate_response_curve,
    average_slope,
    average_vogel_rows,
    backpressure_consistency,
    backtest_anchors,
    bias_by_pad,
    clamp_scenario_whp,
    classify_lift,
    corr_display_plan,
    depletion_signature,
    describe_donor,
    donor_member_wells,
    donor_tokens,
    estimate_header_impacts,
    fit_well_ipr,
    group_correlation_stats,
    pad_updown_lever,
    pf_map_from_selected,
    physics_slope,
    pr_hi_for_formation,
    predict_dbhp_from_curve,
    recent_bhp_anchor,
    resolve_pad_pf,
    sense_check_response,
    sense_check_table,
    solver_error_note,
    summarize_sensitivity,
    vogel_oil,
)
from server.services.tools.pf_scenario import _estimate_gaugeless_ipr

# Smallest (ResP - BHP) for which a generic Vogel IPR on an ESP / flowing well
# is allowed. Below this the assumed ResP is doing the work and the implied
# productivity index is fiction (review 2026-09-01, EVID-F17).
_MIN_GENERIC_DRAWDOWN_PSI = 300.0

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
    sonic/Mach flags, and ``error`` — the raw per-row error string from
    ``batch_run`` ("na" on a converged solve, ``repr(exc)`` on a failed one;
    see :func:`header_engine.solver_error_note`) so a non-converging solve can
    be told apart from a genuine "no response" (P1-21).
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
        jpump_direction=wc.jpump_direction,
        wellname=wc.well_name,
    )
    result_df = batch.batch_run([jp])
    if result_df.empty:
        return dict(
            oil=np.nan,
            pf_rate=np.nan,
            psu=np.nan,
            sonic=False,
            mach=np.nan,
            error="empty batch result",
        )
    r = result_df.iloc[0]
    return dict(
        oil=float(r["qoil_std"]) if pd.notna(r["qoil_std"]) else np.nan,
        pf_rate=float(r["lift_wat"]) if pd.notna(r["lift_wat"]) else np.nan,
        psu=float(r["psu_solv"]) if pd.notna(r["psu_solv"]) else np.nan,
        sonic=bool(r["sonic_status"]) if pd.notna(r.get("sonic_status")) else False,
        mach=float(r["mach_te"]) if pd.notna(r.get("mach_te")) else np.nan,
        error=r.get("error"),
    )


# ── producer universe + lift-type classification ─────────────────────────────


@ttl_cache(config.TTL_WELL_TESTS, maxsize=32)
def fetch_well_overview(months_back: int) -> pd.DataFrame:
    """Per-producer latest-test snapshot for ALL producers (not just JP).

    Columns: well (normalized), well_pad, oil, esp_amps, lift_gas, lift_wat, whp,
    resvr_press, wt_date. The universe + lift-type inputs + generic-IPR inputs
    (oil rate, reservoir pressure) for the all-lift-types flow.

    ``wt_date`` is the date of that latest test — ``_classify_lift`` needs it to
    tell a JP→ESP conversion (amps postdate the install) from an ESP→JP one
    (install postdates the amps). Consumers must tolerate its absence: a cache
    entry written before it was projected won't have the column.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    days = int(months_back * 31)
    q = f"""
    WITH latest AS (
        SELECT vwt.enthid, vwt.well_name, vwt.form_oil AS oil, vwt.esp_amps,
               vwt.lift_gas, vwt.lift_wat, vwt.whp, vwt.wt_date,
               ROW_NUMBER() OVER (PARTITION BY vwt.enthid ORDER BY vwt.wt_date DESC) AS rn
        FROM mpu.wells.vw_well_test vwt
        WHERE vwt.wt_date >= DATE_SUB(current_date(), {days}) AND vwt.allocated = True
    )
    SELECT l.well_name, l.oil, l.esp_amps, l.lift_gas, l.lift_wat, l.whp,
           l.wt_date, r.resvr_press, h.well_pad
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
    if "wt_date" in df.columns:
        df["wt_date"] = pd.to_datetime(df["wt_date"], errors="coerce")
    return df


def _classify_lift(well: str, jp_hist, ov_row) -> str:
    """ESP (esp_amps) / JP (current pump) / gas-lift (lift_gas) / flowing.

    Thin adapter: unpacks the jp_history install + the overview row and defers
    to :func:`header_engine.classify_lift`, which owns the rule (P1-26 recency
    terminator + the ESP→JP date guard). This module used to carry a second,
    divergent copy of that logic while the engine's went unused.

    ``ov_row`` is the well's LATEST test within the caller's recency window
    (``fetch_well_overview``'s ``months_back``). ``wt_date`` may be absent on a
    stale cache entry or a mocked frame — ``.get`` yields None and the engine
    falls back to the P1-26 default, so this degrades safely.
    """
    pump = get_current_pump(jp_hist, well)
    # A row whose nozzle/throat didn't parse isn't really a jet-pump install
    # (legacy ESP/wireline rows land in those columns), so it can't claim the
    # well — pass no install date in that case.
    is_jp = bool(pump and pump.get("nozzle_no") and pump.get("throat_ratio"))
    get = ov_row.get if ov_row is not None else (lambda _k, _d=None: None)
    return classify_lift(
        pump.get("date_set") if is_jp else None,
        get("esp_amps"),
        get("lift_gas"),
        test_date=get("wt_date"),
    )


# ── per-pad power-fluid pressure ─────────────────────────────────────────────


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
    jp_hist, _src = datasources.jp_history_safe()
    if jp_hist is None or jp_hist.empty:
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
        is_jp = (
            lift == "JP"
            and pump is not None
            and pump.get("nozzle_no")
            and pump.get("throat_ratio")
        )
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

        # JP wells: live per-well PF (vw_pressure_daily) beats the pad-table
        # value; pad value covers wells without a valid live reading.
        pf_held = None
        if is_jp:
            live_pf = live_pf_for_seed(wn)
            pf_held = (
                int(round(live_pf["pf_press"]))
                if live_pf
                # default_pad_pf() takes a WELL NAME; handing it the pad
                # letter resolved to the 3,400 fallback for every pad
                # (review 2026-09-01, EVID-F16). Use the pad table directly.
                else int(pad_pf.get(pad, PAD_PF_DEFAULTS.get(pad, PAD_PF_FALLBACK)))
            )

        rows.append(
            {
                "Well": wn,
                "Pad": pad,
                "Lift": lift,
                "Pump": f"{pump['nozzle_no']}{pump['throat_ratio']}" if is_jp else "",
                "Oil (BOPD)": round(float(oil), 0) if pd.notna(oil) else None,
                "PF held (psi)": pf_held,
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
    start = (datetime.now() - relativedelta(months=int(months_back))).strftime(
        "%Y-%m-%d"
    )
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
        y_name="BHP",
        x_name="WHP",
        mean_slope=float(slope),
        median_slope=float(slope),
        slope_std=0.0,
        n_days=30,
        n_good_days=20,
        mean_r2=0.8,
        daily=pd.DataFrame(),
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
    wn,
    row,
    emp_fits: dict,
    emp_well_dfs: dict,
    delta_p: float,
    fit_override: dict | None = None,
    corr_prov: str = "own",
    ipr_sink: dict | None = None,
    res_pres_override: float | None = None,
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

    # Robust "current BHP" anchor (P1-25): median of the last 24 readings above
    # the shut-in/sentinel screen, NOT the single last raw reading — a lone
    # low tail bin (shut-in, dead gauge) could otherwise anchor the whole
    # generic Vogel IPR for the well.
    trend = emp_well_dfs.get(wn)
    bhp_now = np.nan
    bhp_anchor_screened = True
    if trend is not None and "BHP" in getattr(trend, "columns", []):
        bhp_now, bhp_anchor_screened = recent_bhp_anchor(trend["BHP"])

    oil = row.get("Oil (BOPD)")
    pseudo = res_pres_override is not None and not pd.isna(res_pres_override)
    res_pres = (
        float(res_pres_override) if pseudo else float(row.get("ResP (psi)") or 1800.0)
    )
    whp_now = row.get("WHP now (psi)")

    emp_dbhp = np.nan
    emp_doil = np.nan
    if pd.isna(bhp_now):
        verdict = "gaugeless — use Analog"
    elif emp_class in ("no data", "insufficient"):
        # Data absence, not a physical diagnosis (P1-23) — don't book a
        # confident ΔOil = 0 under a mislabeled "slugging" verdict. Leave
        # emp_dbhp/emp_doil as NaN so the well is EXCLUDED from the field
        # uplift sum (Chosen ΔOil / oil_scen both come out NaN and
        # DataFrame.sum() skips NaN) rather than silently zeroed in.
        verdict = "insufficient data" if emp_class == "insufficient" else "no data"
    elif fit is None or emp_class != "responsive":
        verdict = "slugging"
        emp_dbhp = 0.0
        emp_doil = 0.0  # an unresponsive well won't move with the header
    elif oil is None or pd.isna(oil) or float(oil) <= 0:
        verdict = "no test oil rate"
    elif res_pres - bhp_now < _MIN_GENERIC_DRAWDOWN_PSI:
        # An ASSUMED reservoir pressure within a few hundred psi of the
        # measured BHP manufactures a productivity index: with the old
        # max(ResP, BHP+100) floor a 30 psi header move predicted ~30% of the
        # well's oil (review 2026-09-01, EVID-F17). No IPR, no delta oil -
        # the well is excluded from the pad sum rather than invented.
        verdict = "ResP too close to BHP - no IPR"
    else:
        pres = res_pres
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
        and res_pres - bhp_now >= _MIN_GENERIC_DRAWDOWN_PSI
    ):
        ipr_sink[wn] = {
            "res_pres": float(res_pres),
            "qwf": float(oil),
            "pwf": float(bhp_now),
            "form_wc": 0.0,  # qwf is already the oil rate for non-JP
        }
    oil_scen = (
        (float(oil) + emp_doil) if (pd.notna(oil) and pd.notna(emp_doil)) else np.nan
    )

    ipr_src = "empirical (pseudo-Pr)" if pseudo else "empirical (test + assumed ResP)"
    if pd.notna(bhp_now) and not bhp_anchor_screened:
        # No reading passed the shut-in/sentinel screen — recent_bhp_anchor fell
        # back to the raw median, so flag the anchor as suspect rather than
        # presenting it silently as a clean measurement.
        ipr_src += " · BHP anchor unscreened"

    return {
        "Well": wn,
        "Pad": row["Pad"],
        "Lift": row.get("Lift", ""),
        "Formation": row.get("Formation"),
        "Pump": "",
        "IPR src": ipr_src,
        "Fric src": "—",
        "BHP cal err": None,
        "PF held (psi)": None,
        "WHP now (psi)": int(round(whp_now)) if pd.notna(whp_now) else None,
        "WHP scen (psi)": int(round(whp_now + delta_p)) if pd.notna(whp_now) else None,
        "BHP now (psi)": bhp_now,
        "BHP scen (psi)": (
            (bhp_now + emp_dbhp)
            if (pd.notna(emp_dbhp) and pd.notna(bhp_now))
            else np.nan
        ),
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


# ── main tab ─────────────────────────────────────────────────────────────────


# ── results display ───────────────────────────────────────────────────────


# ── v2 review surface: sensitivity, per-pad expanders, IPR, sense check ──────


def _predict_doil(bhp_then, bhp_now, ir, fit, use_pseudo: bool, wc: float = 0.0):
    """Predicted oil change for the actual BHP move, off an IPR. ``use_pseudo`` → the
    single-fit blue IPR (on TOTAL LIQUID → ΔLiquid × (1−WC) = ΔOil); else the model
    IPR (already oil, qwf×(1−WC)). NaN if the inputs aren't there."""
    if pd.isna(bhp_then) or pd.isna(bhp_now):
        return np.nan
    if use_pseudo and fit:
        dliq = vogel_oil(bhp_now, fit["qmax"], fit["pr"]) - vogel_oil(
            bhp_then, fit["qmax"], fit["pr"]
        )
        return dliq * (1.0 - float(wc))  # liquid IPR → oil via water cut
    if ir:
        try:
            from woffl.flow.inflow import InFlow

            fwc = float(ir.get("form_wc", 0.5))
            inflow = InFlow(
                qwf=float(ir["qwf"]) * (1.0 - fwc),
                pwf=float(ir["pwf"]),
                pres=float(ir.get("res_pres", 1800.0)),
            )
            return float(
                inflow.oil_flow(float(bhp_now), "vogel")
                - inflow.oil_flow(float(bhp_then), "vogel")
            )
        except Exception:
            return np.nan
    return np.nan


def _backtest_table(
    pad_df,
    test_df,
    curve,
    pump_changed: set | None = None,
    fit_map: dict | None = None,
    ipr_rows: dict | None = None,
    use_pseudo: bool = False,
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
        dbhp_pred, extr = predict_dbhp_from_curve(
            cwells.get(w), a["whp_then"], a["whp_now"]
        )
        diagnosis = backpressure_consistency(a["d_whp"], a["d_bhp"], a["d_liquid"])
        fm = fit_map.get(w) or {}
        fit, depl = fm.get("fit"), fm.get("depl") or {}
        wc = 0.0
        if tw is not None and "form_wc" in getattr(tw, "columns", []):
            s = pd.to_numeric(
                tw.sort_values("WtDate")["form_wc"], errors="coerce"
            ).dropna()
            if not s.empty:
                wc = float(s.iloc[-1])
        doil_pred = _predict_doil(
            a["bhp_then"], a["bhp_now"], ipr_rows.get(w), fit, use_pseudo, wc
        )
        # Single-fit pseudo-Pr (* = pinned at the formation cap → soft) + the
        # geometric depletion verdict (corr sign), which is the robust read.
        pr_fit = (
            (f"{fit['pr']:.0f}{'*' if fit.get('pr_at_bound') else ''}") if fit else "—"
        )
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
                "Well": w,
                "n tests": a["n_tests"],
                "ΔWHP (psi)": a["d_whp"],
                "ΔBHP actual (psi)": a["d_bhp"],
                "ΔBHP pred (psi)": dbhp_pred,
                "ΔOil actual (BOPD)": a["d_oil"],
                "ΔOil pred (BOPD)": doil_pred,
                "ΔLiquid (BPD)": a["d_liquid"],
                "Pr fit": pr_fit,
                "BHP↔liq corr": corr,
                "Depletion": depl.get("verdict", "—"),
                "Diagnosis": diagnosis,
                "Flags": " ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def _pump_changed_in_window(results_df: pd.DataFrame, months_back: int = 6) -> set:
    """Wells whose CURRENT jet pump was installed inside the test/trend lookback —
    a bigger pump shifts BHP independent of the header, so the back-test row is
    suspect. Uses jp_history install dates vs the months-back window."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    jp_hist, _src = datasources.jp_history_safe()
    if jp_hist is None or jp_hist.empty:
        return set()
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


# ── per-well empirical fit diagnostics ────────────────────────────────────


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




def solve_jp_row(
    row_dict: dict,
    chars: dict | None,
    pump: dict | None,
    vogel_row: dict | None,
    well_fits: dict | None,
    delta_p: float,
) -> dict:
    """One JP well through the physics path. Module-level and picklable.

    Lifted from the tab's render loop, which is where the JP branch lived -
    ``_solve_nonjp_row`` is, as its name says, the OTHER branch. Routing JP
    wells through that one produces rows with no DeltaOil at all, which is
    exactly what a first pass of this port did.

    Solves the well at its current wellhead pressure and again at
    ``WHP + delta_p``; the BHP and oil change fall out of the two solves.
    The scenario WHP is clamped to ``_WHP_FLOOR`` (P1-22): delta_p reaches
    -500 while typical WHPs run 100-400 psi, and the solver returns nonsense
    below the floor. Display and maths both read the CLAMPED value so they
    cannot disagree.

    v1 deliberately omits the tab's donor-IPR and friction-recalibration
    options; it uses the well's own Vogel row and its stored friction
    coefficients. Those are additive and belong to a later pass.
    """
    wn = row_dict["Well"]
    base = {
        "Well": wn,
        "Pad": row_dict.get("Pad"),
        "Lift": "JP",
        "Formation": row_dict.get("Formation"),
    }
    if not chars or not (pump and pump.get("nozzle_no") and pump.get("throat_ratio")):
        return {**base, "Error": "no jp_chars row or no current pump", "Verdict": "n/a"}

    whp_now = row_dict.get("WHP now (psi)")
    try:
        whp_now = float(whp_now)
    except (TypeError, ValueError):
        whp_now = 210.0
    if pd.isna(whp_now):
        whp_now = 210.0
    pf_held = float(row_dict.get("PF held (psi)") or default_pad_pf(wn))

    try:
        wc = _common.build_well_config(wn, {wn: chars}, vogel_row, surf_pres=whp_now)
        well_objs = _common.create_well_objects(wc)
    except Exception as exc:  # noqa: BLE001
        return {**base, "Error": f"setup: {str(exc)[:160]}", "Verdict": "n/a"}

    fric = _common.friction_coefs_from_chars(chars)
    nozzle, throat = pump["nozzle_no"], pump["throat_ratio"]
    whp_scen, clamped = clamp_scenario_whp(whp_now, delta_p, floor=_WHP_FLOOR)

    try:
        res_now = _solve_at_whp(wc, well_objs, nozzle, throat, whp_now, pf_held, fric)
        res_scen = _solve_at_whp(wc, well_objs, nozzle, throat, whp_scen, pf_held, fric)
    except Exception as exc:  # noqa: BLE001
        return {**base, "Error": f"solver: {str(exc)[:160]}", "Verdict": "n/a"}

    d_oil = res_scen["oil"] - res_now["oil"]
    d_bhp = res_scen["psu"] - res_now["psu"]

    emp: dict = {}
    try:
        emp = _empirical_columns(
            well_fits, well_objs[2], float(wc.res_pres), float(res_now["psu"]), delta_p
        )
    except Exception:  # noqa: BLE001 - empirical is the optional half
        emp = {}

    return {
        **base,
        "Pump": f"{nozzle}{throat}",
        "WHP now": round(whp_now, 1),
        "WHP scen": round(whp_scen, 1),
        "WHP clamped": bool(clamped),
        "PF held": round(pf_held, 1),
        "BHP now": round(res_now["psu"], 1),
        "BHP scen": round(res_scen["psu"], 1),
        "DeltaBhp": round(d_bhp, 1),
        "Oil now": round(res_now["oil"], 1),
        "Oil scen": round(res_scen["oil"], 1),
        "DeltaOil": round(d_oil, 1),
        "SonicNow": bool(res_now["sonic"]),
        "SonicScen": bool(res_scen["sonic"]),
        "MachNow": res_now["mach"],
        **emp,
        "Verdict": _verdict(
            bool(res_now["sonic"]),
            bool(res_scen["sonic"]),
            float(d_oil),
            # Row key is "Emp class" (see _empirical_columns). The old
            # "EmpClass" lookup was always None, so the physics-vs-field
            # verdict never fired (review 2026-09-01, EVID-F15).
            emp.get("Emp class"),
            compare_emp=bool(emp),
        ),
        "Error": "",
    }
