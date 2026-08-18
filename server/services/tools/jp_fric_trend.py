"""JP Friction Trend - fitted friction coefficients over a well's test history.

Port of woffl/gui/scotts_tools/jp_fric_trend.py (engine only; the Streamlit
rendering is replaced by the React page).

Two-step per test: calibrate the PF pressure that reproduces the measured
lift water, then fit the friction coefficients that reproduce the measured
BHP at that pressure. Each test uses the pump installed AT THAT TEST and the
test's own WC/GOR - before that fix every historical point was fitted with
today's geometry and a generic fluid, so coefficient "trends" across a JPCO
line were geometry artifacts rather than wear.
"""


from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from server.services import datasources

from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from woffl.flow.inflow import InFlow
from woffl.gui.fric_calibration import calibrate_friction_coefs
from woffl.assembly.pf_calibration import calibrate_pf_for_lift
from woffl.assembly.sim_factories import create_pvt_components
from woffl.pvt.resmix import ResMix

from server.services.tools._common import (
    build_well_config,
    create_well_objects,
    fetch_well_tests_raw,
    friction_coefs_from_chars,
    get_vogel_for_wells,
    worker_ceiling,
)


def _build_well_inputs(well_names: list[str], months_back: int) -> dict[str, pd.DataFrame]:
    """Return {well: tests_df} keeping only tests with all required fields."""
    raw = fetch_well_tests_raw(months_back)
    if raw is None or raw.empty:
        return {}
    needed = ["lift_wat", "WtOilVol", "BHP", "WtDate"]
    available = [c for c in needed if c in raw.columns]
    df = raw.dropna(subset=available).copy()
    if "lift_wat" in df.columns:
        df = df[df["lift_wat"] > 0]
    out: dict[str, pd.DataFrame] = {}
    for wn in well_names:
        sub = df[df["well"] == wn].sort_values("WtDate")
        if not sub.empty:
            out[wn] = sub
    return out


def _calibrate_well(
    well_name: str,
    tests_df: pd.DataFrame,
    chars: dict,
    pump: dict,
    vogel_row: dict | None = None,
) -> pd.DataFrame:
    """Run the two-step calibration over every test for one well.

    Module-level + Streamlit-free so it can be dispatched via
    ``ProcessPoolExecutor``. The main thread tracks progress at the well
    level via ``as_completed``; this function only returns the result rows.

    Each test calibrates with the pump installed AT THE TEST DATE
    (``NozzleAtTest``/``ThroatAtTest`` columns added by the task builder;
    falls back to the current ``pump`` when absent) and with the test's own
    fluid (WC/GOR) and operating point (oil rate at measured BHP) anchoring
    the IPR. Before this, every historical test used today's pump geometry
    and a generic WC=0.5/GOR=250 fluid — the fitted coefficients absorbed
    those errors, and coef shifts across JPCO lines were geometry artifacts.
    """
    rows: list[dict] = []
    wc = build_well_config(well_name, {well_name: chars}, vogel_row, surf_pres=210.0)
    wellbore, wellprof, _inflow_unused, _resmix_unused, prop_pf = create_well_objects(wc)
    # One PVT component set per well, shared by the per-test mixtures.
    oil_pvt, wat_pvt, gas_pvt = create_pvt_components(
        field_model=wc.field_model, oil_api=wc.oil_api, gas_sg=wc.gas_sg,
        wat_sg=wc.wat_sg, bubble_point=wc.bubble_point,
    )
    cur = friction_coefs_from_chars(chars)
    knz = cur.get("knz", 0.01)
    seed_ken = cur.get("ken", 0.03)
    seed_kth = cur.get("kth", 0.30)
    seed_kdi = cur.get("kdi", 0.30)

    for _, test in tests_df.iterrows():
        try:
            nozzle = str(test.get("NozzleAtTest") or pump["nozzle_no"])
            throat = str(test.get("ThroatAtTest") or pump["throat_ratio"])
            pwh = float(test["whp"]) if pd.notna(test.get("whp")) else 210.0

            # Per-test fluid: the test's own WC/GOR, falling back to the
            # well-level (Vogel or default) values when unmeasured.
            wc_t = test.get("form_wc")
            if wc_t is None or pd.isna(wc_t):
                o = float(test.get("WtOilVol") or 0.0)
                w = float(test.get("WtWaterVol") or 0.0)
                wc_t = w / (o + w) if (o + w) > 0 else wc.form_wc
            wc_t = min(max(float(wc_t), 0.0), 0.99)
            gor_t = test.get("fgor")
            gor_t = (
                float(gor_t) if (gor_t is not None and pd.notna(gor_t) and float(gor_t) > 0)
                else wc.form_gor
            )
            res_mix = ResMix(wc=wc_t, fgor=gor_t, oil=oil_pvt, wat=wat_pvt, gas=gas_pvt)

            # IPR anchored on the test's own operating point (oil @ BHP),
            # reservoir pressure from the Vogel fit / chars.
            inflow = InFlow(
                qwf=float(test["WtOilVol"]), pwf=float(test["BHP"]), pres=wc.res_pres
            )

            # Step 1: PF cal
            pf = calibrate_pf_for_lift(
                well_name=well_name,
                target_lift=float(test["lift_wat"]),
                pwh=pwh, tsu=wc.form_temp,
                nozzle=nozzle, throat=throat,
                knz=knz, ken=seed_ken, kth=seed_kth, kdi=seed_kdi,
                wellbore=wellbore, wellprof=wellprof,
                ipr_su=inflow, prop_su=res_mix, prop_pf=prop_pf,
            )
            # Step 2: Coef cal at ppf_surf*
            coef = calibrate_friction_coefs(
                well_name=well_name,
                target_bhp=float(test["BHP"]),
                pwh=pwh, tsu=wc.form_temp,
                ppf_surf=float(pf.ppf_surf),
                nozzle=nozzle, throat=throat,
                knz=knz, ken=seed_ken,
                wellbore=wellbore, wellprof=wellprof,
                ipr_su=inflow, prop_su=res_mix, prop_pf=prop_pf,
            )
            rows.append({
                "WtDate": test["WtDate"],
                "Well": well_name,
                "Nozzle": nozzle,
                "Throat": throat,
                "Pump": f"{nozzle}{throat}",
                "Oil": float(test["WtOilVol"]),
                "Water": float(test.get("WtWaterVol") or 0.0),
                "Gas": float(test.get("WtGasVol") or 0.0),
                "WC": round(wc_t, 3),
                "GOR": round(gor_t, 0),
                "WHP": pwh,
                "BHP": float(test["BHP"]),
                "LiftWat": float(test["lift_wat"]),
                "PpfSurfFound": pf.ppf_surf,
                "LiftResidual": pf.lift_residual,
                "PfConverged": pf.converged,
                "PfBounded": pf.bounded,
                "PfSonic": pf.sonic,
                "Ken": coef.best_ken,
                "Kth": coef.best_kth,
                "Kdi": coef.best_kdi,
                "CoefMatchQuality": coef.match_quality,
                "CoefBounded": coef.bounded,
                "CoefSonic": coef.sonic,
                "BhpError": coef.bhp_error,
                "Status": "ok",
                "Error": "",
            })
        except Exception as e:  # pragma: no cover — per-test safety net
            rows.append({
                "WtDate": test.get("WtDate"),
                "Well": well_name,
                "Status": "error",
                "Error": str(e)[:200],
            })
    return pd.DataFrame(rows)


def combine_results(per_well: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-well result frames. Was a Streamlit session read; the caller
    now owns the accumulation (the React page keeps the selected wells)."""
    frames = [df for df in per_well.values() if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_BOOL_COLS = (
    "PfConverged", "PfBounded", "PfSonic",
    "CoefBounded", "CoefSonic",
)


def _format_jp(nozzle, throat) -> str:
    """Format nozzle + throat into a pump label like '12B'.

    Mirrors jp_history_tab._format_jp but defensive against non-numeric
    nozzles (e.g. 'G' in some legacy history rows) — falls back to the raw
    string instead of crashing on int().
    """
    parts = ""
    if pd.notna(nozzle):
        try:
            parts += str(int(nozzle))
        except (TypeError, ValueError):
            parts += str(nozzle).strip()
    if pd.notna(throat):
        parts += str(throat).strip()
    return parts or "?"


def _add_jpco_overlays(
    fig,
    well_name: str,
    x_min: pd.Timestamp,
    x_max: pd.Timestamp,
) -> None:
    """Overlay vertical dashed lines + JPCO labels on a time-series figure.

    Mirrors the pattern in jp_history_tab._create_history_chart. Shows
    every JP change in the JP history whose Date Set falls within a
    generous range around the chart's data window so the "what pump is in
    at the start" context is visible without distorting the axis.
    """
    jp_hist, _src = datasources.jp_history_safe()
    if jp_hist is None or jp_hist.empty:
        return
    well_jp = (
        jp_hist[jp_hist["Well Name"] == well_name]
            .dropna(subset=["Date Set"])
            .sort_values("Date Set")
            .reset_index(drop=True)
    )
    if well_jp.empty:
        return
    # Trim to JPCOs within the chart range plus a 6-month leading buffer so
    # the most-recent install before the first calibration test stays visible.
    lead = pd.Timedelta(days=180)
    trail = pd.Timedelta(days=30)
    in_range = well_jp[
        (well_jp["Date Set"] >= x_min - lead)
        & (well_jp["Date Set"] <= x_max + trail)
    ].reset_index(drop=True)
    if in_range.empty:
        return
    for idx, row in in_range.iterrows():
        date_str = row["Date Set"].isoformat()
        new_jp = _format_jp(row.get("Nozzle Number"), row.get("Throat Ratio"))
        # The "previous pump" comes from the full history, not the trimmed
        # view — otherwise the first overlay label always reads "Set X".
        full_idx = well_jp.index[well_jp["Date Set"] == row["Date Set"]].min()
        if full_idx == 0:
            label = f"Set {new_jp}"
        else:
            prev = well_jp.iloc[full_idx - 1]
            old_jp = _format_jp(prev.get("Nozzle Number"), prev.get("Throat Ratio"))
            label = (
                f"JPCO {new_jp} (same)" if old_jp == new_jp
                else f"JPCO {old_jp}→{new_jp}"
            )
        y_frac = 0.95 if idx % 2 == 0 else 0.85
        fig.add_shape(
            type="line",
            x0=date_str, x1=date_str, y0=0, y1=1, yref="paper",
            line=dict(dash="dash", color="rgba(211,47,47,0.7)", width=1.5),
        )
        fig.add_annotation(
            x=date_str, y=y_frac, yref="paper",
            text=label, showarrow=False, textangle=-90, xshift=-7,
            font=dict(size=10, color="#D32F2F"),
        )


def _parse_uploaded_csv(csv_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Parse the combined-CSV output from this tab and group by Well.

    The download button writes the result of pd.concat(...) on all wells'
    rows. Reversing that is just read_csv + groupby. Dates and bool columns
    need explicit coercion since CSV doesn't preserve dtypes.
    """
    import io

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if "Well" not in df.columns:
        raise ValueError("CSV missing required column 'Well'")
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], errors="coerce")
    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.lower()
                     .isin(("true", "1", "yes"))
            )
    out: dict[str, pd.DataFrame] = {}
    for well, group in df.groupby("Well", sort=True):
        out[str(well)] = group.reset_index(drop=True)
    return out


