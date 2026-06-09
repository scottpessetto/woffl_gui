"""JP Friction Coefficient Trend Analysis tab.

For each selected well, run the two-step calibration on every historical
test that has both rate and BHP measured:
  1. PF calibration — find ppf_surf such that modeled qnz matches observed
     lift_wat (at the well's current friction coefs).
  2. Coef calibration — find ken/kth/kdi such that modeled BHP matches
     observed BHP (at the ppf_surf from step 1).

Results accumulate in session_state across wells so cross-well trends can
be inspected without re-running per-session. Single-pass per test — the
PF↔coef cross-coupling residual is recorded but not iterated.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.jp_history import get_current_pump, get_pump_at_date
from woffl.flow.inflow import InFlow
from woffl.gui.fric_calibration import calibrate_friction_coefs
from woffl.gui.pf_calibration import calibrate_pf_for_lift
from woffl.gui.utils import create_pvt_components, load_well_characteristics
from woffl.pvt.resmix import ResMix

from ._common import (
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


def _combined_results() -> pd.DataFrame:
    store = st.session_state.get("jp_fric_trend", {})
    frames = [df for df in store.values() if df is not None and not df.empty]
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
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
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


def _render_per_well(df: pd.DataFrame, wn: str) -> None:
    """One-well detail view: result table + time-series + scatter plots."""
    if df is None or df.empty:
        st.info("No results.")
        return
    ok = df[df["Status"] == "ok"].copy()
    err = df[df["Status"] == "error"]

    if ok.empty:
        st.warning(f"{wn}: all tests errored — see error table below.")
    else:
        st.dataframe(
            ok, use_container_width=True, hide_index=True,
            column_config={
                "WtDate": st.column_config.DatetimeColumn(
                    "Test Date", format="YYYY-MM-DD",
                ),
                "Ken": st.column_config.NumberColumn("ken", format="%.4f"),
                "Kth": st.column_config.NumberColumn("kth", format="%.4f"),
                "Kdi": st.column_config.NumberColumn("kdi", format="%.4f"),
                "PpfSurfFound": st.column_config.NumberColumn(
                    "PF (psi)", format="%.0f",
                ),
                "Oil": st.column_config.NumberColumn("Oil", format="%.0f"),
                "Water": st.column_config.NumberColumn("Water", format="%.0f"),
                "Gas": st.column_config.NumberColumn("Gas", format="%.0f"),
                "WC": st.column_config.NumberColumn("WC", format="%.2f"),
                "GOR": st.column_config.NumberColumn("GOR", format="%.0f"),
                "WHP": st.column_config.NumberColumn("WHP", format="%.0f"),
                "BHP": st.column_config.NumberColumn("BHP", format="%.0f"),
                "LiftWat": st.column_config.NumberColumn("LiftWat", format="%.0f"),
                "LiftResidual": st.column_config.NumberColumn(
                    "Lift Res", format="%.1f",
                ),
                "BhpError": st.column_config.NumberColumn(
                    "BHP Err", format="%.1f",
                ),
                "PfBounded": st.column_config.CheckboxColumn("PF Bnd"),
                "PfSonic": st.column_config.CheckboxColumn("PF Son"),
                "CoefBounded": st.column_config.CheckboxColumn("Coef Bnd"),
                "CoefSonic": st.column_config.CheckboxColumn("Coef Son"),
                "CoefMatchQuality": st.column_config.TextColumn("Match"),
            },
        )

        # Time-series of ken / kth / kdi
        fig = go.Figure()
        for col, name in [("Ken", "ken"), ("Kth", "kth"), ("Kdi", "kdi")]:
            fig.add_trace(go.Scatter(
                x=ok["WtDate"], y=ok[col],
                mode="lines+markers", name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>" + name + ": %{y:.4f}<extra></extra>",
            ))
        _add_jpco_overlays(fig, wn, ok["WtDate"].min(), ok["WtDate"].max())
        fig.update_layout(
            title=f"{wn} — friction coefficients over time",
            xaxis_title="Test Date", yaxis_title="Coefficient",
            hovermode="x unified", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: coefs vs rates
        scols = st.columns(3)
        for j, (xcol, xlabel, xfmt) in enumerate([
            ("Oil", "Oil rate (BOPD)", "%{x:.0f}"),
            ("Gas", "Gas rate (MCFD)", "%{x:.0f}"),
            ("WC",  "Water cut (frac)", "%{x:.2f}"),
        ]):
            with scols[j]:
                fs = go.Figure()
                for col, name in [("Ken", "ken"), ("Kth", "kth"), ("Kdi", "kdi")]:
                    fs.add_trace(go.Scatter(
                        x=ok[xcol], y=ok[col], mode="markers", name=name,
                        hovertemplate=xfmt + "<br>" + name + ": %{y:.4f}<extra></extra>",
                    ))
                fs.update_layout(
                    title=f"Coefs vs {xcol}",
                    xaxis_title=xlabel, yaxis_title="Coefficient",
                    height=320, showlegend=(j == 0),
                )
                st.plotly_chart(fs, use_container_width=True)

    if not err.empty:
        with st.expander(f"{len(err)} errored tests"):
            st.dataframe(err[["WtDate", "Status", "Error"]], hide_index=True)


def render_tab() -> None:
    st.header("JP Friction Coefficient Trend Analysis")
    st.caption(
        "Two-step calibration per historical test: find ppf_surf to match "
        "observed lift_wat, then calibrate ken/kth/kdi at that ppf_surf to "
        "match observed BHP. Single-pass — small PF↔coef residual is "
        "recorded but not iterated."
    )

    if "jp_fric_trend" not in st.session_state:
        st.session_state["jp_fric_trend"] = {}

    jp_chars_df = load_well_characteristics()
    if jp_chars_df is None or jp_chars_df.empty:
        st.error("No well characteristics loaded.")
        return
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.error("jp_history not loaded — go back to the main app first.")
        return

    all_wells = sorted(jp_chars_df["Well"].dropna().astype(str).tolist())

    def _has_valid_jp(well_name: str) -> bool:
        """True only if this well has a current JP install with both a
        numeric nozzle number and a throat ratio. Skips wells where the
        history row has corrupt or non-JP equipment data."""
        try:
            p = get_current_pump(jp_hist, well_name)
        except Exception:
            return False
        return bool(p) and bool(p.get("nozzle_no")) and bool(p.get("throat_ratio"))

    jp_wells = [w for w in all_wells if _has_valid_jp(w)]

    c1, c2, c3, c4 = st.columns([2.5, 1.2, 1.2, 1.1])
    with c1:
        selected_wells = st.multiselect(
            "Wells to calibrate",
            options=jp_wells,
            default=st.session_state.get("jp_fric_trend_wells", []),
            key="jp_fric_trend_wells",
        )
    with c2:
        months_back = st.slider(
            "Test lookback (months)", 1, 24, 12,
            key="jp_fric_trend_months",
        )
    with c3:
        ceiling = worker_ceiling()
        if ceiling > 1:
            workers = st.slider(
                "Parallel workers", 1, ceiling, ceiling,
                key="jp_fric_trend_workers",
                help=f"ProcessPool workers for per-well calibration. Capped "
                f"by WOFFL_MAX_WORKERS (current={ceiling}). Defaults to the "
                "cap; drag down to throttle.",
            )
        else:
            workers = 1
            st.caption("Workers: 1 (set WOFFL_MAX_WORKERS to enable)")
    with c4:
        st.write("")
        force_refresh = st.checkbox(
            "Force recompute",
            help="Re-run calibration even for wells already in results",
            key="jp_fric_trend_force",
        )

    col_run, col_clear, _ = st.columns([1.4, 1.2, 4])
    with col_run:
        run_clicked = st.button(
            "Run trend calibration",
            type="primary",
            disabled=not selected_wells,
        )
    with col_clear:
        if st.session_state["jp_fric_trend"]:
            if st.button("Clear all results"):
                st.session_state["jp_fric_trend"] = {}
                st.rerun()

    with st.expander(
        "Load prior results from CSV "
        "(use a previously-downloaded jp_fric_trend_*.csv)",
        expanded=False,
    ):
        uploaded = st.file_uploader(
            "CSV file",
            type=["csv"],
            key="jp_fric_trend_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            merge_mode = st.radio(
                "Merge mode",
                ["Merge with existing wells", "Replace existing wells"],
                horizontal=True,
                key="jp_fric_trend_upload_mode",
                help="Merge: keep wells already in results and add new ones. "
                "Replace: clear current results and load only the CSV.",
            )
            if st.button("Load CSV", key="jp_fric_trend_upload_btn"):
                try:
                    loaded = _parse_uploaded_csv(uploaded.getvalue())
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
                else:
                    if merge_mode.startswith("Replace"):
                        st.session_state["jp_fric_trend"] = loaded
                    else:
                        st.session_state["jp_fric_trend"].update(loaded)
                    n_wells = len(loaded)
                    n_rows = sum(len(df) for df in loaded.values())
                    st.success(
                        f"Loaded {n_wells} wells ({n_rows} rows). Scroll down "
                        "to view results."
                    )

    if run_clicked:
        to_run = selected_wells if force_refresh else [
            w for w in selected_wells
            if w not in st.session_state["jp_fric_trend"]
        ]
        if not to_run:
            st.info(
                "All selected wells already have results. "
                "Check 'Force recompute' to re-run."
            )
        else:
            jp_chars_dict = jp_chars_df.set_index("Well").to_dict("index")
            well_tests = _build_well_inputs(to_run, months_back)
            # Per-well Vogel IPR so reservoir pressure / fluid fall back to
            # measured fits instead of the generic WC=0.5/GOR=250 defaults.
            vogel_map = get_vogel_for_wells(to_run, months_back)

            # Build per-well task tuples (well_name, tests_df, chars, pump, vogel)
            tasks: list[tuple[str, pd.DataFrame, dict, dict, dict | None]] = []
            for wn in to_run:
                if wn not in well_tests:
                    st.warning(
                        f"{wn}: no qualifying tests "
                        "(need rate + BHP + lift_wat)"
                    )
                    continue
                chars = jp_chars_dict.get(wn)
                if not chars:
                    st.warning(f"{wn}: missing from jp_chars")
                    continue
                pump = get_current_pump(jp_hist, wn)
                if pump is None or not pump.get("nozzle_no") or not pump.get("throat_ratio"):
                    st.warning(f"{wn}: no current JP pump")
                    continue

                # Resolve the pump installed at each test's date so coef
                # trends across JPCO lines aren't geometry artifacts.
                tests = well_tests[wn].copy()
                at_noz, at_thr = [], []
                for d in tests["WtDate"]:
                    p = get_pump_at_date(jp_hist, wn, d)
                    ok = bool(p and p.get("nozzle_no") and p.get("throat_ratio"))
                    at_noz.append(p["nozzle_no"] if ok else None)
                    at_thr.append(p["throat_ratio"] if ok else None)
                tests["NozzleAtTest"] = at_noz
                tests["ThroatAtTest"] = at_thr
                n_nopump = int(tests["NozzleAtTest"].isna().sum())
                if n_nopump:
                    st.warning(
                        f"{wn}: {n_nopump} test(s) have no JP install record "
                        "covering their date — skipped"
                    )
                    tests = tests.dropna(subset=["NozzleAtTest", "ThroatAtTest"])
                if tests.empty:
                    continue

                tasks.append((wn, tests, dict(chars), dict(pump), vogel_map.get(wn)))

            if tasks:
                n_tasks = len(tasks)
                progress = st.progress(
                    0.0,
                    text=f"Calibrating {n_tasks} wells "
                         f"({workers} worker{'s' if workers > 1 else ''})",
                )
                done = 0
                if workers == 1:
                    # Sequential — call directly, no pool overhead.
                    for wn, tests, chars, pump, vrow in tasks:
                        try:
                            df = _calibrate_well(wn, tests, chars, pump, vrow)
                            st.session_state["jp_fric_trend"][wn] = df
                        except Exception as e:
                            st.error(f"{wn}: calibration loop failed: {e}")
                        done += 1
                        progress.progress(
                            done / n_tasks,
                            text=f"{wn} done ({done}/{n_tasks})",
                        )
                else:
                    # Parallel via ProcessPoolExecutor. Each worker rebuilds
                    # its own well objects (cheap vs the calibration work).
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        futs = {
                            pool.submit(_calibrate_well, wn, tests, chars, pump, vrow): wn
                            for wn, tests, chars, pump, vrow in tasks
                        }
                        for fut in as_completed(futs):
                            wn = futs[fut]
                            try:
                                df = fut.result()
                                st.session_state["jp_fric_trend"][wn] = df
                            except Exception as e:
                                st.error(f"{wn}: worker failed: {e}")
                            done += 1
                            progress.progress(
                                done / n_tasks,
                                text=f"{wn} done ({done}/{n_tasks})",
                            )
                progress.empty()
                st.success(f"Completed {len(tasks)} wells.")

    store = st.session_state["jp_fric_trend"]
    if not store:
        st.info("Pick wells and click 'Run trend calibration'.")
        return

    # Per-well detail
    st.subheader("Per-well detail")
    view_well = st.selectbox(
        "View well",
        options=list(store.keys()),
        key="jp_fric_trend_view",
    )
    if view_well:
        _render_per_well(store[view_well], view_well)

    # Cross-well rollup (only meaningful with ≥2 wells)
    combined = _combined_results()
    ok_all = combined[combined["Status"] == "ok"].copy() if not combined.empty else pd.DataFrame()
    if len(store) >= 2 and not ok_all.empty:
        st.subheader("Cross-well rollup")
        st.caption(f"{len(ok_all)} successful test calibrations across {ok_all['Well'].nunique()} wells.")

        for coef_col, coef_label in [("Ken", "ken"), ("Kth", "kth"), ("Kdi", "kdi")]:
            r1, r2 = st.columns(2)
            with r1:
                fig = px.strip(
                    ok_all, x="Nozzle", y=coef_col, color="Well",
                    hover_data=["WtDate", "Pump", "Oil"],
                    title=f"{coef_label} vs Nozzle (color = well)",
                    stripmode="overlay",
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            with r2:
                fig = px.box(
                    ok_all, x="Throat", y=coef_col,
                    title=f"{coef_label} by throat ratio (all wells)",
                    points="all",
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Per-well summary**")
        summary = (
            ok_all.groupby("Well")[["Ken", "Kth", "Kdi"]]
                  .agg(["mean", "std", "median"])
                  .round(4)
        )
        st.dataframe(summary, use_container_width=True)

    # Combined CSV download
    if not combined.empty:
        st.download_button(
            "Download combined CSV (all wells)",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name=f"jp_fric_trend_{pd.Timestamp.now():%Y_%m_%d}.csv",
            mime="text/csv",
            key="jp_fric_trend_dl",
        )
