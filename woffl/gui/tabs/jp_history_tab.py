"""Tab 5: JP History & Production Chart

Shows jet pump change history overlaid on well test production data.
Fetches well tests going back as far as the JP history for the selected well.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.gui.params import SimulationParams


def render_tab(params: SimulationParams) -> None:
    """Render the JP History & Production tab."""
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        st.info("Select a well with JP history to view this tab.")
        return

    well_name = params.selected_well

    # Get all JP changes for this well, sorted by date
    well_jp = jp_hist[jp_hist["Well Name"] == well_name].copy()
    well_jp = well_jp.dropna(subset=["Date Set"])
    well_jp = well_jp.sort_values("Date Set").reset_index(drop=True)

    if well_jp.empty:
        st.info(f"No JP history found for {well_name}.")
        return

    earliest_date = well_jp["Date Set"].min()

    st.subheader(f"JP History & Production — {well_name}")
    st.caption(
        f"{len(well_jp)} pump installations on record, "
        f"earliest: {earliest_date.strftime('%b %d, %Y')}"
    )

    # Fetch well tests back to earliest JP change
    test_df, bhp_daily_df, bhp_overlay_df = _fetch_extended_well_tests(
        well_name, earliest_date
    )

    if test_df is not None and not test_df.empty:
        bhp_zero = st.checkbox(
            "BHP axis starts at 0", value=True, key="jp_hist_bhp_zero"
        )
        # Same combined figure as the Solver's pump-history strip — chart on
        # top, pumps-in-hole timeline below, structurally aligned.
        fig, tl = build_history_with_strip_figure(
            well_name, well_jp, test_df, bhp_daily_df, bhp_overlay_df,
            bhp_from_zero=bhp_zero, height=620,
            title=f"{well_name} — Production & JP Change History",
        )
        st.plotly_chart(fig, use_container_width=True)
        render_current_pump_caption(tl, n_installs=len(well_jp))

        _render_report_card(well_name, well_jp, test_df, bhp_daily_df)
    else:
        st.warning(
            "Could not fetch well test data from Databricks for this date range."
        )

    # Always show JP changes table
    _show_jp_table(well_jp)


_EXTENDED_TEST_QUERY = """\
SELECT
    vwt.well_name,
    vwt.wt_date,
    vwt.form_oil AS oil_rate,
    vwt.form_wat AS fwat_rate,
    vwt.lift_wat,
    round(vbdc.bhp_cln_value, 2) AS bhp,
    vpd.tubing_prs AS pf_tubing_prs,
    vpd.inn_ann_prs AS pf_inn_ann_prs
FROM mpu.wells.vw_well_test vwt
LEFT JOIN mpu.wells.vw_bhp_daily_clean vbdc
    ON vwt.enthid = vbdc.enthid
    AND to_date(vwt.wt_date) = vbdc.tag_date
LEFT JOIN (
    -- Test-day PF pressure (same aggregated join as well_test_client) —
    -- feeds the Pump Report Card's per-era PF context and the chart overlay.
    SELECT
        enthid,
        sample_date,
        max(tubing_prs) AS tubing_prs,
        max(inn_ann_prs) AS inn_ann_prs
    FROM mpu.wells.vw_pressure_daily
    GROUP BY enthid, sample_date
) vpd
    ON vwt.enthid = vpd.enthid
    AND to_date(vwt.wt_date) = vpd.sample_date
WHERE vwt.well_name = '{well_name}'
    AND vwt.wt_date BETWEEN '{start_date}' AND '{end_date}'
    AND vwt.allocated = True
ORDER BY vwt.wt_date
"""

_BHP_DAILY_QUERY = """\
SELECT
    vbdc.tag_date,
    round(vbdc.bhp_cln_value, 2) AS bhp
FROM mpu.wells.vw_bhp_daily_clean vbdc
WHERE vbdc.enthid = (
    SELECT DISTINCT enthid
    FROM mpu.wells.vw_well_test
    WHERE well_name = '{well_name}'
    LIMIT 1
)
AND vbdc.tag_date BETWEEN '{start_date}' AND '{end_date}'
ORDER BY vbdc.tag_date
"""


@st.cache_data(ttl=86400, show_spinner=False, max_entries=128)
def _cached_extended_tests(db_name: str, start_date: str, end_date: str):
    """Fetch well tests for extended date range without requiring BHP. Cached 24h."""
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(
        _EXTENDED_TEST_QUERY.format(
            well_name=db_name, start_date=start_date, end_date=end_date
        )
    )
    if df.empty:
        return df

    rename = {
        "well_name": "well",
        "wt_date": "WtDate",
        "bhp": "BHP",
        "oil_rate": "WtOilVol",
        "fwat_rate": "WtWaterVol",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "well" in df.columns:
        df["well"] = df["well"].apply(_normalize_well_name)
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], utc=True).dt.tz_localize(None)

    for col in ["BHP", "WtOilVol", "WtWaterVol", "lift_wat", "pf_tubing_prs", "pf_inn_ann_prs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resolve test-day PF (annulus vs tubing per circulation direction).
    from woffl.assembly.pf_pressure import add_pf_columns

    df = add_pf_columns(df)

    # Only require date + at least one rate — NOT BHP
    df = df.dropna(subset=["WtDate"])
    return df.sort_values("WtDate")


@st.cache_data(ttl=86400, show_spinner=False, max_entries=128)
def _cached_bhp_daily(db_name: str, start_date: str, end_date: str):
    """Fetch daily BHP for all dates (not just well test dates). Cached 24h."""
    from woffl.assembly.databricks_client import execute_query

    df = execute_query(
        _BHP_DAILY_QUERY.format(
            well_name=db_name, start_date=start_date, end_date=end_date
        )
    )
    if df.empty:
        return df

    if "tag_date" in df.columns:
        df["tag_date"] = pd.to_datetime(df["tag_date"], utc=True).dt.tz_localize(None)
    if "bhp" in df.columns:
        df["bhp"] = pd.to_numeric(df["bhp"], errors="coerce")

    return df.dropna(subset=["tag_date", "bhp"]).sort_values("tag_date")


def _fetch_extended_well_tests(
    well_name: str, earliest_date
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch well tests + the BHP series the chart should plot.

    Returns ``(test_df, primary_bhp_df, overlay_bhp_df)``. The three slots:
      * ``test_df`` — well tests (with gauge BHP merged when a gauge is
        active, so test-date scatter lines up with the daily line).
      * ``primary_bhp_df`` — main BHP trace: the memory gauge when one is
        uploaded, otherwise the Databricks daily feed.
      * ``overlay_bhp_df`` — secondary trace shown only when the user has
        BOTH uploaded a gauge AND disregarded Databricks (so the user can
        see the divergence the auto-detection caught). Plotted in a muted
        style and labeled "disregarded" so it doesn't suggest the chart
        is using it.
    """
    from datetime import datetime

    from woffl.assembly.well_test_client import _denormalize_well_name
    from woffl.gui.memory_gauge import (
        apply_to_well_tests,
        daily_bhp_from_gauge,
        get_gauge,
        is_disregarding_databricks_bhp,
    )

    db_name = _denormalize_well_name(well_name)
    start = earliest_date.strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    test_df = None
    primary_bhp_df = None
    overlay_bhp_df = None
    gauge = get_gauge(well_name)
    disregard = is_disregarding_databricks_bhp(well_name)

    # The daily-BHP series is needed either as the primary trace (no gauge,
    # not disregarded) or as the disregarded overlay (gauge + disregard).
    # It's independent of the tests query, so fetch it CONCURRENTLY — a cold
    # first view costs the slower query instead of the two back-to-back.
    need_daily = (gauge is None and not disregard) or (gauge is not None and disregard)
    daily_box: dict = {}
    daily_thread = None
    if need_daily:
        import threading

        from streamlit.runtime.scriptrunner import add_script_run_ctx

        def _daily_worker() -> None:
            try:
                daily_box["df"] = _cached_bhp_daily(db_name, start, end)
            except Exception:
                daily_box["df"] = None

        daily_thread = threading.Thread(
            target=_daily_worker, daemon=True, name=f"bhp-daily-{well_name}"
        )
        add_script_run_ctx(daily_thread)
        daily_thread.start()

    with st.spinner(f"Fetching tests for {well_name} ({start} to {end})..."):
        try:
            test_df = _cached_extended_tests(db_name, start, end)
            if test_df is not None and not test_df.empty:
                if disregard and "BHP" in test_df.columns:
                    # np.nan (not pd.NA) keeps the column float64 so the
                    # downstream chart and BHP-scatter trace work.
                    import numpy as _np
                    test_df["BHP"] = _np.nan
                if gauge is not None:
                    test_df = apply_to_well_tests(test_df, gauge)
        except Exception as e:
            st.warning(f"Databricks query failed: {e}")

        if daily_thread is not None:
            daily_thread.join()
        daily_df = daily_box.get("df")

        if gauge is not None:
            primary_bhp_df = daily_bhp_from_gauge(gauge)
            # Show Databricks alongside the gauge when the user has flagged
            # it as bad — visualizes the divergence the auto-detect caught.
            if disregard:
                overlay_bhp_df = daily_df
        elif not disregard:
            # No gauge AND not disregarding — normal Databricks path.
            primary_bhp_df = daily_df
        # else: disregard only (no gauge) → no BHP shown. User has said
        # the Databricks data is bad and hasn't given us a replacement,
        # so showing nothing is more honest than showing the bad line.

    return test_df, primary_bhp_df, overlay_bhp_df


def _format_jp(nozzle, throat) -> str:
    """Format nozzle + throat into a string like '12B'.

    Delegates to :func:`woffl.assembly.pump_report.format_pump` — the single
    tolerant implementation (float nozzles, S-17's throat-only rows) shared
    with the Pump Report Card's era builder so the chart labels and the era
    table can never disagree.
    """
    from woffl.assembly.pump_report import format_pump

    return format_pump(nozzle, throat)


def _create_history_chart(
    well_name: str,
    test_df: pd.DataFrame,
    jp_changes: pd.DataFrame,
    bhp_daily_df: pd.DataFrame | None = None,
    bhp_overlay_df: pd.DataFrame | None = None,
    bhp_from_zero: bool = False,
) -> go.Figure:
    """Create interactive Plotly chart with stacked production, BHP, and JP change lines."""
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    test_df = test_df.sort_values("WtDate")

    # Oil rate (bottom of stack)
    if "WtOilVol" in test_df.columns:
        fig.add_trace(
            go.Scatter(
                x=test_df["WtDate"],
                y=test_df["WtOilVol"],
                name="Oil (BOPD)",
                mode="lines",
                line=dict(color="#2E7D32", width=1.5),
                fillcolor="rgba(46,125,50,0.4)",
                stackgroup="production",
                hovertemplate="%{x|%Y-%m-%d}<br>Oil: %{y:.0f} BOPD<extra></extra>",
            ),
            secondary_y=False,
        )

    # Formation water (stacked on top of oil)
    if "WtWaterVol" in test_df.columns:
        fig.add_trace(
            go.Scatter(
                x=test_df["WtDate"],
                y=test_df["WtWaterVol"],
                name="Form Water (BWPD)",
                mode="lines",
                line=dict(color="#1565C0", width=1.5),
                fillcolor="rgba(21,101,192,0.3)",
                stackgroup="production",
                hovertemplate="%{x|%Y-%m-%d}<br>Water: %{y:.0f} BWPD<extra></extra>",
            ),
            secondary_y=False,
        )

    # BHP on secondary y-axis — prefer daily BHP, fall back to test-date BHP.
    # Memory-gauge daily BHP is rendered in the same orange to keep the chart
    # readable, but the legend names it explicitly so the user always knows
    # which data source they're looking at.
    from woffl.gui.memory_gauge import has_gauge

    bhp_legend_name = (
        "BHP (Memory Gauge, psi)" if has_gauge(well_name) else "BHP (psi)"
    )
    # Primary BHP trace: prefer the daily series; fall back to the test-date
    # scatter ONLY when no daily BHP is available. (The old code used an elif
    # on the overlay block below, which let the test-date scatter draw on top
    # of the daily line whenever no overlay was set — producing two BHP traces.)
    if bhp_daily_df is not None and not bhp_daily_df.empty:
        fig.add_trace(
            go.Scatter(
                x=bhp_daily_df["tag_date"],
                y=bhp_daily_df["bhp"],
                name=bhp_legend_name,
                mode="lines",
                line=dict(color="#E65100", width=1.5),
                hovertemplate="%{x|%Y-%m-%d}<br>BHP: %{y:.0f} psi<extra></extra>",
            ),
            secondary_y=True,
        )
    elif "BHP" in test_df.columns:
        bhp_data = test_df.dropna(subset=["BHP"])
        if not bhp_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=bhp_data["WtDate"],
                    y=bhp_data["BHP"],
                    name="BHP (psi)",
                    mode="lines+markers",
                    line=dict(color="#E65100", width=2),
                    marker=dict(size=4),
                    hovertemplate="%{x|%Y-%m-%d}<br>BHP: %{y:.0f} psi<extra></extra>",
                ),
                secondary_y=True,
            )

    # Test-day power-fluid pressure (from vw_pressure_daily) on the pressure
    # axis — PF context for the Pump Report Card below the chart.
    if "pf_press" in test_df.columns:
        pf_data = test_df.dropna(subset=["pf_press"])
        if not pf_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=pf_data["WtDate"],
                    y=pf_data["pf_press"],
                    name="PF pressure (psi)",
                    mode="lines+markers",
                    line=dict(color="#6A1B9A", width=1, dash="dot"),
                    marker=dict(size=3),
                    hovertemplate="%{x|%Y-%m-%d}<br>PF: %{y:.0f} psi<extra></extra>",
                ),
                secondary_y=True,
            )

    # Overlay: disregarded Databricks BHP, plotted in a muted dashed gray so the
    # user can see the divergence that prompted the auto-disregard without
    # thinking the chart is using that data. Only fires when both a gauge is
    # active AND the disregard flag is on — see _fetch_extended_well_tests.
    if bhp_overlay_df is not None and not bhp_overlay_df.empty:
        fig.add_trace(
            go.Scatter(
                x=bhp_overlay_df["tag_date"],
                y=bhp_overlay_df["bhp"],
                name="Databricks BHP (disregarded)",
                mode="lines",
                line=dict(color="#9E9E9E", width=1, dash="dash"),
                opacity=0.7,
                hovertemplate="%{x|%Y-%m-%d}<br>Databricks BHP: %{y:.0f} psi<extra></extra>",
            ),
            secondary_y=True,
        )

    # Vertical lines for each JP change with JPCO labels
    for idx in range(len(jp_changes)):
        row = jp_changes.iloc[idx]
        date = row["Date Set"]
        date_str = date.isoformat()
        new_jp = _format_jp(row.get("Nozzle Number"), row.get("Throat Ratio"))

        if idx == 0:
            label = f"Set {new_jp}"
        else:
            prev = jp_changes.iloc[idx - 1]
            old_jp = _format_jp(prev.get("Nozzle Number"), prev.get("Throat Ratio"))
            if old_jp == new_jp:
                label = f"JPCO {new_jp} (same)"
            else:
                label = f"JPCO {old_jp} to {new_jp}"

        # Alternate y-position to reduce label overlap
        y_frac = 0.95 if idx % 2 == 0 else 0.85

        fig.add_shape(
            type="line",
            x0=date_str,
            x1=date_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(dash="dash", color="rgba(211,47,47,0.7)", width=1.5),
        )
        fig.add_annotation(
            x=date_str,
            y=y_frac,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-90,
            xshift=-7,
            font=dict(size=10, color="#D32F2F"),
        )

    fig.update_layout(
        title=f"{well_name} — Production & JP Change History",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )
    fig.update_yaxes(title_text="Rate (BPD)", secondary_y=False)
    if bhp_from_zero:
        fig.update_yaxes(
            title_text="BHP (psi)", rangemode="tozero", showgrid=False, secondary_y=True
        )
    else:
        fig.update_yaxes(title_text="BHP (psi)", showgrid=False, secondary_y=True)

    return fig


def build_history_with_strip_figure(
    well_name: str,
    well_jp: pd.DataFrame,
    test_df: pd.DataFrame | None,
    bhp_daily_df: pd.DataFrame | None,
    bhp_overlay_df: pd.DataFrame | None,
    *,
    bhp_from_zero: bool = True,
    height: int = 470,
    title: str = "",
) -> tuple[go.Figure, pd.DataFrame | None]:
    """Production/BHP/JPCO chart with the pumps-in-hole timeline strip below.

    THE shared figure for the JP History tab and the Solver's pump-history
    strip — one builder so the two can never drift (Scott: "the plot on JP
    history should match the plot on the single well solver with pumps in
    hole below the plot").

    One figure, two stacked subplots sharing the x-axis — STRUCTURAL
    alignment: the same pixel maps to the same date in both rows, so the
    JPCO dashed lines pass straight through the matching color boundaries in
    the strip. Tenure = Date Set → next Date Set (JPCO same-day rule; the
    tracker's Date Pulled produces phantom gaps).

    Returns ``(figure, timeline_df_or_None)`` — the timeline frame (Pump /
    Start / End / Days / Pulled) feeds the callers' current-pump caption.
    ``well_jp`` empty → plain history chart, no strip, timeline None.
    """
    import plotly.express as px
    from plotly.subplots import make_subplots

    well_jp = (
        well_jp.dropna(subset=["Date Set"]).sort_values("Date Set")
        if well_jp is not None and not well_jp.empty
        else pd.DataFrame(columns=["Date Set"])
    )

    have_chart = test_df is not None and not test_df.empty

    if well_jp.empty:
        # No installs on record — the plain history chart is all there is.
        fig = _create_history_chart(
            well_name,
            test_df if have_chart else pd.DataFrame(),
            well_jp,
            bhp_daily_df=bhp_daily_df,
            bhp_overlay_df=bhp_overlay_df,
            bhp_from_zero=bhp_from_zero,
        )
        if title:
            fig.update_layout(title=title)
        return fig, None

    earliest_date = well_jp["Date Set"].min()
    today = pd.Timestamp.today().normalize()
    # Shared x-range so the colored install segments below line up vertically
    # with the JPCO dashed lines in the chart above.
    x_range = [
        earliest_date - pd.Timedelta(days=15),
        today + pd.Timedelta(days=15),
    ]

    # Tenure segments (set-to-set; last install runs to today).
    rows = []
    set_dates = list(well_jp["Date Set"])
    for i, (_, r) in enumerate(well_jp.iterrows()):
        start = r["Date Set"]
        is_last = i == len(set_dates) - 1
        end = today if is_last else set_dates[i + 1]
        if end <= start:  # zero/negative spans render invisibly
            end = start + pd.Timedelta(days=1)
        rows.append(
            {
                "Pump": _format_jp(r.get("Nozzle Number"), r.get("Throat Ratio")),
                "Start": start,
                "End": end,
                "Days": int((end - start).days),
                "Pulled": "in hole" if is_last else end.strftime("%Y-%m-%d"),
            }
        )
    tl = pd.DataFrame(rows)

    palette = px.colors.qualitative.Set2
    pump_order = list(dict.fromkeys(tl["Pump"]))
    color_of = {p: palette[i % len(palette)] for i, p in enumerate(pump_order)}

    if have_chart:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.84, 0.16],
            vertical_spacing=0.06,
            specs=[[{"secondary_y": True}], [{}]],
        )
        base = _create_history_chart(
            well_name,
            test_df,
            well_jp,
            bhp_daily_df=bhp_daily_df,
            bhp_overlay_df=bhp_overlay_df,
            bhp_from_zero=bhp_from_zero,
        )
        for tr in base.data:
            fig.add_trace(
                tr, row=1, col=1,
                secondary_y=(getattr(tr, "yaxis", "y") == "y2"),
            )
        # JPCO dashed lines + labels are paper-referenced — re-added on the
        # combined figure they span BOTH rows, visually tying each change
        # line to its segment boundary in the strip.
        for shp in base.layout.shapes:
            fig.add_shape(shp)
        for ann in base.layout.annotations:
            fig.add_annotation(ann)
        strip_xref, strip_yref = "x2", "y3 domain"
        strip_target = dict(row=2, col=1)
    else:
        fig = go.Figure()
        strip_xref, strip_yref = "x", "y domain"
        strip_target = {}

    # Strip segments as RECTANGLE SHAPES in real date coordinates — the
    # px.timeline/go.Bar encoding (numeric millisecond lengths + datetime
    # base) breaks inside subplots because the strip's axis infers a linear
    # type from the numeric lengths. Shapes carry actual dates; nothing to
    # mis-infer.
    span = x_range[1] - x_range[0]
    mids, labels, hover_custom = [], [], []
    for _, seg in tl.iterrows():
        fig.add_shape(
            type="rect",
            x0=seg["Start"],
            x1=seg["End"],
            y0=0.06,
            y1=0.94,
            xref=strip_xref,
            yref=strip_yref,
            fillcolor=color_of[seg["Pump"]],
            line=dict(width=1, color="white"),
            # below traces AND below the above-layer JPCO shapes, so the
            # dashed change lines visibly cross the strip
            layer="below",
        )
        mids.append(seg["Start"] + (seg["End"] - seg["Start"]) / 2)
        # Only label segments wide enough to carry text — slivers (same-day
        # JPCOs) keep their hover but stay unlabeled to avoid clutter.
        wide_enough = (seg["End"] - seg["Start"]) / span > 0.03
        labels.append(seg["Pump"] if wide_enough else "")
        hover_custom.append(
            [seg["Pump"], seg["Start"].strftime("%Y-%m-%d"), seg["Pulled"], seg["Days"]]
        )

    # Invisible markers + text at segment midpoints provide labels and hover
    # (shapes themselves don't hover). Datetime x keeps the strip axis typed
    # as a date axis.
    fig.add_trace(
        go.Scatter(
            x=mids,
            y=[0.5] * len(mids),
            mode="markers+text",
            marker=dict(size=14, opacity=0),
            text=labels,
            textposition="middle center",
            textfont=dict(size=12, color="#1a1a1a"),
            customdata=hover_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Set %{customdata[1]} → %{customdata[2]}<br>"
                "%{customdata[3]:,} days<extra></extra>"
            ),
            showlegend=False,
        ),
        **strip_target,
    )

    if have_chart:
        fig.update_layout(
            height=height,
            title_text=title,
            hovermode="x unified",
            margin=dict(t=60 if title else 30, b=8, l=65, r=65),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # Months render under the CHART row; the strip row has no axis of
        # its own (shared_xaxes hides upper ticks by default — re-enable).
        fig.update_xaxes(
            range=x_range, row=1, col=1, showticklabels=True, title_text=""
        )
        fig.update_xaxes(range=x_range, row=2, col=1, visible=False)
        fig.update_yaxes(title_text="Rate (BPD)", row=1, col=1, secondary_y=False)
        if bhp_from_zero:
            fig.update_yaxes(
                title_text="BHP (psi)", rangemode="tozero", showgrid=False,
                row=1, col=1, secondary_y=True,
            )
        else:
            fig.update_yaxes(
                title_text="BHP (psi)", showgrid=False,
                row=1, col=1, secondary_y=True,
            )
        fig.update_yaxes(
            visible=False, fixedrange=True, range=[0, 1], row=2, col=1
        )
    else:
        # Fallback when the test/BHP fetch failed: standalone strip with its
        # own date axis so it still has a time reference.
        fig.update_layout(
            height=110,
            margin=dict(l=65, r=65, t=4, b=4),
            showlegend=False,
            yaxis=dict(visible=False, fixedrange=True, range=[0, 1]),
            xaxis=dict(side="bottom", fixedrange=True, range=x_range),
            plot_bgcolor="rgba(0,0,0,0)",
        )

    return fig, tl


def render_current_pump_caption(tl: pd.DataFrame | None, n_installs: int | None = None) -> None:
    """One-line 'current pump X — in hole since …' caption under the chart."""
    if tl is None or tl.empty:
        return
    cur = tl.iloc[-1]
    if cur["Pulled"] == "in hole":
        st.caption(
            f"Current pump **{cur['Pump']}** — in hole since "
            f"{cur['Start'].strftime('%Y-%m-%d')} ({cur['Days']:,} days). "
            f"{n_installs if n_installs is not None else len(tl):,} install(s) on record."
        )


def _render_report_card(
    well_name: str,
    well_jp: pd.DataFrame,
    test_df: pd.DataFrame,
    bhp_daily_df: pd.DataFrame | None,
) -> None:
    """Pump Report Card — per-era metrics table + 'historically best' verdict.

    Pure analysis lives in woffl.assembly.pump_report; this renders it. The
    two 'good' thresholds are visible controls seeded from the well's own
    history (75% of P75 oil; median daily BHP) and editable — time-in-good-
    state is an engineering judgment, not a constant.
    """
    from woffl.assembly.pump_report import build_report, default_good_thresholds

    st.subheader("Pump Report Card")

    oil_seed, bhp_seed = default_good_thresholds(test_df, bhp_daily_df)
    c1, c2 = st.columns(2)
    good_oil = c1.number_input(
        "Good test: oil ≥ (BOPD)",
        min_value=0,
        max_value=5000,
        value=int(oil_seed or 100),
        step=10,
        key=f"prc_good_oil_{well_name}",
        help="Seeded at 75% of this well's P75 oil rate over the window.",
    )
    if bhp_seed is not None:
        good_bhp = c2.number_input(
            "Good BHP ≤ (psi)",
            min_value=0,
            max_value=5000,
            value=int(bhp_seed),
            step=25,
            key=f"prc_good_bhp_{well_name}",
            help="Seeded at the median of the daily BHP series — lower BHP = "
            "better drawdown. Counted on the DAILY series, not just test days.",
        )
    else:
        good_bhp = None
        c2.caption("No BHP data for this well — good-state uses oil only.")

    ranked, verdict = build_report(
        well_jp, test_df, bhp_daily_df, float(good_oil),
        float(good_bhp) if good_bhp is not None else None,
    )
    if not ranked:
        st.info("Not enough test history to build a report card.")
        return

    if verdict is not None:
        best = verdict["best"]
        st.success(
            f"🏆 **Historically, {well_name} has performed best on a "
            f"{best['pump']} pump** (score {best['score']:.0f}/100) — "
            + "; ".join(verdict["reasons"])
            + "."
        )
        for c in verdict["caveats"]:
            st.warning(f"⚠️ {c}")
    else:
        st.info(
            "No era has enough tests to rank (min 3) — table below shows "
            "what's on record."
        )

    rows = []
    for m in ranked:
        rows.append(
            {
                "Pump": m["pump"] + (" (in hole)" if m.get("active") else ""),
                "Era": f"{m['start']:%Y-%m-%d} → "
                + ("today" if m.get("active") else f"{m['end']:%Y-%m-%d}"),
                "Days": m["days"],
                "Sets": m["installs"],
                "Tests": m["n_tests"],
                "Med Oil": m["med_oil"],
                "P90 Oil": m["p90_oil"],
                "Med WC": m["med_wc"],
                "ΔWC": m["wc_drift"],
                "Med PF": m["med_pf"],
                "Oil/PF bbl": m["oil_per_pf"],
                "Med BHP": m["med_bhp"],
                "Good tests %": (
                    m["good_test_frac"] * 100
                    if m["good_test_frac"] is not None
                    else None
                ),
                "Good run (d)": m["good_run_days"],
                "Good BHP %": (
                    m["good_bhp_frac"] * 100
                    if m["good_bhp_frac"] is not None
                    else None
                ),
                "Score": m["score"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Med Oil": st.column_config.NumberColumn(format="%.0f"),
            "P90 Oil": st.column_config.NumberColumn(format="%.0f"),
            "Med WC": st.column_config.NumberColumn(format="%.2f"),
            "ΔWC": st.column_config.NumberColumn(
                format="%+.2f", help="WC drift within the era (last vs first third)"
            ),
            "Med PF": st.column_config.NumberColumn(format="%.0f"),
            "Oil/PF bbl": st.column_config.NumberColumn(format="%.2f"),
            "Med BHP": st.column_config.NumberColumn(format="%.0f"),
            "Good tests %": st.column_config.NumberColumn(format="%.0f%%"),
            "Good BHP %": st.column_config.NumberColumn(
                format="%.0f%%",
                help="Share of era DAYS (daily BHP series) at/below the target",
            ),
            "Score": st.column_config.NumberColumn(format="%.0f"),
            "Sets": st.column_config.NumberColumn(
                help="Slickline installs within the era — same pump re-set "
                "(JPCO churn / wash-out signal)"
            ),
        },
    )
    st.caption(
        "Blank PF/BHP cells = era pre-dates daily pressure/BHP coverage "
        "(oil-only comparison). Eras with < 3 tests are shown but not scored. "
        "Score = 40% oil, 25% time-in-good-state, 20% longevity (capped 1 yr), "
        "15% oil-per-PF efficiency, renormalized over available components."
    )


def _show_jp_table(well_jp: pd.DataFrame) -> None:
    """Display the JP change history as a table."""
    display_cols = {
        "Date Set": "Date Set",
        "Date Pulled": "Date Pulled",
        "Nozzle Number": "Nozzle",
        "Throat Ratio": "Throat",
        "Tubing Diameter": "Tubing OD",
    }
    available = [c for c in display_cols if c in well_jp.columns]
    table = well_jp[available].copy()
    table = table.rename(columns=display_cols)

    for col in ["Date Set", "Date Pulled"]:
        if col in table.columns:
            table[col] = pd.to_datetime(table[col]).dt.strftime("%Y-%m-%d")

    table = table.sort_values("Date Set", ascending=False).reset_index(drop=True)

    with st.expander(f"JP Change History ({len(table)} records)", expanded=False):
        st.dataframe(table, use_container_width=True, hide_index=True)
