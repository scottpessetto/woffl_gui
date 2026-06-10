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
        fig = _create_history_chart(
            well_name, test_df, well_jp,
            bhp_daily_df=bhp_daily_df,
            bhp_overlay_df=bhp_overlay_df,
            bhp_from_zero=bhp_zero,
        )
        st.plotly_chart(fig, use_container_width=True)
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
    round(vbdc.bhp_cln_value, 2) AS bhp
FROM mpu.wells.vw_well_test vwt
LEFT JOIN mpu.wells.vw_bhp_daily_clean vbdc
    ON vwt.enthid = vbdc.enthid
    AND to_date(vwt.wt_date) = vbdc.tag_date
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

    for col in ["BHP", "WtOilVol", "WtWaterVol", "lift_wat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    """Format nozzle + throat into a string like '12B'."""
    parts = ""
    if pd.notna(nozzle):
        parts += str(int(nozzle))
    if pd.notna(throat):
        parts += str(throat).strip()
    return parts or "?"


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
