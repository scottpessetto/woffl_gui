"""Pad Water Cut tab.

Daily pad-level WC over time for pads G/H/I/J. Each well's last allocated
test is forward-filled; well-days with >6 h shut-in are excluded. H and I
treated as on-pad PF recycle, G and J ship lift water back to the plant.
"""

import datetime as _dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=3600, show_spinner="Building pad water-cut series...")
def _cached_pad_watercut(start_date: str, end_date: str) -> pd.DataFrame:
    from woffl.assembly.pad_watercut_client import fetch_pad_watercut
    return fetch_pad_watercut(start_date, end_date)


def render_tab() -> None:
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

    # Explicit load gate: st.tabs executes every tool's body on every rerun,
    # so an unconditional fetch here fired the 3-year, 2-query pull the
    # moment anyone opened Scott's Tools for a DIFFERENT tool.
    if not st.session_state.get("pad_wc_loaded"):
        if st.button("Load pad water-cut history", type="primary", key="pad_wc_load"):
            st.session_state["pad_wc_loaded"] = True
            st.rerun()
        st.info(
            "Click to query the pad water-cut history for the selected window "
            "(cached 1 h once loaded)."
        )
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
