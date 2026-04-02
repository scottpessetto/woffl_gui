"""Well Database Viewer — read-only view of jp_chars.csv."""

import streamlit as st

from woffl.gui.utils import load_well_characteristics


def run_well_database_page():
    st.title("Well Database")
    st.caption("Read-only view of jp_chars.csv — edit the CSV directly to make changes.")

    df = load_well_characteristics()
    if df.empty:
        return

    # Drop the stale unnamed index column if present
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Wells", len(df))
    col2.metric("Schrader", int(df["is_sch"].sum()) if "is_sch" in df.columns else "—")
    col3.metric(
        "Pads",
        df["Well"].str.extract(r"(MP[A-Z])")[0].nunique() if "Well" in df.columns else "—",
    )

    # Search filter
    search = st.text_input("Filter wells", placeholder="e.g. MPH, MPS, MPM-10")
    if search:
        mask = df["Well"].str.contains(search.strip(), case=False, na=False)
        df = df[mask]

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Well": st.column_config.TextColumn("Well"),
            "is_sch": st.column_config.CheckboxColumn("Schrader"),
            "out_dia": st.column_config.NumberColumn("OD (in)", format="%.3f"),
            "thick": st.column_config.NumberColumn("Wall (in)", format="%.3f"),
            "res_pres": st.column_config.NumberColumn("Res Press (psi)", format="%.0f"),
            "form_temp": st.column_config.NumberColumn("Temp (°F)", format="%.0f"),
            "JP_TVD": st.column_config.NumberColumn("JP TVD (ft)", format="%.1f"),
            "JP_MD": st.column_config.NumberColumn("JP MD (ft)", format="%.0f"),
        },
    )

    st.caption(f"Showing {len(df)} wells")
