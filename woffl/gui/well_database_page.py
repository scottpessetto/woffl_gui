"""Well Database Viewer — read-only view of mech + reservoir properties from Databricks."""

import streamlit as st

from woffl.gui.utils import load_well_characteristics


def run_well_database_page():
    st.title("Well Database")
    st.caption(
        "Live view of mpu.wells.vw_prop_mech + vw_prop_resvr (Databricks). "
        "JP_TVD computed locally from deviation surveys."
    )

    df = load_well_characteristics()
    if df.empty:
        return

    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    n_estimated = int(df["tvd_estimated"].sum()) if "tvd_estimated" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wells", len(df))
    col2.metric("Schrader", int(df["is_sch"].sum()) if "is_sch" in df.columns else "—")
    col3.metric(
        "Pads",
        df["Well"].str.extract(r"(MP[A-Z])")[0].nunique() if "Well" in df.columns else "—",
    )
    col4.metric("Estimated TVD", n_estimated, help="Wells lacking a deviation survey")

    search = st.text_input("Filter wells", placeholder="e.g. MPH, MPS, MPM-10")
    if search:
        mask = df["Well"].str.contains(search.strip(), case=False, na=False)
        df = df[mask]

    display_cols = [
        c
        for c in [
            "Well",
            "is_sch",
            "tvd_estimated",
            "out_dia",
            "thick",
            "casing_out_dia",
            "casing_inn_dia",
            "JP_MD",
            "JP_TVD",
            "res_pres",
            "form_temp",
            "oil_api",
            "gas_sg",
            "wat_sg",
            "bubble_point",
        ]
        if c in df.columns
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Well": st.column_config.TextColumn("Well"),
            "is_sch": st.column_config.CheckboxColumn("Schrader"),
            "tvd_estimated": st.column_config.CheckboxColumn(
                "TVD Est.", help="JP_TVD estimated from pad-average TVD/MD ratio"
            ),
            "out_dia": st.column_config.NumberColumn("Tbg OD (in)", format="%.3f"),
            "thick": st.column_config.NumberColumn("Tbg Wall (in)", format="%.3f"),
            "casing_out_dia": st.column_config.NumberColumn("Csg OD (in)", format="%.3f"),
            "casing_inn_dia": st.column_config.NumberColumn("Csg ID (in)", format="%.3f"),
            "JP_MD": st.column_config.NumberColumn("JP MD (ft)", format="%.0f"),
            "JP_TVD": st.column_config.NumberColumn("JP TVD (ft)", format="%.1f"),
            "res_pres": st.column_config.NumberColumn("Res Press (psi)", format="%.0f"),
            "form_temp": st.column_config.NumberColumn("Res Temp (°F)", format="%.0f"),
            "oil_api": st.column_config.NumberColumn("Oil API", format="%.1f"),
            "gas_sg": st.column_config.NumberColumn("Gas SG", format="%.3f"),
            "wat_sg": st.column_config.NumberColumn("Water SG", format="%.3f"),
            "bubble_point": st.column_config.NumberColumn("Pb (psi)", format="%.0f"),
        },
    )

    st.caption(f"Showing {len(df)} wells")
