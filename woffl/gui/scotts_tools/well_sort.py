"""Well Sort tab.

Online vs Shut-In classification + outlier flagging across MPU producers.
Uses live ProdXV status to rescue just-restarted wells from the lagging
shut-in log.
"""

import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner="Loading shut-in history from Databricks...")
def _cached_shut_in_history() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_current_shut_in_history
    return fetch_current_shut_in_history()


@st.cache_data(ttl=3600, show_spinner="Loading well tests from Databricks...")
def _cached_recent_tests(days: int) -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_recent_tests
    return fetch_recent_tests(days=days)


@st.cache_data(ttl=3600, show_spinner="Loading producer list...")
def _cached_producers() -> list[str]:
    from woffl.assembly.well_sort_client import fetch_mpu_producers
    return fetch_mpu_producers()


@st.cache_data(ttl=3600, show_spinner="Loading producer catalog...")
def _cached_producer_catalog() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_producer_catalog
    return fetch_producer_catalog()


@st.cache_data(ttl=3600, show_spinner="Loading all-time last tests...")
def _cached_last_tests_ever() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_last_tests_ever
    return fetch_last_tests_ever()


@st.cache_data(ttl=300, show_spinner="Loading safety-valve status...")
def _cached_xv_status() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_xv_status
    return fetch_xv_status()


def render_tab() -> None:
    from woffl.assembly.well_sort_client import (
        apply_pops_pad,
        build_online_table,
        build_shut_in_table,
        classify_wells,
        export_bench_xlsx,
        split_offline_ltsi,
    )

    st.header("Well Sort")
    st.caption(
        "Online = not in vw_shut_in, OR in vw_shut_in but live ProdXV = open "
        "(daily log lags restarts up to 24 h). Wells absent from the log stay "
        "online regardless of XV — handles flowback edge cases like H-31. "
        "Outlier = |test − 2-mo avg| > 25% on oil or water."
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.8, 1.5, 1.0])
    with ctrl1:
        if st.button("Refresh data", help="Clear cache and re-query Databricks"):
            _cached_shut_in_history.clear()
            _cached_recent_tests.clear()
            _cached_producers.clear()
            _cached_producer_catalog.clear()
            _cached_last_tests_ever.clear()
            _cached_xv_status.clear()
            st.rerun()
    with ctrl2:
        mode_label = st.radio(
            "Display test",
            ["Most recent allocated", "Most recent (any)"],
            horizontal=True,
            key="well_sort_mode",
        )
    with ctrl3:
        stale_days = st.slider(
            "Stale-test threshold (days)", 14, 180, 60,
            key="well_sort_stale_days",
            help="Flag wells whose most-recent test is older than this.",
        )
    with ctrl4:
        st.write("")
        st.write("")
        st.caption("Tests window: 180 d")

    mode = "allocated" if mode_label.startswith("Most recent allocated") else "any"

    shut_in_hist = _cached_shut_in_history()
    tests = _cached_recent_tests(180)
    producers = _cached_producers()
    catalog = _cached_producer_catalog()
    last_tests = _cached_last_tests_ever()
    xv = _cached_xv_status()

    if not producers:
        st.error("No producers returned from vw_well_header.")
        return

    # On-pad separation settings (persist across interactions)
    all_pads = sorted(catalog["well_pad"].dropna().unique().tolist()) if not catalog.empty else []
    pops_pads = st.multiselect(
        "Pads with on-pad production separation",
        options=all_pads,
        default=st.session_state.get(
            "well_sort_pops_pads", ["E", "F", "H", "I", "M", "S"]
        ),
        key="well_sort_pops_pads",
        help="Wells on these pads get PopsPad=True. Per-well overrides apply after.",
    )
    # Per-well PopsPad=True overrides (wells that get True even if their pad
    # doesn't have separation — e.g. MPS-08 in the Apr-20 bench sheet).
    force_true_wells = st.multiselect(
        "Per-well PopsPad=True overrides",
        options=sorted(producers),
        default=st.session_state.get("well_sort_pops_force_true", []),
        key="well_sort_pops_force_true",
        help="These wells are treated as having on-pad separation regardless "
        "of the pad-level setting above.",
    )
    overrides = {w: True for w in force_true_wells}

    online_set, shut_set = classify_wells(
        producers, shut_in_hist, xv_df=xv, trust_xv=True
    )
    online_df = build_online_table(
        tests, shut_in_hist, producers, mode=mode,
        stale_days=stale_days, xv_df=xv, online_wells=online_set,
        catalog_df=catalog,
    )
    shut_df = build_shut_in_table(
        shut_in_hist, tests, xv_df=xv, shut_in_wells=shut_set,
        catalog_df=catalog, last_tests_df=last_tests,
    )
    online_df = apply_pops_pad(online_df, set(pops_pads), overrides)
    shut_df = apply_pops_pad(shut_df, set(pops_pads), overrides)
    offline_df, ltsi_df = split_offline_ltsi(shut_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Online Wells", len(online_df))
    m2.metric("Shut-In Wells", len(shut_df))
    m3.metric(
        "Outliers flagged",
        int(online_df["FlagOutlier"].sum()) if not online_df.empty else 0,
        help="|test − 2-month avg| > 25% on oil or water",
    )
    just_restarted = int(online_df["JustRestarted"].sum()) if not online_df.empty else 0
    m4.metric(
        "Just restarted",
        just_restarted,
        help="Wells in vw_shut_in but ProdXV shows open — rescued to online by "
        "live XV. Reflects the 12-24 h lag in the daily shut-in log.",
    )

    # Bench export (matches MPU_Well_Bench_YYYY_MM_DD.xlsx layout)
    from datetime import date
    xlsx_bytes = export_bench_xlsx(online_df, offline_df, ltsi_df)
    st.download_button(
        f"Download bench xlsx  ({len(online_df)} online, {len(offline_df)} offline, "
        f"{len(ltsi_df)} ltsi)",
        data=xlsx_bytes,
        file_name=f"MPU_Well_Bench_{date.today():%Y_%m_%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="well_sort_dl_bench",
    )

    sub_online, sub_off, sub_ltsi = st.tabs(
        [f"Online ({len(online_df)})",
         f"Offline ({len(offline_df)})",
         f"LTSI ({len(ltsi_df)})"]
    )

    with sub_online:
        if online_df.empty:
            st.info("No online wells with recent tests.")
        else:
            display = online_df.copy()
            display["OilDev"] = display["OilDev"] * 100
            display["WatDev"] = display["WatDev"] * 100
            display["AllocVsInfoOilPct"] = display["AllocVsInfoOilPct"] * 100
            display["WC"] = display["WC"] * 100
            if "TotalWC" in display.columns:
                display["TotalWC"] = display["TotalWC"] * 100

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Well": st.column_config.TextColumn("Well", pinned="left"),
                    "Pad": st.column_config.TextColumn("Pad"),
                    "Reservoir": st.column_config.TextColumn("Reservoir"),
                    "LiftType": st.column_config.TextColumn("Lift Type"),
                    "PopsPad": st.column_config.CheckboxColumn(
                        "OnPad Sep?",
                        help="Pad has on-pad production separation",
                    ),
                    "TestDate": st.column_config.DatetimeColumn(
                        "Test Date", format="YYYY-MM-DD"
                    ),
                    "DaysSinceTest": st.column_config.NumberColumn(
                        "Days since", format="%.0f",
                        help="Days between today and the displayed test date",
                    ),
                    "StaleTest": st.column_config.CheckboxColumn(
                        "Stale?",
                        help=f"Displayed test older than {stale_days} days",
                    ),
                    "ProdXV": st.column_config.NumberColumn(
                        "Prod XV", format="%.0f",
                        help="Production safety valve: 1=open, 0=closed",
                    ),
                    "PFXV": st.column_config.NumberColumn(
                        "PF XV", format="%.0f",
                        help="Power-fluid safety valve: 1=open, 0=closed",
                    ),
                    "XVTime": st.column_config.DatetimeColumn(
                        "XV Time", format="MM-DD HH:mm",
                        help="Timestamp of most recent XV reading",
                    ),
                    "JustRestarted": st.column_config.CheckboxColumn(
                        "Just restarted?",
                        help="XV shows flowing but vw_shut_in still has it as "
                        "shut-in. The daily log hasn't caught up yet.",
                    ),
                    "Allocated": st.column_config.CheckboxColumn("Alloc."),
                    "FallbackUsed": st.column_config.CheckboxColumn(
                        "Fallback",
                        help="No allocated test exists; displayed row is info-only "
                        "(or there are no tests at all for this well)",
                    ),
                    "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
                    "Water": st.column_config.NumberColumn("Water (BWPD)", format="%.0f"),
                    "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
                    "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
                    "TotalWC": st.column_config.NumberColumn("Total WC (%)", format="%.1f"),
                    "GOR": st.column_config.NumberColumn("GOR (scf/bbl)", format="%.0f"),
                    "TotalGOR": st.column_config.NumberColumn("Total GOR", format="%.0f"),
                    "BHP": st.column_config.NumberColumn("BHP (psi)", format="%.0f"),
                    "WHP": st.column_config.NumberColumn("WHP (psi)", format="%.0f"),
                    "LiftWater": st.column_config.NumberColumn(
                        "Lift Water (BWPD)", format="%.0f"
                    ),
                    "LiftGas": st.column_config.NumberColumn(
                        "Lift Gas (MCFD)", format="%.0f"
                    ),
                    "TotalWater": st.column_config.NumberColumn(
                        "Total Water (BWPD)", format="%.0f"
                    ),
                    "TotalGas": st.column_config.NumberColumn(
                        "Total Gas (MCFD)", format="%.0f"
                    ),
                    "EspHz": st.column_config.NumberColumn(
                        "ESP Hz", format="%.1f",
                        help="ESP frequency from displayed test (blank for non-ESP wells)",
                    ),
                    "EspAmps": st.column_config.NumberColumn(
                        "ESP Amps", format="%.0f",
                        help="ESP motor amps from displayed test (blank for non-ESP wells)",
                    ),
                    "Oil_2moAvg": st.column_config.NumberColumn(
                        "Oil 2mo avg", format="%.0f"
                    ),
                    "Wat_2moAvg": st.column_config.NumberColumn(
                        "Wat 2mo avg", format="%.0f"
                    ),
                    "OilDev": st.column_config.NumberColumn(
                        "Oil Δ vs 2mo (%)", format="%.0f"
                    ),
                    "WatDev": st.column_config.NumberColumn(
                        "Wat Δ vs 2mo (%)", format="%.0f"
                    ),
                    "FlagOutlier": st.column_config.CheckboxColumn(
                        "Outlier?",
                        help="|Δ| > 25% on oil or water vs trailing 2-month average",
                    ),
                    "AllocVsInfoOilPct": st.column_config.NumberColumn(
                        "Info vs Alloc Oil Δ (%)",
                        help="Latest info-only oil rate vs latest allocated. "
                        "Large values = allocation drift.",
                        format="%.0f",
                    ),
                    "LatestAllocDate": st.column_config.DatetimeColumn(
                        "Latest Alloc Date", format="YYYY-MM-DD"
                    ),
                    "LatestInfoDate": st.column_config.DatetimeColumn(
                        "Latest Info Date", format="YYYY-MM-DD"
                    ),
                },
            )

            csv = online_df.to_csv(index=False, float_format="%.2f").encode("utf-8")
            st.download_button(
                "Download Online CSV",
                data=csv,
                file_name="well_sort_online.csv",
                mime="text/csv",
                key="well_sort_dl_online",
            )

    def _shut_columns_config():
        return {
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Reservoir": st.column_config.TextColumn("Reservoir"),
            "LiftType": st.column_config.TextColumn("Lift Type"),
            "PopsPad": st.column_config.CheckboxColumn("OnPad Sep?"),
            "ShutInSince": st.column_config.DateColumn(
                "Shut-In Since",
                help="Start of current consecutive full-day shut-in streak",
            ),
            "CurrentCode": st.column_config.TextColumn("Code"),
            "CurrentReason": st.column_config.TextColumn("Reason"),
            "Notes": st.column_config.TextColumn("Notes"),
            "DownHours": st.column_config.NumberColumn("Down hrs", format="%.1f"),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            "Water": st.column_config.NumberColumn("Water (BWPD)", format="%.0f"),
            "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
            "LiftWater": st.column_config.NumberColumn("Lift Wat (BWPD)", format="%.0f"),
            "LiftGas": st.column_config.NumberColumn("Lift Gas (MCFD)", format="%.0f"),
            "TotalWater": st.column_config.NumberColumn("Total Wat (BWPD)", format="%.0f"),
            "TotalGas": st.column_config.NumberColumn("Total Gas (MCFD)", format="%.0f"),
            "EspHz": st.column_config.NumberColumn(
                "ESP Hz", format="%.1f",
                help="ESP frequency from last test (blank for non-ESP wells)",
            ),
            "EspAmps": st.column_config.NumberColumn(
                "ESP Amps", format="%.0f",
                help="ESP motor amps from last test (blank for non-ESP wells)",
            ),
            "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
            "TotalWC": st.column_config.NumberColumn("Total WC (%)", format="%.1f"),
            "GOR": st.column_config.NumberColumn("GOR", format="%.0f"),
            "TotalGOR": st.column_config.NumberColumn("Total GOR", format="%.0f"),
            "LastOnlineDate": st.column_config.DateColumn("Last Online"),
            "LastTestDate": st.column_config.DatetimeColumn(
                "Last Test", format="YYYY-MM-DD",
                help="Absolute-latest test date, any age (not bounded by 180d window)",
            ),
            "NearAvgOil": st.column_config.NumberColumn(
                "Near Avg Oil", format="%.0f",
                help="Avg oil rate over tests within 90 days of last test",
            ),
            "NearAvgWater": st.column_config.NumberColumn("Near Avg Wat", format="%.0f"),
            "NearAvgGas": st.column_config.NumberColumn("Near Avg Gas", format="%.0f"),
            "NTestsNear": st.column_config.NumberColumn(
                "# Near Tests", format="%.0f",
                help="How many tests in the 90-day near-last window",
            ),
            "ProdXV": st.column_config.NumberColumn("Prod XV", format="%.0f"),
            "PFXV": st.column_config.NumberColumn("PF XV", format="%.0f"),
            "XVTime": st.column_config.DatetimeColumn("XV Time", format="MM-DD HH:mm"),
        }

    def _render_shut_section(df: pd.DataFrame, label: str, key: str):
        if df.empty:
            st.info(f"No {label} wells.")
            return
        disp = df.copy()
        for c in ("WC", "TotalWC"):
            if c in disp.columns:
                disp[c] = disp[c] * 100
        st.dataframe(
            disp, use_container_width=True, hide_index=True,
            column_config=_shut_columns_config(),
        )
        st.download_button(
            f"Download {label} CSV",
            data=df.to_csv(index=False, float_format="%.2f").encode("utf-8"),
            file_name=f"well_sort_{key}.csv",
            mime="text/csv",
            key=f"well_sort_dl_{key}",
        )

    with sub_off:
        _render_shut_section(offline_df, "Offline", "offline")

    with sub_ltsi:
        _render_shut_section(ltsi_df, "LTSI", "ltsi")
