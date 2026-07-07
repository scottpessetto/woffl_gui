"""Well Sort tab.

Online vs Shut-In classification + outlier flagging across MPU producers.
Live ProdXV status overrides the daily shut-in log symmetrically: a closed
XV forces shut-in (catches just-shut wells), an open XV rescues a logged-
shut well back to online (catches just-restarted wells).
"""

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# POPS pad config — widget-key GC-safe mirrors (P1-31)
# ---------------------------------------------------------------------------
# The Wells tab's "Pads with on-pad production separation" / per-well override
# multiselects are read back by three OTHER call sites (the Marginal WC tab,
# the per-pad section, and Triage) via
# ``st.session_state.get("well_sort_pops_pads", ...)``. Per this repo's
# CLAUDE.md ("Widget state is garbage-collected when its view isn't
# rendered"), a Scott's Tools sub-tab detour can drop these widget keys
# entirely, which used to silently reset every reader back to the hard-coded
# default. These non-widget mirror keys hold the last known-good value; the
# widgets' fallback defaults are computed from the mirror (a no-op once the
# widget key itself survives, since Streamlit ignores `default` then).
_DEFAULT_POPS_PADS = ["E", "F", "H", "I", "M", "S"]
_POPS_PADS_MIRROR_KEY = "well_sort_pops_pads_mirror"
_POPS_FORCE_TRUE_MIRROR_KEY = "well_sort_pops_force_true_mirror"


def _pops_pads_fallback() -> list[str]:
    """Best available default for the POPs-pads multiselect (P1-31).

    Prefers the mirror (the last value the widget actually held) over the
    hard-coded field default, so a GC'd widget key restores the user's
    selection instead of silently reverting.
    """
    return list(st.session_state.get(_POPS_PADS_MIRROR_KEY, _DEFAULT_POPS_PADS))


def _pops_force_true_fallback() -> list[str]:
    """Best available default for the per-well PopsPad=True overrides (P1-31)."""
    return list(st.session_state.get(_POPS_FORCE_TRUE_MIRROR_KEY, []))


@st.cache_data(ttl=3600, show_spinner="Loading shut-in history from Databricks...")
def _cached_shut_in_history() -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_current_shut_in_history

    return fetch_current_shut_in_history()


@st.cache_data(ttl=3600, show_spinner="Loading 30-day shut-in history...")
def _cached_recent_shut_in_history(days: int = 60) -> pd.DataFrame:
    from woffl.assembly.well_sort_client import fetch_recent_shut_in_history

    return fetch_recent_shut_in_history(days=days)


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
        "Online = ProdXV open AND not in vw_shut_in (or in the log but XV "
        "shows open — log lags up to 24 h either way). ProdXV closed forces "
        "shut-in even when the well isn't yet in the log. "
        "Outlier = |test − 2-mo avg| > 25% on oil or water."
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.1, 1.8, 1.5, 1.0])
    with ctrl1:
        if st.button("Refresh data", help="Clear cache and re-query Databricks"):
            _cached_shut_in_history.clear()
            _cached_recent_shut_in_history.clear()
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
            "Stale-test threshold (days)",
            14,
            180,
            60,
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
    if xv.empty:
        st.caption(
            "ℹ️ Safety-valve (XV) status unavailable — ProdXV/PFXV columns are "
            "blank and classification falls back to the shut-in log only. (On "
            "Databricks this usually means the app lacks access to the "
            "`reporting` catalog.)"
        )

    if not producers:
        st.error("No producers returned from vw_well_header.")
        return

    # On-pad separation settings (persist across interactions). Defaults AND
    # remembered selections are intersected with the live options — a value
    # missing from options (empty catalog on a transient Databricks failure,
    # a well dropping out of vw_well_header) raises StreamlitAPIException
    # and kills the whole page.
    all_pads = (
        sorted(catalog["well_pad"].dropna().unique().tolist())
        if not catalog.empty
        else []
    )
    if "well_sort_pops_pads" in st.session_state:
        st.session_state["well_sort_pops_pads"] = [
            p for p in st.session_state["well_sort_pops_pads"] if p in all_pads
        ]
    pops_pads = st.multiselect(
        "Pads with on-pad production separation",
        options=all_pads,
        default=[
            p
            for p in st.session_state.get("well_sort_pops_pads", _pops_pads_fallback())
            if p in all_pads
        ],
        key="well_sort_pops_pads",
        help="Wells on these pads get PopsPad=True. Per-well overrides apply after.",
    )
    # Mirror to a non-widget key so a later tab detour (which GC's the widget
    # key above) can restore this instead of silently reverting to the
    # hard-coded default (P1-31).
    st.session_state[_POPS_PADS_MIRROR_KEY] = list(pops_pads)

    # Per-well PopsPad=True overrides (wells that get True even if their pad
    # doesn't have separation — e.g. MPS-08 in the Apr-20 bench sheet).
    producer_opts = sorted(producers)
    if "well_sort_pops_force_true" in st.session_state:
        st.session_state["well_sort_pops_force_true"] = [
            w
            for w in st.session_state["well_sort_pops_force_true"]
            if w in producer_opts
        ]
    force_true_wells = st.multiselect(
        "Per-well PopsPad=True overrides",
        options=producer_opts,
        default=[
            w
            for w in st.session_state.get(
                "well_sort_pops_force_true", _pops_force_true_fallback()
            )
            if w in producer_opts
        ],
        key="well_sort_pops_force_true",
        help="These wells are treated as having on-pad separation regardless "
        "of the pad-level setting above.",
    )
    # Same GC-survival mirror as pops_pads above (P1-31).
    st.session_state[_POPS_FORCE_TRUE_MIRROR_KEY] = list(force_true_wells)
    overrides = {w: True for w in force_true_wells}

    online_set, shut_set = classify_wells(
        producers, shut_in_hist, xv_df=xv, trust_xv=True
    )
    online_df = build_online_table(
        tests,
        shut_in_hist,
        producers,
        mode=mode,
        stale_days=stale_days,
        xv_df=xv,
        online_wells=online_set,
        catalog_df=catalog,
    )
    shut_df = build_shut_in_table(
        shut_in_hist,
        tests,
        xv_df=xv,
        shut_in_wells=shut_set,
        catalog_df=catalog,
        last_tests_df=last_tests,
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

    # Bench export (matches MPU_Well_Bench_YYYY_MM_DD.xlsx layout). Built ON
    # CLICK with an immediate auto-download — the eager build re-ran the
    # openpyxl per-cell formatting on every rerun of this page just to feed
    # the download button.
    from datetime import date

    if st.button(
        f"Download bench xlsx  ({len(online_df)} online, {len(offline_df)} offline, "
        f"{len(ltsi_df)} ltsi)",
        key="well_sort_dl_bench",
    ):
        from woffl.gui.components.download import autodownload

        xlsx_bytes = export_bench_xlsx(online_df, offline_df, ltsi_df)
        autodownload(
            xlsx_bytes,
            f"MPU_Well_Bench_{date.today():%Y_%m_%d}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ws_bench_dl",
        )

    sub_online, sub_off, sub_ltsi, sub_changes = st.tabs(
        [
            f"Online ({len(online_df)})",
            f"Offline ({len(offline_df)})",
            f"LTSI ({len(ltsi_df)})",
            "30-Day Changes",
        ]
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
                        "Days since",
                        format="%.0f",
                        help="Days between today and the displayed test date",
                    ),
                    "StaleTest": st.column_config.CheckboxColumn(
                        "Stale?",
                        help=f"Displayed test older than {stale_days} days",
                    ),
                    "ProdXV": st.column_config.NumberColumn(
                        "Prod XV",
                        format="%.0f",
                        help="Production safety valve: 1=open, 0=closed",
                    ),
                    "PFXV": st.column_config.NumberColumn(
                        "PF XV",
                        format="%.0f",
                        help="Power-fluid safety valve: 1=open, 0=closed",
                    ),
                    "XVTime": st.column_config.DatetimeColumn(
                        "XV Time",
                        format="MM-DD HH:mm",
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
                    "Water": st.column_config.NumberColumn(
                        "Water (BWPD)", format="%.0f"
                    ),
                    "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
                    "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
                    "TotalWC": st.column_config.NumberColumn(
                        "Total WC (%)", format="%.1f"
                    ),
                    "GOR": st.column_config.NumberColumn(
                        "GOR (scf/bbl)", format="%.0f"
                    ),
                    "TotalGOR": st.column_config.NumberColumn(
                        "Total GOR", format="%.0f"
                    ),
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
                        "ESP Hz",
                        format="%.1f",
                        help="ESP frequency from displayed test (blank for non-ESP wells)",
                    ),
                    "EspAmps": st.column_config.NumberColumn(
                        "ESP Amps",
                        format="%.0f",
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
            "LiftWater": st.column_config.NumberColumn(
                "Lift Wat (BWPD)", format="%.0f"
            ),
            "LiftGas": st.column_config.NumberColumn("Lift Gas (MCFD)", format="%.0f"),
            "TotalWater": st.column_config.NumberColumn(
                "Total Wat (BWPD)", format="%.0f"
            ),
            "TotalGas": st.column_config.NumberColumn(
                "Total Gas (MCFD)", format="%.0f"
            ),
            "EspHz": st.column_config.NumberColumn(
                "ESP Hz",
                format="%.1f",
                help="ESP frequency from last test (blank for non-ESP wells)",
            ),
            "EspAmps": st.column_config.NumberColumn(
                "ESP Amps",
                format="%.0f",
                help="ESP motor amps from last test (blank for non-ESP wells)",
            ),
            "WC": st.column_config.NumberColumn("WC (%)", format="%.1f"),
            "TotalWC": st.column_config.NumberColumn("Total WC (%)", format="%.1f"),
            "GOR": st.column_config.NumberColumn("GOR", format="%.0f"),
            "TotalGOR": st.column_config.NumberColumn("Total GOR", format="%.0f"),
            "LastOnlineDate": st.column_config.DateColumn("Last Online"),
            "LastTestDate": st.column_config.DatetimeColumn(
                "Last Test",
                format="YYYY-MM-DD",
                help="Absolute-latest test date, any age (not bounded by 180d window)",
            ),
            "NearAvgOil": st.column_config.NumberColumn(
                "Near Avg Oil",
                format="%.0f",
                help="Avg oil rate over tests within 90 days of last test",
            ),
            "NearAvgWater": st.column_config.NumberColumn(
                "Near Avg Wat", format="%.0f"
            ),
            "NearAvgGas": st.column_config.NumberColumn("Near Avg Gas", format="%.0f"),
            "NTestsNear": st.column_config.NumberColumn(
                "# Near Tests",
                format="%.0f",
                help="How many tests in the 90-day near-last window",
            ),
            "ProdXV": st.column_config.NumberColumn("Prod XV", format="%.0f"),
            "PFXV": st.column_config.NumberColumn("PF XV", format="%.0f"),
            "XVTime": st.column_config.DatetimeColumn("XV Time", format="MM-DD HH:mm"),
        }

    _shut_col_order = [
        "Well",
        "Pad",
        "Reservoir",
        "LiftType",
        "ShutInSince",
        "CurrentCode",
        "CurrentReason",
        "Notes",
        "DownHours",
        "Oil",
        "Water",
        "Gas",
        "LastOnlineDate",
        "LastTestDate",
        "WC",
        "TotalWC",
        "GOR",
        "TotalGOR",
        "LiftWater",
        "LiftGas",
        "TotalWater",
        "TotalGas",
        "EspHz",
        "EspAmps",
        "NearAvgOil",
        "NearAvgWater",
        "NearAvgGas",
        "NTestsNear",
        "ProdXV",
        "PFXV",
        "XVTime",
        "PopsPad",
    ]

    def _render_shut_section(df: pd.DataFrame, label: str, key: str):
        if df.empty:
            st.info(f"No {label} wells.")
            return
        disp = df.copy()
        for c in ("WC", "TotalWC"):
            if c in disp.columns:
                disp[c] = disp[c] * 100
        ordered = [c for c in _shut_col_order if c in disp.columns]
        extras = [c for c in disp.columns if c not in ordered]
        disp = disp[ordered + extras]
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
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

    with sub_changes:
        from woffl.assembly.well_sort_client import (
            NOTABLE_DOWN_HOURS,
            compute_recent_down_events,
        )

        th_col, _ = st.columns([1, 3])
        with th_col:
            threshold = st.slider(
                "Min hrs/day to count as 'down'",
                1,
                24,
                int(NOTABLE_DOWN_HOURS),
                key="well_sort_down_threshold",
                help="A day counts as a down day when its total down_hours "
                "meets this threshold. Lower captures partial-day events "
                "(e.g. a 12-hr jet pump changeout); higher restricts to "
                "near-full-day shut-ins.",
            )
        st.caption(
            f"One row per shut-in event (consecutive days with ≥ {threshold} "
            "down hrs) overlapping the last 30 days. Ongoing events float "
            "to the top, then sorted by Days descending. A single 12-hr "
            "changeout shows as a 1-day event."
        )
        recent_hist = _cached_recent_shut_in_history(60)
        events_df = compute_recent_down_events(
            recent_hist,
            producers,
            catalog_df=catalog,
            window_days=30,
            down_hours_threshold=float(threshold),
        )
        if events_df.empty:
            st.info("No shut-in events in the last 30 days.")
        else:
            ongoing = int(events_df["Ongoing"].sum())
            one_day = int((events_df["Days"] == 1).sum())
            n_wells = events_df["Well"].nunique()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Events", len(events_df))
            c2.metric("Still down", ongoing)
            c3.metric("1-day events", one_day)
            c4.metric("Wells affected", n_wells)

            st.dataframe(
                events_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Well": st.column_config.TextColumn("Well", pinned="left"),
                    "Pad": st.column_config.TextColumn("Pad"),
                    "Reservoir": st.column_config.TextColumn("Reservoir"),
                    "Started": st.column_config.DateColumn("Started"),
                    "Ended": st.column_config.DateColumn(
                        "Ended",
                        help="Blank when the event is still ongoing",
                    ),
                    "Days": st.column_config.NumberColumn(
                        "Days",
                        format="%.0f",
                        help="Consecutive days at or above the threshold",
                    ),
                    "MaxHrs": st.column_config.NumberColumn(
                        "Max Hrs",
                        format="%.1f",
                        help="Peak single-day down_hours in this event",
                    ),
                    "TotalHrs": st.column_config.NumberColumn(
                        "Total Hrs",
                        format="%.1f",
                        help="Sum of down_hours across the event",
                    ),
                    "Code": st.column_config.TextColumn("Code"),
                    "Reason": st.column_config.TextColumn("Reason"),
                    "Notes": st.column_config.TextColumn("Notes"),
                    "Ongoing": st.column_config.CheckboxColumn(
                        "Ongoing?",
                        help="Event extends to the latest log date",
                    ),
                },
            )
            st.download_button(
                "Download 30-Day Events CSV",
                data=events_df.to_csv(index=False).encode("utf-8"),
                file_name="well_sort_30day_events.csv",
                mime="text/csv",
                key="well_sort_dl_changes",
            )


# ---------------------------------------------------------------------------
# Marginal Water Cut calculator
# ---------------------------------------------------------------------------


# Per-POPs-pad water handling capacity (BWPD). Used as starter presets
# for the Per-Pad Marginal WC calculator on the Marginal WC tab. The user
# can override each value in the UI — edits persist via session_state.
# Update these numbers here if the field hardware changes.
PUMP_LIMIT_PRESETS: dict[str, int] = {
    "E": 20000,
    "F": 28000,
    "H": 30000,
    "I": 32000,
    "M": 55000,
    "S": 35000,
}


# Which water stream the pad's pump actually handles. Some POPs pads only
# separate power-fluid water (formation water passes through to the central
# facility); others handle the full produced stream. The marginal-WC calc
# measures cumulative water against the pad's pump limit using whichever
# stream the pad actually constrains.
POPS_PUMP_HANDLES: dict[str, str] = {
    # Full POPs: pad handles formation + lift water
    "E": "total",
    "F": "total",
    "M": "total",
    # PF-only POPs: pad pump only sees lift (power-fluid) water; formation
    # water passes through to the central facility
    "I": "lift",
    "H": "lift",
    "S": "lift",
}


def _build_online_full(stale_days: int = 60) -> "pd.DataFrame":
    """Build the full online table with the PopsPad column set.

    Shares the cached Databricks fetchers + classification helpers with
    the main Well Sort tab. POPs-pad selections + per-well overrides are
    inherited from session_state (set on the Wells tab), with the same
    defaults if the user hasn't visited that tab yet. Both the field-wide
    marginal-WC calc and the per-pad calc go through this — they just
    apply different filters on top.
    """
    from woffl.assembly.well_sort_client import (
        apply_pops_pad,
        build_online_table,
        classify_wells,
    )

    shut_in_hist = _cached_shut_in_history()
    tests = _cached_recent_tests(180)
    producers = _cached_producers()
    catalog = _cached_producer_catalog()
    xv = _cached_xv_status()

    # P1-31: fall back to the GC-survival mirror, not a hard-coded default,
    # if the Wells tab's widget keys aren't present this rerun.
    pops_pads = st.session_state.get("well_sort_pops_pads", _pops_pads_fallback())
    force_true = st.session_state.get(
        "well_sort_pops_force_true", _pops_force_true_fallback()
    )
    overrides = {w: True for w in force_true}

    online_set, _ = classify_wells(producers, shut_in_hist, xv_df=xv, trust_xv=True)
    online_df = build_online_table(
        tests,
        shut_in_hist,
        producers,
        mode="allocated",
        stale_days=stale_days,
        xv_df=xv,
        online_wells=online_set,
        catalog_df=catalog,
    )
    return apply_pops_pad(online_df, set(pops_pads), overrides)


def _build_online_non_pops(stale_days: int = 60) -> "pd.DataFrame":
    """Online wells with PopsPad=True filtered out.

    Used by the field-wide marginal-WC calc — POPs-pad wells run their own
    water handling, so they don't compete for central facility capacity.
    """
    full = _build_online_full(stale_days)
    if full.empty:
        return full
    return full[~full["PopsPad"]].copy()


def compute_field_marginal_wc(threshold_pct: float = 2.0) -> dict | None:
    """Compute today's field-wide marginal WC at the given cumulative-water threshold.

    Sort online non-POPs wells by TotalWC descending, walk the list adding
    TotalWater. The marginal WC is the WC of the first well at which the
    cumulative water crosses the threshold fraction of field water. This
    rejects single-well noise (a 0.99-WC stripper making 40 BWPD lifts the
    cumulative by ~zero, so the marginal lands on a meaningful well).

    Returns a dict with the headline result + the ranked dataframe for
    display, or None when no online non-POPs wells with valid TotalWC /
    TotalWater are available. Public so the Batch Run "Import from Well
    Sort" button can call it without re-implementing the math.
    """
    non_pops = _build_online_non_pops()
    if non_pops.empty:
        return None

    valid = non_pops[
        non_pops["TotalWC"].notna() & non_pops["TotalWater"].notna()
    ].copy()
    valid = valid[valid["TotalWater"] > 0].copy()
    if valid.empty:
        return None

    valid = valid.sort_values("TotalWC", ascending=False).reset_index(drop=True)
    total_field_water = float(valid["TotalWater"].sum())
    valid["CumWater"] = valid["TotalWater"].cumsum()
    valid["CumWaterPct"] = (valid["CumWater"] / total_field_water) * 100.0

    above_mask = valid["CumWaterPct"] >= threshold_pct
    if above_mask.any():
        marg_idx = int(above_mask.idxmax())
    else:
        # Threshold higher than 100% (impossible by construction). Use the
        # bottom well as a defensive fallback.
        marg_idx = len(valid) - 1

    marg_row = valid.iloc[marg_idx]
    return {
        "marginal_wc": float(marg_row["TotalWC"]),
        "well": str(marg_row["Well"]),
        "pad": str(marg_row.get("Pad", "—")),
        "total_field_water": total_field_water,
        "well_count": int(len(valid)),
        "threshold_pct": float(threshold_pct),
        "marg_idx": marg_idx,
        "ranked_df": valid,
    }


def compute_pad_marginal_wc(pad: str, pump_limit: float) -> dict | None:
    """Per-pad marginal WC + headroom against the pad pump's capacity.

    The pad's pump only sees one water stream:
      * **lift** for PF-only POPs (I, H, S) — pump only handles power-fluid
        water; formation water passes through to the central facility.
      * **total** for full POPs (E, F, M) — pump handles formation + PF.

    Per-well "pad WC" = ``water / (water + oil)`` using whichever stream
    the pad pump actually sees. For PF-only pads this is a new metric
    ("PFWC"); for full POPs it equals the standard TotalWC.

    The pad's **marginal WC** is just the max of that per-well WC — i.e.
    the worst-performing well on the pad pump. The pump limit doesn't
    influence the marginal; instead it gives the **headroom** number
    (positive = "X BWPD available to allocate", negative = "OVER by X").

    Returns dict with the headline result + the ranked dataframe (sorted
    by pad WC descending), or None if the pad has no online wells with a
    usable water + oil pair.
    """
    full = _build_online_full()
    if full.empty:
        return None

    pad_df = full[full["Pad"] == pad].copy()
    if pad_df.empty:
        return None

    water_basis = POPS_PUMP_HANDLES.get(pad, "total")
    water_col = "LiftWater" if water_basis == "lift" else "TotalWater"
    if water_col not in pad_df.columns or "Oil" not in pad_df.columns:
        return None

    valid = pad_df[pad_df[water_col].notna() & pad_df["Oil"].notna()].copy()
    # Drop rows where both rates are zero — no per-well WC defined.
    valid = valid[(valid[water_col] > 0) | (valid["Oil"] > 0)].copy()
    if valid.empty:
        return None

    # Per-well WC measured against the pad pump's stream.
    denom = valid[water_col] + valid["Oil"]
    valid["WC_pad"] = valid[water_col] / denom.where(denom > 0, 1.0)

    # Sort worst-first so the marginal lands at the top of the table.
    valid = valid.sort_values("WC_pad", ascending=False).reset_index(drop=True)

    total_pad_water = float(valid[water_col].sum())
    marg_idx = 0
    marg_row = valid.iloc[marg_idx]
    marginal_wc = float(marg_row["WC_pad"])

    pump_limit_f = float(pump_limit) if pump_limit and pump_limit > 0 else 0.0
    headroom = pump_limit_f - total_pad_water if pump_limit_f > 0 else None

    return {
        "marginal_wc": marginal_wc,
        "well": str(marg_row["Well"]),
        "pad": pad,
        "pad_water": total_pad_water,
        "pump_limit": pump_limit_f,
        "headroom": float(headroom) if headroom is not None else None,
        "well_count": int(len(valid)),
        "marg_idx": marg_idx,
        "ranked_df": valid,
        "water_basis": water_basis,
        "water_col": water_col,
    }


def render_marginal_wc_tab() -> None:
    """Field-wide marginal-WC calculator using a cumulative-water threshold.

    Thin UI wrapper around compute_field_marginal_wc().
    """
    st.header("Marginal Water Cut Calculator")
    st.caption(
        "Estimates today's marginal WC by walking the worst-WC online wells "
        "downward and stopping where cumulative water crosses the threshold. "
        "POPs-pad wells are excluded — they don't compete for the central "
        "facility's water-handling capacity. POPs-pad selections inherit "
        "from the **Wells** tab."
    )

    # Threshold slider + import button row
    col_thr, col_btn = st.columns([3, 2])
    with col_thr:
        cum_threshold_pct = st.slider(
            "Cumulative water threshold (% of field water)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            format="%.1f",
            key="marg_wc_threshold_pct",
            help=(
                "Walk down the sorted-by-WC list, summing water. The marginal "
                "WC lands on the first well at which cumulative water crosses "
                "this % of total field (non-POPs) water. Higher = stricter "
                "(more wells get excluded as 'noise'); lower = more sensitive."
            ),
        )

    result = compute_field_marginal_wc(threshold_pct=cum_threshold_pct)
    if result is None:
        st.warning(
            "No online non-POPs wells with valid TotalWC / TotalWater — "
            "check the Wells tab to verify online classification and "
            "POPs-pad settings."
        )
        return

    valid = result["ranked_df"]
    marg_idx = result["marg_idx"]
    marginal_wc = result["marginal_wc"]
    marg_well = result["well"]
    marg_pad = result["pad"]
    total_field_water = result["total_field_water"]

    # Hero metric
    st.metric(
        "Today's Marginal WC",
        f"{marginal_wc:.3f}",
        delta=f"set by {marg_well} ({marg_pad}-Pad)",
        delta_color="off",
    )
    st.caption(
        f"Worst-WC online wells totaling **{cum_threshold_pct:.1f}%** "
        f"({valid.iloc[marg_idx]['CumWater']:,.0f} BWPD) of field water "
        f"are above this WC. Field water (non-POPs): "
        f"**{total_field_water:,.0f} BWPD** across **{len(valid)}** wells."
    )

    # Import button
    with col_btn:
        st.write("")  # vertical spacing to align with slider
        st.write("")
        if st.button(
            f"Import {marginal_wc:.3f} to sidebar",
            type="primary",
            use_container_width=True,
            key="marg_wc_import_btn",
            help=(
                "Updates the sidebar's Marginal Watercut. The Single Well "
                "Analysis batch recommender + the inline Marginal WC quickfix "
                "on the Batch Run tab will use this value on the next render."
            ),
        ):
            st.session_state["marginal_watercut"] = marginal_wc
            # Pop the widget key so the sidebar's _number_input helper
            # re-initializes from the logical key on the next render.
            st.session_state.pop("marginal_watercut_input", None)
            # Also pop the batch-run quickfix key so it picks up the new value.
            st.session_state.pop("_batch_marg_wc_box", None)
            st.success(f"Imported {marginal_wc:.3f} into sidebar Marginal Watercut.")

    # Ranked-wells table
    st.subheader("Ranked Online Wells (non-POPs)")

    display = valid.copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display["Marginal"] = display.index == marg_idx
    cols = [
        "Rank",
        "Marginal",
        "Well",
        "Pad",
        "Reservoir",
        "Oil",
        "TotalWater",
        "TotalWC",
        "CumWater",
        "CumWaterPct",
    ]
    cols = [c for c in cols if c in display.columns]
    display = display[cols]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Marginal": st.column_config.CheckboxColumn(
                "Marginal?",
                help="Well at which cumulative water crosses the threshold",
            ),
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Reservoir": st.column_config.TextColumn("Reservoir"),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            "TotalWater": st.column_config.NumberColumn(
                "Total Water (BWPD)",
                format="%.0f",
            ),
            "TotalWC": st.column_config.NumberColumn(
                "Total WC",
                format="%.3f",
            ),
            "CumWater": st.column_config.NumberColumn(
                "Cum Water (BWPD)",
                format="%.0f",
                help="Running total of TotalWater from the worst-WC well downward",
            ),
            "CumWaterPct": st.column_config.NumberColumn(
                "Cum %",
                format="%.1f%%",
                help="Cumulative water as a percentage of total field water",
            ),
        },
    )

    # ------------------------------------------------------------------
    # Per-Pad Marginal WC
    # ------------------------------------------------------------------
    st.divider()
    _render_pad_marginal_wc_section()


def _pad_limit_key(pad: str) -> str:
    """Session-state key for the editable pump limit of one pad."""
    return f"_pad_pump_limit_{pad}"


def _pad_limit_mirror_key(pad: str) -> str:
    """Non-widget mirror of a pad's pump limit — survives widget-key GC.

    ``_pad_limit_key`` is used directly as the ``st.number_input`` key; a
    Scott's Tools tab detour can GC that widget key, which used to silently
    re-seed the input from ``PUMP_LIMIT_PRESETS`` and drop the user's edit
    (P1-31). This mirror holds the last value the widget actually held so the
    input can restore it instead.
    """
    return f"_pad_pump_limit_mirror_{pad}"


def _reset_pad_limit(pad: str) -> None:
    """on_click callback — restore a pad's pump limit to its preset.

    Runs before the widget renders on the next rerun, so writing to the
    widget's session-state key is safe here.
    """
    preset = int(PUMP_LIMIT_PRESETS.get(pad, 0))
    st.session_state[_pad_limit_key(pad)] = preset
    st.session_state[_pad_limit_mirror_key(pad)] = preset


def _seed_pad_limit_widget(pad: str) -> None:
    """Seed a pad's pump-limit widget key before it renders (P1-31).

    * Widget key already holds a value (survived the rerun) → no-op; the
      widget owns it from here (Streamlit ignores any default we'd pass).
    * Widget key absent — either first visit, or a Scott's Tools tab detour
      just GC'd it — re-seed from the non-widget mirror (the user's last
      edit) when one exists, falling back to the static preset only when it
      doesn't. Without this, a GC'd key used to always re-seed from the
      preset, silently discarding the user's edit.
    """
    pump_limit_key = _pad_limit_key(pad)
    if pump_limit_key in st.session_state:
        return
    mirror_key = _pad_limit_mirror_key(pad)
    st.session_state[pump_limit_key] = int(
        st.session_state.get(mirror_key, PUMP_LIMIT_PRESETS.get(pad, 0))
    )


def _render_pad_marginal_wc_section() -> None:
    """Per-pad marginal-WC calculator (POPs pads).

    Picks a single POPs pad, applies its water-handling capacity (pump
    limit, editable, seeded from PUMP_LIMIT_PRESETS), and computes the
    marginal WC via compute_pad_marginal_wc. Includes an Import button
    that pushes the per-pad value into the sidebar's Marginal Watercut.
    """
    st.subheader("Per-Pad Marginal Water Cut")
    st.caption(
        "POPs pads have their own water-handling pumps. The marginal WC is "
        "simply the WC of the single worst-performing online well on the pad "
        "pump (a plain max, not a shedding calculation) — measured against "
        "whichever stream that pad's pump actually sees. **E / F / M** handle "
        "the full produced stream (formation + PF). **I / H / S** are "
        "PF-only — their pump only sees lift water, and formation water "
        "passes through to the central facility. The pump limit below does "
        "**not** change which well is marginal; it only sets the **headroom** "
        "(capacity still available, or how far over capacity the pad is)."
    )

    # P1-31: fall back to the GC-survival mirror, not a hard-coded default.
    pops_pads = sorted(
        st.session_state.get("well_sort_pops_pads", _pops_pads_fallback())
    )
    if not pops_pads:
        st.info(
            "No POPs pads configured — add one on the Wells tab "
            "(`Pads with on-pad production separation`)."
        )
        return

    # Pick pad + edit pump limit
    col_pad, col_limit, col_reset = st.columns([1.5, 2.5, 2])
    with col_pad:
        pad = st.selectbox(
            "POPs Pad",
            options=pops_pads,
            key="marg_wc_pad",
            help="POPs pad to compute the marginal WC for.",
        )

    # Water basis (lift / total) drives all the labels below.
    water_basis = POPS_PUMP_HANDLES.get(pad, "total")
    basis_label = "PF water" if water_basis == "lift" else "total water"

    pump_limit_key = _pad_limit_key(pad)
    mirror_key = _pad_limit_mirror_key(pad)
    _seed_pad_limit_widget(pad)

    with col_limit:
        st.number_input(
            f"{pad}-Pad pump limit (BWPD, {basis_label})",
            min_value=0,
            max_value=200_000,
            step=1_000,
            key=pump_limit_key,
            help=(
                f"{pad}-Pad water-handling capacity measured against "
                f"{basis_label}. The marginal WC is computed against this "
                "limit. Edits persist for the session."
            ),
        )
    with col_reset:
        preset = PUMP_LIMIT_PRESETS.get(pad, 0)
        st.caption(" ")
        st.button(
            f"Reset to preset ({preset:,} BWPD)",
            on_click=_reset_pad_limit,
            kwargs={"pad": pad},
            key=f"marg_wc_pad_reset_{pad}",
            use_container_width=True,
            help="Restore this pad's pump limit to its default preset.",
        )

    pump_limit = int(st.session_state[pump_limit_key])
    # Keep the mirror current so a later GC of the widget key restores this
    # value rather than the static preset (P1-31).
    st.session_state[mirror_key] = pump_limit
    pad_result = compute_pad_marginal_wc(pad=pad, pump_limit=pump_limit)

    # Surface a previous-rerun import message (st.button click triggers a
    # rerun before st.success would render)
    import_msg = st.session_state.pop("_pad_marg_import_msg", None)
    if import_msg:
        st.success(import_msg)

    if pad_result is None:
        st.warning(
            f"No online wells on **{pad}-Pad** with valid TotalWC / "
            f"{basis_label}. Check the Wells tab."
        )
        return

    # Summary row: pad water vs limit
    pad_water = pad_result["pad_water"]
    headroom = pad_result["headroom"]
    marginal_wc = pad_result["marginal_wc"]
    marg_well = pad_result["well"]
    marg_idx = pad_result["marg_idx"]
    pad_valid = pad_result["ranked_df"]

    summary_cols = st.columns(4)
    summary_cols[0].metric(
        f"{pad}-Pad {basis_label.title()}",
        f"{pad_water:,.0f} BWPD",
        help=(
            f"Sum of {basis_label} across {pad_result['well_count']} "
            f"online wells on the pad. ({pad_result['water_col']} column.)"
        ),
    )
    summary_cols[1].metric(
        "Pump Limit",
        f"{pump_limit:,.0f} BWPD",
        help="Editable above. Reset returns to the preset.",
    )
    if headroom is not None:
        # Phrasing differs by basis: for PF-only pads the headroom is
        # specifically PF capacity to allocate to more wells.
        capacity_label = "PF" if water_basis == "lift" else "water"
        if headroom >= 0:
            delta_label = f"{headroom:,.0f} BWPD {capacity_label} available to allocate"
        else:
            delta_label = f"OVER by {abs(headroom):,.0f} BWPD"
        summary_cols[2].metric(
            "Headroom",
            f"{headroom:+,.0f} BWPD",
            delta=delta_label,
            delta_color=("normal" if headroom >= 0 else "inverse"),
            help="Positive = pad fits under limit; negative = over capacity.",
        )
    else:
        summary_cols[2].metric("Headroom", "—")
    summary_cols[3].metric(
        f"{pad}-Pad Marginal WC",
        f"{marginal_wc:.3f}",
        delta=f"set by {marg_well}",
        delta_color="off",
    )

    # Import-to-sidebar
    if st.button(
        f"📊 Import {pad}-Pad Marginal WC ({marginal_wc:.3f}) to sidebar",
        type="primary",
        key=f"marg_wc_pad_import_{pad}",
        help=(
            "Writes this per-pad marginal WC to the sidebar's "
            "Marginal Watercut. Single Well Analysis batch recommender "
            "and the Batch Run quickfix update on the next render."
        ),
    ):
        st.session_state["marginal_watercut"] = marginal_wc
        st.session_state.pop("marginal_watercut_input", None)
        st.session_state.pop("_batch_marg_wc_box", None)
        st.session_state["_pad_marg_import_msg"] = (
            f"Imported {marginal_wc:.3f} ({pad}-Pad, {basis_label} basis, "
            f"set by {marg_well}) into sidebar Marginal Watercut."
        )
        st.rerun()

    # Pad-level ranked table — sorted by per-well "pad WC" descending.
    # For PF-only pads we expose two basis-specific columns named clearly
    # for CSV-export readability:
    #   PFRate = LiftWater renamed
    #   PFWC   = water/(water+oil) using PF water  (= WC_pad)
    # For full POPs we use the existing TotalWater + TotalWC names; our
    # WC_pad equals TotalWC by definition, so we drop the duplicate column.
    if water_basis == "lift":
        pad_display = pad_valid.rename(
            columns={
                "LiftWater": "PFRate",
                "WC_pad": "PFWC",
            }
        )
        water_col_display = "PFRate"
        water_col_label = "PF Rate (BWPD)"
        wc_col_display = "PFWC"
        wc_col_label = "PF WC"
    else:
        pad_display = pad_valid.drop(columns=["WC_pad"])
        water_col_display = "TotalWater"
        water_col_label = "Total Water (BWPD)"
        wc_col_display = "TotalWC"
        wc_col_label = "Total WC"

    st.markdown(
        f"**Online wells on {pad}-Pad** — sorted by {wc_col_label} desc; "
        f"top row = marginal (worst-WC well on the pad pump)."
    )
    pad_display.insert(0, "Rank", range(1, len(pad_display) + 1))
    cols = [
        "Rank",
        "Well",
        "Pad",
        "Reservoir",
        "Oil",
        water_col_display,
        wc_col_display,
    ]
    cols = [c for c in cols if c in pad_display.columns]
    pad_display = pad_display[cols]

    st.dataframe(
        pad_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Reservoir": st.column_config.TextColumn("Reservoir"),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            water_col_display: st.column_config.NumberColumn(
                water_col_label,
                format="%.0f",
                help=(
                    "Water stream the pad's pump actually handles "
                    f"({pad_result['water_col']} from the source data)."
                ),
            ),
            wc_col_display: st.column_config.NumberColumn(
                wc_col_label,
                format="%.3f",
                help=(
                    f"{wc_col_label} = {water_col_display} / "
                    f"({water_col_display} + Oil). Marginal WC = max."
                ),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Triage (beta) — keep / SI / BOL decision view
# ---------------------------------------------------------------------------
#
# Experimental decision layer that sits ALONGSIDE the original Wells tab
# (left untouched for back-to-back comparison). Driving rule, per Scott: a
# well's water cut vs the field MARGINAL WC sets the lean —
#   * online well, WC above marginal  -> shut-in (SI) candidate
#   * shut well,   WC below marginal  -> bring-on-line (BOL) candidate
# A poor LATEST test against a healthy recent HISTORY is deliberately NOT
# acted on: it's flagged to verify / BOL-trial, because a single bad test
# often recovers (Scott BOLs these to check). The history signal reuses what
# the engine already computes — the 2-month outlier deviation (online) and
# the 90-day near-last-test average (shut).


def _effective_wc(row) -> tuple[float, str | None]:
    """Effective last-test WC and the basis it was read on.

    Returns ``(wc, basis)`` — basis ``"total"`` (``totl_wc``, the same basis
    as the marginal line), ``"form"`` (formation-water fallback when the test
    carries no total WC — reads LOW vs the total-WC line on lifted wells, so
    callers must flag it rather than judge silently, P1-27), or ``None`` when
    no WC exists at all.
    """
    v = row.get("TotalWC")
    if pd.notna(v):
        return float(v), "total"
    v = row.get("WC")
    if pd.notna(v):
        return float(v), "form"
    return float("nan"), None


def _form_basis_note(wc_basis: str | None) -> str:
    """Why-suffix flagging a decision made on form-basis WC (P1-27)."""
    if wc_basis != "form":
        return ""
    return " [form-basis WC — test has no total WC; reads low vs the " "total-WC line]"


def add_online_decision(online_df: pd.DataFrame, marginal_wc: float) -> pd.DataFrame:
    """Augment the online table with a keep/SI decision vs the marginal WC.

    Adds: Decision (emoji-tagged string), Why (plain-language reason),
    WCvsMarginal (Total WC − marginal), WCBasis ("total"/"form" — which WC
    the decision used; a form-basis fallback reads low vs the total-WC line
    and is flagged in Why, P1-27), and a hidden ``_rank`` for sorting.

    Rule:
      POPS-pad well                              -> ⚪ POPS (own handling)
      stale / no test                            -> ⚠️ Verify (unknown state)
      WC ≤ marginal                              -> ✅ Keep online
      WC > marginal AND latest test anomalous    -> ⚠️ Verify before SI
        (oil outlier-LOW or water outlier-HIGH — either way one fluky test
         shouldn't condemn the well, P1-28)
      WC > marginal otherwise                    -> 🔴 SI candidate
    """
    from woffl.assembly.well_sort_client import OUTLIER_PCT

    df = online_df.copy()
    if df.empty:
        for c in ("WCvsMarginal", "_rank"):
            df[c] = pd.Series(dtype="float")
        for c in ("Decision", "Why", "WCBasis"):
            df[c] = pd.Series(dtype="object")
        return df

    decisions, whys, deltas, bases, ranks = [], [], [], [], []
    for _, r in df.iterrows():
        wc, wc_basis = _effective_wc(r)
        deltas.append(wc - marginal_wc if pd.notna(wc) else float("nan"))
        bases.append(wc_basis)
        basis_note = _form_basis_note(wc_basis)
        pops = bool(r.get("PopsPad"))
        stale = bool(r.get("StaleTest"))
        oil_dev = r.get("OilDev")
        wat_dev = r.get("WatDev")
        if pops:
            decisions.append("⚪ POPS (own handling)")
            whys.append(
                "On a POPs pad — water separated on-pad; judge with the "
                "per-pad Marginal WC calc, not the field line."
            )
            ranks.append(4)
        elif pd.isna(wc) or stale:
            decisions.append("⚠️ Verify — stale/no test")
            whys.append("No recent representative test; re-test before any SI call.")
            ranks.append(1)
        elif wc <= marginal_wc:
            decisions.append("✅ Keep online")
            whys.append(
                f"WC {wc * 100:.0f}% ≤ marginal {marginal_wc * 100:.0f}% — worth its water."
                + basis_note
            )
            ranks.append(2)
        else:
            # Outlier protection: a single anomalous test — oil reading LOW
            # or water reading HIGH — inflates the apparent WC; verify with a
            # re-test instead of condemning the well on it (P1-28).
            oil_down = pd.notna(oil_dev) and float(oil_dev) < -OUTLIER_PCT
            wat_up = pd.notna(wat_dev) and float(wat_dev) > OUTLIER_PCT
            if oil_down or wat_up:
                anomalies = []
                if oil_down:
                    anomalies.append(
                        f"oil is {abs(float(oil_dev)) * 100:.0f}% below the 2-mo avg"
                    )
                if wat_up:
                    anomalies.append(
                        f"water is {float(wat_dev) * 100:.0f}% above the 2-mo avg"
                    )
                decisions.append("⚠️ Verify before SI")
                whys.append(
                    f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}%, but the "
                    f"latest test looks anomalous — {' and '.join(anomalies)} — "
                    "confirm with a re-test before SI (may recover)." + basis_note
                )
                ranks.append(1)
            else:
                decisions.append("🔴 SI candidate")
                whys.append(
                    f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}% — "
                    "water not worth the oil." + basis_note
                )
                ranks.append(0)
    df["WCvsMarginal"] = deltas
    df["WCBasis"] = bases
    df["Decision"] = decisions
    df["Why"] = whys
    df["_rank"] = ranks
    return df


def add_shut_decision(shut_df: pd.DataFrame, marginal_wc: float) -> pd.DataFrame:
    """Augment the shut-in (offline) table with a BOL decision vs marginal WC.

    Adds: Decision, Why, WCvsMarginal, NearAvgWC (90-day history WC),
    NearAvgWCBasis / WCBasis ("total"/"form"), and a hidden ``_rank``.

    The 90-day history WC is compared to a TOTAL-WC marginal line, so it is
    computed on the same basis wherever possible: the near-window SQL only
    averages FORMATION water (``AVG(form_wat)``), so when the last test
    carries lift-water data it is folded in as the window's lift proxy. Where
    only form-basis history exists it is NOT allowed to grant a BOL trial —
    form WC reads systematically low vs the total-WC line on lifted wells —
    the row is flagged to verify instead (P1-27).

    Rule:
      no test                                       -> ⚠️ Verify (no test)
      last WC ≤ marginal                            -> 🟢 BOL candidate
      last WC > marginal BUT 90-day TOTAL-basis
        hist WC ≤ marg                              -> 🔬 BOL trial (recovery?)
      last WC > marginal, form-basis hist ≤ marg    -> ⚠️ Verify (basis unreliable)
      last WC > marginal and history also high      -> ⏸️ Leave shut
    """
    df = shut_df.copy()
    if df.empty:
        for c in ("WCvsMarginal", "NearAvgWC", "_rank"):
            df[c] = pd.Series(dtype="float")
        for c in ("Decision", "Why", "WCBasis", "NearAvgWCBasis"):
            df[c] = pd.Series(dtype="object")
        return df

    decisions, whys, deltas = [], [], []
    near_wcs, near_bases, wc_bases, ranks = [], [], [], []
    for _, r in df.iterrows():
        wc, wc_basis = _effective_wc(r)
        deltas.append(wc - marginal_wc if pd.notna(wc) else float("nan"))
        wc_bases.append(wc_basis)
        basis_note = _form_basis_note(wc_basis)
        # 90-day near-last-test history WC: the "was it healthy recently?"
        # signal that protects against condemning a well on one bad test.
        # NearAvgWater is formation water only, but the marginal line is
        # total WC — when the last test has lift-water data, fold it in so
        # history and line share a basis; a form-only history is flagged
        # below rather than decided on (P1-27).
        na_oil, na_wat = r.get("NearAvgOil"), r.get("NearAvgWater")
        lift_wat = r.get("LiftWater")
        nwc, nwc_basis = float("nan"), None
        if pd.notna(na_oil) and pd.notna(na_wat):
            hist_oil, hist_wat = float(na_oil), float(na_wat)
            if pd.notna(lift_wat):
                hist_wat += float(lift_wat)
                nwc_basis = "total"
            else:
                nwc_basis = "form"
            if (hist_oil + hist_wat) > 0:
                nwc = hist_wat / (hist_oil + hist_wat)
            else:
                nwc_basis = None
        near_wcs.append(nwc)
        near_bases.append(nwc_basis)

        if pd.isna(wc):
            decisions.append("⚠️ Verify — no test")
            whys.append("No usable test on record; test before BOL.")
            ranks.append(2)
        elif wc <= marginal_wc:
            decisions.append("🟢 BOL candidate")
            whys.append(
                f"Last WC {wc * 100:.0f}% ≤ marginal {marginal_wc * 100:.0f}% — "
                "worth bringing on." + basis_note
            )
            ranks.append(0)
        elif pd.notna(nwc) and nwc <= marginal_wc:
            if nwc_basis == "total":
                decisions.append("🔬 BOL trial")
                whys.append(
                    f"Last WC {wc * 100:.0f}% > marginal, but the 90-day history WC "
                    f"{nwc * 100:.0f}% (total basis, incl lift water) was below — "
                    "BOL to see if the oil rate has recovered." + basis_note
                )
                ranks.append(1)
            else:
                # Form-basis history reads LOW vs the total-WC line — don't
                # grant a BOL trial on it; flag for a re-test instead.
                decisions.append("⚠️ Verify — form-basis history")
                whys.append(
                    f"Last WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}%; "
                    f"the 90-day history WC {nwc * 100:.0f}% is below the line but "
                    "formation-water basis only (no lift-water data) — unreliable "
                    "vs the total-WC line. Re-test before any BOL call."
                )
                ranks.append(2)
        else:
            # Safe on either basis: a form-basis history above the line
            # implies the total-basis history is above it too.
            decisions.append("⏸️ Leave shut")
            whys.append(
                f"WC {wc * 100:.0f}% > marginal {marginal_wc * 100:.0f}% (history too) — "
                "water not worth it." + basis_note
            )
            ranks.append(3)
    df["WCvsMarginal"] = deltas
    df["NearAvgWC"] = near_wcs
    df["NearAvgWCBasis"] = near_bases
    df["WCBasis"] = wc_bases
    df["Decision"] = decisions
    df["Why"] = whys
    df["_rank"] = ranks
    return df


_TRIAGE_ONLINE_COMPACT = [
    "Decision",
    "Why",
    "Well",
    "Pad",
    "Oil",
    "TotalWC",
    "WCvsMarginal",
    "GOR",
    "DaysSinceTest",
    "StaleTest",
    "FlagOutlier",
    "PopsPad",
]
_TRIAGE_ONLINE_DETAIL = _TRIAGE_ONLINE_COMPACT + [
    "Reservoir",
    "LiftType",
    "Water",
    "LiftWater",
    "TotalWater",
    "Gas",
    "TotalGas",
    "Oil_2moAvg",
    "OilDev",
    "BHP",
    "WHP",
    "ProdXV",
    "XVTime",
    "TestDate",
    "Allocated",
    "FallbackUsed",
]

_TRIAGE_SHUT_COMPACT = [
    "Decision",
    "Why",
    "Well",
    "Pad",
    "Oil",
    "TotalWC",
    "WCvsMarginal",
    "NearAvgWC",
    "NearAvgOil",
    "ShutInSince",
    "CurrentCode",
    "CurrentReason",
    "LastTestDate",
]
_TRIAGE_SHUT_DETAIL = _TRIAGE_SHUT_COMPACT + [
    "Reservoir",
    "LiftType",
    "Water",
    "Gas",
    "LiftWater",
    "TotalWater",
    "NearAvgWater",
    "NTestsNear",
    "Notes",
    "DownHours",
    "LastOnlineDate",
    "ProdXV",
]


def _triage_online_cfg(marginal_wc: float, stale_days: int) -> dict:
    return {
        "Decision": st.column_config.TextColumn(
            "Decision", pinned="left", width="medium"
        ),
        "Why": st.column_config.TextColumn("Why", width="large"),
        "Well": st.column_config.TextColumn("Well", pinned="left"),
        "Pad": st.column_config.TextColumn("Pad"),
        "Reservoir": st.column_config.TextColumn("Reservoir"),
        "LiftType": st.column_config.TextColumn("Lift Type"),
        "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
        "Water": st.column_config.NumberColumn("Form Water (BWPD)", format="%.0f"),
        "LiftWater": st.column_config.NumberColumn("Lift Water (BWPD)", format="%.0f"),
        "TotalWater": st.column_config.NumberColumn(
            "Total Water (BWPD)", format="%.0f"
        ),
        "Gas": st.column_config.NumberColumn("Gas (MCFD)", format="%.0f"),
        "TotalGas": st.column_config.NumberColumn("Total Gas (MCFD)", format="%.0f"),
        "TotalWC": st.column_config.NumberColumn(
            "Total WC (%)",
            format="%.1f",
            help="Latest-test total water cut (%). Compared to the marginal line.",
        ),
        "WCvsMarginal": st.column_config.NumberColumn(
            "WC − Marg (pp)",
            format="%+.1f",
            help=f"Total WC minus the marginal WC, in percentage points. "
            f"Positive = above the line (SI lean). Marginal = {marginal_wc * 100:.0f}%.",
        ),
        "GOR": st.column_config.NumberColumn("GOR (scf/bbl)", format="%.0f"),
        "DaysSinceTest": st.column_config.NumberColumn("Days since", format="%.0f"),
        "StaleTest": st.column_config.CheckboxColumn(
            "Stale?",
            help=f"Latest test older than {stale_days} days",
        ),
        "FlagOutlier": st.column_config.CheckboxColumn(
            "Outlier?",
            help="Latest test deviates >25% from the 2-month average on oil or water",
        ),
        "PopsPad": st.column_config.CheckboxColumn(
            "POPS?",
            help="Pad has on-pad water separation",
        ),
        "Oil_2moAvg": st.column_config.NumberColumn("Oil 2mo avg", format="%.0f"),
        "OilDev": st.column_config.NumberColumn("Oil Δ vs 2mo (%)", format="%.0f"),
        "BHP": st.column_config.NumberColumn("BHP (psi)", format="%.0f"),
        "WHP": st.column_config.NumberColumn("WHP (psi)", format="%.0f"),
        "ProdXV": st.column_config.NumberColumn(
            "Prod XV",
            format="%.0f",
            help="1=open, 0=closed",
        ),
        "XVTime": st.column_config.DatetimeColumn("XV Time", format="MM-DD HH:mm"),
        "TestDate": st.column_config.DatetimeColumn("Test Date", format="YYYY-MM-DD"),
        "Allocated": st.column_config.CheckboxColumn("Alloc."),
        "FallbackUsed": st.column_config.CheckboxColumn("Fallback"),
    }


def _triage_shut_cfg(marginal_wc: float) -> dict:
    return {
        "Decision": st.column_config.TextColumn(
            "Decision", pinned="left", width="medium"
        ),
        "Why": st.column_config.TextColumn("Why", width="large"),
        "Well": st.column_config.TextColumn("Well", pinned="left"),
        "Pad": st.column_config.TextColumn("Pad"),
        "Reservoir": st.column_config.TextColumn("Reservoir"),
        "LiftType": st.column_config.TextColumn("Lift Type"),
        "Oil": st.column_config.NumberColumn(
            "Last Oil (BOPD)",
            format="%.0f",
            help="Oil from the last test on record",
        ),
        "Water": st.column_config.NumberColumn("Last Form Water (BWPD)", format="%.0f"),
        "LiftWater": st.column_config.NumberColumn(
            "Last Lift Water (BWPD)", format="%.0f"
        ),
        "TotalWater": st.column_config.NumberColumn(
            "Last Total Water (BWPD)", format="%.0f"
        ),
        "Gas": st.column_config.NumberColumn("Last Gas (MCFD)", format="%.0f"),
        "TotalWC": st.column_config.NumberColumn("Last Total WC (%)", format="%.1f"),
        "WCvsMarginal": st.column_config.NumberColumn(
            "WC − Marg (pp)",
            format="%+.1f",
            help=f"Last-test Total WC minus the marginal WC, in percentage points. "
            f"Negative = below the line (BOL lean). Marginal = {marginal_wc * 100:.0f}%.",
        ),
        "NearAvgWC": st.column_config.NumberColumn(
            "90-day Hist WC (%)",
            format="%.1f",
            help="Form WC (%) averaged over tests within 90 days of the last test — "
            "the 'was it healthy recently' signal behind BOL-trial.",
        ),
        "NearAvgOil": st.column_config.NumberColumn("90-day Avg Oil", format="%.0f"),
        "NearAvgWater": st.column_config.NumberColumn(
            "90-day Avg Water", format="%.0f"
        ),
        "NTestsNear": st.column_config.NumberColumn("# Near Tests", format="%.0f"),
        "ShutInSince": st.column_config.DateColumn("Shut-In Since"),
        "LastOnlineDate": st.column_config.DateColumn("Last Online"),
        "LastTestDate": st.column_config.DatetimeColumn(
            "Last Test", format="YYYY-MM-DD"
        ),
        "CurrentCode": st.column_config.TextColumn("Code"),
        "CurrentReason": st.column_config.TextColumn("Reason"),
        "Notes": st.column_config.TextColumn("Notes"),
        "DownHours": st.column_config.NumberColumn("Down hrs", format="%.1f"),
        "ProdXV": st.column_config.NumberColumn("Prod XV", format="%.0f"),
    }


def _render_triage_online(
    df: pd.DataFrame,
    marginal_wc: float,
    stale_days: int,
    only_action: bool,
    show_all: bool,
) -> None:
    if df.empty:
        st.info("No online wells.")
        return
    work = df.copy()
    if only_action:
        work = work[work["_rank"].isin([0, 1])]
    if work.empty:
        st.success("No online wells above the marginal WC — nothing to review.")
        return
    work = work.sort_values(
        ["_rank", "WCvsMarginal"], ascending=[True, False]
    ).reset_index(drop=True)
    cols = _TRIAGE_ONLINE_DETAIL if show_all else _TRIAGE_ONLINE_COMPACT
    cols = [c for c in cols if c in work.columns]
    disp = work[cols].copy()
    for c in ("TotalWC", "WCvsMarginal"):
        if c in disp.columns:
            disp[c] = disp[c] * 100
    if "OilDev" in disp.columns:
        disp["OilDev"] = disp["OilDev"] * 100
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config=_triage_online_cfg(marginal_wc, stale_days),
    )
    st.download_button(
        "Download Online triage CSV",
        data=work.drop(columns=["_rank"], errors="ignore")
        .to_csv(index=False, float_format="%.4f")
        .encode("utf-8"),
        file_name="well_sort_triage_online.csv",
        mime="text/csv",
        key="triage_dl_online",
    )


def _render_triage_shut(
    df: pd.DataFrame,
    marginal_wc: float,
    only_action: bool,
    show_all: bool,
) -> None:
    if df.empty:
        st.info("No shut-in (offline) wells.")
        return
    work = df.copy()
    if only_action:
        work = work[work["_rank"].isin([0, 1, 2])]
    if work.empty:
        st.success("No shut-in wells below the marginal WC — no BOL candidates.")
        return
    work = work.sort_values(
        ["_rank", "WCvsMarginal"], ascending=[True, True]
    ).reset_index(drop=True)
    cols = _TRIAGE_SHUT_DETAIL if show_all else _TRIAGE_SHUT_COMPACT
    cols = [c for c in cols if c in work.columns]
    disp = work[cols].copy()
    for c in ("TotalWC", "WCvsMarginal", "NearAvgWC"):
        if c in disp.columns:
            disp[c] = disp[c] * 100
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config=_triage_shut_cfg(marginal_wc),
    )
    st.download_button(
        "Download Shut/BOL triage CSV",
        data=work.drop(columns=["_rank"], errors="ignore")
        .to_csv(index=False, float_format="%.4f")
        .encode("utf-8"),
        file_name="well_sort_triage_shut.csv",
        mime="text/csv",
        key="triage_dl_shut",
    )


def render_triage_tab() -> None:
    """Beta keep/SI/BOL decision view, driven by each well's WC vs the marginal.

    Sits alongside the original Wells tab (untouched) for back-to-back
    comparison. Reuses the same cached fetchers, classification, and POPs-pad
    settings as the Wells tab — only the decision layer + decluttered display
    are new.
    """
    from woffl.assembly.well_sort_client import (
        apply_pops_pad,
        build_online_table,
        build_shut_in_table,
        classify_wells,
        split_offline_ltsi,
    )

    st.header("Well Sort — Triage (beta)")
    st.caption(
        "Experimental decision view. Each well's water cut is compared to the "
        "field **marginal WC**: online wells above the line are shut-in (SI) "
        "candidates; shut wells below the line are bring-on-line (BOL) "
        "candidates. A poor latest test over a healthy recent history is "
        "flagged to **verify / BOL-trial** rather than acted on. The original "
        "**Wells** tab is unchanged — compare the two and tell me when this is "
        "ready to take over. POPs-pad settings are shared with the Wells tab."
    )

    c1, c2, c3, c4 = st.columns([1.1, 1.7, 1.5, 1.5])
    with c1:
        if st.button(
            "Refresh data",
            key="triage_refresh",
            help="Clear cache and re-query Databricks",
        ):
            for fn in (
                _cached_shut_in_history,
                _cached_recent_tests,
                _cached_producers,
                _cached_producer_catalog,
                _cached_last_tests_ever,
                _cached_xv_status,
            ):
                fn.clear()
            st.rerun()
    with c2:
        threshold = st.number_input(
            "Marginal WC buffer (% of field water)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.5,
            format="%.1f",
            key="triage_marg_threshold",
            help="Noise buffer on the marginal WC. Skips the worst-WC wells that "
            "make up this % of the field's water before reading the marginal — so "
            "one tiny 99%-WC stripper doesn't set the line. 0 = use the literal "
            "worst well; bigger buffer → lower marginal WC. Typical 1–3%.",
        )
    with c3:
        stale_days = st.slider(
            "Stale-test threshold (days)",
            14,
            180,
            60,
            key="triage_stale_days",
            help="Wells whose latest test is older than this are sent to 'verify'.",
        )
    with c4:
        only_action = st.toggle(
            "Only wells needing a decision",
            value=True,
            key="triage_only_action",
            help="Hide healthy 'keep' / 'leave shut' wells; show only "
            "SI / BOL / verify candidates.",
        )
        show_all = st.toggle(
            "Show all columns",
            value=False,
            key="triage_show_all",
            help="Expand from the triage columns to the full detail set.",
        )

    shut_in_hist = _cached_shut_in_history()
    tests = _cached_recent_tests(180)
    producers = _cached_producers()
    catalog = _cached_producer_catalog()
    last_tests = _cached_last_tests_ever()
    xv = _cached_xv_status()
    if xv.empty:
        # P1-33: match the Wells tab's warning — Triage was silently
        # degrading to the shut-in log alone with no indication why.
        st.caption(
            "ℹ️ Safety-valve (XV) status unavailable — ProdXV/PFXV columns are "
            "blank and classification falls back to the shut-in log only. (On "
            "Databricks this usually means the app lacks access to the "
            "`reporting` catalog.)"
        )

    if not producers:
        st.error("No producers returned from vw_well_header.")
        return

    # Inherit POPs-pad config from the Wells tab (same session keys, GC-safe
    # mirror fallback per P1-31).
    pops_pads = list(st.session_state.get("well_sort_pops_pads", _pops_pads_fallback()))
    force_true = st.session_state.get(
        "well_sort_pops_force_true", _pops_force_true_fallback()
    )
    overrides = {w: True for w in force_true}

    online_set, shut_set = classify_wells(
        producers, shut_in_hist, xv_df=xv, trust_xv=True
    )
    online_df = build_online_table(
        tests,
        shut_in_hist,
        producers,
        mode="allocated",
        stale_days=stale_days,
        xv_df=xv,
        online_wells=online_set,
        catalog_df=catalog,
    )
    shut_df = build_shut_in_table(
        shut_in_hist,
        tests,
        xv_df=xv,
        shut_in_wells=shut_set,
        catalog_df=catalog,
        last_tests_df=last_tests,
    )
    online_df = apply_pops_pad(online_df, set(pops_pads), overrides)
    shut_df = apply_pops_pad(shut_df, set(pops_pads), overrides)
    offline_df, _ltsi_df = split_offline_ltsi(shut_df)

    marg = compute_field_marginal_wc(threshold_pct=threshold)
    if marg is None:
        st.warning(
            "Can't compute the field marginal WC (no online non-POPs wells with "
            "valid water data). Check the Wells tab."
        )
        return
    marginal_wc = marg["marginal_wc"]

    online_dec = add_online_decision(online_df, marginal_wc)
    offline_dec = add_shut_decision(offline_df, marginal_wc)

    n_si = int((online_dec["_rank"] == 0).sum()) if not online_dec.empty else 0
    n_verify = int((online_dec["_rank"] == 1).sum()) if not online_dec.empty else 0
    n_bol = int((offline_dec["_rank"] == 0).sum()) if not offline_dec.empty else 0
    n_trial = int((offline_dec["_rank"] == 1).sum()) if not offline_dec.empty else 0

    # Raw (no-buffer) marginal = the single worst-WC well — usually a tiny
    # stripper. Shown next to the buffered marginal so the buffer's effect is
    # obvious ("94% instead of 99% because one small well was online").
    ranked = marg.get("ranked_df")
    raw_wc = raw_well = None
    raw_water = 0.0
    if ranked is not None and not ranked.empty:
        raw_row = ranked.iloc[0]  # list is sorted by TotalWC desc
        raw_wc = float(raw_row["TotalWC"])
        raw_well = str(raw_row["Well"])
        raw_water = float(raw_row["TotalWater"])

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "Marginal WC (cut line)",
        f"{marginal_wc * 100:.0f}%",
        delta=f"set by {marg['well']} ({marg['pad']})",
        delta_color="off",
        help="The decision line, AFTER the buffer. Online wells above it lean SI; "
        "shut wells below it lean BOL.",
    )
    mc2.metric(
        "Buffer",
        f"{threshold:.1f}% of field water",
        help="Set in the box above. Skips the worst-WC wells making up this % of "
        "field water before reading the marginal — stops one tiny high-WC well "
        "from setting the line.",
    )
    if raw_wc is not None:
        mc3.metric(
            "Worst single well (no buffer)",
            f"{raw_wc * 100:.0f}%",
            delta=f"{raw_well} · {raw_water:,.0f} BWPD",
            delta_color="off",
            help="What the line would be with NO buffer — usually a small stripper. "
            "The buffer skips past it.",
        )
    else:
        mc3.metric("Worst single well (no buffer)", "—")

    if raw_wc is not None and raw_well != marg["well"]:
        st.caption(
            f"With a **{threshold:.1f}%** buffer the marginal WC is "
            f"**{marginal_wc * 100:.0f}%** (set by {marg['well']}) instead of "
            f"**{raw_wc * 100:.0f}%** from {raw_well} — a small well making only "
            f"{raw_water:,.0f} BWPD. Lower the buffer toward 0 to follow the worst "
            "well; raise it to lean on bigger-volume wells."
        )
    elif raw_wc is not None:
        st.caption(
            f"The {threshold:.1f}% buffer isn't skipping any wells yet — the "
            f"marginal WC ({marginal_wc * 100:.0f}%) is still set by the worst "
            f"well {marg['well']}. Raise the buffer to skip small high-WC wells."
        )

    st.markdown("**Decisions**")
    h2, h3, h4, h5 = st.columns(4)
    h2.metric(
        "🔴 SI candidates",
        n_si,
        help="Online, WC above marginal, latest test looks representative.",
    )
    h3.metric(
        "⚠️ Verify",
        n_verify,
        help="WC above marginal but latest oil is an outlier-low, or stale/no test "
        "— confirm before SI.",
    )
    h4.metric(
        "🟢 BOL candidates",
        n_bol,
        help="Shut, last WC below marginal — worth bringing on.",
    )
    h5.metric(
        "🔬 BOL trials",
        n_trial,
        help="Shut, last test poor but recent history was good — BOL to confirm recovery.",
    )

    with st.expander("What the decisions mean"):
        st.markdown(
            "- ✅ **Keep online** — WC at/below marginal; water is worth the oil.\n"
            "- 🔴 **SI candidate** — online, WC above marginal, and the latest test looks real.\n"
            "- ⚠️ **Verify** — WC above marginal but the latest oil dropped sharply vs the "
            "2-month average (or no recent test). Often recovers — re-test or BOL-trial before SI.\n"
            "- 🟢 **BOL candidate** — shut, last WC below marginal; bring it on.\n"
            "- 🔬 **BOL trial** — shut, last test poor but the 90-day history WC was below "
            "marginal; BOL to see if the rate recovered.\n"
            "- ⏸️ **Leave shut** — shut, WC above marginal on both the last test and the history.\n"
            "- ⚪ **POPS (own handling)** — pad separates its own water; the field marginal line "
            "doesn't apply. Use the **Marginal WC** tab's per-pad calc.\n\n"
            "LTSI wells (long-term shut-in / mechanical) are out of scope here — see the Wells tab."
        )

    sub_online, sub_shut = st.tabs(
        [
            f"Online — SI review ({len(online_dec)})",
            f"Shut — BOL review ({len(offline_dec)})",
        ]
    )
    with sub_online:
        _render_triage_online(
            online_dec, marginal_wc, stale_days, only_action, show_all
        )
    with sub_shut:
        _render_triage_shut(offline_dec, marginal_wc, only_action, show_all)
