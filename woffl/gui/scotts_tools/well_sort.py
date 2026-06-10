"""Well Sort tab.

Online vs Shut-In classification + outlier flagging across MPU producers.
Live ProdXV status overrides the daily shut-in log symmetrically: a closed
XV forces shut-in (catches just-shut wells), an open XV rescues a logged-
shut well back to online (catches just-restarted wells).
"""

import pandas as pd
import streamlit as st


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

    # On-pad separation settings (persist across interactions). Defaults AND
    # remembered selections are intersected with the live options — a value
    # missing from options (empty catalog on a transient Databricks failure,
    # a well dropping out of vw_well_header) raises StreamlitAPIException
    # and kills the whole page.
    all_pads = sorted(catalog["well_pad"].dropna().unique().tolist()) if not catalog.empty else []
    if "well_sort_pops_pads" in st.session_state:
        st.session_state["well_sort_pops_pads"] = [
            p for p in st.session_state["well_sort_pops_pads"] if p in all_pads
        ]
    pops_pads = st.multiselect(
        "Pads with on-pad production separation",
        options=all_pads,
        default=[
            p
            for p in st.session_state.get(
                "well_sort_pops_pads", ["E", "F", "H", "I", "M", "S"]
            )
            if p in all_pads
        ],
        key="well_sort_pops_pads",
        help="Wells on these pads get PopsPad=True. Per-well overrides apply after.",
    )
    # Per-well PopsPad=True overrides (wells that get True even if their pad
    # doesn't have separation — e.g. MPS-08 in the Apr-20 bench sheet).
    producer_opts = sorted(producers)
    if "well_sort_pops_force_true" in st.session_state:
        st.session_state["well_sort_pops_force_true"] = [
            w for w in st.session_state["well_sort_pops_force_true"] if w in producer_opts
        ]
    force_true_wells = st.multiselect(
        "Per-well PopsPad=True overrides",
        options=producer_opts,
        default=[
            w
            for w in st.session_state.get("well_sort_pops_force_true", [])
            if w in producer_opts
        ],
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
        [f"Online ({len(online_df)})",
         f"Offline ({len(offline_df)})",
         f"LTSI ({len(ltsi_df)})",
         "30-Day Changes"]
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

    _shut_col_order = [
        "Well", "Pad", "Reservoir", "LiftType",
        "ShutInSince", "CurrentCode", "CurrentReason", "Notes", "DownHours",
        "Oil", "Water", "Gas",
        "LastOnlineDate", "LastTestDate",
        "WC", "TotalWC", "GOR", "TotalGOR",
        "LiftWater", "LiftGas", "TotalWater", "TotalGas",
        "EspHz", "EspAmps",
        "NearAvgOil", "NearAvgWater", "NearAvgGas", "NTestsNear",
        "ProdXV", "PFXV", "XVTime", "PopsPad",
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

    with sub_changes:
        from woffl.assembly.well_sort_client import (
            NOTABLE_DOWN_HOURS, compute_recent_down_events,
        )
        th_col, _ = st.columns([1, 3])
        with th_col:
            threshold = st.slider(
                "Min hrs/day to count as 'down'", 1, 24,
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
            recent_hist, producers, catalog_df=catalog,
            window_days=30, down_hours_threshold=float(threshold),
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
                events_df, use_container_width=True, hide_index=True,
                column_config={
                    "Well": st.column_config.TextColumn("Well", pinned="left"),
                    "Pad": st.column_config.TextColumn("Pad"),
                    "Reservoir": st.column_config.TextColumn("Reservoir"),
                    "Started": st.column_config.DateColumn("Started"),
                    "Ended": st.column_config.DateColumn(
                        "Ended", help="Blank when the event is still ongoing",
                    ),
                    "Days": st.column_config.NumberColumn(
                        "Days", format="%.0f",
                        help="Consecutive days at or above the threshold",
                    ),
                    "MaxHrs": st.column_config.NumberColumn(
                        "Max Hrs", format="%.1f",
                        help="Peak single-day down_hours in this event",
                    ),
                    "TotalHrs": st.column_config.NumberColumn(
                        "Total Hrs", format="%.1f",
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

    pops_pads = st.session_state.get(
        "well_sort_pops_pads", ["E", "F", "H", "I", "M", "S"]
    )
    force_true = st.session_state.get("well_sort_pops_force_true", [])
    overrides = {w: True for w in force_true}

    online_set, _ = classify_wells(producers, shut_in_hist, xv_df=xv, trust_xv=True)
    online_df = build_online_table(
        tests, shut_in_hist, producers, mode="allocated",
        stale_days=stale_days, xv_df=xv, online_wells=online_set,
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

    valid = pad_df[
        pad_df[water_col].notna() & pad_df["Oil"].notna()
    ].copy()
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
            st.success(
                f"Imported {marginal_wc:.3f} into sidebar Marginal Watercut."
            )

    # Ranked-wells table
    st.subheader("Ranked Online Wells (non-POPs)")

    display = valid.copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display["Marginal"] = display.index == marg_idx
    cols = [
        "Rank", "Marginal", "Well", "Pad", "Reservoir",
        "Oil", "TotalWater", "TotalWC", "CumWater", "CumWaterPct",
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
                "Marginal?", help="Well at which cumulative water crosses the threshold",
            ),
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Pad": st.column_config.TextColumn("Pad"),
            "Reservoir": st.column_config.TextColumn("Reservoir"),
            "Oil": st.column_config.NumberColumn("Oil (BOPD)", format="%.0f"),
            "TotalWater": st.column_config.NumberColumn(
                "Total Water (BWPD)", format="%.0f",
            ),
            "TotalWC": st.column_config.NumberColumn(
                "Total WC", format="%.3f",
            ),
            "CumWater": st.column_config.NumberColumn(
                "Cum Water (BWPD)", format="%.0f",
                help="Running total of TotalWater from the worst-WC well downward",
            ),
            "CumWaterPct": st.column_config.NumberColumn(
                "Cum %", format="%.1f%%",
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


def _reset_pad_limit(pad: str) -> None:
    """on_click callback — restore a pad's pump limit to its preset.

    Runs before the widget renders on the next rerun, so writing to the
    widget's session-state key is safe here.
    """
    preset = PUMP_LIMIT_PRESETS.get(pad, 0)
    st.session_state[_pad_limit_key(pad)] = int(preset)


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
        "the WC of the highest-WC well still kept online after shedding the "
        "worst-WC wells down to the pump's capacity. **E / F / M** handle "
        "the full produced stream (formation + PF). **I / H / S** are "
        "PF-only — their pump only sees lift water, and formation water "
        "passes through to the central facility."
    )

    pops_pads = sorted(
        st.session_state.get(
            "well_sort_pops_pads", ["E", "F", "H", "I", "M", "S"]
        )
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

    # Seed the pump-limit widget key from the preset on first visit so the
    # number_input picks it up; subsequent edits persist in session_state.
    pump_limit_key = _pad_limit_key(pad)
    if pump_limit_key not in st.session_state:
        st.session_state[pump_limit_key] = int(PUMP_LIMIT_PRESETS.get(pad, 0))

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
        "Pump Limit", f"{pump_limit:,.0f} BWPD",
        help="Editable above. Reset returns to the preset.",
    )
    if headroom is not None:
        # Phrasing differs by basis: for PF-only pads the headroom is
        # specifically PF capacity to allocate to more wells.
        capacity_label = "PF" if water_basis == "lift" else "water"
        if headroom >= 0:
            delta_label = (
                f"{headroom:,.0f} BWPD {capacity_label} available to allocate"
            )
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
        pad_display = pad_valid.rename(columns={
            "LiftWater": "PFRate",
            "WC_pad": "PFWC",
        })
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
        "Rank", "Well", "Pad", "Reservoir",
        "Oil", water_col_display, wc_col_display,
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
                water_col_label, format="%.0f",
                help=(
                    "Water stream the pad's pump actually handles "
                    f"({pad_result['water_col']} from the source data)."
                ),
            ),
            wc_col_display: st.column_config.NumberColumn(
                wc_col_label, format="%.3f",
                help=(
                    f"{wc_col_label} = {water_col_display} / "
                    f"({water_col_display} + Oil). Marginal WC = max."
                ),
            ),
        },
    )
