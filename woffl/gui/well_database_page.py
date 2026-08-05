"""Well Database Viewer — read-only view of mech + reservoir properties from Databricks."""

import pandas as pd
import streamlit as st

from woffl.gui.utils import load_well_characteristics


def _render_unavailable(detail: str) -> None:
    """Error + retry instead of a silently blank page.

    Failures are never cached (see utils.load_well_characteristics), so the
    Retry button's rerun re-probes Databricks; clear() is belt-and-braces in
    case a stale success entry is what's actually being served.
    """
    st.error(f"Well properties unavailable: {detail}")
    if st.button("Retry", key="well_db_retry"):
        load_well_characteristics.clear()
        st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def _latest_test_dates() -> dict:
    """{well: latest ALLOCATED test date} — the recency proxy for "online".

    Fail-soft to {}: the aging filter then disables itself rather than
    silently dropping every well.
    """
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    try:
        df = execute_query(
            """
            SELECT well_name, max(wt_date) AS last_test
            FROM mpu.wells.vw_well_test
            WHERE allocated = True
            GROUP BY well_name
            """
        )
        return {
            _normalize_well_name(str(r["well_name"]).strip()): r["last_test"]
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


def run_well_database_page():
    st.title("Well Database")
    st.caption(
        "Live view of mpu.wells.vw_prop_mech + vw_prop_resvr (Databricks). "
        "JP_TVD computed locally from deviation surveys."
    )

    try:
        df = load_well_characteristics()
    except Exception as e:
        _render_unavailable(str(e))
        return
    if df.empty:
        _render_unavailable("the well-properties source returned no rows")
        return

    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

    n_estimated = int(df["tvd_estimated"].sum()) if "tvd_estimated" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wells", len(df))
    col2.metric("Schrader", int(df["is_sch"].sum()) if "is_sch" in df.columns else "—")
    col3.metric(
        "Pads",
        (
            df["Well"].str.extract(r"(MP[A-Z])")[0].nunique()
            if "Well" in df.columns
            else "—"
        ),
    )
    col4.metric("Estimated TVD", n_estimated, help="Wells lacking a deviation survey")

    search = st.text_input("Filter wells", placeholder="e.g. MPH, MPS, MPM-10")
    if search:
        # regex=False: str.contains defaults to regex, so typing "(" or "["
        # raised an uncaught re.error and red-screened the page.
        mask = df["Well"].str.contains(
            search.strip(), case=False, na=False, regex=False
        )
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
            "casing_out_dia": st.column_config.NumberColumn(
                "Csg OD (in)", format="%.3f"
            ),
            "casing_inn_dia": st.column_config.NumberColumn(
                "Csg ID (in)", format="%.3f"
            ),
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

    # ── Aging jet pumps: who hasn't had a change in a long time ──────────────
    # Tenure = set-to-set (the JPCO rule — Date Pulled is never consulted).
    st.divider()
    st.subheader("Aging jet pumps")
    st.caption(
        "Days each well's CURRENT pump has been in hole (latest *Date Set* → "
        "today). A long tenure can mean a well worth a wash-out check, a "
        "resize look, or just a reliable pump — the Solver's JP History view "
        "has the per-well story."
    )

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or getattr(jp_hist, "empty", True):
        st.info(
            "JP history isn't loaded yet — open any well in Single Well "
            "Analysis once (the app warms it on startup) and revisit."
        )
    else:
        from woffl.assembly.jp_history import pump_ages
        from woffl.assembly.pump_report import format_pump

        ages = pump_ages(jp_hist)
        only_known = st.checkbox(
            "Only wells in the table above",
            value=True,
            key="wdb_age_known_only",
            help=(
                "Drops tracker wells not in vw_prop_mech (converted to ESP, "
                "retired, or never characterized)."
            ),
        )
        if only_known and "Well" in df.columns and not ages.empty:
            ages = ages[ages["Well Name"].isin(set(df["Well"].astype(str)))]

        # "Online recently" filter (Scott 2026-07-31): a well SI'd for years
        # shows a very old pump — true but not actionable the same way.
        # Evidence = latest allocated well test within the window (the same
        # recency proxy Well Sort's stale-days logic uses).
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            online_only = st.checkbox(
                "Only wells online recently",
                value=True,
                key="wdb_age_online_only",
                help=(
                    "Keeps wells with an ALLOCATED well test inside the "
                    "window — the app's standard recency proxy for online."
                ),
            )
        with fc2:
            online_days = st.number_input(
                "…with a test in the last (days)",
                min_value=7,
                max_value=365,
                value=60,
                step=7,
                key="wdb_age_online_days",
                disabled=not online_only,
            )
        if online_only and not ages.empty:
            from woffl.assembly.jp_history import filter_recently_online

            last_tests = _latest_test_dates()
            if last_tests:
                ages = filter_recently_online(ages, last_tests, int(online_days))
            else:
                st.caption(
                    "⚠ Well-test dates unavailable — online filter skipped."
                )

        if ages.empty:
            st.info("No jet pump installs on record for these wells.")
        else:
            threshold = st.number_input(
                "Flag pumps older than (days)",
                min_value=30,
                max_value=3650,
                value=365,
                step=30,
                key="wdb_age_threshold",
            )
            over = ages[ages["Days In Hole"] >= threshold]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("JP wells tracked", len(ages))
            a2.metric(f"Older than {threshold:,} d", len(over))
            a3.metric("Oldest", f"{int(ages['Days In Hole'].max()):,} d")
            a4.metric("Median age", f"{int(ages['Days In Hole'].median()):,} d")

            view = ages.copy()
            view["Pump"] = [
                format_pump(n, t)
                for n, t in zip(
                    view.get("Nozzle Number"), view.get("Throat Ratio")
                )
            ]
            view["Years"] = (view["Days In Hole"] / 365.25).round(1)
            view["Over"] = view["Days In Hole"] >= threshold
            show_cols = [
                c
                for c in ("Well Name", "Pump", "Date Set", "Days In Hole",
                          "Years", "Installs", "Last Test", "Over")
                if c in view.columns
            ]
            export = view[show_cols].rename(
                columns={"Well Name": "Well", "Days In Hole": "Days"}
            )
            st.dataframe(
                export,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date Set": st.column_config.DatetimeColumn(
                        "Set", format="YYYY-MM-DD"
                    ),
                    "Days": st.column_config.NumberColumn(format="%d"),
                    "Years": st.column_config.NumberColumn(format="%.1f"),
                    "Installs": st.column_config.NumberColumn(
                        help="Installs on record — frequent changers vs never-touched"
                    ),
                    "Last Test": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD",
                        help="Latest allocated well test (the online proxy)",
                    ),
                    "Over": st.column_config.CheckboxColumn(
                        f"> {threshold:,} d"
                    ),
                },
            )

            # Single-click Excel export of EXACTLY the filtered view above
            # (the house rule: build the bytes in the handler, auto-download).
            if st.button("⬇ Export to Excel", key="wdb_age_export"):
                import io

                from woffl.gui.components.download import autodownload

                buf = io.BytesIO()
                xdf = export.copy()
                for col in ("Date Set", "Last Test"):
                    if col in xdf.columns:  # excel-safe: drop timezones
                        xdf[col] = pd.to_datetime(
                            xdf[col], utc=True, errors="coerce"
                        ).dt.tz_localize(None)
                xdf.to_excel(
                    buf, index=False, sheet_name="Aging Jet Pumps",
                    engine="openpyxl",
                )
                stamp = pd.Timestamp.today().strftime("%Y-%m-%d")
                autodownload(
                    buf.getvalue(),
                    f"aging_jet_pumps_{stamp}.xlsx",
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet",
                )

    # ── Save history: the prop_hist audit trail for one well ─────────────────
    # Every 📌 save, pad-review push, friction calibration, IPR pin and the
    # original DART bulk load — value, timestamp, user (Scott, 2026-07-30).
    st.divider()
    st.subheader("Save history")
    st.caption(
        "Everything ever written to `mpu.wells.prop_hist` for a well — the "
        "append-only audit trail behind saved IPRs, calibrations and property "
        "edits. The **current** rows are what the well opens with."
    )

    wells = sorted(df["Well"].dropna().astype(str)) if "Well" in df.columns else []
    if not wells:
        st.info("No wells available.")
        return
    pick, refresh = st.columns([5, 1])
    with pick:
        hist_well = st.selectbox("Well", wells, key="wdb_hist_well")

    from woffl.assembly.prop_hist_client import format_alaska
    from woffl.gui.prop_history import fetch_prop_history, shape_history

    with refresh:
        st.write("")  # nudge the button down onto the selectbox's baseline
        if st.button(
            "↻ Refresh",
            key="wdb_hist_refresh",
            help="Re-read prop_hist now. The history is cached for 5 minutes "
            "so that sorting or switching wells doesn't re-query.",
        ):
            fetch_prop_history.clear()
            st.rerun()

    try:
        with st.spinner(f"Loading property history for {hist_well}…"):
            shaped = shape_history(fetch_prop_history(hist_well))
    except Exception as e:
        st.warning(f"Could not load property history: {e}")
        return

    if shaped is None:
        st.info(
            f"No saved property rows for {hist_well} yet — nothing has been "
            "written to prop_hist for this well."
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total saves", shaped["n_edits"])
    m2.metric("Properties touched", shaped["n_props"])
    m3.metric(
        "Last save",
        format_alaska(shaped["last_edit"], "%Y-%m-%d %H:%M"),
        help="Alaska time (AKDT/AKST); stored as UTC.",
    )
    m4.metric("Editors", len(shaped["editors"]), help=", ".join(shaped["editors"]))

    st.markdown("##### Current stored state (what the well opens with)")
    st.dataframe(
        shaped["latest"][
            ["category", "prop_name", "display_value", "units", "derivation",
             "entry_datetime_ak", "entry_user", "comment"]
        ].rename(
            columns={
                "category": "Category", "prop_name": "Property",
                "display_value": "Value", "units": "Units",
                "derivation": "How", "entry_datetime_ak": "Saved (AK)",
                "entry_user": "By", "comment": "Why",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Saved (AK)": st.column_config.DatetimeColumn(
                "Saved (AK)",
                format="YYYY-MM-DD HH:mm",
                help="Alaska time (AKDT/AKST). Stored as UTC — the column is an "
                "ordering key, so it can't be written in a zone that repeats an "
                "hour every fall.",
            ),
            # Saved IPR rate is DERIVED (oil ÷ (1 − WC)), so it matches no
            # well test whenever the WC is locked — spell the arithmetic out
            # rather than let it read as a bad number.
            "How": st.column_config.TextColumn(
                "How",
                width="medium",
                help="Shown when a stored value is computed rather than "
                "measured.",
            ),
            # The engineer's note for the save this value came from — the
            # whole point of woffl_eng_comment. Wide, because a truncated
            # reason is no reason.
            "Why": st.column_config.TextColumn(
                "Why",
                width="large",
                help="Note the engineer left when they saved this value.",
            ),
        },
    )

    with st.expander(f"Full history — all {shaped['n_edits']} save(s)"):
        st.dataframe(
            shaped["history"][
                ["entry_datetime_ak", "prop_name", "display_value", "units",
                 "derivation", "entry_user", "is_current", "comment"]
            ].rename(
                columns={
                    "entry_datetime_ak": "When (AK)", "prop_name": "Property",
                    "display_value": "Value", "units": "Units",
                    "derivation": "How", "entry_user": "By",
                    "is_current": "Current", "comment": "Why",
                }
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "When (AK)": st.column_config.DatetimeColumn(
                    "When (AK)", format="YYYY-MM-DD HH:mm"
                ),
                "Current": st.column_config.CheckboxColumn(
                    "Current", help="The live row — superseded rows unchecked"
                ),
                "How": st.column_config.TextColumn("How", width="medium"),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
        )
        st.caption(
            "Rows never delete — a save supersedes, '(cleared)' rows are NULL "
            "tombstones (e.g. an un-pinned IPR anchor). The 2026-04-16 block "
            "is the original DART bulk load."
        )
