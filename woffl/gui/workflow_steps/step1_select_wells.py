"""Step 1: Select Wells — pad/date selection from Databricks or CSV upload."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from woffl.assembly.well_test_client import (
    filter_wells_by_pad,
    get_pad_names,
)
from woffl.gui.workflow_page import _clear_downstream

# Same default as well_sort.py. Read live from Well Sort's session_state when
# the user has visited that tab — keeps the POPS list as a single source of
# truth across the app.
_DEFAULT_POPS_PADS = ("E", "F", "H", "I", "M", "S")

# Pads supported by the Phase 1 pad-scope optimization. These are the
# PF-only POPs pads — their pad pump only handles lift water, which maps
# directly to the existing PowerFluidConstraint in NetworkOptimizer. Full
# POPs pads (E, F, M) need a TotalWaterConstraint we haven't built yet.
_PAD_SCOPE_PADS = ["I", "H", "S"]


def _get_pops_pads() -> set[str]:
    """Active POPS-pad set: Well Sort's session_state if present, else default."""
    raw = st.session_state.get("well_sort_pops_pads", _DEFAULT_POPS_PADS)
    return set(raw or ())


def _active_pad_scope() -> str | None:
    """Currently active pad-scope letter, or None when in Field-wide mode.

    Reads the live Step-1 widgets when present, falling back to the
    ``_uw_scope`` / ``_uw_scope_pad`` shadow keys written by the Load
    handler. The widget keys (``uw_scope`` / ``uw_pad_scope_pad``) are
    garbage-collected the moment Step 1 stops rendering, so Steps 3/4 read
    THIS — reading the widget keys directly made Pad mode silently degrade
    to Field-wide by the time the user reached Configure & Run.
    """
    scope = st.session_state.get("uw_scope", st.session_state.get("_uw_scope"))
    if scope != "Pad":
        return None
    return st.session_state.get(
        "uw_pad_scope_pad", st.session_state.get("_uw_scope_pad")
    )


def render_step1():
    st.subheader("Step 1: Select Wells")

    data_source = st.radio(
        "Data Source",
        ["Databricks", "Upload CSV Template"],
        horizontal=True,
        key="uw_data_source_radio",
    )

    if data_source == "Databricks":
        _render_databricks_path()
    else:
        _render_csv_upload_path()


def _render_databricks_path():
    """Pad selection + date range → load well tests from Databricks."""
    from woffl.assembly.well_test_client import _normalize_well_name
    from woffl.gui.utils import load_well_characteristics
    from woffl.gui.well_test_cache import (
        _cached_mpu_well_names,
        _cached_well_test_query,
    )

    # Fetch well names and filter to JP wells
    try:
        with st.spinner("Fetching well list from Databricks..."):
            db_well_names = _cached_mpu_well_names()
    except Exception as e:
        st.error(f"Could not fetch well names from Databricks: {e}")
        return

    jp_wells = set(load_well_characteristics()["Well"].tolist())
    all_well_names = [w for w in db_well_names if _normalize_well_name(w) in jp_wells]

    if not all_well_names:
        st.warning("No JP wells found. Check Databricks connectivity and jp_chars.csv.")
        return

    all_pads = get_pad_names(all_well_names)

    # --- Scope: Field-wide vs single pad ---------------------------------
    # Pad scope locks the optimizer to one pad's wells and (in Step 3)
    # pre-fills the PF constraint from that pad's pump limit. Phase 1
    # only supports PF-only POPs pads (I, H, S) since those map cleanly
    # to PowerFluidConstraint. Full POPs (E, F, M) need a separate
    # constraint class — coming in Phase 2.
    scope_col, pad_col = st.columns([1, 2])
    with scope_col:
        # index restores the selection from the shadow key after a step
        # detour garbage-collects the widget state (Streamlit ignores index
        # when the widget state survived).
        scope = st.radio(
            "Scope",
            options=["Field-wide", "Pad"],
            horizontal=True,
            key="uw_scope",
            index=1 if st.session_state.get("_uw_scope") == "Pad" else 0,
            help=(
                "**Field-wide** = optimize across multiple pads under "
                "a total-PF cap. **Pad** = optimize all wells on a "
                "single PF-only POPs pad subject to that pad's pump "
                "capacity (Phase 1: I/H/S only)."
            ),
        )
    with pad_col:
        if scope == "Pad":
            current_choice = st.session_state.get(
                "uw_pad_scope_pad",
                st.session_state.get("_uw_scope_pad", _PAD_SCOPE_PADS[0]),
            )
            available = [p for p in _PAD_SCOPE_PADS if p in all_pads]
            if not available:
                st.warning(
                    "No PF-only POPs pads (I, H, S) found in the well list. "
                    "Switch to Field-wide scope."
                )
                return
            if current_choice not in available:
                current_choice = available[0]
            pad_choice = st.selectbox(
                "Pad",
                options=available,
                index=available.index(current_choice),
                key="uw_pad_scope_pad",
                help=(
                    "PF-only POPs pad. Each pad's PF pump only handles "
                    "lift water; formation water passes through to central."
                ),
            )
        else:
            # Clear pad-mode tracking when the user switches back to
            # Field-wide so re-entering pad mode triggers a fresh
            # auto-tick on the next change.
            st.session_state.pop("uw_pad_scope_pad", None)
            st.session_state.pop("_uw_last_scope_pad", None)
            pad_choice = None

    # Auto-tick the pad's checkbox + clear others when pad mode is on and
    # the user changes the dropdown. Runs BEFORE the checkbox grid below
    # so the widget keys can be written safely.
    if scope == "Pad" and pad_choice:
        last_pad = st.session_state.get("_uw_last_scope_pad")
        if last_pad != pad_choice:
            for pad in all_pads:
                st.session_state[f"uw_pad_{pad}"] = (pad == pad_choice)
            st.session_state["_uw_last_scope_pad"] = pad_choice

    # Quick-select buttons. We mutate the pad-checkbox session_state keys
    # BEFORE the widgets render below; Streamlit forbids writing to a
    # widget key after the widget exists, so the rerun puts the new values
    # into the freshly-rendered checkboxes.
    st.write("### Select Pads")
    qs_cols = st.columns([1.4, 1.4, 1.4, 4])
    pops_set = _get_pops_pads()
    with qs_cols[0]:
        if st.button(
            "Select Non-POPS pads",
            help=f"Sets pads not in {sorted(pops_set)} (POPS list from Well Sort). "
            "Adjust the POPS list on the Well Sort tab if needed.",
        ):
            for pad in all_pads:
                st.session_state[f"uw_pad_{pad}"] = pad not in pops_set
            st.rerun()
    with qs_cols[1]:
        if st.button("Select all pads"):
            for pad in all_pads:
                st.session_state[f"uw_pad_{pad}"] = True
            st.rerun()
    with qs_cols[2]:
        if st.button("Clear pad selection"):
            for pad in all_pads:
                st.session_state[f"uw_pad_{pad}"] = False
            st.rerun()

    # Restore checked pads from the shadow saved at Load time — the checkbox
    # widget keys are GC'd whenever Step 1 isn't rendered, so without this a
    # detour to another step unchecked everything.
    saved_pads = set(st.session_state.get("_uw_selected_pads", []) or [])
    pad_cols = st.columns(min(len(all_pads), 6))
    selected_pads = []
    for i, pad in enumerate(all_pads):
        pad_wells = filter_wells_by_pad(all_well_names, [pad])
        with pad_cols[i % len(pad_cols)]:
            if st.checkbox(
                f"Pad {pad} ({len(pad_wells)})",
                value=pad in saved_pads,
                key=f"uw_pad_{pad}",
            ):
                selected_pads.append(pad)

    if not selected_pads:
        st.info("Select at least one pad above.")
        return

    filtered_well_names = filter_wells_by_pad(all_well_names, selected_pads)

    # Per-well exclude list — lets the user trim individual wells out of the
    # selected pads before pulling tests. Useful when the non-POPS quick-select
    # pulls in a well or two you don't want in this optimization run.
    exclude_opts = sorted(filtered_well_names)
    excluded = st.multiselect(
        "Wells to exclude (optional)",
        options=exclude_opts,
        # Restore from the shadow after widget-state GC; intersect with the
        # live options so a stale well name can't raise.
        default=[
            w
            for w in st.session_state.get(
                "uw_well_exclude", st.session_state.get("_uw_well_exclude", [])
            )
            if w in exclude_opts
        ],
        key="uw_well_exclude",
        help="Drop specific wells from the selected pads before loading tests.",
    )
    if excluded:
        filtered_well_names = [w for w in filtered_well_names if w not in excluded]
        if not filtered_well_names:
            st.info("All wells excluded — pick fewer exclusions.")
            return

    # Date range (value= restores from the Load-time shadow after widget GC)
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = st.session_state.get(
            "_uw_start_date", date.today() - timedelta(days=90)
        )
        start_date = st.date_input(
            "Start Date", value=default_start, key="uw_start_date"
        )
    with col_end:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get("_uw_end_date", date.today()),
            key="uw_end_date",
        )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    st.caption(f"{len(filtered_well_names)} wells across {len(selected_pads)} pads")

    if st.button(
        "Load Well Tests from Databricks",
        type="primary",
        use_container_width=True,
        key="uw_load_tests",
    ):
        try:
            with st.spinner("Querying Databricks for well tests (cached 24h)..."):
                df, dropped_wells = _cached_well_test_query(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    tuple(filtered_well_names),
                )

            if df.empty:
                st.error("No well test data returned for selected pads/dates.")
                return

            # Clear downstream state and store results
            _clear_downstream(2)
            st.session_state["uw_well_test_df"] = df
            st.session_state["uw_dropped_wells"] = dropped_wells
            st.session_state["uw_csv_shortcut"] = False
            # Shadow the Step-1 selections into non-widget keys — the widget
            # keys are GC'd when Step 1 stops rendering, which made Pad
            # scope (and the pad/exclusion/date selections) silently vanish
            # by Step 3. These shadows are what Steps 3/4 read.
            st.session_state["_uw_scope"] = scope
            st.session_state["_uw_scope_pad"] = pad_choice
            st.session_state["_uw_selected_pads"] = list(selected_pads)
            st.session_state["_uw_well_exclude"] = list(excluded)
            st.session_state["_uw_start_date"] = start_date
            st.session_state["_uw_end_date"] = end_date
            st.session_state["uw_current_step"] = 2
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 1), 2
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error querying Databricks: {str(e)}")
            st.exception(e)

    # Summary banner when data is already loaded — the step indicator at
    # the top of the workflow already provides forward navigation, so we
    # don't need a second "Proceed" button here.
    if "uw_well_test_df" in st.session_state and not st.session_state.get(
        "uw_csv_shortcut", False
    ):
        df = st.session_state["uw_well_test_df"]
        st.success(
            f"Well tests loaded: {len(df)} tests for {df['well'].nunique()} wells. "
            "Click **Step 2** above to continue."
        )


def _render_csv_upload_path():
    """Upload an optimization template CSV to skip Databricks + IPR analysis."""
    from woffl.assembly.network_optimizer import load_wells_from_dataframe

    uploaded = st.file_uploader(
        "Upload Optimization Template CSV",
        type=["csv"],
        key="uw_csv_upload",
        help="Upload a CSV with well configurations (same format as the optimization template). "
        "This skips Databricks loading and IPR analysis.",
    )

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            well_configs = load_wells_from_dataframe(input_df)

            _clear_downstream(2)
            st.session_state["uw_well_configs"] = well_configs
            st.session_state["uw_template_df"] = input_df
            st.session_state["uw_csv_shortcut"] = True
            st.session_state["uw_current_step"] = 2
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 1), 2
            )

            st.success(f"Loaded {len(well_configs)} well configurations from CSV")
            st.rerun()

        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
