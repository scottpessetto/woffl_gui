"""WOFFL Streamlit GUI Application

This is the main entry point for the WOFFL Streamlit GUI application.
It provides a web interface for interacting with the WOFFL package's jetpump functionality.

The application supports three modes:
- Single Well Analysis: Detailed analysis of one well with multiple visualization tabs
- Multi-Well Optimization: Optimize pump sizing across multiple wells
- Well Test Analysis: Generate Vogel IPR from well tests + Databricks BHP data
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from woffl.assembly.jp_history import parse_jp_history
from woffl.gui.sidebar import render_sidebar
from woffl.gui.single_well_page import run_single_well_page, show_welcome_message

_JP_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "jetpump_history.xlsx"
)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_jp_history():
    """Fetch JP history from Databricks mpu_tracker. Cached 24h."""
    from woffl.assembly.databricks_client import fetch_jp_history

    return fetch_jp_history()


def _prefetch_well_sort_data() -> None:
    """Fire-and-forget Databricks prefetch so the Marginal WC import button
    on the Batch Run tab feels instant.

    Spawns a daemon thread that warms the @st.cache_data caches behind the
    Well Sort tab (shut-in history, recent tests, producer list, catalog,
    XV status). When the user later clicks "Import from Well Sort" on Batch
    Run, every fetcher hits warm cache and the marginal WC is computed
    from pandas ops only.

    Once per session, guarded via session_state so we don't spawn a new
    thread on every rerun.
    """
    if st.session_state.get("_well_sort_prefetched"):
        return
    st.session_state["_well_sort_prefetched"] = True

    def _worker() -> None:
        # Import inside the worker so the module is loaded lazily — only
        # when the prefetch actually runs.
        from woffl.gui.scotts_tools.well_sort import (
            _cached_producer_catalog,
            _cached_producers,
            _cached_recent_tests,
            _cached_shut_in_history,
            _cached_xv_status,
        )

        # Warm each cache independently; one fetch failure shouldn't stop
        # the others (e.g. transient XV status flake shouldn't block tests).
        for fn, args in (
            (_cached_shut_in_history, ()),
            (_cached_recent_tests, (180,)),
            (_cached_producers, ()),
            (_cached_producer_catalog, ()),
            (_cached_xv_status, ()),
        ):
            try:
                fn(*args)
            except Exception:
                pass

    import threading

    from streamlit.runtime.scriptrunner import add_script_run_ctx

    thread = threading.Thread(
        target=_worker, daemon=True, name="well-sort-prefetch"
    )
    add_script_run_ctx(thread)  # lets the thread call @st.cache_data fns cleanly
    thread.start()


# Well-test pre-fetch lives in gui.utils (fetch_all_well_tests) so the single-
# source-of-truth lookback window is shared between startup and the per-well
# slicer get_well_tests_for_well. Imported lazily at the call site below.


def main():
    """Main function for the Streamlit application."""
    st.set_page_config(
        page_title="WOFFL Haus",
        page_icon="🧇",  # waffle — shows as the browser-tab favicon
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("WOFFL Haus 🧇")
    st.caption("*Built on Kaelin Ellis's WOFFL Jet Pump Model*")

    # Global startup prefetch — JP history, recent well tests, and well
    # properties are three INDEPENDENT Databricks pulls, so warm them
    # concurrently: a cold start costs the slowest query instead of the sum
    # (they used to run back-to-back behind three sequential spinners).
    # Workers only fill the process-wide st.cache_data entries; session_state
    # writes and fallback handling stay on the main thread below.
    with st.sidebar:
        needs_jp = "jp_history_df" not in st.session_state
        needs_tests = "all_well_tests_df" not in st.session_state
        if needs_jp or needs_tests:
            import threading

            from streamlit.runtime.scriptrunner import add_script_run_ctx

            from woffl.gui.utils import (
                DEFAULT_TEST_MONTHS,
                _load_well_characteristics_cached,
                fetch_all_well_tests,
            )

            results: dict = {}
            errors: dict = {}

            def _warm(name, fn, *args):
                try:
                    results[name] = fn(*args)
                except Exception as e:
                    errors[name] = e

            jobs: list[tuple] = [("chars", _load_well_characteristics_cached)]
            if needs_jp:
                jobs.append(("jp_history", _cached_jp_history))
            if needs_tests:
                # Pass the lookback explicitly: st.cache_data keys on the args
                # AS PASSED, so the old no-arg call cached under a different
                # key than get_well_tests_for_well's (months,) call — the
                # identical full-field query ran twice per day.
                jobs.append(("well_tests", fetch_all_well_tests, DEFAULT_TEST_MONTHS))

            with st.spinner("Loading well data from Databricks..."):
                threads = []
                for name, fn, *args in jobs:
                    t = threading.Thread(
                        target=_warm,
                        args=(name, fn, *args),
                        daemon=True,
                        name=f"startup-{name}",
                    )
                    add_script_run_ctx(t)
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()

            if needs_jp:
                if "jp_history" in results:
                    st.session_state["jp_history_df"] = results["jp_history"]
                    st.session_state["jp_history_source"] = "Databricks"
                else:
                    e = errors.get("jp_history")
                    if _JP_HISTORY_PATH.exists():
                        st.session_state["jp_history_df"] = parse_jp_history(
                            str(_JP_HISTORY_PATH)
                        )
                        st.session_state["jp_history_source"] = "Excel (fallback)"
                        st.warning(f"Databricks unavailable, using bundled Excel: {e}")
                    else:
                        st.warning(f"Could not load JP history: {e}")

            if needs_tests:
                if "well_tests" in results:
                    st.session_state["all_well_tests_df"] = results["well_tests"]
                else:
                    st.warning(
                        f"Could not fetch well tests: {errors.get('well_tests')}"
                    )
                    st.session_state["all_well_tests_df"] = None

        # Well properties — instant when the parallel warm above succeeded;
        # the wrapper also sets the missing-survey list into session_state.
        from woffl.gui.utils import load_well_characteristics

        try:
            load_well_characteristics()
        except Exception as e:
            st.warning(f"Could not load well properties: {e}")

        missing = st.session_state.get("wells_missing_surveys") or []
        if missing:
            st.warning(
                f"{len(missing)} well(s) missing deviation surveys — JP_TVD estimated "
                f"via pad-average TVD/MD ratio. Run "
                f"`python -m woffl.jp_data.check_missing_surveys` then "
                f"`pull_surveys.py` to fix."
            )
            with st.expander("Wells with estimated JP_TVD"):
                st.write(", ".join(missing))

    # Warm the Well Sort Databricks caches in a background thread so the
    # Batch Run "Import marginal WC" button can pull instantly. Once per
    # session — no-op on subsequent reruns.
    _prefetch_well_sort_data()

    modes = [
        "Single Well Analysis",
        "Optimization Workflow",
        "Well Database",
        "Well Sort",
    ]
    if st.session_state.get("_scotts_tools", False):
        modes.append("Scott's Tools")

    # Mode selection. The key matters: without one, the widget identity is
    # derived from its options, so unlocking Scott's Tools (options change)
    # created a NEW radio and bounced the user back to Single Well.
    app_mode = st.radio(
        "Select Analysis Mode:",
        modes,
        horizontal=True,
        key="app_mode_radio",
        help=(
            "Single Well: Analyze one well in detail. "
            "Optimization: Select wells → Review IPR → Optimize → Results. "
            "Well Database: View live well properties from Databricks. "
            "Well Sort: Online/offline classification + marginal WC calculator."
        ),
    )

    if app_mode == "Optimization Workflow":
        from woffl.gui.workflow_page import run_workflow_page

        run_workflow_page()

    elif app_mode == "Well Database":
        from woffl.gui.well_database_page import run_well_database_page

        run_well_database_page()

    elif app_mode == "Well Sort":
        from woffl.gui.well_sort_page import run_well_sort_page

        run_well_sort_page()

    elif app_mode == "Scott's Tools":
        from woffl.gui.scotts_tools import run_scotts_tools_page

        run_scotts_tools_page()

    else:
        # Single Well Analysis mode
        run_button, params = render_sidebar()

        if run_button:
            st.session_state.sw_sim_active = True

        if st.session_state.get("sw_sim_active", False):
            run_single_well_page(params)
        else:
            show_welcome_message()

    # Easter egg — renders at the very bottom of the sidebar (after all page-specific content)
    if not st.session_state.get("_scotts_tools", False):
        with st.sidebar:
            st.divider()
            code = st.text_input(
                "", placeholder="", label_visibility="collapsed", key="_egg_input"
            )
            if code.strip().lower() == "scott":
                st.session_state["_scotts_tools"] = True
                st.rerun()


if __name__ == "__main__":
    main()
