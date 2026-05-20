"""Test Harness tab — drives the curated case set from tests/harness_cases.py.

Each click of "Run all cases" iterates over ``ALL_CASES`` (defined in
``tests/harness_cases.py``) and renders pass/fail + a drilldown of
expected-vs-actual values. Cases share the same ``@st.cache_data``
fetchers as the rest of the app, so the harness runs against today's
live Databricks data with warm caches.

Add new cases in ``tests/harness_cases.py``. The page auto-discovers
them via the ``ALL_CASES`` registry.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _import_cases():
    """Locate and import tests/harness_cases.py.

    The Streamlit app runs from ``woffl_gui/`` with ``sys.path`` containing
    only the package root; ``tests/`` is a sibling, not a package, so we
    have to add the repo root to ``sys.path`` before importing.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from tests import harness_cases

    return harness_cases


def _status_icon(passed: bool) -> str:
    return "✅" if passed else "❌"


def render_tab() -> None:
    """Render the Test Harness tab."""
    st.header("Test Harness")
    st.caption(
        "Curated regression cases that exercise the optimizer, marginal-WC "
        "math, and Databricks plumbing against today's live data. Add cases "
        "by editing `tests/harness_cases.py` — the registry on this page "
        "auto-discovers new functions on the next rerun."
    )

    try:
        harness = _import_cases()
    except Exception as e:
        st.error(f"Could not import `tests/harness_cases.py`: {e}")
        return

    cases = harness.ALL_CASES
    st.caption(f"**{len(cases)} cases** registered.")

    col_run, col_clear, col_filter = st.columns([1.3, 1, 2.5])
    with col_run:
        run_clicked = st.button(
            "Run all cases", type="primary", use_container_width=True,
            help="Execute every case against today's data; results render below.",
        )
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("_harness_results", None)
            st.session_state.pop("_harness_ran_at", None)
            st.rerun()
    with col_filter:
        show_only_failures = st.checkbox(
            "Show only failures",
            value=False,
            key="_harness_show_only_failures",
        )

    if run_clicked:
        results: list = []
        progress = st.progress(0.0, text="Running cases…")
        for i, case_fn in enumerate(cases):
            try:
                result = case_fn()
            except Exception as e:
                # Defensive net — the case should catch its own exceptions,
                # but if it leaks we still surface a useful row.
                result = harness.CaseResult(
                    name=case_fn.__name__,
                    description=(case_fn.__doc__ or ""),
                    passed=False,
                    summary=f"Unhandled {type(e).__name__}: {e}",
                    error=str(e),
                )
            results.append(result)
            progress.progress(
                (i + 1) / len(cases),
                text=f"{result.name} done ({i + 1}/{len(cases)})",
            )
        progress.empty()
        st.session_state["_harness_results"] = results
        st.session_state["_harness_ran_at"] = datetime.now()

    results = st.session_state.get("_harness_results")
    if not results:
        st.info("Click **Run all cases** to execute the test plan.")
        return

    ran_at = st.session_state.get("_harness_ran_at")
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    header_cols = st.columns([2, 2, 3])
    header_cols[0].metric("Passed", f"{passed_count} / {len(results)}")
    header_cols[1].metric("Failed", failed_count)
    if ran_at:
        header_cols[2].caption(
            f"Ran at **{ran_at:%H:%M:%S}** on {ran_at:%Y-%m-%d}."
        )

    if failed_count == 0:
        st.success("All cases passed.")
    else:
        st.error(f"{failed_count} case(s) failed — expand below for details.")

    st.divider()

    for result in results:
        if show_only_failures and result.passed:
            continue
        icon = _status_icon(result.passed)
        header = f"{icon}  **{result.name}** — {result.summary}"
        with st.expander(header, expanded=not result.passed):
            if result.description:
                st.caption(result.description.strip())
            if result.error:
                st.error(f"Exception: {result.error}")
            if result.details:
                st.json(result.details, expanded=False)
