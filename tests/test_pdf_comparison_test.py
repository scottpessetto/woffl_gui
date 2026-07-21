"""Tests for ``pdf_export._comparison_test_row``.

The PDF's "Compared Against" card and hero deltas must use the SAME test the
Solver hero resolves on screen. Regression for the MPE-19 report bug: the PDF
previously grabbed the literal most-recent test, which postdated the memory
gauge's coverage (so its BHP was NaN) and carried a partial PF allocation —
the PDF showed a +1,509 BWPD power-fluid delta while the on-screen hero
(compared against the anchor's 07-09 test) showed -129 BWPD.

Resolution chain under test (mirrors ``jetpump_solver._render_test_picker``):
gauge-coverage filter → synced: IPR-anchor test / decoupled: picker shadow
date → most-recent fallback.

Streamlit is mocked before importing the modules so their module-level
``import streamlit as st`` resolves without a running server.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mock streamlit *before* importing the GUI modules (mirrors test_solver_ipr_sync).
_st_mock = MagicMock()
_st_mock.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) else lambda fn: fn
)
sys.modules.setdefault("streamlit", _st_mock)

from woffl.gui import pdf_export  # noqa: E402
from woffl.gui.tabs import jetpump_solver  # noqa: E402


class _FakeSt:
    """Minimal st replacement with a real dict session_state."""

    def __init__(self):
        self.session_state = {}


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    # Both modules read session state: pdf_export for the decouple/shadow
    # keys, jetpump_solver (via _resolve_anchor_test_row) for the anchor sig.
    monkeypatch.setattr(pdf_export, "st", fs)
    monkeypatch.setattr(jetpump_solver, "st", fs)
    return fs


def _tests_df() -> pd.DataFrame:
    """Three tests; the newest (07-18) has no BHP — like a test taken the
    day after gauge coverage ends on a well with no permanent gauge."""
    return pd.DataFrame(
        {
            "WtDate": pd.to_datetime(["2026-07-18", "2026-07-09", "2026-06-01"]),
            "BHP": [float("nan"), 548.0, 600.0],
            "WtTotalFluid": [440.0, 431.0, 420.0],
            "WtOilVol": [154.0, 151.0, 150.0],
            "lift_wat": [1136.0, 2774.0, 2700.0],
        }
    )


@pytest.fixture
def well_tests(monkeypatch):
    df = _tests_df()
    monkeypatch.setattr(
        "woffl.gui.utils.get_well_tests_for_well", lambda w: df.copy()
    )
    return df


def _gauge_through(end_date: str):
    """Stub gauge whose daily coverage runs 2025-08-29 → end_date."""
    days = pd.date_range("2025-08-29", end_date, freq="D")
    return SimpleNamespace(
        daily_df=pd.DataFrame({"tag_date": days, "bhp": 500.0})
    )


def test_synced_resolves_anchor_test_not_literal_latest(
    fake_st, well_tests, monkeypatch
):
    """Even without a gauge, the BHP-less latest test can't be the anchor —
    the PDF must compare against the anchor's test (07-09), not 07-18."""
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)

    row = pdf_export._comparison_test_row("MPE-19")

    assert row is not None
    assert row["WtDate"].strftime("%Y-%m-%d") == "2026-07-09"


def test_gauge_coverage_excludes_post_gauge_test(
    fake_st, well_tests, monkeypatch
):
    """Gauge ends 07-17: the 07-18 test isn't comparable (no gauge BHP for
    its date) and must be filtered out, exactly like the Solver's picker."""
    monkeypatch.setattr(
        "woffl.gui.memory_gauge.get_gauge",
        lambda w: _gauge_through("2026-07-17"),
    )

    row = pdf_export._comparison_test_row("MPE-19")

    assert row is not None
    assert row["WtDate"].strftime("%Y-%m-%d") == "2026-07-09"


def test_decoupled_uses_picker_shadow_date(fake_st, well_tests, monkeypatch):
    """'Use a different test for comparison' checked → the PDF follows the
    independent picker's persisted (shadow-key) selection."""
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    fake_st.session_state["sw_ipr_decouple_MPE-19"] = True
    fake_st.session_state["sw_test_picker_date_MPE-19"] = "2026-06-01"

    row = pdf_export._comparison_test_row("MPE-19")

    assert row is not None
    assert row["WtDate"].strftime("%Y-%m-%d") == "2026-06-01"


def test_no_tests_returns_none(fake_st, monkeypatch):
    monkeypatch.setattr(
        "woffl.gui.utils.get_well_tests_for_well", lambda w: None
    )
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)

    assert pdf_export._comparison_test_row("MPE-19") is None


def test_no_gauge_covered_tests_returns_none(fake_st, well_tests, monkeypatch):
    """Gauge active but covering none of the test dates → no comparable test
    (the Solver shows an info message and blanks the hero deltas)."""
    monkeypatch.setattr(
        "woffl.gui.memory_gauge.get_gauge",
        lambda w: _gauge_through("2025-09-30"),
    )

    assert pdf_export._comparison_test_row("MPE-19") is None
