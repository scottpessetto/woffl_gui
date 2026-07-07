"""Tests for step3_configure_optimize._get_all_well_tests retry logic (P1-18).

Before this fix, app.py's startup warm-load pinned
``st.session_state["all_well_tests_df"] = None`` on any Databricks failure,
and ``needs_tests = "all_well_tests_df" not in st.session_state`` only checks
key PRESENCE — so once poisoned, the warm-load never retried for the rest of
the session, and Step 3/4's ``_build_actual_maps`` read the (permanently
``None``) session_state value directly with no retry of its own. One
transient startup blip killed "Current vs Optimized" actuals until a full
app restart.

The fix has two parts:
  1. app.py no longer pins the poisoned ``None`` on failure (leaves the key
     absent instead, exactly like the sibling JP-history branch already did)
     so the warm-load naturally retries on the next rerun.
  2. step3's own read path (``_get_all_well_tests``) independently retries
     the cached fetcher before ever consulting session_state, mirroring the
     self-healing pattern in ``woffl.gui.utils.get_well_tests_for_well`` and
     ``scotts_tools._common.get_vogel_for_wells``.

These tests cover part 2 directly (it's the piece with real extractable
logic); part 1 is a one-line "don't pin None" removal, verified by reading
woffl/gui/app.py, and flagged for Scott's live click-through since exercising
app.py's startup warm-load needs a full Streamlit script run.
"""

import pandas as pd
import pytest

import woffl.gui.utils as gutils
import woffl.gui.workflow_steps.step3_configure_optimize as step3


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well": ["MPB-28"],
            "WtDate": pd.to_datetime(["2026-07-01"]),
            "WtOilVol": [100.0],
            "lift_wat": [2000.0],
            "BHP": [900.0],
        }
    )


class TestDirectFetchSucceeds:
    def test_returns_live_fetch_ignoring_session_state(self, monkeypatch):
        live_df = _sample_df()
        stale_df = pd.DataFrame({"well": ["STALE"]})
        monkeypatch.setattr(gutils, "fetch_all_well_tests", lambda months: live_df)
        monkeypatch.setattr(step3.st, "session_state", {"all_well_tests_df": stale_df})

        result = step3._get_all_well_tests()

        assert result is live_df


class TestDirectFetchFailsFallsBackToSessionState:
    def test_exception_falls_back_to_session_state(self, monkeypatch):
        cached_df = _sample_df()

        def _boom(months):
            raise RuntimeError("Databricks blip")

        monkeypatch.setattr(gutils, "fetch_all_well_tests", _boom)
        monkeypatch.setattr(step3.st, "session_state", {"all_well_tests_df": cached_df})

        result = step3._get_all_well_tests()

        assert result is cached_df

    def test_empty_frame_falls_back_to_session_state(self, monkeypatch):
        """An empty (not-failed) fetch is also treated as unusable —
        prefer the last known-good session_state snapshot over an empty
        frame that would make _build_actual_maps report zero actuals."""
        cached_df = _sample_df()
        monkeypatch.setattr(
            gutils, "fetch_all_well_tests", lambda months: pd.DataFrame()
        )
        monkeypatch.setattr(step3.st, "session_state", {"all_well_tests_df": cached_df})

        result = step3._get_all_well_tests()

        assert result is cached_df


class TestBothSourcesUnavailable:
    def test_no_crash_when_session_state_key_absent(self, monkeypatch):
        """The P1-18 app.py fix leaves the key ABSENT (not None) on a
        startup failure -- confirm the retry helper tolerates that (and the
        legacy poisoned-None case) without raising, returning None so
        _build_actual_maps degrades to empty maps rather than crashing."""

        def _boom(months):
            raise RuntimeError("Databricks still down")

        monkeypatch.setattr(gutils, "fetch_all_well_tests", _boom)
        monkeypatch.setattr(step3.st, "session_state", {})

        assert step3._get_all_well_tests() is None

    def test_no_crash_when_session_state_value_is_none(self, monkeypatch):
        def _boom(months):
            raise RuntimeError("Databricks still down")

        monkeypatch.setattr(gutils, "fetch_all_well_tests", _boom)
        monkeypatch.setattr(step3.st, "session_state", {"all_well_tests_df": None})

        assert step3._get_all_well_tests() is None


class TestBuildActualMapsToleratesNone:
    def test_build_actual_maps_returns_empty_dicts_for_none(self):
        oil, pf, bhp = step3._build_actual_maps(["MPB-28"], None)
        assert oil == {} and pf == {} and bhp == {}
