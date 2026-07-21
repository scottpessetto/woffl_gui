"""Tests for the pad-review save hook's IPR-anchor pin push (W3 of the
woffl-prop-hist-persistence plan).

Covers ``step_review_wells._maybe_pin_saved_ipr`` (called from
``_save_and_advance`` AFTER the store write) and the structural guarantee
that the batch auto-match apply path never calls it (plan W3(d) — batch
anchors are machine-fit, not user-reviewed, so must never auto-pin).

``push_prop`` / ``resolve_entry_user`` are always mocked at the
``woffl.gui.ipr_anchor`` call boundary (or lower, in ``pin_ipr_anchor``
integration tests) — zero live Databricks calls from this module.
"""

import pandas as pd
import pytest

from woffl.gui.tabs import jetpump_solver
from woffl.gui.workflow_steps import step_review_wells as srw


class _FakeSt:
    """Minimal st replacement recording toast/caption/warning calls."""

    def __init__(self):
        self.toasts = []
        self.captions = []
        self.warnings = []

    def toast(self, text, icon=None, **kw):
        self.toasts.append(text)

    def caption(self, text, *a, **k):
        self.captions.append(text)

    def warning(self, text, *a, **k):
        self.warnings.append(text)


class _FakeJpsSt:
    """Minimal st stand-in for jetpump_solver's module-level ``st`` — only
    ``session_state`` is touched by ``_resolve_anchor_test_row`` in this
    module's call path."""

    def __init__(self):
        self.session_state = {}


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    monkeypatch.setattr(srw, "st", fs)
    monkeypatch.setattr(jetpump_solver, "st", _FakeJpsSt())
    return fs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)


def _tests_df(with_uid=True):
    df = pd.DataFrame(
        {
            "WtDate": [pd.Timestamp("2026-05-10"), pd.Timestamp("2026-04-01")],
            "BHP": [1200.0, 1300.0],
            "WtTotalFluid": [1000.0, 980.0],
        }
    )
    df["wt_uid"] = [301.0, 302.0] if with_uid else float("nan")
    return df


class TestMaybePinSavedIpr:
    def test_gate_off_no_push_attempted_save_still_implied_ok(
        self, fake_st, monkeypatch
    ):
        """Gate off: no push attempted, and nothing shown (silent) — the
        store save itself happens in the caller and is unaffected."""
        calls = []
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.pin_ipr_anchor",
            lambda *a, **k: calls.append(1) or (True, "should not run"),
        )
        srw._maybe_pin_saved_ipr("MPB-28")
        assert calls == []
        assert fake_st.toasts == []
        assert fake_st.captions == []
        assert fake_st.warnings == []

    def test_pushes_resolved_wt_uid_and_clears_cache(self, fake_st, monkeypatch):
        """Gate on: resolves the CURRENT anchor (most-recent test, no sig
        set) and pushes its wt_uid; a successful push clears the Solver's
        pin memo and shows a toast."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setattr(srw, "get_well_tests_for_well", lambda w: _tests_df())

        pushed = {}

        def _fake_pin(well_name, anchor_row):
            pushed["well"] = well_name
            pushed["uid"] = anchor_row.get("wt_uid")
            pushed["date"] = anchor_row.get("WtDate")
            return True, "📌 IPR saved to Databricks — test 2026-05-10"

        monkeypatch.setattr("woffl.gui.ipr_anchor.pin_ipr_anchor", _fake_pin)
        cleared = []
        monkeypatch.setattr(
            jetpump_solver, "_clear_pin_cache", lambda w: cleared.append(w)
        )

        srw._maybe_pin_saved_ipr("MPB-28")

        assert pushed["well"] == "MPB-28"
        assert pushed["uid"] == 301.0  # most-recent test's uid
        assert pushed["date"] == pd.Timestamp("2026-05-10")
        assert cleared == ["MPB-28"]
        assert fake_st.toasts == ["📌 IPR saved to Databricks — test 2026-05-10"]
        assert fake_st.warnings == []

    def test_manual_test_anchor_skips_with_caption(self, fake_st, monkeypatch):
        """A manual/provisional-test anchor (NaN wt_uid) is never pinnable —
        skip with a caption, no push attempted, no warning."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setattr(
            srw, "get_well_tests_for_well", lambda w: _tests_df(with_uid=False)
        )
        called = []
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.push_prop",
            lambda *a, **k: called.append(1),
        )

        srw._maybe_pin_saved_ipr("MPB-28")

        assert called == []  # push_prop never reached
        assert fake_st.toasts == []
        assert fake_st.warnings == []
        assert any("IPR not saved to Databricks" in c for c in fake_st.captions)

    def test_no_tests_skips_with_caption(self, fake_st, monkeypatch):
        """No test frame at all (e.g. Databricks hiccup on the tests fetch)
        -> no resolvable anchor -> skip with a caption."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setattr(srw, "get_well_tests_for_well", lambda w: None)

        srw._maybe_pin_saved_ipr("MPB-28")

        assert fake_st.toasts == []
        assert fake_st.warnings == []
        assert any("IPR not saved to Databricks" in c for c in fake_st.captions)

    def test_get_well_tests_raising_skips_with_caption(self, fake_st, monkeypatch):
        """get_well_tests_for_well raising must not propagate — degrade to
        the no-resolvable-test skip."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")

        def _raise(w):
            raise RuntimeError("databricks down")

        monkeypatch.setattr(srw, "get_well_tests_for_well", _raise)

        srw._maybe_pin_saved_ipr("MPB-28")

        assert fake_st.toasts == []
        assert any("IPR not saved to Databricks" in c for c in fake_st.captions)

    def test_push_failure_shows_warning_and_does_not_raise(self, fake_st, monkeypatch):
        """A push_prop failure (connection, enthid resolution, ...) surfaces
        as st.warning — the store save (which already happened in the
        caller) must not be affected; this function must not raise."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setattr(srw, "get_well_tests_for_well", lambda w: _tests_df())
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.pin_ipr_anchor",
            lambda well, row: (
                False,
                "Could not save IPR to Databricks: No enthid found for well 'MPB-28'.",
            ),
        )

        srw._maybe_pin_saved_ipr("MPB-28")  # must not raise

        assert fake_st.toasts == []
        assert fake_st.captions == []
        assert len(fake_st.warnings) == 1
        assert "Could not save IPR to Databricks" in fake_st.warnings[0]


class TestBatchAutomatchNeverPins:
    """Plan W3(d): batch auto-match anchors are machine-fit, not
    user-reviewed, so applying them to the store must never push an IPR pin.
    A full render-level test of ``_render_batch_automatch`` would require
    mocking the entire Streamlit data_editor + joint_match pipeline; this
    structural guard is the pragmatic regression net — it fails immediately
    if a future edit wires pinning into the batch-apply path."""

    def test_batch_automatch_source_never_references_pin_helpers(self):
        """Checks for CALLS (name + open-paren), not mere mentions — the
        apply block carries a deliberate one-line comment naming
        ``_maybe_pin_saved_ipr`` to explain why it's absent, which a bare
        substring check would trip over."""
        import inspect

        src = inspect.getsource(srw._render_batch_automatch)
        assert "_maybe_pin_saved_ipr(" not in src
        assert "pin_ipr_anchor(" not in src
        assert "push_prop(" not in src
