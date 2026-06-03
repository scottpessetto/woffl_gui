"""Tests for the Solver tab's IPR → sidebar seed-sync.

Covers ``jetpump_solver._sync_chosen_ipr_to_sidebar``, which seeds the chosen
IPR-anchor test's curve + fluid (qwf, pwf, res_pres, form_wc, form_gor) into the
sidebar so Batch Run, PF Range, and the top Solver all use it — and the engineer
can then override any field by editing the sidebar. The critical properties:

  * the default state (most-recent anchor) never writes,
  * a real selection change writes all five values and reruns ONCE
    (no infinite rerun loop),
  * manual sidebar edits survive when the selection signature is unchanged,
  * switching back to "Most recent" restores the recent-fit values,
  * the GOR floor (marginal-well auto-recovery) is honored when seeding.

Streamlit is mocked before importing the module so the module-level
``import streamlit as st`` resolves without a running server.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mock streamlit *before* importing the solver tab (and its transitive
# woffl.gui imports), mirroring tests/test_utils.py.
_st_mock = MagicMock()
_st_mock.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) else lambda fn: fn
)
sys.modules.setdefault("streamlit", _st_mock)

from woffl.gui.tabs import jetpump_solver  # noqa: E402


class _Rerun(Exception):
    """Stand-in for Streamlit's RerunException raised by st.rerun()."""


class _FakeSt:
    """Minimal st replacement: a real dict session_state + counting rerun."""

    def __init__(self):
        self.session_state = {}
        self.rerun_count = 0

    def rerun(self):
        self.rerun_count += 1
        raise _Rerun()


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    monkeypatch.setattr(jetpump_solver, "st", fs)
    return fs


# Recent-fit values, matching what sidebar._auto_populate_from_ipr would set.
_RECENT = {"qwf": 750, "pwf": 500, "res_pres": 1700, "form_wc": 0.55, "form_gor": 300}


def _seed_recent(fs):
    """Sidebar as the auto-populate leaves it for a recent-fit well."""
    for k, v in _RECENT.items():
        fs.session_state[k] = v
        fs.session_state[f"{k}_input"] = v


WELL = "MPB-28"
SIG_KEY = f"sw_ipr_applied_sig_{WELL}"


def _call(fs, *, mode, date, qwf, pwf, res_p, form_wc, fgor):
    """Invoke the seed-sync; return True if it triggered a rerun, else False."""
    try:
        jetpump_solver._sync_chosen_ipr_to_sidebar(
            WELL,
            anchor_mode=mode,
            anchor_date=date,
            qwf_oil=qwf,
            pwf=pwf,
            res_p=res_p,
            form_wc=form_wc,
            fgor=fgor,
        )
        return False
    except _Rerun:
        return True


# A specific-anchor fit that differs from the recent fit on every field.
_SPEC = dict(qwf=812.7, pwf=540.4, res_p=1853.9, form_wc=0.32, fgor=210.6)
_SPEC_DATE = pd.Timestamp("2026-05-10")


def test_default_anchor_never_writes(fake_st):
    """Most-recent anchor == the auto-populated state: no-op, no rerun."""
    _seed_recent(fake_st)
    reran = _call(
        fake_st, mode="recent", date=None,
        qwf=750.0, pwf=500.0, res_p=1700.0, form_wc=0.55, fgor=300,
    )
    assert reran is False
    assert fake_st.rerun_count == 0
    assert fake_st.session_state["qwf"] == 750
    assert fake_st.session_state["form_gor"] == 300
    # Widget keys untouched (nothing popped) so the sidebar widgets keep state.
    assert "form_gor_input" in fake_st.session_state


def test_specific_anchor_seeds_all_five_and_reruns_once(fake_st):
    """Picking a specific test seeds all five fields and reruns exactly once."""
    _seed_recent(fake_st)
    reran = _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC)
    assert reran is True
    assert fake_st.rerun_count == 1
    # int()/round casting matches the sidebar auto-populate.
    assert fake_st.session_state["qwf"] == 812
    assert fake_st.session_state["pwf"] == 540
    assert fake_st.session_state["res_pres"] == 1853
    assert fake_st.session_state["form_wc"] == 0.32
    assert fake_st.session_state["form_gor"] == 210
    # All five widget keys dropped so _number_input re-seeds from logical keys.
    for k in ("qwf", "pwf", "res_pres", "form_wc", "form_gor"):
        assert f"{k}_input" not in fake_st.session_state
    assert fake_st.session_state[SIG_KEY] == ("specific", "2026-05-10")
    assert "_ipr_sync_msg" in fake_st.session_state


def test_no_infinite_loop_after_apply(fake_st):
    """Re-running with the SAME selection after the seed must not rerun again."""
    _seed_recent(fake_st)
    assert _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC) is True
    # Second call (post-rerun): sidebar holds the seeded values and the
    # signature matches → early return, no further rerun. Loop-prevention.
    reran_again = _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC)
    assert reran_again is False
    assert fake_st.rerun_count == 1


def test_manual_sidebar_edit_survives_unchanged_selection(fake_st):
    """Manual WC/GOR edits aren't stomped while the selection is unchanged."""
    _seed_recent(fake_st)
    assert _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC) is True
    # User hand-edits WC and GOR in the sidebar.
    fake_st.session_state["form_wc"] = 0.10
    fake_st.session_state["form_gor"] = 600
    # Same anchor selection next render (the fit still says 0.32 / 210) →
    # signature unchanged → the function leaves the manual edits alone.
    reran = _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC)
    assert reran is False
    assert fake_st.session_state["form_wc"] == 0.10
    assert fake_st.session_state["form_gor"] == 600


def test_switch_back_to_recent_restores(fake_st):
    """Returning to the most-recent anchor seeds the recent-fit values back."""
    _seed_recent(fake_st)
    assert _call(fake_st, mode="specific", date=_SPEC_DATE, **_SPEC) is True
    assert fake_st.session_state["form_wc"] == 0.32
    # Switch the anchor back to "Most recent" — signature changes, recent-fit
    # values restored.
    reran = _call(
        fake_st, mode="recent", date=None,
        qwf=750.0, pwf=500.0, res_p=1700.0, form_wc=0.55, fgor=300,
    )
    assert reran is True
    assert fake_st.session_state["qwf"] == 750
    assert fake_st.session_state["form_wc"] == 0.55
    assert fake_st.session_state["form_gor"] == 300
    assert fake_st.session_state[SIG_KEY] == ("recent", None)


def test_gor_floor_is_honored_when_seeding(fake_st):
    """A test GOR below the marginal-well recovery floor is clamped up.

    _trigger_gor_reset records ``_well_min_gor[well]`` after a solver failure;
    re-seeding from a low-GOR test must not drop GOR back under that floor.
    """
    _seed_recent(fake_st)
    fake_st.session_state["_well_min_gor"] = {WELL: 250}
    reran = _call(
        fake_st, mode="specific", date=_SPEC_DATE,
        qwf=812.7, pwf=540.4, res_p=1853.9, form_wc=0.32, fgor=180,
    )
    assert reran is True
    # max(int(180), 250) == 250 — the recovery floor wins.
    assert fake_st.session_state["form_gor"] == 250


# ---------------------------------------------------------------------------
# Picker-restore tests — the path the live bug actually hit.
#
# The previous round of tests exercised _sync_chosen_ipr_to_sidebar in
# isolation and never simulated a tab switch, so they were green while the
# feature was broken. These run the REAL picker functions
# (_render_ipr_anchor_control / _render_test_picker) after simulating the
# Streamlit "widget state garbage-collected on view switch" behavior — i.e.
# the widget keys are dropped but the non-widget marker/shadow survive.
#
# Caveat (honest): _FakeStWidgets models Streamlit's documented keyed-widget
# contract (session_state value wins when the key is present; `index` is the
# default only when the key is absent). If real Streamlit deviates from that
# contract these tests can't catch it — the live click-through is the ground
# truth. But unlike before, these at least exercise the actual failure path.
# ---------------------------------------------------------------------------


class _FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeStWidgets:
    """st fake modeling the keyed-widget contract the restore depends on."""

    def __init__(self):
        self.session_state = {}
        self.last_options = {}  # label -> options list, for assertions

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn() for _ in range(n)]

    def selectbox(self, label, options, index=0, key=None, help=None, **kw):
        opts = list(options)
        self.last_options[label] = opts
        # Key present with a still-valid value → that value wins (index ignored).
        if key is not None and self.session_state.get(key) in opts:
            return self.session_state[key]
        # Key absent (GC'd) or stale → fall back to `index` and record it.
        val = opts[index] if opts else None
        if key is not None:
            self.session_state[key] = val
        return val

    def checkbox(self, label, value=False, key=None, help=None, **kw):
        if key is not None and key in self.session_state:
            return bool(self.session_state[key])
        val = bool(value)
        if key is not None:
            self.session_state[key] = val
        return val

    def info(self, *a, **k):
        pass


@pytest.fixture
def picker_st(monkeypatch):
    fs = _FakeStWidgets()
    monkeypatch.setattr(jetpump_solver, "st", fs)
    return fs


def _make_test_df():
    return pd.DataFrame(
        {
            "WtDate": [
                pd.Timestamp("2026-05-10"),
                pd.Timestamp("2026-04-01"),
                pd.Timestamp("2026-03-01"),
            ],
            "BHP": [1200.0, 1300.0, 1400.0],
            "WtOilVol": [500.0, 480.0, 460.0],
            "WtTotalFluid": [1000.0, 980.0, 960.0],
            "lift_wat": [300.0, 290.0, 280.0],
        }
    )


def test_anchor_restores_specific_after_tab_detour(picker_st):
    """The exact live-bug path: after Batch drops the widget state, the anchor
    selector restores the previously-applied specific test from the surviving
    sig instead of snapping back to 'Most recent'."""
    df = _make_test_df()
    # State on return from Batch: seed marker survived (non-widget key); the
    # selectbox widget keys were garbage-collected (absent).
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-04-01")
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "specific"
    assert anchor_date == pd.Timestamp("2026-04-01")


def test_anchor_defaults_recent_without_sig(picker_st):
    """Fresh well, no marker yet → 'Most recent' (unchanged default behavior)."""
    df = _make_test_df()
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"
    assert anchor_date is None


def test_anchor_restores_median_after_detour(picker_st):
    df = _make_test_df()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("median", None)
    mode, _ = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "median"


def test_anchor_live_widget_value_wins_over_sig(picker_st):
    """No-GC case: when the widget state survived, the live selection wins even
    if it differs from the sig (so explicitly switching back to recent sticks)."""
    df = _make_test_df()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-04-01")
    picker_st.session_state["_sw_ipr_anchor_sel_WELL"] = "Most recent"
    mode, _ = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"


def test_compare_picker_restores_after_detour(picker_st, monkeypatch):
    """The 'Test to compare against' picker restores from its date shadow after
    a tab detour drops the widget state."""
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    df = _make_test_df()
    picker_st.session_state["sw_test_picker_date_WELL"] = "2026-04-01"
    row = jetpump_solver._render_test_picker("WELL", df, synced=False)
    assert row["WtDate"] == pd.Timestamp("2026-04-01")


# ---------------------------------------------------------------------------
# Compare-against ↔ IPR-anchor sync (synced by default; decouple checkbox).
# _make_test_df has BHP [1200, 1300, 1400] on [05-10, 04-01, 03-01], so the
# median-BHP test (1300) is 2026-04-01 and the most-recent is 2026-05-10.
# ---------------------------------------------------------------------------


def test_resolve_anchor_row_modes(picker_st):
    """_resolve_anchor_test_row maps the anchor sig to the right test row."""
    df = (
        _make_test_df()
        .sort_values("WtDate", ascending=False)
        .reset_index(drop=True)
    )
    # No sig → most recent.
    assert jetpump_solver._resolve_anchor_test_row("WELL", df)[
        "WtDate"
    ] == pd.Timestamp("2026-05-10")
    # Specific date.
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-03-01")
    assert jetpump_solver._resolve_anchor_test_row("WELL", df)[
        "WtDate"
    ] == pd.Timestamp("2026-03-01")
    # Median BHP (1300) → 2026-04-01.
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("median", None)
    assert jetpump_solver._resolve_anchor_test_row("WELL", df)[
        "WtDate"
    ] == pd.Timestamp("2026-04-01")


def test_compare_picker_synced_to_specific_anchor(picker_st, monkeypatch):
    """Synced (default): the comparison picker is slaved to the anchor's test."""
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    df = _make_test_df()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-04-01")
    row = jetpump_solver._render_test_picker("WELL", df, synced=True)
    assert row["WtDate"] == pd.Timestamp("2026-04-01")


def test_compare_picker_synced_to_median_anchor(picker_st, monkeypatch):
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    df = _make_test_df()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("median", None)
    row = jetpump_solver._render_test_picker("WELL", df, synced=True)
    assert row["WtDate"] == pd.Timestamp("2026-04-01")  # the median-BHP test


def test_compare_picker_synced_recent_when_no_sig(picker_st, monkeypatch):
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    df = _make_test_df()
    row = jetpump_solver._render_test_picker("WELL", df, synced=True)
    assert row["WtDate"] == pd.Timestamp("2026-05-10")  # most recent


def test_compare_picker_decoupled_ignores_anchor(picker_st, monkeypatch):
    """Decoupled: the comparison picker uses its own selection, not the anchor."""
    monkeypatch.setattr("woffl.gui.memory_gauge.get_gauge", lambda w: None)
    df = _make_test_df()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-04-01")
    # User independently picked the oldest test (persisted in its shadow).
    picker_st.session_state["sw_test_picker_date_WELL"] = "2026-03-01"
    row = jetpump_solver._render_test_picker("WELL", df, synced=False)
    assert row["WtDate"] == pd.Timestamp("2026-03-01")  # NOT the anchor's 04-01


def test_ipr_anchor_dropdown_shows_jp_size(picker_st):
    """The IPR anchor's specific-test dropdown shows the pump installed at each
    test's date, like the 'Test to compare against' dropdown."""
    df = _make_test_df()
    # One JP install (12B) set before all the tests, so every test's option
    # should carry "12B".
    picker_st.session_state["jp_history_df"] = pd.DataFrame(
        {
            "Well Name": ["WELL"],
            "Date Set": [pd.Timestamp("2026-01-01")],
            "Nozzle Number": [12.0],
            "Throat Ratio": ["B"],
            "Tubing Diameter": [4.5],
        }
    )
    # Force "specific" mode so the date dropdown (with the pump) renders.
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("specific", "2026-04-01")
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    anchor_opts = picker_st.last_options.get("Anchor test", [])
    assert anchor_opts, "specific mode should render the 'Anchor test' dropdown"
    assert all("12B" in opt for opt in anchor_opts)
