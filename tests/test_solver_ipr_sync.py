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
from datetime import datetime, timezone
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


@pytest.fixture(autouse=True)
def _no_live_prop_hist_calls(monkeypatch):
    """Safety net for EVERY test in this module (added alongside the W2
    IPR-anchor pin feature): ``_render_ipr_anchor_control`` now looks up a
    saved pin via ``prop_hist_client.fetch_latest_prop`` on every render.
    Without this, tests that call the control (or anything that reaches it)
    would try a REAL Databricks connection. Default = no saved pin (``None``)
    — the pin-specific tests below override this per-scenario.
    """
    monkeypatch.setattr(
        "woffl.assembly.prop_hist_client.fetch_latest_prop",
        lambda *a, **k: None,
    )


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
        fake_st,
        mode="recent",
        date=None,
        qwf=750.0,
        pwf=500.0,
        res_p=1700.0,
        form_wc=0.55,
        fgor=300,
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
        fake_st,
        mode="recent",
        date=None,
        qwf=750.0,
        pwf=500.0,
        res_p=1700.0,
        form_wc=0.55,
        fgor=300,
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
        fake_st,
        mode="specific",
        date=_SPEC_DATE,
        qwf=812.7,
        pwf=540.4,
        res_p=1853.9,
        form_wc=0.32,
        fgor=180,
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
        self.captions = []  # st.caption(...) calls, for pin-provenance tests
        self.toasts = []  # st.toast(...) calls, for W3 pin/clear tests
        self.warnings = []  # st.warning(...) calls, for W3 pin/clear tests
        self.clicks = set()  # button keys to report as "clicked" this render
        self.button_calls = []  # every button key rendered, for W3 pin tests
        self.rerun_count = 0

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn() for _ in range(n)]

    def button(self, label, key=None, help=None, **kw):
        self.button_calls.append(key)
        return key in self.clicks

    def toast(self, text, icon=None, **kw):
        self.toasts.append(text)

    def warning(self, text, *a, **k):
        self.warnings.append(text)

    def rerun(self):
        self.rerun_count += 1
        raise _Rerun()

    def selectbox(
        self, label, options, index=0, key=None, help=None, format_func=None, **kw
    ):
        opts = list(options)
        # Record the DISPLAYED labels, applying format_func like real Streamlit,
        # so option-text assertions work whether the caller passes label strings
        # directly or indices + a format_func (the dup-label-safe idiom).
        self.last_options[label] = (
            [format_func(o) for o in opts] if format_func is not None else opts
        )
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

    def caption(self, text, *a, **k):
        self.captions.append(text)


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
    df = _make_test_df().sort_values("WtDate", ascending=False).reset_index(drop=True)
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


# ---------------------------------------------------------------------------
# Saved IPR-anchor pin (mpu.wells.prop_hist's ipr_wt_uid — W2 of the
# woffl-prop-hist-persistence plan). The module-level `_no_live_prop_hist_calls`
# autouse fixture makes `fetch_latest_prop` return None (no pin) by default;
# these tests override it per-scenario via monkeypatch, at the SOURCE module
# (`woffl.assembly.prop_hist_client.fetch_latest_prop`) so the real
# `_load_pinned_anchor` code path — including its memoization and exception
# handling — is exercised, not bypassed.
# ---------------------------------------------------------------------------


def _mock_fetch_latest_prop(
    monkeypatch,
    *,
    value,
    entry_datetime=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
    entry_user="Scott",
):
    # fetch_latest_prop's second tuple element is entry_datetime -- a full
    # timestamp (schema migrated off the old date-only entry_date column,
    # see prop_hist_client) -- not a bare date string.
    monkeypatch.setattr(
        "woffl.assembly.prop_hist_client.fetch_latest_prop",
        lambda well_name, prop_id: (value, entry_datetime, entry_user),
    )


def _make_test_df_with_uid():
    """_make_test_df's three rows (05-10 / 04-01 / 03-01), each tagged with a
    wt_uid like a real Databricks-sourced test frame would carry."""
    df = _make_test_df()
    df["wt_uid"] = [301.0, 302.0, 303.0]
    return df


def test_pin_defaults_fresh_well_to_specific_and_shows_caption(picker_st, monkeypatch):
    """A saved pin whose test IS in the frame defaults a fresh well load (no
    sig yet) to ('specific', that test's date) via the existing sig-restore
    machinery, and the caption reports who/when it was saved."""
    _mock_fetch_latest_prop(
        monkeypatch,
        value=302,
        entry_datetime=datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc),
        entry_user="Scott",
    )
    df = _make_test_df_with_uid()
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "specific"
    assert anchor_date == pd.Timestamp("2026-04-01")  # wt_uid 302's test date
    captions = "\n".join(picker_st.captions)
    assert "Saved IPR" in captions
    assert "2026-04-01" in captions
    assert "Scott" in captions
    assert "2026-06-01" in captions  # _format_pin_date renders the timestamp as a date


def test_pin_does_not_override_existing_session_choice(picker_st, monkeypatch):
    """A pin must NEVER override a selection already active this session —
    only a genuinely fresh well load (no sig yet) consults it."""
    _mock_fetch_latest_prop(monkeypatch, value=302)
    df = _make_test_df_with_uid()
    picker_st.session_state["sw_ipr_applied_sig_WELL"] = ("median", None)
    mode, _ = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "median"
    # No "Saved IPR" caption either — the active anchor isn't the pin's test.
    assert not any("Saved IPR" in c for c in picker_st.captions)


def test_pin_uid_not_in_frame_falls_back_with_caption(picker_st, monkeypatch):
    """A pin whose wt_uid has aged out of the current test window falls back
    to the normal 'most recent' default and shows the aged-out caption."""
    _mock_fetch_latest_prop(monkeypatch, value=999)  # no test carries this uid
    df = _make_test_df_with_uid()
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"
    assert anchor_date is None
    captions = "\n".join(picker_st.captions)
    assert "not in the current test window" in captions
    assert "999" in captions


def test_pin_null_value_means_no_pin(picker_st, monkeypatch):
    """A NULL/None prop_value is the un-pin marker (W3) — treated as no saved
    pin, same as if prop_hist had no row at all. There is no sign-based
    sentinel: real wt_uid values are signed and span both positive and
    negative ranges, so only None/NaN means "no pin"."""
    _mock_fetch_latest_prop(monkeypatch, value=None)
    df = _make_test_df_with_uid()
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"
    assert anchor_date is None
    # No provenance caption. (A separate "gate off" caption from the W3
    # save/clear pin controls is expected here — ALLOW_DATABRICKS_WRITES
    # isn't set in tests — but it's unrelated to pin PROVENANCE, which is
    # what this test covers.)
    assert not any("Saved IPR" in c for c in picker_st.captions)


def test_pin_nan_value_means_no_pin(picker_st, monkeypatch):
    """A NaN prop_value (however the connector surfaces a NULL numeric
    column) is likewise treated as no saved pin."""
    _mock_fetch_latest_prop(monkeypatch, value=float("nan"))
    df = _make_test_df_with_uid()
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"
    assert anchor_date is None
    assert not any("Saved IPR" in c for c in picker_st.captions)


def test_pin_negative_wt_uid_applies_as_a_valid_pin(picker_st, monkeypatch):
    """A NEGATIVE wt_uid present in the test frame must auto-apply as a
    valid pin, exactly like a positive one — real wt_uid values in
    vw_well_test are signed and span roughly -3.6M to +3.1M (almost all
    negative in practice, e.g. C-045's real saved pin at -3576674)."""
    _mock_fetch_latest_prop(
        monkeypatch,
        value=-302,
        entry_datetime=datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc),
        entry_user="Scott",
    )
    df = _make_test_df_with_uid()
    df["wt_uid"] = [-301.0, -302.0, -303.0]
    mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "specific"
    assert anchor_date == pd.Timestamp("2026-04-01")  # wt_uid -302's test date
    captions = "\n".join(picker_st.captions)
    assert "Saved IPR" in captions
    assert "2026-04-01" in captions


def test_pin_lookup_raises_falls_back_silently(picker_st, monkeypatch, caplog):
    """ANY exception from fetch_latest_prop (offline, missing grant, bad row)
    degrades to the normal default — never raises, never blocks the Solver,
    just a logger.warning (never st.error)."""
    import logging

    def _raise(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr("woffl.assembly.prop_hist_client.fetch_latest_prop", _raise)
    df = _make_test_df_with_uid()
    with caplog.at_level(logging.WARNING, logger="woffl.gui.tabs.jetpump_solver"):
        mode, anchor_date = jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert mode == "recent"
    assert anchor_date is None
    # No provenance caption (see note above re: the unrelated W3 gate-off caption).
    assert not any("Saved IPR" in c for c in picker_st.captions)
    assert any("pin lookup failed" in r.message.lower() for r in caplog.records)


def test_pin_lookup_memoized_per_well_per_session(picker_st, monkeypatch):
    """fetch_latest_prop is called at most once per well per session — a
    second render (e.g. the seed-triggered rerun) reuses the memo rather than
    re-hitting Databricks."""
    calls = []

    def _tracked(well_name, prop_id):
        calls.append(well_name)
        return (302, datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc), "Scott")

    monkeypatch.setattr("woffl.assembly.prop_hist_client.fetch_latest_prop", _tracked)
    df = _make_test_df_with_uid()
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert len(calls) == 1


def test_pin_lookup_failure_also_memoized(picker_st, monkeypatch):
    """A failed lookup is cached too — a dead connection must not be retried
    on every rerun."""
    calls = []

    def _raise(*a, **k):
        calls.append(1)
        raise RuntimeError("offline")

    monkeypatch.setattr("woffl.assembly.prop_hist_client.fetch_latest_prop", _raise)
    df = _make_test_df_with_uid()
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert len(calls) == 1


def test_clear_pin_cache_forces_requery(picker_st, monkeypatch):
    """The W3 seam: _clear_pin_cache pops the per-well memo so the very next
    lookup re-queries prop_hist instead of replaying the pre-save value."""
    calls = []

    def _tracked(well_name, prop_id):
        calls.append(well_name)
        return (302, datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc), "Scott")

    monkeypatch.setattr("woffl.assembly.prop_hist_client.fetch_latest_prop", _tracked)
    df = _make_test_df_with_uid()
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    jetpump_solver._clear_pin_cache("WELL")
    jetpump_solver._render_ipr_anchor_control("WELL", df)
    assert len(calls) == 2


def test_find_test_row_by_wt_uid(picker_st):
    """ipr_anchor.find_test_row_by_wt_uid — the pure lookup _load_pinned_anchor
    delegates to."""
    from woffl.gui.ipr_anchor import find_test_row_by_wt_uid

    df = _make_test_df_with_uid()
    row = find_test_row_by_wt_uid(df, 302)
    assert row is not None
    assert row["WtDate"] == pd.Timestamp("2026-04-01")

    assert find_test_row_by_wt_uid(df, 999) is None
    assert find_test_row_by_wt_uid(pd.DataFrame(), 302) is None
    assert find_test_row_by_wt_uid(None, 302) is None
    # No wt_uid column at all (older cached frame / synthetic test data).
    assert find_test_row_by_wt_uid(_make_test_df(), 302) is None


# ---------------------------------------------------------------------------
# Solver "Save IPR as well default" / "Clear saved IPR" buttons (W3 of the
# woffl-prop-hist-persistence plan) — _render_ipr_pin_controls, rendered
# inside _render_ipr_anchor_control right under the provenance caption.
# ---------------------------------------------------------------------------


def _enable_writes(monkeypatch):
    monkeypatch.setattr("woffl.gui.ipr_anchor.writes_enabled", lambda: True)


class TestSaveIprPinButton:
    def test_hidden_with_caption_when_gate_off(self, picker_st, monkeypatch):
        """Default env (no ALLOW_DATABRICKS_WRITES) — no button at all, just
        a one-line explanatory caption."""
        df = _make_test_df_with_uid()
        jetpump_solver._render_ipr_anchor_control("WELL", df)
        assert "sw_save_ipr_pin_WELL" not in picker_st.button_calls
        assert "sw_clear_ipr_pin_WELL" not in picker_st.button_calls
        assert any("ALLOW_DATABRICKS_WRITES" in c for c in picker_st.captions)

    def test_hidden_with_caption_when_anchor_not_pinnable(self, picker_st, monkeypatch):
        """Gate on, but the resolved anchor row has no wt_uid column at all
        (older frame / synthetic data) — the button is hidden, not just
        disabled, with a reason caption."""
        _enable_writes(monkeypatch)
        df = _make_test_df()  # no wt_uid column
        jetpump_solver._render_ipr_anchor_control("WELL", df)
        assert "sw_save_ipr_pin_WELL" not in picker_st.button_calls
        assert any(
            "Save IPR as well default" in c and "unavailable" in c
            for c in picker_st.captions
        )

    def test_button_shown_and_click_pushes_resolved_wt_uid(
        self, picker_st, monkeypatch
    ):
        """Clicking the button pushes the CURRENTLY-resolved anchor's wt_uid
        (via ipr_anchor.pin_ipr_anchor), clears the pin memo, toasts, and
        reruns."""
        _enable_writes(monkeypatch)
        df = _make_test_df_with_uid()

        pushed = {}

        def _fake_pin(well_name, anchor_row):
            pushed["well"] = well_name
            pushed["uid"] = anchor_row.get("wt_uid")
            return True, "📌 IPR saved to Databricks — test 2026-05-10"

        monkeypatch.setattr("woffl.gui.ipr_anchor.pin_ipr_anchor", _fake_pin)
        cleared = []
        monkeypatch.setattr(
            jetpump_solver, "_clear_pin_cache", lambda w: cleared.append(w)
        )

        picker_st.clicks.add("sw_save_ipr_pin_WELL")
        with pytest.raises(_Rerun):
            jetpump_solver._render_ipr_anchor_control("WELL", df)

        assert "sw_save_ipr_pin_WELL" in picker_st.button_calls
        assert pushed["well"] == "WELL"
        assert pushed["uid"] == 301.0  # most-recent test (no sig set yet)
        assert cleared == ["WELL"]
        assert picker_st.toasts == ["📌 IPR saved to Databricks — test 2026-05-10"]
        assert picker_st.rerun_count == 1

    def test_click_skip_shows_caption_not_warning_no_rerun(
        self, picker_st, monkeypatch
    ):
        _enable_writes(monkeypatch)
        df = _make_test_df_with_uid()
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.pin_ipr_anchor",
            lambda w, r: (False, "IPR not saved to Databricks: some reason."),
        )
        picker_st.clicks.add("sw_save_ipr_pin_WELL")
        jetpump_solver._render_ipr_anchor_control("WELL", df)  # must not raise/rerun
        assert picker_st.rerun_count == 0
        assert picker_st.warnings == []
        assert any("some reason" in c for c in picker_st.captions)

    def test_click_failure_shows_warning_no_rerun(self, picker_st, monkeypatch):
        _enable_writes(monkeypatch)
        df = _make_test_df_with_uid()
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.pin_ipr_anchor",
            lambda w, r: (False, "Could not save IPR to Databricks: connection reset"),
        )
        picker_st.clicks.add("sw_save_ipr_pin_WELL")
        jetpump_solver._render_ipr_anchor_control("WELL", df)  # must not raise/rerun
        assert picker_st.rerun_count == 0
        assert any("connection reset" in w for w in picker_st.warnings)


class TestClearIprPinButton:
    def test_not_rendered_when_no_pin(self, picker_st, monkeypatch):
        """No saved pin (the module default) -> the Clear button never
        renders, even with writes enabled."""
        _enable_writes(monkeypatch)
        df = _make_test_df_with_uid()
        jetpump_solver._render_ipr_anchor_control("WELL", df)
        assert "sw_clear_ipr_pin_WELL" not in picker_st.button_calls

    def test_rendered_when_pin_exists_and_click_clears(self, picker_st, monkeypatch):
        _enable_writes(monkeypatch)
        _mock_fetch_latest_prop(monkeypatch, value=302, entry_user="Scott")
        df = _make_test_df_with_uid()

        cleared_push = []
        monkeypatch.setattr(
            "woffl.gui.ipr_anchor.clear_ipr_pin",
            lambda w: (cleared_push.append(w) or (True, "Saved IPR cleared")),
        )
        cache_cleared = []
        monkeypatch.setattr(
            jetpump_solver, "_clear_pin_cache", lambda w: cache_cleared.append(w)
        )

        picker_st.clicks.add("sw_clear_ipr_pin_WELL")
        with pytest.raises(_Rerun):
            jetpump_solver._render_ipr_anchor_control("WELL", df)

        assert "sw_clear_ipr_pin_WELL" in picker_st.button_calls
        assert cleared_push == ["WELL"]
        assert cache_cleared == ["WELL"]
        assert picker_st.toasts == ["Saved IPR cleared"]
        assert picker_st.rerun_count == 1

    def test_rendered_for_stale_pin_too(self, picker_st, monkeypatch):
        """A pin whose test aged out of the frame (status 'stale') still
        offers Clear — the row persists in Databricks even though the
        Solver can't currently show it as applied."""
        _enable_writes(monkeypatch)
        _mock_fetch_latest_prop(monkeypatch, value=999)  # no test carries uid 999
        df = _make_test_df_with_uid()
        jetpump_solver._render_ipr_anchor_control("WELL", df)
        assert "sw_clear_ipr_pin_WELL" in picker_st.button_calls
