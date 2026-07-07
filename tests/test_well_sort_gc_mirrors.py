"""Tests for the widget-key GC-survival mirrors in well_sort.py (P1-31).

Per this repo's CLAUDE.md ("Widget state is garbage-collected when its view
isn't rendered"), a Scott's Tools tab detour can drop a widget's
session_state key entirely. Before this fix, well_sort.py's POPS-pads
multiselect, the per-well PopsPad=True override multiselect, and each POPs
pad's pump-limit number_input fell back to a hard-coded default whenever
their widget key was absent — silently discarding the user's actual
selection/edit.

The fix mirrors the selection to a non-widget session_state key on every
render and computes the widget's fallback default from that mirror instead
of the hard-coded default, so restoring after a GC is a no-op (the widget
key still wins when it survives) but a genuine GC recovers the last known
value rather than reverting.

These tests exercise the small extracted, pure(ish) helpers directly against
a fake ``st.session_state`` dict — they don't drive the full ``render_tab()``
Streamlit UI (that needs the live click-through Scott already uses for this
class of bug; see CLAUDE.md's note on tab-switch GC not being reproducible
via a mocked plain-dict session_state).
"""

import streamlit as st

import woffl.gui.scotts_tools.well_sort as well_sort


def _fresh_session_state(monkeypatch):
    fake = {}
    monkeypatch.setattr(st, "session_state", fake)
    return fake


class TestPopsPadsFallback:
    def test_no_mirror_uses_hardcoded_default(self, monkeypatch):
        _fresh_session_state(monkeypatch)
        assert well_sort._pops_pads_fallback() == well_sort._DEFAULT_POPS_PADS

    def test_mirror_present_wins_over_hardcoded_default(self, monkeypatch):
        fake = _fresh_session_state(monkeypatch)
        fake[well_sort._POPS_PADS_MIRROR_KEY] = ["B", "G"]
        assert well_sort._pops_pads_fallback() == ["B", "G"]

    def test_returns_a_copy_not_the_stored_list(self, monkeypatch):
        # Callers filter this list; it must not alias session_state's own list.
        fake = _fresh_session_state(monkeypatch)
        fake[well_sort._POPS_PADS_MIRROR_KEY] = ["B", "G"]
        result = well_sort._pops_pads_fallback()
        result.append("Z")
        assert fake[well_sort._POPS_PADS_MIRROR_KEY] == ["B", "G"]


class TestPopsForceTrueFallback:
    def test_no_mirror_defaults_empty(self, monkeypatch):
        _fresh_session_state(monkeypatch)
        assert well_sort._pops_force_true_fallback() == []

    def test_mirror_present_wins(self, monkeypatch):
        fake = _fresh_session_state(monkeypatch)
        fake[well_sort._POPS_FORCE_TRUE_MIRROR_KEY] = ["MPS-08"]
        assert well_sort._pops_force_true_fallback() == ["MPS-08"]


class TestPadLimitMirror:
    def test_seed_uses_preset_on_first_visit(self, monkeypatch):
        """Neither the widget key nor the mirror exist yet -> preset."""
        _fresh_session_state(monkeypatch)
        well_sort._seed_pad_limit_widget("E")
        assert st.session_state[well_sort._pad_limit_key("E")] == (
            well_sort.PUMP_LIMIT_PRESETS["E"]
        )

    def test_seed_restores_from_mirror_after_gc(self, monkeypatch):
        """Widget key GC'd (absent) but the mirror survived with an edited
        value -> re-seed from the mirror, NOT the static preset. This is the
        exact bug P1-31 describes: before the fix, this case always fell back
        to the preset, silently discarding the user's edit."""
        fake = _fresh_session_state(monkeypatch)
        edited_value = well_sort.PUMP_LIMIT_PRESETS["E"] + 5_000
        fake[well_sort._pad_limit_mirror_key("E")] = edited_value
        assert well_sort._pad_limit_key("E") not in fake

        well_sort._seed_pad_limit_widget("E")

        assert st.session_state[well_sort._pad_limit_key("E")] == edited_value

    def test_seed_is_noop_when_widget_key_survived(self, monkeypatch):
        """Widget key already present (no GC this rerun) -> left untouched,
        even if it disagrees with the mirror or the preset."""
        fake = _fresh_session_state(monkeypatch)
        fake[well_sort._pad_limit_key("E")] = 99_999
        fake[well_sort._pad_limit_mirror_key("E")] = 12_345

        well_sort._seed_pad_limit_widget("E")

        assert st.session_state[well_sort._pad_limit_key("E")] == 99_999

    def test_reset_pad_limit_sets_widget_and_mirror_to_preset(self, monkeypatch):
        fake = _fresh_session_state(monkeypatch)
        fake[well_sort._pad_limit_key("M")] = 1
        fake[well_sort._pad_limit_mirror_key("M")] = 1

        well_sort._reset_pad_limit("M")

        preset = well_sort.PUMP_LIMIT_PRESETS["M"]
        assert st.session_state[well_sort._pad_limit_key("M")] == preset
        assert st.session_state[well_sort._pad_limit_mirror_key("M")] == preset
