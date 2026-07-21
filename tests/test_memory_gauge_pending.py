"""Tests for the memory-gauge pending-upload stash.

Regression for the "gauge data dropped on a view detour" bug: files dropped
into the Solver's ``st.file_uploader`` but not yet Applied used to live ONLY
in the uploader's widget state. The Single-Well page's segmented control runs
only the active view per rerun, so a Solver → Batch Run → Solver detour let
Streamlit garbage-collect that widget state and the uploaded files silently
vanished.

The fix parses uploads immediately and stashes them per-well via
``memory_gauge.get_pending_files`` / ``set_pending_files`` under a plain
(non-widget) session key that survives the detour. These tests pin the CRUD
contract and the GC-survival invariant.

NOTE (from CLAUDE.md): unit tests that mock ``session_state`` as a plain dict
cannot reproduce Streamlit's real widget-state GC — the GC simulation below
only asserts the stash lives outside the widget keys that GC deletes. Verify
the full flow live: upload a file, DON'T click Apply, switch to Batch Run,
return to Solver → the pending file + Apply button must still be there.

Streamlit is mocked before importing the module so the module-level
``import streamlit as st`` resolves without a running server.
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mock streamlit *before* importing the module (mirrors test_solver_ipr_sync).
_st_mock = MagicMock()
_st_mock.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) else lambda fn: fn
)
sys.modules.setdefault("streamlit", _st_mock)

from woffl.gui import memory_gauge  # noqa: E402


class _FakeSt:
    """Minimal st replacement with a real dict session_state."""

    def __init__(self):
        self.session_state = {}


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    monkeypatch.setattr(memory_gauge, "st", fs)
    return fs


def _mk_file(name: str) -> memory_gauge.MemoryGaugeFile:
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"])
    df = pd.DataFrame({"timestamp": ts, "pressure": [1000.0, 1010.0]})
    return memory_gauge.MemoryGaugeFile(
        source_filename=name,
        raw_df=df,
        start_date=ts[0],
        end_date=ts[1],
        sample_count=2,
        uploaded_at=datetime.now(),
    )


def test_get_pending_default_empty(fake_st):
    assert memory_gauge.get_pending_files("MPB-35") == []


def test_set_and_get_roundtrip_is_per_well(fake_st):
    files = [_mk_file("run1.xlsx"), _mk_file("run2.xlsx")]
    memory_gauge.set_pending_files("MPB-35", files)

    assert memory_gauge.get_pending_files("MPB-35") == files
    # Other wells are untouched.
    assert memory_gauge.get_pending_files("MPS-17") == []


def test_empty_list_clears_entry(fake_st):
    memory_gauge.set_pending_files("MPB-35", [_mk_file("run1.xlsx")])
    memory_gauge.set_pending_files("MPB-35", [])

    assert memory_gauge.get_pending_files("MPB-35") == []
    # The well's entry is fully removed, not left as an empty list.
    assert "MPB-35" not in fake_st.session_state.get(
        memory_gauge._PENDING_KEY, {}
    )


def test_pending_survives_widget_state_gc(fake_st):
    """The stash must live outside the widget keys Streamlit GCs on a detour.

    Simulates the Solver → Batch Run → Solver round trip: every widget key
    the memory-gauge section creates (uploader, buttons, checkbox) is
    deleted — Streamlit drops widget state for anything that didn't render —
    while plain session keys persist. The pending files must survive.
    """
    well = "MPB-35"
    files = [_mk_file("run1.xlsx")]
    memory_gauge.set_pending_files(well, files)

    # Widget keys as the Solver's gauge section creates them.
    fake_st.session_state[f"mg_upload_{well}_0"] = ["<uploader widget state>"]
    fake_st.session_state[f"mg_apply_btn_{well}"] = False
    fake_st.session_state[f"mg_disregard_cb_{well}"] = False

    # Simulated widget-state GC: every non-underscore (widget) key vanishes.
    for key in [k for k in fake_st.session_state if not k.startswith("_")]:
        del fake_st.session_state[key]

    assert memory_gauge.get_pending_files(well) == files


def test_apply_contract_moves_pending_into_gauge(fake_st):
    """Mirrors the Apply handler: add each pending file, then clear the stash."""
    well = "MPB-35"
    files = [_mk_file("run1.xlsx"), _mk_file("run2.xlsx")]
    memory_gauge.set_pending_files(well, files)

    for pf in memory_gauge.get_pending_files(well):
        memory_gauge.add_file_to_gauge(well, pf)
    memory_gauge.set_pending_files(well, [])

    gauge = memory_gauge.get_gauge(well)
    assert gauge is not None
    assert [f.source_filename for f in gauge.files] == ["run1.xlsx", "run2.xlsx"]
    assert memory_gauge.get_pending_files(well) == []
