"""JP History chart builder — the PF-pressure trace toggle.

The PF trace shipped always-on; the tab now exposes it as a checkbox
("Show PF pressure"). These pin that ``show_pf`` reaches BOTH figure
paths — the plain chart and the pumps-in-hole strip figure that the
Solver also renders — so the toggle can't be silently dropped on one of
them (the two call sites are easy to miss: the strip builder calls
``_create_history_chart`` twice).
"""

import pandas as pd

from woffl.gui.tabs.jp_history_tab import (
    _create_history_chart,
    build_history_with_strip_figure,
)

PF_TRACE_NAME = "PF pressure (psi)"


def _tests_frame():
    return pd.DataFrame(
        {
            "WtDate": pd.to_datetime(["2026-01-10", "2026-03-14", "2026-05-02"]),
            "WtOilVol": [220.0, 205.0, 198.0],
            "WtWaterVol": [1400.0, 1520.0, 1610.0],
            "BHP": [980.0, 1010.0, 1035.0],
            "pf_press": [2740.0, 2695.0, 2810.0],
        }
    )


def _jp_changes():
    return pd.DataFrame(
        {
            "Date Set": pd.to_datetime(["2025-11-02", "2026-04-08"]),
            "Nozzle Number": ["12", "13"],
            "Throat Ratio": ["B", "A"],
        }
    )


def _trace_names(fig):
    return {tr.name for tr in fig.data}


class TestPlainChart:
    def test_pf_trace_present_by_default(self):
        fig = _create_history_chart("MPB-28", _tests_frame(), _jp_changes())
        assert PF_TRACE_NAME in _trace_names(fig)

    def test_pf_trace_hidden_when_toggled_off(self):
        fig = _create_history_chart(
            "MPB-28", _tests_frame(), _jp_changes(), show_pf=False
        )
        assert PF_TRACE_NAME not in _trace_names(fig)

    def test_toggle_leaves_other_traces_alone(self):
        """Only the PF trace moves — BHP and production must be untouched."""
        on = _trace_names(_create_history_chart("MPB-28", _tests_frame(), _jp_changes()))
        off = _trace_names(
            _create_history_chart(
                "MPB-28", _tests_frame(), _jp_changes(), show_pf=False
            )
        )
        assert on - off == {PF_TRACE_NAME}


class TestStripFigure:
    """The strip builder calls _create_history_chart on two branches (with and
    without JP installs on record) — show_pf must reach both."""

    def test_pf_trace_present_by_default(self):
        fig, _tl = build_history_with_strip_figure(
            "MPB-28", _jp_changes(), _tests_frame(), None, None
        )
        assert PF_TRACE_NAME in _trace_names(fig)

    def test_pf_trace_hidden_when_toggled_off(self):
        fig, _tl = build_history_with_strip_figure(
            "MPB-28", _jp_changes(), _tests_frame(), None, None, show_pf=False
        )
        assert PF_TRACE_NAME not in _trace_names(fig)

    def test_no_install_branch_honors_toggle(self):
        """S-67-style well: tests but no JP installs — the no-strip branch."""
        empty_jp = pd.DataFrame(columns=["Date Set"])
        on, _ = build_history_with_strip_figure(
            "MPS-67", empty_jp, _tests_frame(), None, None
        )
        off, _ = build_history_with_strip_figure(
            "MPS-67", empty_jp, _tests_frame(), None, None, show_pf=False
        )
        assert PF_TRACE_NAME in _trace_names(on)
        assert PF_TRACE_NAME not in _trace_names(off)
