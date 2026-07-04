"""Import-level smoke tests for the GUI page modules.

The 565-test suite couldn't see a NameError inside a page module (P0-1: the pad
batch auto-match called a function that was never imported, and the per-well
``except Exception`` reported it as a data problem). These tests make "every
page module at least imports, and the auto-match input builder resolves" a
hard gate.

``woffl.gui.app`` is deliberately excluded — it warms Databricks fetches at
module level.
"""

import importlib

import pandas as pd
import pytest

PAGE_MODULES = [
    "woffl.gui.pad_helpers",
    "woffl.gui.params",
    "woffl.gui.sidebar",
    "woffl.gui.tab_helpers",
    "woffl.gui.utils",
    "woffl.gui.single_well_page",
    "woffl.gui.workflow_page",
    "woffl.gui.well_database_page",
    "woffl.gui.well_sort_page",
    "woffl.gui.pad_page",
    "woffl.gui.s_pad_page",
    "woffl.gui.i_pad_page",
    "woffl.gui.m_pad_page",
    "woffl.gui.pad_optimize",
    "woffl.gui.pad_plant_base",
    "woffl.gui.s_pad_plant",
    "woffl.gui.i_pad_plant",
    "woffl.gui.m_pad_plant",
    "woffl.gui.s_pad_ipr_report",
    "woffl.gui.memory_gauge",
    "woffl.gui.joint_match",
    "woffl.gui.ipr_anchor",
    "woffl.gui.ipr_backmatch",
    "woffl.gui.ipr_viz",
    "woffl.gui.optimization_viz",
    "woffl.gui.fric_calibration",
    "woffl.gui.pf_calibration",
    "woffl.gui.pdf_export",
    "woffl.gui.explainers",
    "woffl.gui.well_test_cache",
    "woffl.gui.components.download",
    "woffl.gui.components.dataframe_display",
    "woffl.gui.tabs.jetpump_solver",
    "woffl.gui.tabs.batch_run",
    "woffl.gui.tabs.power_fluid_range",
    "woffl.gui.tabs.jp_history_tab",
    "woffl.gui.tabs.pressure_profile",
    "woffl.gui.tabs.well_profile",
    "woffl.gui.tabs.pump_equivalent",
    "woffl.gui.workflow_steps.step1_select_wells",
    "woffl.gui.workflow_steps.step2_review_ipr",
    "woffl.gui.workflow_steps.step2_5_precalibrate",
    "woffl.gui.workflow_steps.step3_configure_optimize",
    "woffl.gui.workflow_steps.step4_results",
    "woffl.gui.workflow_steps.step_review_wells",
    "woffl.gui.workflow_steps.well_review_store",
    "woffl.gui.scotts_tools.page",
    "woffl.gui.scotts_tools._common",
    "woffl.gui.scotts_tools.pf_scenario",
    "woffl.gui.scotts_tools.header_impact",
    "woffl.gui.scotts_tools.header_engine",
    "woffl.gui.scotts_tools.header_trend",
    "woffl.gui.scotts_tools.header_report",
    "woffl.gui.scotts_tools.jp_calibration",
    "woffl.gui.scotts_tools.jp_fric_trend",
    "woffl.gui.scotts_tools.jp_washout",
    "woffl.gui.scotts_tools.pad_watercut",
    "woffl.gui.scotts_tools.well_sort",
]


@pytest.mark.parametrize("mod", PAGE_MODULES)
def test_page_module_imports(mod):
    importlib.import_module(mod)


class TestBatchAutomatchInputs:
    """Regression for P0-1: _batch_automatch_inputs must resolve its helpers.

    jp_hist=None exits with 'no current pump' — but only AFTER the
    _recent_test_rates call that used to raise NameError, so reaching that
    reason proves the helper chain is intact.
    """

    def _patch_data(self, monkeypatch):
        import woffl.gui.utils as gutils

        tests = pd.DataFrame(
            {
                "WtDate": pd.to_datetime(["2026-05-01", "2026-06-01"]),
                "WtOilVol": [100.0, 120.0],
                "lift_wat": [2000.0, 2200.0],
                "BHP": [900.0, 910.0],
                "whp": [250.0, 260.0],
            }
        )
        monkeypatch.setattr(
            gutils, "get_well_data", lambda w: {"JP_TVD": 4000, "is_sch": True}
        )
        monkeypatch.setattr(gutils, "get_well_tests_for_well", lambda w, *a, **k: tests)

    def test_reaches_pump_check_not_nameerror(self, monkeypatch):
        from woffl.gui.workflow_steps import step_review_wells as srw

        self._patch_data(monkeypatch)
        kwargs, raw, why = srw._batch_automatch_inputs("MPS-99", None, 3400.0)
        assert kwargs is None and raw is None
        assert why == "no current pump"

    def test_helpers_are_the_shared_ones(self):
        # R-1 Phase C: the render path lives in pad_page; the S/I/M modules
        # are thin specs and no longer touch the helpers directly.
        from woffl.gui import pad_helpers, pad_page
        from woffl.gui.workflow_steps import step_review_wells as srw

        assert srw._recent_test_rates is pad_helpers.recent_test_rates
        assert pad_page._recent_test_rates is pad_helpers.recent_test_rates
        assert pad_page._parse_pump is pad_helpers.parse_pump

    def test_parse_pump(self):
        from woffl.gui.pad_helpers import parse_pump

        assert parse_pump("12B") == ("12", "B")
        assert parse_pump(" 10A ") == ("10", "A")
        assert parse_pump("Shut in") is None
        assert parse_pump("") is None
        assert parse_pump("B") is None
