"""Tests for the P1-32 fix in woffl.gui.scotts_tools.jp_washout.

P1-32: ``_build_scan_input`` computed ``pump_changed`` (stored under the key
``"PumpChangedSinceTest"``) but it was never propagated past that point —
``_calibrate_one`` built its own result dict from scratch and dropped the
field, so wash-out flags recommended changeouts for pumps that had already
been changed out since the scanned test.

These tests isolate ``_calibrate_one``'s propagation logic by mocking out
the physics-heavy helpers it calls (``build_well_config``,
``create_well_objects``, ``calibrate_pf_for_lift``), covering both the
success and the exception result branches.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from woffl.gui.pf_calibration import PfCalibrationResult
from woffl.gui.scotts_tools.jp_washout import _calibrate_one


def _row_dict(pump_changed: bool) -> dict:
    return {
        "Well": "MPB-28",
        "Pad": "B",
        "Pump": "12B",
        "Nozzle": "12",
        "Throat": "B",
        "PumpChangedSinceTest": pump_changed,
        "_vogel": None,
        "WtDate": pd.Timestamp("2026-01-01"),
        "Oil": 500.0,
        "Water": 1000.0,
        "Gas": 200.0,
        "LiftWat": 3000.0,
        "WHP": 210.0,
        "BHP": 800.0,
        "_chars": {"Well": "MPB-28"},
    }


def _fake_pf_result() -> PfCalibrationResult:
    return PfCalibrationResult(
        well_name="MPB-28",
        target_lift=3000.0,
        ppf_surf=3200.0,
        modeled_qnz=3000.0,
        modeled_qoil=500.0,
        modeled_bhp=800.0,
        lift_residual=0.0,
        converged=True,
        bounded=False,
        sonic=False,
        iterations=8,
    )


class TestPumpChangedPropagation:
    @pytest.mark.parametrize("pump_changed", [True, False])
    def test_success_branch_propagates_flag(self, pump_changed):
        row_dict = _row_dict(pump_changed)
        with (
            patch(
                "woffl.gui.scotts_tools.jp_washout.build_well_config"
            ) as mock_build_wc,
            patch(
                "woffl.gui.scotts_tools.jp_washout.create_well_objects",
                return_value=(object(), object(), object(), object(), object()),
            ),
            patch(
                "woffl.gui.scotts_tools.jp_washout.friction_coefs_from_chars",
                return_value={"knz": 0.01, "ken": 0.03, "kth": 0.30, "kdi": 0.30},
            ),
            patch(
                "woffl.gui.scotts_tools.jp_washout.calibrate_pf_for_lift",
                return_value=_fake_pf_result(),
            ),
        ):
            # wc.form_temp is read by _calibrate_one — give the fake
            # WellConfig-ish return value that attribute.
            mock_build_wc.return_value.form_temp = 150.0
            result = _calibrate_one(row_dict)

        assert result["Status"] == "ok"
        assert result["PumpChangedSinceTest"] is pump_changed

    @pytest.mark.parametrize("pump_changed", [True, False])
    def test_error_branch_still_propagates_flag(self, pump_changed):
        # Force the try block to raise so we exercise the except branch —
        # before the fix this branch (like the success branch) built its
        # dict from scratch with no PumpChangedSinceTest key at all.
        row_dict = _row_dict(pump_changed)
        with patch(
            "woffl.gui.scotts_tools.jp_washout.build_well_config",
            side_effect=RuntimeError("boom"),
        ):
            result = _calibrate_one(row_dict)

        assert result["Status"] == "error"
        assert result["PumpChangedSinceTest"] is pump_changed

    def test_missing_key_defaults_false(self):
        # Defensive: a row dict without the key (shouldn't happen given
        # _build_scan_input always sets it, but _calibrate_one must not
        # crash) falls back to False rather than raising KeyError.
        row_dict = _row_dict(False)
        del row_dict["PumpChangedSinceTest"]
        with patch(
            "woffl.gui.scotts_tools.jp_washout.build_well_config",
            side_effect=RuntimeError("boom"),
        ):
            result = _calibrate_one(row_dict)
        assert result["PumpChangedSinceTest"] is False
