"""Tests for pure-logic helpers in woffl.gui.tabs.batch_run.

Covers two P1 fixes from docs/code_review_2026-07-01.md:

- P1-17: ``batch_success_stats`` — counts of total/successful nozzle x
  throat combos, feeding the "Successful Runs" / "Success Rate" metrics
  that mirror power_fluid_range.py's Analysis Summary. Previously failed
  combos were silently dropped with no surfaced count.
- P1-14: ``nm_comparison_factor`` — the Nelder-Mead section compared a
  CALIBRATED seed pump row against a RAW (uncalibrated) Nelder-Mead
  result whenever the calibration toggle was on, producing a phantom
  oil-rate "uplift"/shortfall driven purely by the calibration factor.
  The factor returned here reconciles both sides onto the same basis.
"""

import numpy as np
import pandas as pd

from woffl.assembly.calibration import CalibrationResult
from woffl.gui.tabs.batch_run import batch_success_stats, nm_comparison_factor


class TestBatchSuccessStats:
    def test_all_successful(self):
        df = pd.DataFrame({"qoil_std": [100.0, 200.0, 300.0]})
        total, successful, pct = batch_success_stats(df)
        assert (total, successful) == (3, 3)
        assert pct == 100.0

    def test_partial_failure_counted_not_hidden(self):
        # 2 of 5 combos failed to converge (qoil_std NaN) — this is exactly
        # the P1-17 scenario: previously these were filtered out with no
        # visible count anywhere in the tab.
        df = pd.DataFrame({"qoil_std": [100.0, np.nan, 300.0, np.nan, 500.0]})
        total, successful, pct = batch_success_stats(df)
        assert total == 5
        assert successful == 3
        assert pct == 60.0

    def test_all_failed(self):
        df = pd.DataFrame({"qoil_std": [np.nan, np.nan]})
        total, successful, pct = batch_success_stats(df)
        assert (total, successful, pct) == (2, 0, 0.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"qoil_std": []})
        assert batch_success_stats(df) == (0, 0, 0.0)


def _make_calibration(factor: float) -> CalibrationResult:
    return CalibrationResult(
        well_name="MPB-28",
        current_nozzle="12",
        current_throat="B",
        model_oil=1000.0,
        actual_oil=1000.0 * factor,
        model_pf=3000.0,
        actual_pf=3000.0,
        model_bhp=800.0,
        actual_bhp=800.0,
        calibration_factor=factor,
    )


class TestNmComparisonFactor:
    def test_no_calibration_result_is_noop(self):
        # Calibration unavailable (no JP history / no actuals) — factor
        # must be a no-op regardless of the (irrelevant) applied flag.
        assert nm_comparison_factor(None, calibration_applied=True) == 1.0
        assert nm_comparison_factor(None, calibration_applied=False) == 1.0

    def test_calibration_available_but_not_applied_is_noop(self):
        # Toggle unchecked (or PF-blocked) — batch_pump.df was never
        # mutated, so the seed row is raw; the NM result must stay raw too.
        cal = _make_calibration(0.7)
        assert nm_comparison_factor(cal, calibration_applied=False) == 1.0

    def test_calibration_applied_returns_the_factor(self):
        # This is the P1-14 bug scenario: factor 0.7 previously wasn't
        # applied to the NM side, producing a phantom ~40% uplift
        # (1/0.7 - 1 ≈ 43%) when comparing a calibrated seed to raw NM.
        cal = _make_calibration(0.7)
        assert nm_comparison_factor(cal, calibration_applied=True) == 0.7

    def test_reconciled_basis_removes_phantom_uplift(self):
        # End-to-end sanity: with the factor applied to the raw NM oil
        # rate, an NM result that is physically identical to the seed's
        # pre-calibration raw rate now compares as "near-optimal" (delta
        # ~0) instead of showing a spurious large delta.
        cal = _make_calibration(0.7)
        raw_seed_oil = 1000.0
        calibrated_seed_oil = raw_seed_oil * cal.calibration_factor  # 700.0
        raw_nm_oil = 1000.0  # NM found essentially the same operating point

        factor = nm_comparison_factor(cal, calibration_applied=True)
        reconciled_nm_oil = raw_nm_oil * factor

        delta = reconciled_nm_oil - calibrated_seed_oil
        assert abs(delta) < 1e-9

        # The old (buggy) comparison would have shown a phantom uplift:
        buggy_delta = raw_nm_oil - calibrated_seed_oil
        assert buggy_delta == 300.0  # the phantom ~43% "uplift" at factor 0.7
