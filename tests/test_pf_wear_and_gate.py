"""Tests for the live-PF era of the PF-mismatch check.

1. render_pf_mismatch_warning: a MEASURED test-day PF pressure must never
   gate BHP calibration (the old gate existed because pressure was assumed);
   tests without a daily reading keep the legacy severe-tier gate.
2. estimate_nozzle_wear: with pressure pinned at the measured value, the
   effective nozzle-area factor is the free variable — bisection must
   recover a known factor and flag out-of-range targets as bounded.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from woffl.gui import pf_calibration as pfc
from woffl.gui.utils import render_pf_mismatch_warning

# ── warning gate ───────────────────────────────────────────────────────────


class TestPFMismatchGate:
    def _run(self, modeled, actual, measured_pf):
        with patch("woffl.gui.utils.st") as st_mock:
            st_mock.session_state = MagicMock()
            shown, blocked = render_pf_mismatch_warning(
                modeled,
                actual,
                3168,
                test_date_str="2026-06-17",
                well_name="MPB-28",
                measured_pf=measured_pf,
            )
            return shown, blocked, st_mock

    def test_severe_with_measured_pf_never_blocks(self):
        # 1000 modeled vs 2000 actual = 50% off — severe. Measured PF present
        # → diagnostic warning, calibration allowed.
        shown, blocked, st_mock = self._run(1000.0, 2000.0, measured_pf=2436.0)
        assert shown is True
        assert blocked is False
        st_mock.warning.assert_called_once()
        st_mock.error.assert_not_called()

    def test_severe_without_measured_pf_blocks(self):
        # Legacy path: pressure is an assumption — severe still gates.
        shown, blocked, st_mock = self._run(1000.0, 2000.0, measured_pf=None)
        assert shown is True
        assert blocked is True
        st_mock.error.assert_called_once()

    def test_moderate_with_measured_pf_is_soft(self):
        # 1180 vs 1000 = 18% — moderate tier. Info, not blocking.
        shown, blocked, st_mock = self._run(1180.0, 1000.0, measured_pf=2436.0)
        assert shown is True
        assert blocked is False
        st_mock.info.assert_called_once()
        st_mock.error.assert_not_called()

    def test_within_tolerance_silent(self):
        shown, blocked, _ = self._run(1050.0, 1000.0, measured_pf=2436.0)
        assert (shown, blocked) == (False, False)

    def test_measured_pf_message_names_wear(self):
        _, _, st_mock = self._run(1000.0, 2000.0, measured_pf=2436.0)
        msg = st_mock.warning.call_args[0][0]
        assert "measured" in msg
        assert "wash-out" in msg


# ── nozzle-wear estimator ──────────────────────────────────────────────────


def _fake_solver_factory(rate_per_ft2: float):
    """jetpump_solver stand-in: qnz proportional to nozzle area (Bernoulli
    at fixed pressure), constant psu/qoil, never sonic."""

    def fake(*, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, ipr_su,
             prop_su, prop_pf, jpump_direction):
        qnz = rate_per_ft2 * jpump.anz
        return 1200.0, False, 400.0, 600.0, qnz, 0.4

    return fake


class TestEstimateNozzleWear:
    def _catalog_qnz(self, rate_per_ft2):
        from woffl.geometry.jetpump import JetPump

        return rate_per_ft2 * JetPump("12", "B").anz

    def _run(self, target_lift, rate_per_ft2=1.0e7):
        with patch.object(
            pfc, "jetpump_solver", _fake_solver_factory(rate_per_ft2)
        ):
            return pfc.estimate_nozzle_wear(
                well_name="MPB-28",
                target_lift=target_lift,
                ppf_surf=2436.0,
                pwh=165.0,
                tsu=75.0,
                nozzle="12",
                throat="B",
                knz=0.01, ken=0.03, kth=0.3, kdi=0.3,
                wellbore=None, wellprof=None,
                ipr_su=None, prop_su=None, prop_pf=None,
            )

    def test_recovers_known_wear_factor(self):
        rate = 1.0e7
        catalog_qnz = self._catalog_qnz(rate)
        r = self._run(target_lift=1.2 * catalog_qnz, rate_per_ft2=rate)
        assert r.converged and not r.bounded
        # qnz ∝ area, so a 1.2× rate needs a 1.2× area.
        assert r.wear_factor == pytest.approx(1.2, abs=0.01)
        assert r.dnz_effective == pytest.approx(
            r.dnz_catalog * np.sqrt(1.2), rel=1e-3
        )
        assert r.modeled_qnz == pytest.approx(1.2 * catalog_qnz, rel=0.01)

    def test_healthy_pump_factor_near_one(self):
        rate = 1.0e7
        r = self._run(target_lift=self._catalog_qnz(rate), rate_per_ft2=rate)
        assert r.converged
        assert r.wear_factor == pytest.approx(1.0, abs=0.01)
        assert r.equivalent_nozzle == "12"

    def test_equivalent_nozzle_steps_up_when_washed(self):
        # 12 nozzle dia 0.2099; 13 is 0.237 → area ratio (0.237/0.2099)² ≈ 1.27.
        rate = 1.0e7
        r = self._run(target_lift=1.27 * self._catalog_qnz(rate), rate_per_ft2=rate)
        assert r.converged
        assert r.equivalent_nozzle == "13"

    def test_unreachable_target_bounded_high(self):
        rate = 1.0e7
        r = self._run(target_lift=5.0 * self._catalog_qnz(rate), rate_per_ft2=rate)
        assert r.bounded and not r.converged
        assert r.wear_factor == pytest.approx(pfc.WEAR_HI_DEFAULT)
        assert r.lift_residual < 0

    def test_overshoot_target_bounded_low(self):
        rate = 1.0e7
        r = self._run(target_lift=0.1 * self._catalog_qnz(rate), rate_per_ft2=rate)
        assert r.bounded and not r.converged
        assert r.wear_factor == pytest.approx(pfc.WEAR_LO_DEFAULT)
        assert r.lift_residual > 0

    def test_solver_failure_returns_nan(self):
        def dead(**kwargs):
            raise ValueError("no solution")

        with patch.object(pfc, "jetpump_solver", dead):
            r = pfc.estimate_nozzle_wear(
                well_name="X", target_lift=2000.0, ppf_surf=2436.0,
                pwh=165.0, tsu=75.0, nozzle="12", throat="B",
                knz=0.01, ken=0.03, kth=0.3, kdi=0.3,
                wellbore=None, wellprof=None,
                ipr_su=None, prop_su=None, prop_pf=None,
            )
        assert not r.converged and not r.bounded
        assert np.isnan(r.modeled_qnz)
