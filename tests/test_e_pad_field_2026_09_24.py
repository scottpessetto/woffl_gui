"""E-Pad booster calibrated to the E-41 surface-kit rate test (2026-09-24).

The test (meta ``field_test``): ramped until the drive's current limit, 889 A,
at 29,491 BWPD and 3,400 psi discharge from 2,704 psi CFP suction, 53.1 Hz;
any more rate lost discharge pressure. Baseline: 27,789 BWPD, 826 A, 2,725 psi
suction, same speed and discharge. These tests pin that the default plant is
that booster, that the stored defaults are the derivation from the test, and
that the calibration stays on the unit it was measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from server import schemas
from woffl.gui import e_pad_booster as epb
from woffl.gui.e_pad_plant import PLANT, EPadPlant

LIMIT = dict(rate=29491.0, hz=53.1, amps=889.0, suction=2704.0, discharge=3400.0)
BASE = dict(rate=27789.0, hz=53.1, amps=826.0, suction=2725.0, discharge=3400.0)


def _duty(plant: EPadPlant, q: float) -> tuple[float, float]:
    hz = plant.build.max_hz_at_flow(q, plant.specific_gravity(), plant.hz_max, plant.amps_per_bhp, plant.amp_limit)
    return hz, plant.build.amps(q, hz, plant.specific_gravity(), plant.amps_per_bhp)


def test_the_default_plant_delivers_the_tested_rate_at_the_header_on_the_current_limit():
    q = PLANT.budget_at_pressure(LIMIT["discharge"])
    assert q == pytest.approx(LIMIT["rate"], rel=1e-3)
    hz, amps = _duty(PLANT, q)
    assert hz == pytest.approx(LIMIT["hz"], abs=0.05)
    assert amps == pytest.approx(LIMIT["amps"], abs=1.0)
    assert PLANT.max_discharge_pressure(q) == pytest.approx(LIMIT["discharge"], abs=1.0)


def test_more_rate_than_tested_cannot_hold_the_header():
    # "Any additional increase in rate resulted in a loss of discharge pressure."
    beyond = PLANT.max_discharge_pressure(LIMIT["rate"] * 1.01)
    assert beyond is None or beyond < LIMIT["discharge"]


def test_the_uncalibrated_catalog_overstated_capacity_by_about_ten_percent():
    as_new = EPadPlant(suction_psi=2800.0, amps_per_bhp=0.1435, amp_limit=None, field_calibrated=False)
    assert as_new.budget_at_pressure(3400.0) == pytest.approx(32400.0)
    assert 2800.0 < as_new.budget_at_pressure(3400.0) - PLANT.budget_at_pressure(3400.0) < 3000.0


def test_the_baseline_point_is_reproduced_within_the_stated_tolerance():
    plant = EPadPlant(suction_psi=BASE["suction"])
    b = plant.build
    sg = plant.specific_gravity()
    amps = b.amps(BASE["rate"], BASE["hz"], sg, plant.amps_per_bhp)
    assert amps == pytest.approx(BASE["amps"], rel=0.06)  # 873 A vs 826 A measured
    # Below the limit the unit holds the header with margin (discharge held at
    # 3,400; the model makes about 3,477 psi there at 53.1 Hz).
    made = BASE["suction"] + b.dp_psi(BASE["rate"], BASE["hz"], sg, plant.condition)
    assert BASE["discharge"] <= made < BASE["discharge"] + 100.0


def test_stored_defaults_are_the_derivation_from_the_test():
    cal = epb.field_calibration("SM25000_26STG")
    d = epb.defaults()
    assert d["amps_per_bhp"] == pytest.approx(cal["amps_per_bhp"], abs=1e-4)
    assert d["suction_psi"] == cal["point"]["suction_psi"] == LIMIT["suction"]
    assert d["amp_limit_a"] == cal["point"]["amps"] == LIMIT["amps"]
    assert cal["condition"] == pytest.approx(0.766, abs=1e-3)
    assert cal["ror_hi_60hz"] == pytest.approx(LIMIT["rate"] * 60.0 / LIMIT["hz"])
    run = schemas.OptimizeRunRequest(kind="pad", pad="E")
    screen = schemas.EPadBoosterRequest()
    assert run.e_pad_suction_psi == screen.suction_psi == d["suction_psi"]
    assert run.e_pad_amp_limit_a == screen.amp_limit_a == d["amp_limit_a"]
    assert screen.amps_per_bhp == d["amps_per_bhp"]
    assert screen.suction_psi + screen.dp_psid == d["target_discharge_psi"]
    # The client's run form carries the same defaults.
    form = (Path(__file__).resolve().parents[1] / "web" / "src" / "state" / "runForm.ts").read_text(encoding="utf-8")
    assert "ePadSuction: 2704," in form and 'ePadAmpLimit: "889",' in form


def test_the_calibration_stays_on_the_unit_it_was_measured_on():
    alt = EPadPlant("SN35000_18STG")
    assert alt.field_calibration is None and alt.condition == 1.0
    assert alt.build.ror_60hz == (12400.0, 49500.0)
    as_new = EPadPlant(field_calibrated=False)
    assert as_new.condition == 1.0 and as_new.build.ror_60hz == (8100.0, 32400.0)
    # The motor (amps per BHP, 889 A) belongs to the kit, so it applies to both.
    assert alt.amp_limit == as_new.amp_limit == PLANT.amp_limit == 889.0
    # A stated condition wins over the calibration; the demonstrated range stays.
    stated = EPadPlant(condition=0.9)
    assert stated.condition == 0.9 and stated.flow_ceiling() == pytest.approx(PLANT.flow_ceiling())


def test_an_explicit_none_amp_limit_removes_the_cap():
    uncapped = EPadPlant(amp_limit=None)
    assert uncapped.amp_limit is None
    assert uncapped.budget_at_pressure(3400.0) > PLANT.budget_at_pressure(3400.0)


def test_a_default_e_pad_run_uses_the_tested_booster():
    from server.services.optimizer_runs import _pad_plant_for_run

    plant = _pad_plant_for_run("E", schemas.OptimizeRunRequest(kind="pad", pad="E"))
    assert plant.budget_at_pressure(3400.0) == pytest.approx(LIMIT["rate"], rel=1e-3)
    assert plant.curve_report()["nameplate"]["validated"].startswith("Calibrated to one E-41 rate-test point")
