"""Capacity review regressions: real plant curves and allocation, synthetic wells."""
from types import SimpleNamespace

import pandas as pd
import pytest

from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig
from woffl.gui import pad_optimize as po
from woffl.gui.e_pad_plant import EPadPlant
from woffl.gui.pad_plant_base import SPadPlant
from tests.test_pad_optimize import FreePlant


def _wells(*names):
    return [WellConfig(well_name=n, res_pres=1500, form_temp=100,
                       jpump_tvd=4000, qwf=500, pwf=500, use_survey=False) for n in names]


def _responses(monkeypatch, tables):
    def run(opt, max_workers=None):
        opt.batch_results = {}
        for well in opt.wells:
            values = tables[well.well_name]
            values = values(opt.power_fluid.pressure) if callable(values) else values
            oil, lift, formation = values if values else (float("nan"),) * 3
            row = dict(nozzle="12", throat="B", error="na" if values else "failed",
                       qoil_std=oil, lift_wat=lift, form_wat=formation,
                       totl_wat=lift+formation, psu_solv=500., sonic_status=False, mach_te=.5)
            opt.batch_results[well.well_name] = SimpleNamespace(df=pd.DataFrame([row]))
        return opt.batch_results
    monkeypatch.setattr(NetworkOptimizer, "run_all_batch_simulations", run)


def _run(wells, plant, **kwargs):
    return po.run_optimization(wells, plant, 3, ["12"], ["B"], "milp", None,
                               n_steps=3, refine_rounds=0, **kwargs)


def test_e_low_flow_undeliverable_pin_is_rejected_before_recommendation(monkeypatch):
    _responses(monkeypatch, {"E": (400., 3600., 400.)})
    with pytest.raises(RuntimeError, match="cannot hold"):
        _run(_wells("E"), EPadPlant(), setpoint_psi=3500.)


def test_e_sweep_selects_lower_deliverable_header(monkeypatch):
    def response(p):
        oil = 400 + .2 * (p-3000)
        return oil, 4000-oil, oil
    _responses(monkeypatch, {"E": response})
    _, _, meta = _run(_wells("E"), EPadPlant())
    assert meta["header_psi"] == 3000.
    assert meta["total_oil_bopd"] == 400.
    assert meta["feasible"] is True
    assert meta["coupling_residual_psi"] == 0.


def test_s_finds_interior_root_after_remote_selected_pump_failure(monkeypatch):
    _responses(monkeypatch, {"S": lambda p: (400., 40000., 400.) if p <= 3600 else None})
    rows, _, meta = _run(_wells("S"), SPadPlant(), required_wells={"S"})
    assert len(rows) == 1
    assert meta["header_psi"] == pytest.approx(SPadPlant().header_at_flow(40000., 3), abs=10.)
    assert meta["converged"] and meta["feasible"]


def test_s_below_minimum_is_conditional_and_never_invents_recycle(monkeypatch):
    _responses(monkeypatch, {"S": (400., 20000., 400.)})
    rows, _, meta = _run(_wells("S"), SPadPlant())
    assert len(rows) == 1 and meta["hydraulically_feasible"]
    assert meta["feasible"] is False and meta["in_range"] is False
    assert meta["total_machine_water_bpd"] == 20000.
    assert "recycle" in meta["operating_assumptions"][0]


def test_auto_maximizes_oil_while_manual_water_price_remains_intentional(monkeypatch):
    plant = FreePlant()
    plant.budget_at_pressure = lambda *a: 1000.
    _responses(monkeypatch, {"A": (100., 1000., 0.), "B": (61., 600., 0.)})
    rows, _, meta = _run(_wells("A", "B"), plant)
    assert [r.well_name for r in rows] == ["A"]
    assert meta["lambda_used"] == 0.
    assert meta["diagnostic_lambda"] == pytest.approx(.1)
    rows, _, meta = _run(_wells("A", "B"), plant, water_price=.1)
    assert [r.well_name for r in rows] == ["B"]
    assert meta["lambda_used"] == .1


def test_required_pad_wells_cannot_be_economically_shut_in(monkeypatch):
    plant = FreePlant()
    plant.budget_at_pressure = lambda *a: 1000.
    _responses(monkeypatch, {"A": (100., 1000., 0.), "B": (61., 600., 0.)})
    rows, _, _ = _run(_wells("A", "B"), plant, required_wells={"B"})
    assert [r.well_name for r in rows] == ["B"]
    with pytest.raises(RuntimeError):
        _run(_wells("A", "B"), plant, required_wells={"A", "B"})


def test_all_failed_pad_candidates_raise_instead_of_an_empty_plan(monkeypatch):
    _responses(monkeypatch, {"A": None})
    with pytest.raises(RuntimeError, match="candidate"):
        _run(_wells("A"), FreePlant())


def test_e_pressure_limits_and_amp_report_describe_one_operating_point():
    with pytest.raises(ValueError, match="cap must exceed suction"):
        EPadPlant(suction_psi=3600., max_header_psi=3500.)
    low_cap = EPadPlant(suction_psi=2800., max_header_psi=2900.)
    assert low_cap.pressure_window() == (2900., 2900.)
    plant = EPadPlant(amp_limit=50.)
    row = plant.envelope([18000.], at_pressure=3400.)[0]
    pump = row["pumps"][0]
    assert row["feasible"]
    assert pump["amps"] == pytest.approx(50.)
    assert pump["hz"] < 60.
    assert pump["dP"] == pytest.approx(plant.build.dp_psi(18000., pump["hz"], plant.specific_gravity()))
    assert row["max_discharge_psi"] == pytest.approx(plant.suction_psi()+pump["dP"])


def test_e_envelope_rejects_requested_header_above_available_pressure():
    assert EPadPlant().envelope([4000.], at_pressure=3500.)[0]["feasible"] is False
