"""Regression guards for the reproduced September 7 correctness defects."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pandas as pd
import pytest

from tools import review_errors_2026_09_07 as probe


@pytest.mark.parametrize("wc", [0.0, 0.5, 0.8, 0.986])
@pytest.mark.parametrize("gor", [0., 50., 250., 1000.])
@pytest.mark.parametrize("temp", [60., 100., 180.])
def test_component_mass_conservation(wc, gor, temp):
    from woffl.flow.singlephase import bpd_to_ft3s
    from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

    mix = ResMix(wc, gor, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    oil_std = bpd_to_ft3s(100.)
    water_std = oil_std * wc / (1-wc)
    expected_mass = oil_std * mix.rho_oil_std + water_std * mix.rho_wat_std + 100 * gor / 86400 * mix.rho_gas_std
    for pressure in [0., 500., 1000., 1749., 1750., 1751., 2000., 3000.]:
        mix.condition(pressure, temp)
        oil, water, gas = mix.insitu_volm_flow(100.)
        assert oil == pytest.approx(oil_std * mix.oil.oil_fvf(), rel=1e-10)
        assert water * mix.wat.density == pytest.approx(water_std * mix.rho_wat_std, abs=1e-10)
        assert sum(q * rho for q, rho in zip((oil, water, gas), mix.rho_comp())) == pytest.approx(expected_mass, rel=1e-10)
        assert sum((oil, water, gas)) * mix.rho_mix() == pytest.approx(expected_mass, rel=1e-10)
        assert mix.insitu_mass_flow(100.) == pytest.approx(
            tuple(q * rho for q, rho in zip((oil, water, gas), mix.rho_comp())), rel=1e-10
        )
        assert min(oil, water, gas) >= -1e-12


def test_saved_watercut_preserves_oil_and_optimizer_acceptance():
    for row in probe.saved_watercut():
        assert row["restored_wc"] == row["saved_wc"]
        assert row["restored_oil"] == pytest.approx(row["saved_oil"])
        assert row["optimizer_accepts"]


def test_batch_wear_only_applies_to_installed_size():
    row = probe.batch_wear()
    for batch in row.values():
        assert batch == pytest.approx({"12B:installed": 1.2, "12B:replacement": 1.0, "13B:replacement": 1.0})


def test_wear_resets_after_pump_replacement():
    assert probe.wear_after_replacement()["restored_area_factor"] == 1.0


def test_event_fallback_uses_saved_physics():
    assert probe.calibration_fallback() == dict(seed_kth=.6, seed_kdi=.7, nozzle_area_factor=1.2, mach_crit=1.5)


def test_event_fallback_does_not_use_prior_pump_test():
    from server.services import event_calibration as ec
    with patch.object(ec.tests_svc, "tests_json", return_value=[
        dict(date="2026-08-01", bhp=500., form_wc=.6, fgor=300., pf_press=3168.),
        dict(date="2026-07-01", bhp=450., form_wc=.6, fgor=300., pf_press=3168.),
    ]):
        assert ec._latest_test_target("W", "2026-09-01") is None
        assert ec._latest_test_target("W", "2026-07-15")["bhp"] == 500.


def test_cfp_steep_surface_requires_pressure_balance():
    row = probe.cfp_settling()
    assert row["feasible"]
    assert abs(row["residual_psi"]) < .5


def test_zero_annulus_rejected_at_schema_and_library():
    from server.schemas import SimParams
    from woffl.geometry.pipe import Pipe, PipeInPipe
    with pytest.raises(ValueError):
        SimParams(tubing_od=4.5, casing_od=5.5, casing_thickness=.5)
    with pytest.raises(ValueError):
        PipeInPipe(Pipe(4.5, .5), Pipe(5.5, .5))
    assert PipeInPipe(Pipe(4.5, .5), Pipe(5.6, .5)).ann_area > 0


def test_write_cleanup_does_not_replay_success():
    assert probe.retry_after_commit()["successful_executions"] == 1


def test_write_execution_failure_is_not_replayed():
    from woffl.assembly import databricks_client as dc
    calls = []
    class Cursor:
        rowcount = 1
        def execute(self, *args):
            calls.append("possibly committed")
            raise ConnectionError("lost execution response")
        def close(self):
            pass
    conn = NS(cursor=lambda: Cursor(), close=lambda: None)
    with patch.object(dc, "_CONN_LOCAL", NS(conn=conn)), patch.object(dc, "_new_connection", return_value=conn):
        with pytest.raises(ConnectionError):
            dc._write_via_connector("INSERT INTO fake VALUES (:x)", {"x": 1})
    assert len(calls) == 1


def test_choke_plan_obeys_total_machine_capacity():
    row = probe.choke_machine_budget()
    assert row["selected_total_water_bpd"] <= row["capacity_bpd"]


def test_milp_priced_tie_prefers_oil():
    row = probe.auto_price()
    assert row["derived_lambda"] == 1.0
    assert row["auto_oil"] == row["unpriced_oil"] == 80.


def test_mckp_priced_tie_prefers_oil():
    from woffl.assembly.network import optimize_jet_pumps
    df = pd.DataFrame(dict(nozzle=["12", "13"], throat=["B", "B"],
        qoil_std=[80., 100.], lift_wat=[80., 100.], form_wat=[0., 0.], totl_wat=[80., 100.]))
    out = optimize_jet_pumps([NS(wellname="W", df=df)], 90., allow_shutin=True, all_configs=True, water_price=1.)
    assert out.qoil_std.sum() == 80.


def test_evidence_prediction_independent_of_search_ceiling():
    from woffl.assembly.network_optimizer import WellConfig
    from woffl.gui.pad_optimize import _apply_suction_evidence
    cfg = WellConfig(well_name="W", res_pres=1700., form_temp=70., jpump_tvd=4065.)
    ev = {"W": dict(floor=450., floor_source="era", psu_ref=500., ppf_ref=3200., beta=.1, beta_source="well")}
    values = []
    for top in (3200., 3500.):
        grid = [{"W": (100., 1000., 600., True)}, {"W": (100., 1200., 600., True)}]
        assert _apply_suction_evidence(grid, [3000., top], ["W"], ev, {"W": cfg})
        values.append(grid[0]["W"])
    assert values[0] == values[1]
    assert values[0][2] == 520.
