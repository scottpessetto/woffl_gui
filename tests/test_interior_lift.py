"""Regression: negative endpoint residuals can hide a real interior lift root."""
import math

import pytest

from woffl.assembly import solopump as so
from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig
from woffl.flow import jetflow
from woffl.flow.errors import JetPumpError
from woffl.geometry import JetPump


def test_interior_lift_selects_first_forward_crossing_and_consistent_rates():
    def evaluate(p):
        return -100 * (p - 2.1) * (p - 7.7), 10 * p, 20 * p, 1000 + p, .2

    assert evaluate(0)[0] < 0 and evaluate(10)[0] < 0
    result = so._interior_lift_solution(0., 10., evaluate)
    assert result[0] == pytest.approx(2.1, abs=1e-7)
    assert result[1] is False
    assert result[2:] == evaluate(result[0])[1:]
    assert abs(evaluate(result[0])[0]) < 1e-4


@pytest.mark.parametrize("fault", ["hole", "sign_jump", "nonfinite", "negative_rate", "no_lift"])
def test_interior_lift_rejects_unclosed_or_infeasible_roots_with_bounded_work(fault):
    calls = []

    def evaluate(p):
        calls.append(p)
        residual = -1000 * (p - .333) * (p - .75)
        if fault == "hole" and .32 < p < .35:
            raise JetPumpError("no throat solution")
        if fault == "sign_jump":
            residual = 100. if .333 <= p <= .75 else -100.
        if fault == "no_lift":
            residual = -100.
        oil = math.nan if fault == "nonfinite" else -1. if fault == "negative_rate" else 100.
        return residual, oil, 100., 1000., .2

    assert so._interior_lift_solution(0., 1., evaluate) is None
    assert 1 <= len(calls) <= 192


def b30_case():
    # Frozen September 2, 2026 conditions from the September 8 local snapshot.
    # No live data, per-test anchor, or calibrated loss coefficient.
    cfg = WellConfig(
        well_name="MPB-30", res_pres=1519., form_temp=170., jpump_tvd=6570., jpump_md=6926.,
        tubing_od=4.5, tubing_thickness=.29000000000000004, casing_od=7., casing_thickness=.362,
        form_wc=.7699999809265137, form_gor=3138., field_model="Kuparuk", surf_pres=204.,
        qwf=1378., pwf=1176., oil_api=24., gas_sg=.65, wat_sg=1., bubble_point=2250.,
        installed_nozzle="12", installed_throat="D", rho_pf=63.648,
    )
    bore, profile, ipr, mix, pf = NetworkOptimizer._create_well_objects(cfg)
    args = (221.094, cfg.form_temp, 2565.553, JetPump("12", "D"), bore, profile, ipr, mix, pf, "reverse")
    return cfg, args


def test_b30_interior_lift_recovers_a_closed_balance_with_original_ipr():
    cfg, args = b30_case()
    pwh, temp, ppf, pump, bore, profile, ipr, mix, pf, direction = args
    lower, *_ = jetflow.psu_minimize(temp, pump.ken, pump.ate, ipr, mix)
    upper = cfg.res_pres - 10
    assert so.discharge_residual(lower, *args)[0] < 0
    assert so.discharge_residual(upper, *args)[0] < 0

    result = so.jetpump_solver(*args)
    suction, sonic, oil, water, qpf, mach = result
    assert lower < suction < upper
    assert not sonic and 0 < mach < 1
    assert oil > 0 and water > 0 and qpf > 0
    # Independently rebuild mutable PVT objects for the closure check.
    _cfg, fresh = b30_case()
    residual, *rates = so.discharge_residual(suction, *fresh)
    assert abs(residual) < 1e-3
    assert rates == pytest.approx(result[2:], rel=1e-12)
    assert oil == pytest.approx(ipr.oil_flow(suction, method="vogel"), rel=1e-12)
    assert water / (oil + water) == pytest.approx(cfg.form_wc, abs=1e-12)
    # The selected branch has the same orientation as the ordinary bracket.
    assert so.discharge_residual(suction - 1., *fresh)[0] < 0
    assert so.discharge_residual(suction + 1., *fresh)[0] > 0


def test_existing_successful_solver_path_never_runs_interior_search(monkeypatch):
    # The public solve wrapper is exercised by the existing assembly suite;
    # fail loudly if a normal reference solve starts paying for the new scan.
    def unexpected(*args, **kwargs):
        raise AssertionError("interior scan on an already bracketed solve")

    monkeypatch.setattr(so, "_interior_lift_solution", unexpected)
    from tests.test_asm_solopump import _solve
    oil, water = _solve(JetPump("13", "C"))
    assert oil > 0 and water > 0
