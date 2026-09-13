"""Patch 46: allocation feasibility, status and decision-option regressions."""

from decimal import Decimal
from itertools import product
import random
from types import SimpleNamespace

import pandas as pd
import pytest
from ortools.sat.python import cp_model

from woffl.assembly.network import AllocationError, optimize_jet_pumps
from woffl.assembly.network_optimizer import NetworkOptimizer, reconcile_wells
from woffl.assembly.optimization_algorithms import (
    derive_lambda, derive_pad_marginal_wc, mckp_optimization, milp_optimization,
)
from woffl.assembly.pump_candidates import CLEAN_PUMP, scoped_pumps
from woffl.geometry.jetpump import JetPump


ENGINES = [milp_optimization, mckp_optimization]


def make_optimizer(groups, capacity=1000.0, price=0.0, required=()):
    """groups maps well name to (oil, lift water) options; no physics or I/O."""
    opt = SimpleNamespace(wells=[SimpleNamespace(well_name=name) for name in groups],
        batch_results={}, power_fluid=SimpleNamespace(total_rate=capacity),
        water_price=price, required_wells=set(required), optimization_results=None,
        allocation_status=None)
    for name, options in groups.items():
        rows = [dict(nozzle="12", throat=chr(65+j), pump_state="installed" if j == 0 else "replacement",
            qoil_std=oil, lift_wat=water, form_wat=0.0, totl_wat=water,
            psu_solv=500.0, sonic_status=False, mach_te=.5, molwr=0.0, motwr=0.0,
            error="na", semi=True) for j, (oil, water) in enumerate(options)]
        opt.batch_results[name] = SimpleNamespace(wellname=name, df=pd.DataFrame(rows))
    opt.get_pump_performance = lambda *a, **kw: NetworkOptimizer.get_pump_performance(opt, *a, **kw)
    return opt


@pytest.mark.parametrize("engine", ENGINES)
def test_exact_fit_fractional_resources_keep_both_wells(engine):
    opt = make_optimizer({"A": [(500., 1000.001)], "B": [(100., 500.001)]}, 1500.002)
    results = engine(opt)
    assert sum(r.predicted_oil_rate for r in results) == 600.0
    assert sum(r.predicted_lift_water for r in results) <= 1500.002
    assert opt.allocation_status["status"] == "optimal"
    assert opt.allocation_status["objective_bound"] == pytest.approx(600.0)
    if engine == mckp_optimization:
        assert opt.allocation_status["requested_solver"] == "cp-sat"
        assert opt.allocation_status["refinement_reason"] == "fractional resource coefficients"


@pytest.mark.parametrize("engine", ENGINES)
def test_original_capacity_not_exceeded_near_boundary(engine):
    opt = make_optimizer({"A": [(500., 1000.001)]}, 1000.)
    assert engine(opt) == []
    assert opt.allocation_status["status"] == "optimal"
    assert opt.allocation_status["shut_in_wells"] == ["A"]


@pytest.mark.parametrize("engine", ENGINES)
def test_capacity_float_dust_uses_the_declared_original_unit_tolerance(engine):
    opt = make_optimizer({"A": [(500., 1500.)]}, 1500. - 1e-10)
    assert sum(r.predicted_oil_rate for r in engine(opt)) == 500.
    assert opt.allocation_status["selected_water_bpd"] <= (
        opt.allocation_status["capacity_bpd"] + opt.allocation_status["resource_tolerance_bpd"])
    beyond = make_optimizer({"A": [(500., 1500.)]}, 1500. - 1e-5)
    assert engine(beyond) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_all_shutin_is_an_optimal_completed_allocation(engine):
    opt = make_optimizer({"A": [(100., 1000.)]}, 2000., price=.2)
    assert engine(opt) == []
    assert opt.allocation_status["status"] == "optimal"
    accounting = reconcile_wells(opt, []).iloc[0]
    assert accounting["Status"] == "not allocated"
    assert "shut in" in accounting["Detail"]


@pytest.mark.parametrize("engine", ENGINES)
def test_required_subset_is_served_even_when_optional_well_is_more_profitable(engine):
    opt = make_optimizer({"Existing": [(100., 1000.)], "Future": [(10., 500.)]}, required=["Future"])
    result = engine(opt)
    assert [r.well_name for r in result] == ["Future"]
    assert opt.allocation_status["required_wells"] == ["Future"]


@pytest.mark.parametrize("engine", ENGINES)
def test_infeasible_required_online_set_raises_typed_failure(engine):
    opt = make_optimizer({"A": [(100., 1000.)], "B": [(10., 500.)]}, required=["A", "B"])
    with pytest.raises(AllocationError) as caught:
        engine(opt)
    assert caught.value.status == "infeasible"
    assert opt.allocation_status["status"] == "infeasible"
    assert opt.optimization_results is None


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("required", ["Missing", "Invalid"])
def test_required_unknown_or_invalid_well_is_not_silently_skipped(engine, required):
    opt = make_optimizer({"A": [(100., 100.)], "Invalid": [(100., float("nan"))]}, required=[required])
    with pytest.raises(AllocationError) as caught:
        engine(opt)
    assert caught.value.status == "unsupported"
    assert caught.value.details["unsupported_wells"] == [required]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("bad", [(100., float("nan")), (float("inf"), 100.), (100., -100.), (-100., 100.)])
def test_invalid_rate_rows_cannot_create_oil_or_spare_capacity(engine, bad):
    opt = make_optimizer({"A": [bad, (50., 100.)]})
    result = engine(opt)
    assert len(result) == 1 and result[0].predicted_oil_rate == 50.
    assert result[0].recommended_throat == "B"
    assert opt.allocation_status["excluded_candidates"] == {"A": 1}


def test_milp_solver_error_is_not_a_shutdown_result(monkeypatch):
    import scipy.optimize
    opt = make_optimizer({"A": [(100., 100.)]})
    monkeypatch.setattr(scipy.optimize, "milp", lambda **kw: SimpleNamespace(
        status=4, success=False, message="injected failure", x=None))
    with pytest.raises(AllocationError, match="injected failure"):
        milp_optimization(opt)
    assert opt.allocation_status["status"] == "error"
    assert opt.optimization_results is None
    assert reconcile_wells(opt, []).iloc[0]["Status"] == "allocation failed"


@pytest.mark.parametrize("engine", ENGINES)
def test_candidate_lookup_failure_clears_previous_qualified_plan(engine):
    opt = make_optimizer({"A": [(100., 100.)]})
    assert engine(opt)
    def broken(*a, **kw):
        raise RuntimeError("candidate lookup failed")
    opt.get_pump_performance = broken
    with pytest.raises(AllocationError, match="candidate lookup failed"):
        engine(opt)
    assert opt.optimization_results is None
    assert opt.allocation_status["status"] == "error"


def test_cp_unknown_is_not_misreported_as_capacity_or_shutdown(monkeypatch):
    opt = make_optimizer({"A": [(100., 100.)]})
    monkeypatch.setattr(cp_model.CpSolver, "solve", lambda *a, **kw: cp_model.UNKNOWN)
    with pytest.raises(AllocationError) as caught:
        mckp_optimization(opt)
    assert caught.value.status == "unknown"
    assert opt.allocation_status["status"] == "unknown"
    assert opt.optimization_results is None


def test_milp_limit_preserves_qualified_incumbent_and_gap(monkeypatch):
    import numpy as np
    import scipy.optimize
    opt = make_optimizer({"A": [(100., 100.)]})
    monkeypatch.setattr(scipy.optimize, "milp", lambda **kw: SimpleNamespace(
        status=1, success=False, message="limit with incumbent", x=np.array([1.]),
        mip_dual_bound=-110., mip_gap=.1))
    result = milp_optimization(opt)
    assert result[0].predicted_oil_rate == 100.
    assert opt.allocation_status["status"] == "feasible"
    assert opt.allocation_status["objective_bound"] == 110.
    assert opt.allocation_status["gap"] == .1


def test_failed_oil_tie_break_retains_qualified_primary_plan(monkeypatch):
    import scipy.optimize
    original = scipy.optimize.milp
    calls = 0
    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected secondary failure")
        return original(**kwargs)
    monkeypatch.setattr(scipy.optimize, "milp", fail_second)
    opt = make_optimizer({"A": [(100., 100.)]}, price=.1)
    assert milp_optimization(opt)[0].predicted_oil_rate == 100.
    assert opt.allocation_status["status"] == "optimal"
    assert "injected secondary failure" in opt.allocation_status["oil_tie_break"]


def test_direct_api_retains_exactly_one_default_and_dataframe_status():
    opt = make_optimizer({"A": [(100., 1000.001)], "B": [(10., 500.001)]})
    wells = list(opt.batch_results.values())
    with pytest.raises(AllocationError, match="infeasible"):
        optimize_jet_pumps(wells, 1000.)
    table = optimize_jet_pumps(wells, 1500.002)
    assert len(table) == 2
    assert table.qoil_std.sum() == 110.
    assert table.attrs["allocation_status"]["status"] == "optimal"


def test_concave_hull_shadow_price_does_not_buy_later_segment_first():
    opt = make_optimizer({"A": [(1., 1000.), (101., 1100.)]})
    lam, slack = derive_lambda(opt.batch_results, 500.)
    assert not slack and lam == pytest.approx(101 / 1100)
    wc, slack = derive_pad_marginal_wc(opt.batch_results, 500.)
    assert not slack and wc == pytest.approx(1 / (1 + lam))


def test_installed_pump_outside_replacement_grid_is_always_an_option():
    pumps = scoped_pumps(["12"], ["A", "B"], ("16", "X"),
                         {"ken": .1, "kth": .6, "kdi": .8, "nozzle_area_factor": 1.2})
    assert [(p.noz_no, p.rat_ar, p.pump_state) for p in pumps] == [
        ("16", "X", "installed"), ("12", "A", "replacement"), ("12", "B", "replacement")]
    assert pumps[0].ken == .1
    assert pumps[0].anz == pytest.approx(JetPump("16", "X").anz * 1.2)
    assert all(p.ken == CLEAN_PUMP["ken"] for p in pumps[1:])


@pytest.mark.parametrize("engine", ENGINES)
def test_small_fractional_allocations_match_exhaustive_oracle_with_required_well(engine):
    rng = random.Random(46)
    for _ in range(12):
        groups = {f"W{i}": [(rng.randint(10, 300), rng.randint(0, 800) + .001) for _ in range(3)] for i in range(3)}
        capacity = rng.randint(300, 2000) + .003
        price = .125
        feasible = []
        for picked in product(groups["W0"], [None, *groups["W1"]], [None, *groups["W2"]]):
            oil = sum((Decimal(str(p[0])) for p in picked if p), Decimal(0))
            water = sum((Decimal(str(p[1])) for p in picked if p), Decimal(0))
            if water <= Decimal(str(capacity)):
                feasible.append(float(oil - Decimal(str(price)) * water))
        opt = make_optimizer(groups, capacity, price, ["W0"])
        if not feasible:
            with pytest.raises(AllocationError):
                engine(opt)
        else:
            result = engine(opt)
            assert any(r.well_name == "W0" for r in result)
            assert sum(r.predicted_oil_rate - price*r.predicted_lift_water for r in result) == pytest.approx(max(feasible))
