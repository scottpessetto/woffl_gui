"""Optimization review 2026-09-24: choke-plan and scenario-settle fixes.

* A well that solves only above a candidate header is shut in there, never
  credited with its measured test rates.
* "Today" no longer collapses to plant suction when a reduced bank cannot
  carry today's measured draw.
* The evidence correction uses the ladder's verdict for the today point.
* The forced-header pricing runs each well on its own held pump only.
* A fixed-plan settle is converged only when its residual closed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import woffl.gui.pad_optimize as po
from tests.test_pad_optimize import CurvePlant, _perf, _wells, fake_core  # noqa: F401 - fixture

LEVELS = [2000.0, 2500.0, 3000.0]


class StubPlant:
    coupling = "free_pressure"
    max_header_psi = 3500.0
    infeasible_sweep_msg = "no feasible header"
    water_key = "lift_wat"

    def __init__(self, budget=None, header=3000.0, over=False):
        self._budget = budget or {2000.0: 9000.0, 2500.0: 5000.0, 3000.0: 4000.0}
        self._header = header
        self._over = over

    def pressure_window(self, n_pumps=None):
        return (LEVELS[0], LEVELS[-1])

    def budget_at_pressure(self, pressure, n_pumps=None):
        return self._budget[pressure]

    def warm_start_psi(self, n_pumps=None):
        return 3000.0

    def delivered_header(self, q_total, setpoint=None, n_pumps=None):
        # A reduced bank (n_pumps=1) cannot carry today's draw.
        if self._over and n_pumps == 1:
            return 217.0, True
        return self._header, False

    def header_at_flow(self, q_total, n_pumps=None):
        return self._header

    def suction_psi(self):
        return 217.0

    def flow_window(self, n_pumps=None):
        return (0.0, 60000.0)

    def flags(self, q_total, n_pumps=None):
        return {"in_range": True, "recirc": False, "over_capacity": False}


def test_a_well_that_only_solves_higher_is_shut_in_not_held_at_its_test(monkeypatch):
    curves = {
        # A cannot lift below 2500 psi delivered.
        "A": {2500.0: (250.0, 2500.0), 3000.0: (300.0, 3000.0)},
        "B": {2000.0: (150.0, 1500.0), 2500.0: (180.0, 1800.0), 3000.0: (200.0, 2000.0)},
    }
    monkeypatch.setattr(po, "_model_at_forced_header", lambda cfgs, header, cc: {
        c.well_name: curves[c.well_name].get(header) for c in cfgs})
    configs = [SimpleNamespace(well_name=w) for w in ("A", "B")]
    rows, meta = po.run_choke_optimization(
        configs, StubPlant(), None, {"A": ("12", "B"), "B": ("11", "C")},
        {"A": (300.0, 3000.0), "B": (200.0, 2000.0)}, n_levels=3)
    sweep = {s["header_psi"]: s["total_oil_bopd"] for s in meta["sweep"]}
    # At 2,000 psi A makes nothing (it used to be credited its 300 BOPD test).
    assert sweep[2000.0] == pytest.approx(150.0)
    assert meta["header_psi"] == 2500.0 and meta["total_oil_bopd"] == pytest.approx(430.0)
    assert {r["well"]: r["basis"] for r in rows} == {"A": "model", "B": "model"}


def test_today_uses_measured_well_pressures_when_the_reduced_bank_cannot_carry_it(monkeypatch):
    seen = []

    def forced(cfgs, header, cc):
        seen.append(header)
        at = (lambda w: header[w]) if isinstance(header, dict) else (lambda w: header)
        return {c.well_name: (0.1 * (at(c.well_name) - 1000.0), 2.0 * at(c.well_name), 900.0, False)
                for c in cfgs}

    monkeypatch.setattr(po, "_model_at_forced_header", forced)
    configs = [SimpleNamespace(well_name="A", ppf_surf_well=3100.0),
               SimpleNamespace(well_name="B", ppf_surf_well=2900.0)]
    rows, meta = po.run_choke_optimization(
        configs, StubPlant(over=True), 1, {"A": ("12", "B"), "B": ("12", "B")},
        {"A": (210.0, 6200.0), "B": (190.0, 5800.0)}, n_levels=3)
    assert meta["today_basis"] == "measured_well_pf"
    assert {"A": 3100.0, "B": 2900.0} in seen
    assert 2900.0 <= meta["header_today_psi"] <= 3100.0
    # Measured and modeled today agree, so a plan that holds today's
    # pressures projects no phantom gain.
    for r in rows:
        if r["projected_oil"] is not None and r["action"] == "full":
            assert r["projected_oil"] <= r["test_oil"] * 1.01


def test_today_without_measurements_never_prices_at_the_suction_collapse(monkeypatch):
    seen = []

    def forced(cfgs, header, cc):
        seen.append(header)
        return {c.well_name: (100.0, 1000.0, 900.0, False) for c in cfgs}

    monkeypatch.setattr(po, "_model_at_forced_header", forced)
    configs = [SimpleNamespace(well_name="A")]
    _rows, meta = po.run_choke_optimization(
        configs, StubPlant(over=True), 1, {"A": ("12", "B")}, {"A": (100.0, 70000.0)}, n_levels=3)
    assert meta["header_today_psi"] >= LEVELS[0]
    assert 217.0 not in seen


def test_the_today_point_takes_the_ladders_evidence_verdict(monkeypatch):
    curve = {2000.0: (80.0, 900.0, 900.0, False), 2500.0: (100.0, 1000.0, 800.0, False),
             3000.0: (105.0, 1100.0, 780.0, True)}
    monkeypatch.setattr(po, "_model_at_forced_header", lambda cfgs, header, cc: {
        c.well_name: curve.get(header) for c in cfgs})
    plant = StubPlant(budget={2000.0: 5000.0, 2500.0: 5000.0, 3000.0: 1050.0}, header=2500.0)
    cfg = SimpleNamespace(well_name="A", qwf=1000.0, pwf=800.0, res_pres=1500.0, form_wc=0.0)
    evidence = {"A": {"floor": 700.0, "floor_source": "era", "beta": 0.10, "beta_source": "well",
                      "psu_ref": 700.0, "ppf_ref": 2900.0}}
    rows, meta = po.run_choke_optimization([cfg], plant, None, {"A": ("12", "B")}, {"A": (100.0, 1000.0)},
                                           n_levels=3, evidence=evidence)
    r = rows[0]
    assert meta["header_today_psi"] == meta["header_psi"] == 2500.0
    assert r["suction_basis"] == "evidence" and r["action"] == "full"
    # Plan and today are the same corrected point: no +6 BOPD artifact.
    assert r["projected_oil"] == pytest.approx(r["test_oil"])
    assert meta["projected_d_oil_bopd"] == pytest.approx(0.0, abs=1e-9)


def test_forced_header_prices_only_each_wells_own_pump(fake_core):
    fake_core.Optimizer.perf_table = {("A", "12", "B"): _perf(1000.0, 100.0), ("B", "9", "C"): _perf(800.0, 50.0)}
    out = po._model_at_forced_header(_wells("A", "B", "C"), 3000.0, {"A": ("12", "B"), "B": ("9", "C")})
    opt = fake_core.Optimizer.instances[-1]
    assert opt.well_grids == {"A": (["12"], ["B"]), "B": (["9"], ["C"])}
    assert [w.well_name for w in opt.wells] == ["A", "B"]  # C has no held pump: not simulated
    assert out["A"][0] == 100.0 and out["B"][0] == 50.0 and out["C"] is None


def test_forced_header_accepts_per_well_pressures_and_clamps_the_constraint(fake_core):
    fake_core.Optimizer.perf_table = {("A", "12", "B"): _perf(1000.0, 100.0)}
    wells = _wells("A", "B")
    po._model_at_forced_header(wells, {"A": 2950.0, "B": None}, {"A": ("12", "B"), "B": ("12", "B")})
    opt = fake_core.Optimizer.instances[-1]
    assert [w.well_name for w in opt.wells] == ["A"] and wells[0].ppf_surf_well == 2950.0
    po._model_at_forced_header(_wells("A"), 217.0, {"A": ("12", "B")})
    assert fake_core.Optimizer.instances[-1].power_fluid.pressure == 1000.0


def test_a_fixed_plan_across_a_demand_jump_is_not_certified(fake_core):
    # PF demand jumps 10k -> 40k at 2,700 psi on the 3000 - 0.01q curve: no
    # pressure balances the plan. Bisection shrinks onto the jump.
    fake_core.Optimizer.perf_table = {("W1", "12", "B"): lambda p: _perf(40000 if p >= 2700 else 10000, 500)}
    _rows, meta = po.evaluate_fixed_scenario(_wells("W1"), CurvePlant(), 3, {"W1": ("12", "B")})
    assert meta["converged"] is False


def test_e_pad_sweep_ceiling_finds_the_amp_limited_peak():
    from woffl.gui.e_pad_plant import EPadPlant

    # Catalog motor scale (0.1435 A/BHP) so a 40 A cap moves the peak.
    plant = EPadPlant(amp_limit=40.0, max_header_psi=5000.0, amps_per_bhp=0.1435,
                      suction_psi=2800.0, field_calibrated=False)
    # The nominal knee reads about 3,870 psi; the true peak is about 3,928.
    assert plant.pressure_window()[1] > plant.max_discharge_pressure(plant.knee_flow()) + 40.0


def test_booster_screen_labels_a_low_flow_block_as_the_low_range():
    from woffl.gui import e_pad_booster as eb

    d = eb.defaults()
    for build in eb.candidates():
        r = eb.solve_candidate(build, dp_target=1775.0, suction_psi=d["suction_psi"], sg=d["sg"],
                               condition=1.0, hz_max=60.0, amps_per_bhp=d["amps_per_bhp"], amp_limit=None)
        assert r["duty"] is None and r["limited_by"] == eb.LIMIT_ROR_LOW


def test_stress_cases_reject_the_low_flow_point_the_run_rejects():
    from server.services.plan_robustness import score_plan
    from woffl.gui.e_pad_plant import EPadPlant

    plant = EPadPlant()
    # 2,000 BPD sits below where the booster frontier exists: no 3,400 psi
    # header without an unmodeled recycle. The run rejects it...
    assert po._plant_operating_check(plant, 2000.0, 3400.0, None)["hydraulically_feasible"] is False
    # ...and so must the stress case (delivered_header "held" the setpoint).
    score = score_plan("Current", {"W": ("12", "B", "installed")}, [SimpleNamespace(well_name="W")],
                       {("W", "12", "B", "installed"): {"oil_rate": 100.0, "lift_water": 2000.0, "formation_water": 0.0}},
                       plant, None, 3400.0, 0.0)
    assert score["feasible"] is False


# -- free-pressure sweep ----------------------------------------------------------


def test_every_failed_allocation_is_named_not_blamed_on_the_plant(fake_core):
    from tests.test_pad_optimize import FreePlant
    from woffl.assembly.network import AllocationError

    def infeasible(opt):
        raise AllocationError("Required wells were not allocated", status="infeasible")

    fake_core.optimize_fn = infeasible
    with pytest.raises(RuntimeError) as err:
        po.run_optimization(_wells("W1"), FreePlant(), None, ["12"], ["B"], "milp", None, n_steps=3)
    assert "No header in the sweep has a valid allocation (infeasible)" in str(err.value)
    assert "amp" not in str(err.value).lower()


def test_a_solver_error_marks_the_sweep_incomplete(fake_core):
    from tests.test_pad_optimize import FreePlant, _result
    from woffl.assembly.network import AllocationError

    def flaky(opt):
        if abs(opt.power_fluid.pressure - 2600.0) < 1e-6:
            raise AllocationError("MILP time limit reached", status="unknown")
        return [_result("W1", 1000.0, 5000.0 - abs(opt.power_fluid.pressure - 2600.0))]

    fake_core.optimize_fn = flaky
    _, _, meta = po.run_optimization(_wells("W1"), FreePlant(), None, ["12"], ["B"], "milp", None,
                                     n_steps=11, refine_rounds=0)
    assert meta["sweep_complete"] is False
    assert [f["status"] for f in meta["failed_trials"]] == ["unknown"]


def test_a_degenerate_window_solves_its_one_header_once(fake_core):
    from tests.test_pad_optimize import FreePlant, _result

    class Pinned(FreePlant):
        def pressure_window(self, n_pumps=None):
            return (3000.0, 3000.0)

    fake_core.optimize_fn = lambda opt: [_result("W1", 1000.0, 100.0)]
    _, _, meta = po.run_optimization(_wells("W1"), Pinned(), None, ["12"], ["B"], "milp", None, n_steps=11)
    assert len(meta["sweep"]) == 1


def test_refinement_bisects_toward_a_failed_neighbour(fake_core):
    from tests.test_pad_optimize import FreePlant, _result
    from woffl.assembly.network import AllocationError

    # Oil rises with pressure up to a feasibility edge between 2,600 and
    # 2,700 psi: the optimum sits against the failed neighbour.
    def edge(opt):
        p = opt.power_fluid.pressure
        if p > 2650.0:
            raise AllocationError("infeasible above the edge", status="infeasible")
        return [_result("W1", 1000.0, p)]

    fake_core.optimize_fn = edge
    _, _, meta = po.run_optimization(_wells("W1"), FreePlant(), None, ["12"], ["B"], "milp", None,
                                     n_steps=11, refine_rounds=2)
    assert meta["header_psi"] > 2600.0


def test_mckp_refined_by_milp_skips_the_redundant_cross_check(fake_core):
    from tests.test_pad_optimize import FreePlant, _result

    calls = []

    def solve(opt):
        calls.append(fake_core.method)
        opt.allocation_status = {"status": "optimal", "requested_solver": "cp-sat",
                                 "refinement_reason": "fractional resource coefficients"}
        return [_result("W1", 1000.0, 100.0)]

    fake_core.optimize_fn = solve
    _, _, meta = po.run_optimization(_wells("W1"), FreePlant(), None, ["12"], ["B"], "mckp", None,
                                     setpoint_psi=2500.0)
    assert "skipped" in meta["solver_agreement"]
    assert calls == ["mckp"]  # no second MILP solve


def test_ipr_curve_never_steps_above_a_non_integer_reservoir_pressure():
    # pres * 24 / 24 rounds above pres for some values; InFlow then raised
    # and the whole choke run crashed at its landing-table curve.
    bad = []
    for k in range(2000):
        pres = 1000.0 + k * 0.37
        cfg = SimpleNamespace(qwf=900.0, pwf=600.0, res_pres=pres, form_wc=0.5)
        try:
            curve = po._vogel_ipr_curve(cfg)
        except ValueError:
            bad.append(pres)
            continue
        assert curve[0][1] <= round(pres, 1) and curve[-1] == [curve[-1][0], 0.0]
    assert bad == []
