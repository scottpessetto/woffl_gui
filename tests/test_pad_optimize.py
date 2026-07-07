"""Tests for the unified pad optimization core (R-1 Phase B).

``woffl.gui.pad_optimize`` is pure compute driven by a PadPlant, so everything
here runs against toy plants and a fake NetworkOptimizer/optimize (the
``test_optimization_algos`` mock-batch style, taken one step further: the
optimizer itself is faked so we can shape total PF / oil as a function of the
trial header pressure). Covered:

* fixed_curve fixed point: converges (sets ``converged``), oscillation hits
  the iteration cap with ``converged=False`` (P0-9)
* free_pressure sweep: keeps the max-oil pressure, captures the WINNING
  optimizer + its reconciliation (P0-5), skips zero-budget steps
* scenario evaluator fallback chain: chosen pump infeasible → fallback choice
  → best feasible batch row → ★ test-rate → "infeasible"
* ★/ripple rescale math: [0.3, 1.2] clamps, unchanged-wells-only ratio
* existing-baseline evaluator: bias column, ★ + "(est NT)" labeling
* match_check: ✓/⚠/✗ flag bands and the plant-derived header
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import woffl.gui.pad_optimize as po
from woffl.assembly.network_optimizer import WellConfig
from woffl.gui.pad_plant_base import PadPlant

# ── Toy plants ──────────────────────────────────────────────────────────────


class CurvePlant(PadPlant):
    """fixed_curve toy: header = 3000 - 0.01 * total_flow (linear S-Pad-alike).

    flow_window (10k, 50k) → warm start = header(0.6 * 50k) = 2700 psi;
    clamp_window = (1000, 5000) (suction 220, no operational cap).
    """

    coupling = "fixed_curve"
    n_pump_options = [3, 2]
    max_header_psi = None

    def specific_gravity(self):
        return 1.0

    def suction_psi(self):
        return 220.0

    def header_at_flow(self, q_total, n_pumps=None):
        return 3000.0 - 0.01 * q_total

    def budget_at_pressure(self, pressure, n_pumps=None):
        return 50000.0

    def flow_window(self, n_pumps=None):
        return (10000.0, 50000.0)

    def pressure_window(self, n_pumps=None):
        return (2500.0, 3000.0)

    def flags(self, q_total, n_pumps=None):
        return {
            "in_range": 10000.0 <= q_total <= 50000.0,
            "recirc": False,
            "over_capacity": q_total > 50000.0,
        }

    def envelope(self, flows, n_pumps=None, at_pressure=None):
        return [
            {
                "flow": q,
                "max_discharge_psi": self.header_at_flow(q),
                "feasible": True,
                "pumps": [],
            }
            for q in flows
        ]


class FreePlant(PadPlant):
    """free_pressure toy: frontier = 3400 - 0.02 * flow, None past 100k."""

    coupling = "free_pressure"
    n_pump_options = []
    max_header_psi = 3500.0

    def specific_gravity(self):
        return 1.0

    def suction_psi(self):
        return 217.0

    def header_at_flow(self, q_total, n_pumps=None):
        if q_total > 100000.0:
            return None
        return 3400.0 - 0.02 * q_total

    def budget_at_pressure(self, pressure, n_pumps=None):
        return max(0.0, (3400.0 - pressure) / 0.02)

    def flow_window(self, n_pumps=None):
        return (0.0, 60000.0)

    def pressure_window(self, n_pumps=None):
        return (2000.0, 3000.0)

    def flags(self, q_total, n_pumps=None):
        over = q_total > 100000.0
        return {"in_range": not over, "recirc": False, "over_capacity": over}

    def envelope(self, flows, n_pumps=None, at_pressure=None):
        return [
            {
                "flow": q,
                "max_discharge_psi": self.header_at_flow(q),
                "feasible": True,
                "amp_limited": True,
                "pumps": [{"name": "LP", "hz": 55.0}],
            }
            for q in flows
        ]


# ── Fake optimizer plumbing ─────────────────────────────────────────────────


class FakeOptimizer:
    """Stands in for NetworkOptimizer. Pump performance comes from a
    dict-or-callable table; callables receive the constraint pressure so a
    test can shape performance as a function of the trial header."""

    perf_table: dict = {}
    batch_dfs: dict = {}
    instances: list = []

    def __init__(self, well_configs, pf, nozzles, throats, marginal_watercut=0.6):
        self.well_configs = well_configs
        self.power_fluid = pf
        self.nozzles = nozzles
        self.throats = throats
        self.marginal_watercut = marginal_watercut
        self.batch_results = {
            w: SimpleNamespace(df=df) for w, df in self.batch_dfs.items()
        }
        type(self).instances.append(self)

    def run_all_batch_simulations(self, max_workers=None):
        self.ran = True

    def get_pump_performance(self, well, nozzle, throat):
        v = self.perf_table.get((well, nozzle, throat))
        if callable(v):
            v = v(self.power_fluid.pressure)
        return v


def _batch_df(rows):
    """Mini BatchPump.df: rows = [(nozzle, throat, qoil_std)]."""
    return pd.DataFrame([{"nozzle": n, "throat": t, "qoil_std": q} for n, t, q in rows])


def _wells(*names):
    return [
        WellConfig(well_name=n, res_pres=1500, form_temp=70, jpump_tvd=4000)
        for n in names
    ]


def _result(well, lift_water, oil):
    return SimpleNamespace(
        well_name=well,
        recommended_nozzle="12",
        recommended_throat="B",
        predicted_lift_water=lift_water,
        predicted_oil_rate=oil,
    )


@pytest.fixture
def fake_core(monkeypatch):
    """Patch the modules pad_optimize imports from: NetworkOptimizer,
    optimize, reconcile_wells, worker_ceiling. Returns a mutable namespace the
    test configures (``optimize_fn`` shapes the run results per pressure)."""
    import woffl.assembly.network_optimizer as no_mod
    import woffl.assembly.optimization_algorithms as oa_mod
    import woffl.gui.scotts_tools._common as common_mod

    class Optimizer(FakeOptimizer):
        perf_table = {}
        batch_dfs = {}
        instances = []

    ns = SimpleNamespace(
        Optimizer=Optimizer,
        optimize_fn=lambda opt: [],
        reconcile_calls=[],
    )

    def fake_optimize(opt, method="milp", water_key=None):
        ns.method, ns.water_key = method, water_key
        return ns.optimize_fn(opt)

    def fake_reconcile(opt, results):
        ns.reconcile_calls.append(opt)
        return {"at_pressure": opt.power_fluid.pressure, "n_results": len(results)}

    monkeypatch.setattr(no_mod, "NetworkOptimizer", Optimizer)
    monkeypatch.setattr(no_mod, "reconcile_wells", fake_reconcile)
    monkeypatch.setattr(oa_mod, "optimize", fake_optimize)
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    return ns


# ── run_optimization: fixed_curve fixed point ───────────────────────────────


class TestFixedPointRun:
    def test_converges_and_sets_converged(self, fake_core):
        # constant demand 20k BPD → curve settles at 2800 psi; warm start 2700
        # → deltas 100, 40, 16, 6.4 → converged on iteration 4
        fake_core.optimize_fn = lambda opt: [
            _result("W1", 10000.0, 100.0),
            _result("W2", 10000.0, 150.0),
        ]
        progress = []
        results, optimizer, meta = po.run_optimization(
            _wells("W1", "W2"),
            CurvePlant(),
            3,
            ["12"],
            ["B"],
            "milp",
            0.7,
            progress=lambda *a: progress.append(a),
        )
        assert meta["converged"] is True
        assert meta["header_psi"] == pytest.approx(2800.0)
        assert meta["total_pf_bpd"] == pytest.approx(20000.0)
        assert meta["total_oil_bopd"] == pytest.approx(250.0)
        assert len(meta["history"]) == 4 and len(progress) == 4
        assert meta["history"][0]["curve_psi"] == pytest.approx(2800.0)
        # station extras + flags
        assert meta["per_pump_bpd"] == pytest.approx(20000.0 / 3)
        assert meta["station_cap_bpd"] == pytest.approx(50000.0)
        assert meta["in_range"] is True and meta["over_capacity"] is False
        # the marginal-WC gate is the RUN's, not the scenario constant
        assert optimizer.marginal_watercut == 0.7
        # reconciliation computed from the returned optimizer
        assert meta["reconciliation"]["at_pressure"] == pytest.approx(
            optimizer.power_fluid.pressure
        )

    def test_oscillation_hits_cap_with_converged_false(self, fake_core):
        # threshold flip: high header → big draw → low curve, and vice versa —
        # the damped iterate never lands within tolerance (P0-9 flag)
        def flip(opt):
            q = 60000.0 if opt.power_fluid.pressure >= 2700.0 else 0.0
            return [_result("W1", q, 50.0)]

        fake_core.optimize_fn = flip
        results, optimizer, meta = po.run_optimization(
            _wells("W1"), CurvePlant(), 3, ["12"], ["B"], "milp", 1.0
        )
        assert meta["converged"] is False
        assert len(meta["history"]) == 8  # the iteration cap


# ── run_optimization: free_pressure sweep ───────────────────────────────────


class TestPressureSweepRun:
    def test_keeps_max_oil_pressure_and_winning_optimizer(self, fake_core):
        # oil peaks at 2600 psi across the 2000..3000 sweep (11 steps of 100)
        fake_core.optimize_fn = lambda opt: [
            _result("W1", 1000.0, 5000.0 - abs(opt.power_fluid.pressure - 2600.0))
        ]
        results, optimizer, meta = po.run_optimization(
            _wells("W1"), FreePlant(), None, ["12"], ["B"], "mckp", 1.0, n_steps=11
        )
        assert meta["header_psi"] == pytest.approx(2600.0)
        assert meta["total_oil_bopd"] == pytest.approx(5000.0)
        assert len(meta["sweep"]) == 11
        assert meta["converged"] is True  # a sweep has no fixed point to miss
        # the returned optimizer is the WINNING step's, and the
        # reconciliation was computed from it (P0-5)
        assert optimizer.power_fluid.pressure == pytest.approx(2600.0)
        assert meta["reconciliation"]["at_pressure"] == pytest.approx(2600.0)
        assert fake_core.reconcile_calls == [optimizer]
        # frontier extras
        assert meta["frontier_cap_bpd"] == pytest.approx((3400.0 - 2600.0) / 0.02)
        assert meta["suction_psi"] == pytest.approx(217.0)
        assert meta["amp_limited"] is True and meta["pumps"][0]["name"] == "LP"

    def test_skips_zero_budget_steps(self, fake_core):
        # budget vanishes above 2800 psi → those steps never build an optimizer
        plant = FreePlant()
        plant.budget_at_pressure = lambda p, n_pumps=None: (
            0.0 if p > 2800.0 else 10000.0
        )
        fake_core.optimize_fn = lambda opt: [
            _result("W1", 500.0, opt.power_fluid.pressure)
        ]
        progress = []
        results, optimizer, meta = po.run_optimization(
            _wells("W1"),
            plant,
            None,
            ["12"],
            ["B"],
            "milp",
            1.0,
            n_steps=11,
            progress=lambda *a: progress.append(a),
        )
        assert len(meta["sweep"]) == 9  # 2900, 3000 skipped
        assert len(progress) == 11  # but progress still ticks every step
        assert meta["header_psi"] == pytest.approx(2800.0)  # most oil = highest P

    def test_no_feasible_pressure_raises_plant_message(self, fake_core):
        plant = FreePlant()
        plant.budget_at_pressure = lambda p, n_pumps=None: 0.0
        with pytest.raises(RuntimeError, match="No feasible header pressure"):
            po.run_optimization(
                _wells("W1"), plant, None, ["12"], ["B"], "milp", 1.0, n_steps=5
            )


# ── evaluate_fixed_scenario: fallback chain ─────────────────────────────────


class TestFixedScenarioFallbackChain:
    def test_chain_fallback_then_best_feasible_then_infeasible(self, fake_core):
        opt_cls = fake_core.Optimizer
        opt_cls.perf_table = {
            # W1: chosen pump solves
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 500.0},
            # W2: chosen fails, optimizer fallback solves
            ("W2", "11", "A"): {"oil_rate": 80.0, "lift_water": 400.0},
            # W3: chosen + no fallback; best feasible batch row solves
            ("W3", "10", "C"): {"oil_rate": 60.0, "lift_water": 300.0},
            # W4: nothing solves anywhere
        }
        opt_cls.batch_dfs = {
            "W3": _batch_df([("10", "C", 60.0), ("9", "A", 20.0)]),
            "W4": _batch_df([("10", "C", float("nan"))]),
        }
        choices = {w: ("12", "B") for w in ("W1", "W2", "W3", "W4")}
        per_well, meta = po.evaluate_fixed_scenario(
            _wells("W1", "W2", "W3", "W4"),
            CurvePlant(),
            3,
            choices,
            fallback_choices={"W2": ("11", "A")},
        )
        rows = {r["well"]: r for r in per_well}
        assert rows["W1"]["pump"] == "12B" and rows["W1"]["note"] == ""
        assert rows["W2"]["pump"] == "12B✗→11A" and rows["W2"]["oil"] == 80.0
        assert rows["W3"]["pump"] == "12B✗→10C" and rows["W3"]["oil"] == 60.0
        assert rows["W4"]["note"] == "infeasible" and rows["W4"]["oil"] == 0.0
        assert "✗ no feasible pump" in rows["W4"]["pump"]
        # every constructed optimizer used the scenario marginal-WC constant
        assert all(o.marginal_watercut == 1.0 for o in opt_cls.instances)
        assert meta["total_oil_bopd"] == pytest.approx(240.0)

    def test_star_takes_precedence_over_fallback(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
        }
        per_well, meta = po.evaluate_fixed_scenario(
            _wells("W1", "W5"),
            CurvePlant(),
            3,
            {"W1": ("12", "B"), "W5": ("14", "C")},
            fallback_choices={"W5": ("12", "B")},  # would solve — must NOT be used
            test_rates={"W1": (100.0, 200.0), "W5": (50.0, 40.0)},
            current_choices={"W1": ("12", "B")},
        )
        rows = {r["well"]: r for r in per_well}
        assert rows["W5"]["note"] == "star" and rows["W5"]["pump"] == "14C ★"
        # ratios from W1 are exactly 1.0 → ★ keeps its measured rate
        assert rows["W5"]["oil"] == pytest.approx(50.0)
        assert rows["W5"]["pf"] == pytest.approx(40.0)

    def test_shut_in_choice(self, fake_core):
        per_well, meta = po.evaluate_fixed_scenario(
            _wells("W1"), CurvePlant(), 3, {"W1": None}
        )
        assert per_well[0]["pump"] == "SHUT IN" and per_well[0]["oil"] == 0.0


# ── ★/ripple rescale math ───────────────────────────────────────────────────


class TestRippleRescale:
    def test_unchanged_only_ratio_and_high_clamp(self, fake_core):
        fake_core.Optimizer.perf_table = {
            # W1 UNCHANGED: oil ratio 110/100 = 1.1, pf ratio 240/200 = 1.2
            ("W1", "12", "B"): {"oil_rate": 110.0, "lift_water": 240.0},
            # W2 CHANGED pump: huge ratio that must be EXCLUDED from the avg
            ("W2", "11", "A"): {"oil_rate": 999.0, "lift_water": 999.0},
        }
        per_well, meta = po.evaluate_fixed_scenario(
            _wells("W1", "W2", "W3"),
            CurvePlant(),
            3,
            {"W1": ("12", "B"), "W2": ("11", "A"), "W3": ("13", "B")},
            test_rates={
                "W1": (100.0, 200.0),
                "W2": (10.0, 10.0),
                "W3": (50.0, 40.0),
            },
            current_choices={"W1": ("12", "B"), "W2": ("12", "B")},
        )
        rows = {r["well"]: r for r in per_well}
        # W2's 99.9x ratio excluded (changed pump) — avg is W1's 1.1 / 1.2,
        # NOT the 1.2-clamped blend it would be if W2 leaked in
        assert rows["W3"]["oil"] == pytest.approx(50.0 * 1.1)
        assert rows["W3"]["pf"] == pytest.approx(40.0 * 1.2)
        # totals + header recomputed from the rescaled rows
        total_pf = 240.0 + 999.0 + 48.0
        assert meta["total_pf_bpd"] == pytest.approx(total_pf)
        assert meta["header_psi"] == pytest.approx(3000.0 - 0.01 * total_pf)

    def test_ratio_clamps_at_bounds(self, fake_core):
        # model 10x the test → oil ratio clamps to 1.2; model at 10% → 0.3
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 1000.0, "lift_water": 20.0},
        }
        per_well, _ = po.evaluate_fixed_scenario(
            _wells("W1", "W3"),
            CurvePlant(),
            3,
            {"W1": ("12", "B"), "W3": ("13", "B")},
            test_rates={"W1": (100.0, 200.0), "W3": (50.0, 40.0)},
            current_choices={"W1": ("12", "B")},
        )
        rows = {r["well"]: r for r in per_well}
        assert rows["W3"]["oil"] == pytest.approx(50.0 * 1.2)  # 10.0 → clamp 1.2
        assert rows["W3"]["pf"] == pytest.approx(40.0 * 0.3)  # 0.1 → clamp 0.3


# ── evaluate_existing_scenario ──────────────────────────────────────────────


class TestExistingScenario:
    def test_relative_change_and_bias(self, fake_core):
        fake_core.Optimizer.perf_table = {
            # current pump model: 120 oil / 300 pf (vs measured 100 / 200)
            ("W1", "12", "B"): {"oil_rate": 120.0, "lift_water": 300.0},
            # scenario pump model: half the oil, half the pf
            ("W1", "11", "A"): {"oil_rate": 60.0, "lift_water": 150.0},
        }
        per_well, meta = po.evaluate_existing_scenario(
            _wells("W1"),
            CurvePlant(),
            3,
            {"W1": ("11", "A")},
            {"W1": ("12", "B")},
            test_rates={"W1": (100.0, 200.0)},
        )
        row = per_well[0]
        # displayed = measured x model relative change (bias cancels)
        assert row["oil"] == pytest.approx(100.0 * (60.0 / 120.0))
        assert row["pf"] == pytest.approx(200.0 * (150.0 / 300.0))
        # bias = model-at-current / measured
        assert row["bias"] == pytest.approx(1.2)
        assert meta["converged"] is True

    def test_star_estimated_fallback_labeling(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
            # W2's best feasible batch row (its scenario pump never solves)
            ("W2", "10", "C"): {"oil_rate": 30.0, "lift_water": 60.0},
        }
        fake_core.Optimizer.batch_dfs = {"W2": _batch_df([("10", "C", 30.0)])}
        per_well, _ = po.evaluate_existing_scenario(
            _wells("W1", "W2"),
            CurvePlant(),
            3,
            {"W1": ("12", "B"), "W2": ("11", "A")},
            {"W1": ("12", "B")},  # W2 has no current pump and no test rate
            test_rates={"W1": (100.0, 200.0)},
        )
        rows = {r["well"]: r for r in per_well}
        # no measured anchor → estimated from the best feasible pump, labeled
        assert rows["W2"]["pump"] == "11A ★ (est 10C)"
        assert rows["W2"]["oil"] == pytest.approx(30.0)
        assert rows["W2"]["bias"] is None


# ── match_check ─────────────────────────────────────────────────────────────


class TestMatchCheck:
    @pytest.mark.parametrize(
        "ratio,verdict",
        [
            (None, "— no data"),
            (0.80, "✓ match"),
            (1.0, "✓ match"),
            (1.25, "✓ match"),
            (0.79, "⚠ off"),
            (0.50, "⚠ off"),
            (1.26, "⚠ off"),
            (2.0, "⚠ off"),
            (0.49, "✗ BUST"),
            (2.01, "✗ BUST"),
            (3.0, "✗ BUST"),
        ],
    )
    def test_flag_bands(self, ratio, verdict):
        assert po.match_flag(ratio) == verdict

    def test_rows_and_plant_header(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
        }
        wells = _wells("W1", "W2")
        rows, header = po.match_check(
            wells,
            CurvePlant(),
            3,
            {"W1": ("12", "B"), "W2": None},
            {"W1": (100.0, 400.0), "W2": (None, None)},
        )
        # header from the plant's curve at the measured total PF (400 BPD)
        assert header == pytest.approx(3000.0 - 0.01 * 400.0)
        assert all(wc.ppf_surf_well == header for wc in wells)
        opt = fake_core.Optimizer.instances[-1]
        assert opt.power_fluid.pressure == pytest.approx(header)
        assert opt.marginal_watercut == 1.0
        by_well = {r["well"]: r for r in rows}
        assert by_well["W1"]["oil_flag"] == "✓ match"  # 100/100
        assert by_well["W1"]["pf_flag"] == "⚠ off"  # 200/400 = 0.5
        assert by_well["W2"]["pump"] == "—"
        assert by_well["W2"]["oil_flag"] == "— no data"


# ── P1-13: PowerFluidConstraint.rho_pf plumbing ─────────────────────────────


class TestRhoPfDefault:
    """docs/code_review_2026-07-01.md P1-13: the 5 PowerFluidConstraint call
    sites in this module used to spell out ``rho_pf=62.4`` independently.
    They now share one ``_RHO_PF_DEFAULT`` constant. Confirms (a) the
    constant is numerically identical to PowerFluidConstraint's own default
    (so de-duplicating it changed no behavior — rho_pf isn't read anywhere
    downstream of the dataclass's own range validation) and (b) it actually
    reaches the constructed PowerFluidConstraint at every call site."""

    def test_constant_matches_dataclass_default(self):
        from woffl.assembly.network_optimizer import PowerFluidConstraint

        assert (
            po._RHO_PF_DEFAULT
            == PowerFluidConstraint(total_rate=1000, pressure=2000).rho_pf
        )

    def test_fixed_curve_run(self, fake_core):
        fake_core.optimize_fn = lambda opt: [_result("W1", 10000.0, 100.0)]
        _, optimizer, _ = po.run_optimization(
            _wells("W1"), CurvePlant(), 3, ["12"], ["B"], "milp", 0.7
        )
        assert optimizer.power_fluid.rho_pf == po._RHO_PF_DEFAULT

    def test_free_pressure_run(self, fake_core):
        fake_core.optimize_fn = lambda opt: [_result("W1", 1000.0, 50.0)]
        _, optimizer, _ = po.run_optimization(
            _wells("W1"), FreePlant(), None, ["12"], ["B"], "mckp", 1.0, n_steps=5
        )
        assert optimizer.power_fluid.rho_pf == po._RHO_PF_DEFAULT

    def test_evaluate_fixed_scenario(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
        }
        po.evaluate_fixed_scenario(_wells("W1"), CurvePlant(), 3, {"W1": ("12", "B")})
        opt = fake_core.Optimizer.instances[-1]
        assert opt.power_fluid.rho_pf == po._RHO_PF_DEFAULT

    def test_evaluate_existing_scenario(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
        }
        po.evaluate_existing_scenario(
            _wells("W1"),
            CurvePlant(),
            3,
            {"W1": ("12", "B")},
            {"W1": ("12", "B")},
            test_rates={"W1": (100.0, 200.0)},
        )
        opt = fake_core.Optimizer.instances[-1]
        assert opt.power_fluid.rho_pf == po._RHO_PF_DEFAULT

    def test_match_check(self, fake_core):
        fake_core.Optimizer.perf_table = {
            ("W1", "12", "B"): {"oil_rate": 100.0, "lift_water": 200.0},
        }
        po.match_check(
            _wells("W1"),
            CurvePlant(),
            3,
            {"W1": ("12", "B")},
            {"W1": (100.0, 200.0)},
        )
        opt = fake_core.Optimizer.instances[-1]
        assert opt.power_fluid.rho_pf == po._RHO_PF_DEFAULT


# ── settled_header ──────────────────────────────────────────────────────────


class TestSettledHeader:
    def test_fixed_curve_no_cap(self):
        plant = CurvePlant()
        assert po.settled_header(plant, 20000.0, 999.0) == (2800.0, False)
        # no draw → fallback, uncapped (S-Pad has no operational cap)
        assert po.settled_header(plant, 0.0, 999.0) == (999.0, False)

    def test_free_pressure_cap_and_collapse(self):
        plant = FreePlant()
        # capped at the operational limit
        assert po.settled_header(plant, 1000.0, 0.0) == (3380.0, False)
        # past the frontier → collapses to suction, flagged over-capacity
        assert po.settled_header(plant, 200000.0, 0.0) == (217.0, True)
        # no draw → min(cap, fallback) — the pages' Existing-baseline display
        assert po.settled_header(plant, 0.0, 0.0) == (0.0, False)
