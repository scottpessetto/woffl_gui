"""Joint CFP optimization across J/G/C/B (W3).

Runs against a fake NetworkOptimizer/optimize (same pattern as
``tests/test_pad_optimize.py``) so the sweep logic is tested without real
physics. The physics has its own pins in ``tests/test_cfp_plant.py`` and
``tests/test_cfp_pad_plant.py``.

The things worth protecting here are the ones that would quietly produce a
wrong recommendation:

* each pad gets its OWN delivered PF, and C-Pad's is its booster's, not the
  plant's — one common header pressure would silently model C-Pad wrong;
* the water basis is TOTAL water, not lift (Scott's decision — the plant's
  capacity basis is formation + lift);
* a pressure the plant cannot sustain never wins, and a skipped trial records
  WHY (a silently missing trial looks identical to one that made less oil);
* the run always advertises that the curve is provisional.
"""

from types import SimpleNamespace

import pytest

from woffl.assembly import cfp_plant as cfp
from woffl.assembly.network_optimizer import WellConfig
from woffl.gui import cfp_optimize as co
from woffl.gui.cfp_pad_plant import CFPPlant, CFPMachineSubsetUnvalidated

PLANT = CFPPlant()


# ── fakes ───────────────────────────────────────────────────────────────────


class FakeOptimizer:
    instances: list = []

    def __init__(self, well_configs, pf, nozzles, throats, marginal_watercut=1.0):
        self.well_configs = well_configs
        self.power_fluid = pf
        self.marginal_watercut = marginal_watercut
        self.batch_results = {}
        # Snapshot the per-well PF assignment at construction — the sweep
        # mutates the same WellConfig objects each trial, so a later trial would
        # otherwise overwrite what we want to assert.
        self.pf_by_well = {wc.well_name: wc.ppf_surf_well for wc in well_configs}
        type(self).instances.append(self)

    def run_all_batch_simulations(self, max_workers=None):
        self.ran = True


def _result(well, total_water, oil):
    return SimpleNamespace(
        well_name=well,
        recommended_nozzle="12",
        recommended_throat="B",
        predicted_total_water=total_water,
        predicted_lift_water=total_water,
        predicted_oil_rate=oil,
    )


def _wells(*names):
    return [
        WellConfig(well_name=n, res_pres=1500, form_temp=70, jpump_tvd=4000)
        for n in names
    ]


def _pad_configs():
    return {
        "B": _wells("MPB-28", "MPB-37"),
        "G": _wells("MPG-18"),
        "J": _wells("MPJ-29"),
        "C": _wells("MPC-14"),
    }


@pytest.fixture
def fake_core(monkeypatch):
    import woffl.assembly.network_optimizer as no_mod
    import woffl.assembly.optimization_algorithms as oa_mod
    import woffl.assembly.parallelism as common_mod

    class Optimizer(FakeOptimizer):
        instances = []

    ns = SimpleNamespace(
        Optimizer=Optimizer,
        # default: 5,000 BWPD of pad water, oil rising with the trial pressure
        optimize_fn=lambda opt: [
            _result("MPB-28", 5000.0, opt.power_fluid.pressure / 10.0)
        ],
        water_keys=[],
        gate=(0.94, 1234.0),
    )

    def fake_optimize(opt, method="milp", water_key=None):
        ns.water_keys.append(water_key)
        return ns.optimize_fn(opt)

    def fake_parsimony(results, opt, water_key, threshold):
        ns.parsimony_water_key = water_key
        return results, []

    def fake_gate(batch_results, cap, water_key="lift_wat"):
        ns.gate_water_key = water_key
        return ns.gate

    monkeypatch.setattr(no_mod, "NetworkOptimizer", Optimizer)
    monkeypatch.setattr(oa_mod, "optimize", fake_optimize)
    monkeypatch.setattr(oa_mod, "apply_parsimony", fake_parsimony)
    monkeypatch.setattr(oa_mod, "derive_pad_marginal_wc", fake_gate)
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    return ns


def _run(fake_core, **kw):
    params = dict(
        pad_configs=_pad_configs(),
        plant=PLANT,
        n_machines=3,
        nozzles=["12"],
        throats=["B"],
        method="milp",
        marginal_wc=None,
        exogenous_bwpd=60000.0,
        c_pad_pf_psi=3400.0,
        n_steps=5,
    )
    params.update(kw)
    return co.run_joint_optimization(**params)


# ── per-pad delivery ────────────────────────────────────────────────────────


class TestDeliveredByPad:
    def test_c_pad_uses_its_own_booster_not_the_plant(self):
        per_pad, _ = co.delivered_by_pad(
            PLANT, 2792.0, ["B", "G", "J", "C"], c_pad_pf_psi=3400.0
        )
        assert per_pad["C"] == 3400.0
        # …and it is nothing like what the plant delivers.
        assert per_pad["C"] > max(per_pad[p] for p in ("B", "G", "J")) + 500

    def test_plant_pads_differ_from_each_other(self):
        per_pad, _ = co.delivered_by_pad(
            PLANT, 2792.0, ["B", "G", "J"], c_pad_pf_psi=3400.0
        )
        assert len({per_pad["B"], per_pad["G"], per_pad["J"]}) == 3

    def test_measured_anchor_used_when_supplied(self):
        per_pad, _ = co.delivered_by_pad(
            PLANT,
            cfp.MEASURED_DISCHARGE_PSI,
            ["B"],
            c_pad_pf_psi=3400.0,
            measured_pad_pf={"B": 2623.0},
        )
        assert per_pad["B"] == pytest.approx(2623.0)

    def test_low_discharge_clamps_and_reports(self):
        """A pad's delivered PF can fall under PowerFluidConstraint's 1,000 psi
        floor; clamping silently would hide it."""
        per_pad, clamped = co.delivered_by_pad(
            PLANT, 1100.0, ["B", "C"], c_pad_pf_psi=3400.0
        )
        assert per_pad["B"] == 1000.0
        assert "B" in clamped
        assert "C" not in clamped


# ── the sweep ───────────────────────────────────────────────────────────────


class TestSweep:
    def test_machine_subset_gate_fires_before_any_compute(self, fake_core):
        with pytest.raises(CFPMachineSubsetUnvalidated):
            _run(fake_core, n_machines=2)
        assert fake_core.Optimizer.instances == []

    def test_picks_the_max_oil_pressure(self, fake_core):
        # oil rises with pressure, so the highest feasible trial should win
        results, opt, meta = _run(fake_core)
        assert meta["feasible"] is True
        feasible_ps = [rec["P"] for rec in meta["sweep"]]
        assert meta["header_psi"] == max(feasible_ps)

    def test_water_basis_is_total_not_lift(self, fake_core):
        """Scott's decision: the plant's capacity basis is TOTAL water."""
        _run(fake_core)
        assert set(fake_core.water_keys) == {"totl_wat"}
        assert fake_core.parsimony_water_key == "totl_wat"
        assert fake_core.gate_water_key == "totl_wat"

    def test_each_well_gets_its_pads_pressure(self, fake_core):
        _run(fake_core)
        inst = fake_core.Optimizer.instances[0]
        pf = inst.pf_by_well
        # both B wells share B's delivery…
        assert pf["MPB-28"] == pf["MPB-37"]
        # …C-Pad is on its own booster…
        assert pf["MPC-14"] == 3400.0
        # …and the three plant-fed pads all differ.
        assert len({pf["MPB-28"], pf["MPG-18"], pf["MPJ-29"]}) == 3

    def test_skips_and_records_when_exogenous_exceeds_capacity(self, fake_core):
        """At 130k of exogenous water the high-pressure trials can't even move
        the water we don't control."""
        _results, _opt, meta = _run(fake_core, exogenous_bwpd=130000.0)
        assert meta["skipped"], "a dropped trial must say why"
        assert all("reason" in s for s in meta["skipped"])
        assert any("exogenous" in s["reason"] for s in meta["skipped"])

    def test_all_infeasible_returns_plant_message_not_a_crash(self, fake_core):
        results, opt, meta = _run(fake_core, exogenous_bwpd=500000.0)
        assert results == [] and opt is None
        assert meta["feasible"] is False
        assert meta["message"] == PLANT.infeasible_sweep_msg
        assert meta["n_feasible"] == 0

    def test_unsustainable_pressure_never_wins(self, fake_core):
        """If the solution's water load exceeds what the plant passes at that
        pressure, the trial is dropped even though it made the most oil."""
        fake_core.optimize_fn = lambda opt: [_result("MPB-28", 1e9, 1e9)]
        results, _opt, meta = _run(fake_core)
        assert meta["feasible"] is False
        assert any("water load exceeds" in s["reason"] for s in meta["skipped"])

    def test_meta_advertises_the_provisional_curve(self, fake_core):
        _r, _o, meta = _run(fake_core)
        assert meta["provisional_curve"] is True
        assert meta["measured_discharge_psi"] == cfp.MEASURED_DISCHARGE_PSI
        assert meta["machine_subset_available"] is False

    def test_meta_carries_the_operating_point(self, fake_core):
        _r, _o, meta = _run(fake_core)
        assert meta["plant_load_bwpd"] == pytest.approx(
            meta["pad_water_bwpd"] + meta["exogenous_bwpd"]
        )
        assert meta["plant_load_bwpd"] <= meta["plant_capacity_bwpd"] + 1e-6
        assert set(meta["per_pad_pf"]) == {"B", "G", "J", "C"}
        assert meta["pinned"] in ("interior", "pinned_low", "pinned_high")
        assert isinstance(meta["trusted_band"], bool)

    def test_progress_is_called_per_step(self, fake_core):
        seen = []
        _run(fake_core, progress=lambda i, n, P, w, o: seen.append((i, n)))
        assert [s[0] for s in seen] == [1, 2, 3, 4, 5]
        assert all(s[1] == 5 for s in seen)

    def test_no_wells_rejected(self, fake_core):
        with pytest.raises(ValueError):
            _run(fake_core, pad_configs={"B": []})


# ── per-pad roll-up ─────────────────────────────────────────────────────────


def test_summarize_by_pad():
    pad_configs = _pad_configs()
    results = [
        _result("MPB-28", 4000.0, 300.0),
        _result("MPB-37", 1000.0, 100.0),
        _result("MPJ-29", 2000.0, 250.0),
    ]
    rows = co.summarize_by_pad(results, pad_configs)
    by_pad = {r["pad"]: r for r in rows}
    assert by_pad["B"]["wells"] == 2
    assert by_pad["B"]["oil_bopd"] == pytest.approx(400.0)
    assert by_pad["B"]["total_water_bwpd"] == pytest.approx(5000.0)
    assert by_pad["J"]["wells"] == 1
    assert "G" not in by_pad  # no result for G in this set


# ── model accuracy roll-up (pure) ───────────────────────────────────────────
# The dashboard's break-even WC and BOPD-per-1,000-BWPD are computed FROM the
# per-well JP models. If those don't reproduce the wells' own measured tests the
# answer is arithmetic on noise, so the roll-up has to say so plainly.


def _row(well, oil_flag, pf_flag, oil_ratio=1.0, pf_ratio=1.0):
    return {
        "well": well, "oil_flag": oil_flag, "pf_flag": pf_flag,
        "oil_ratio": oil_ratio, "pf_ratio": pf_ratio,
    }


class TestMatchSummary:
    def test_all_matching_is_good(self):
        rows = [_row(f"W{i}", "✓ match", "✓ match") for i in range(10)]
        s = co.match_summary(rows)
        assert s["trust"] == "good"
        assert s["both_ok"] == 10 and s["frac_ok"] == 1.0

    def test_a_single_bust_drops_it_below_good(self):
        """A bust well means a number in the answer is simply wrong — that must
        not be averaged away by nine good ones."""
        rows = [_row(f"W{i}", "✓ match", "✓ match") for i in range(9)]
        rows.append(_row("BAD", "✗ BUST", "✓ match", oil_ratio=4.0))
        s = co.match_summary(rows)
        assert s["trust"] != "good"
        assert s["oil_bust"] == 1

    def test_mostly_broken_is_poor_and_says_direction_only(self):
        rows = [_row(f"W{i}", "✗ BUST", "✗ BUST", 5.0, 5.0) for i in range(6)]
        rows += [_row(f"G{i}", "✓ match", "✓ match") for i in range(4)]
        s = co.match_summary(rows)
        assert s["trust"] == "poor"
        assert "not reliable" in s["reason"]

    def test_oil_and_pf_counted_separately(self):
        """They fail for different reasons — loose IPR busts oil, wrong nozzle
        or wear state busts PF."""
        rows = [_row("A", "✓ match", "✗ BUST", 1.0, 3.0),
                _row("B", "✗ BUST", "✓ match", 3.0, 1.0)]
        s = co.match_summary(rows)
        assert s["oil_bust"] == 1 and s["pf_bust"] == 1
        assert s["both_ok"] == 0

    def test_worst_wells_surface_first(self):
        rows = [_row("fine", "✓ match", "✓ match", 1.02, 0.99),
                _row("awful", "✗ BUST", "✗ BUST", 6.0, 5.0),
                _row("meh", "⚠ off", "✓ match", 1.6, 1.0)]
        assert co.match_summary(rows)["worst"][0]["well"] == "awful"

    def test_empty_is_none_not_a_crash(self):
        s = co.match_summary([])
        assert s["trust"] == "none" and s["n"] == 0
