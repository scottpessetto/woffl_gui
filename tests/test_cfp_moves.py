"""Today's Moves engine — anchored delta optimization (docs/cfp_moves_methodology.md).

Synthetic surfaces with hand-computable numbers so every assertion is a
by-hand check of the formulation:

* RESP — online B-pad well, oil RESPONDS to discharge (0.5 BOPD/psi on 12B).
* PIG  — online J-pad well: 5 BOPD, 2,000 BPD water. The SI candidate.
* BOL1 — offline G-pad well: 200 BOPD, 4,000 BPD water if brought on.
* FLAT — online C-pad well: own booster, no pressure response.

Anchor: P0 = 2,800 psi, s = 15 psi per 1,000 BPD, trip 2,900 / margin 20.
Baseline water W0 = 5,000 + 2,000 + 1,000 = 8,000; baseline oil = 405.
"""

import pytest

from woffl.gui import cfp_moves as cm
from woffl.gui.cfp_moves import (
    MOVE_BRING_ON,
    MOVE_RESIZE,
    MOVE_SHUT_IN,
    OFF,
    SI,
    AnchoredPlant,
    Surfaces,
    WellSurface,
    anchor,
    best_plan,
    moves_summary,
    option_at,
    pair_moves,
    rank_single_moves,
    settle,
    shadow_price_today,
    sweep_frontier,
)

GRID = [2500.0, 2600.0, 2700.0, 2800.0, 2880.0]
P0 = 2800.0


def _opt(oil_at_2800, water, slope=0.0):
    return {
        "nozzle": "?",
        "throat": "?",
        "_grid": GRID,
        "oil": [oil_at_2800 + slope * (p - 2800.0) for p in GRID],
        "water": [water] * len(GRID),
    }


def _surfaces():
    s = Surfaces(p_grid=GRID, p0=P0)
    s.wells["RESP"] = WellSurface(
        well="RESP", pad="B", online=True, current="12B",
        options={"12B": _opt(300.0, 5000.0, slope=0.5),
                 "10A": _opt(280.0, 3000.0, slope=0.4)},
    )
    s.wells["PIG"] = WellSurface(
        well="PIG", pad="J", online=True, current="12B",
        options={"12B": _opt(5.0, 2000.0)},
    )
    s.wells["BOL1"] = WellSurface(
        well="BOL1", pad="G", online=False, current=None,
        options={"11B": _opt(200.0, 4000.0)},
    )
    s.wells["FLAT"] = WellSurface(
        well="FLAT", pad="C", online=True, current="12B",
        options={"12B": _opt(100.0, 1000.0)},
    )
    return s


def _plant(s):
    return anchor(s, psi_per_kbpd=15.0, trip_psi=2900.0, trip_margin_psi=20.0)


# ── interpolation ───────────────────────────────────────────────────────────


class TestInterp:
    def test_linear_between_grid_points(self):
        ws = _surfaces().wells["RESP"]
        oil, water = option_at(ws, "12B", 2750.0)
        assert oil == pytest.approx(300.0 + 0.5 * -50.0)
        assert water == pytest.approx(5000.0)

    def test_idle_labels_are_exactly_zero(self):
        ws = _surfaces().wells["RESP"]
        assert option_at(ws, SI, 2777.0) == (0.0, 0.0)

    def test_non_converged_points_are_skipped(self):
        ws = _surfaces().wells["RESP"]
        ws.options["12B"]["oil"][0] = None  # failed at 2,500
        ws.options["12B"]["water"][0] = None
        oil, _ = option_at(ws, "12B", 2500.0)  # held at nearest valid (2,600)
        assert oil == pytest.approx(300.0 + 0.5 * -200.0)

    def test_option_with_no_converged_points_is_not_a_choice(self):
        ws = _surfaces().wells["RESP"]
        ws.options["10A"]["oil"] = [None] * len(GRID)
        assert "10A" not in ws.labels()
        assert SI in ws.choice_labels()


# ── the anchor: measured state, no exogenous anything ───────────────────────


class TestAnchor:
    def test_baseline_water_is_the_models_own_sum(self):
        s = _surfaces()
        assert _plant(s).baseline_water == pytest.approx(8000.0)

    def test_baseline_settles_to_exactly_today(self):
        """The anchored model must reproduce today by construction — this is
        the property that makes every unknown cancel."""
        s = _surfaces()
        state = settle(s.baseline_choices(), s, _plant(s))
        assert state["pressure"] == pytest.approx(P0)
        assert state["oil"] == pytest.approx(405.0)
        assert not state["at_trip"]

    def test_offline_wells_are_idle_in_the_baseline(self):
        s = _surfaces()
        assert s.baseline_choices()["BOL1"] == OFF

    def test_shedding_water_raises_pressure_by_the_slope(self):
        s = _surfaces()
        plant = _plant(s)
        p, at_trip = plant.pressure_at(6000.0)  # shed 2,000 BPD
        assert p == pytest.approx(2830.0)
        assert not at_trip

    def test_the_trip_cap_is_the_kink(self):
        """Shed past the cap and disposal re-trims: pressure saturates, flag
        set — further shedding is pure oil loss."""
        s = _surfaces()
        plant = _plant(s)
        p, at_trip = plant.pressure_at(1000.0)  # raw would be 2,905
        assert p == pytest.approx(2880.0)
        assert at_trip


# ── single moves: the knob board ────────────────────────────────────────────


class TestSingleMoves:
    def setup_method(self):
        self.s = _surfaces()
        self.plant = _plant(self.s)
        self.moves = rank_single_moves(self.s, self.plant)

    def _move(self, well, to):
        return next(m for m in self.moves if m["well"] == well and m["to"] == to)

    def test_si_the_pig_pays_through_the_pressure_gain(self):
        """SI PIG: −2,000 BPD → +30 psi → RESP +15 BOPD, PIG −5 → fleet +10."""
        m = self._move("PIG", SI)
        assert m["type"] == MOVE_SHUT_IN
        assert m["pressure_delta"] == pytest.approx(30.0)
        assert m["fleet_oil_delta"] == pytest.approx(10.0)
        assert m["own_oil_delta"] == pytest.approx(-5.0)

    def test_bol_pays_despite_dropping_pressure(self):
        """BOL1 on: +4,000 BPD → −60 psi → RESP −30, BOL1 +200 → fleet +170."""
        m = self._move("BOL1", "11B")
        assert m["type"] == MOVE_BRING_ON
        assert m["pressure_delta"] == pytest.approx(-60.0)
        assert m["fleet_oil_delta"] == pytest.approx(170.0)

    def test_downsizing_the_responsive_well_does_not_pay_here(self):
        """RESP 12B→10A sheds 2,000 BPD (+30 psi) but its own oil drops more
        than the fleet gains: 292 vs 300, others flat → fleet −8."""
        m = self._move("RESP", "10A")
        assert m["type"] == MOVE_RESIZE
        assert m["pressure_delta"] == pytest.approx(30.0)
        assert m["fleet_oil_delta"] == pytest.approx(-8.0)

    def test_ranked_best_first(self):
        deltas = [m["fleet_oil_delta"] for m in self.moves]
        assert deltas == sorted(deltas, reverse=True)
        assert self.moves[0]["well"] == "BOL1"

    def test_current_option_is_not_a_move(self):
        assert not any(
            m["well"] == "RESP" and m["to"] == "12B" for m in self.moves
        )


# ── pairs: BOL offset by a pressure raiser ──────────────────────────────────


class TestPairs:
    def test_bol_plus_si_beats_both_halves(self):
        """BOL1 + SI PIG: net +2,000 BPD → −30 psi → RESP 285, fleet 585 —
        +180 vs +170 (BOL alone) and +10 (SI alone)."""
        s = _surfaces()
        plant = _plant(s)
        pairs = pair_moves(s, plant)
        assert pairs, "the offsetting pair must be found"
        top = pairs[0]
        assert top["bring_on"]["well"] == "BOL1"
        assert top["offset"]["well"] == "PIG" and top["offset"]["to"] == SI
        assert top["fleet_oil_delta"] == pytest.approx(180.0)
        assert top["pressure_delta"] == pytest.approx(-30.0)

    def test_pairs_never_reuse_the_same_well(self):
        s = _surfaces()
        for p in pair_moves(s, _plant(s)):
            assert p["bring_on"]["well"] != p["offset"]["well"]


# ── the equal-slope frontier and the plan ───────────────────────────────────


class TestFrontierAndPlan:
    def test_water_falls_as_the_price_rises(self):
        s = _surfaces()
        frontier = sweep_frontier(s, _plant(s))
        by_lam = sorted(frontier, key=lambda st: st["lam"])
        waters = [st["water"] for st in by_lam]
        assert all(a >= b - 1e-6 for a, b in zip(waters, waters[1:]))

    def test_plan_finds_the_bol_plus_si_combination(self):
        """The sweep must land on {BOL1 on, PIG SI'd}: oil 585, +180 vs today
        — better than either single move, found without enumerating pairs."""
        s = _surfaces()
        plant = _plant(s)
        plan = best_plan(sweep_frontier(s, plant), s.baseline_choices(), s)
        assert plan is not None
        assert plan["oil"] == pytest.approx(585.0)
        acts = {(a["well"], a["to"]) for a in plan["actions"]}
        assert ("BOL1", "11B") in acts
        assert ("PIG", SI) in acts
        types = {a["well"]: a["type"] for a in plan["actions"]}
        assert types["BOL1"] == MOVE_BRING_ON
        assert types["PIG"] == MOVE_SHUT_IN

    def test_frontier_pressures_respect_the_trip_cap(self):
        s = _surfaces()
        plant = _plant(s)
        for st in sweep_frontier(s, plant):
            assert st["pressure"] <= plant.cap + 1e-9


# ── the shadow price ────────────────────────────────────────────────────────


class TestShadowPrice:
    def test_equals_the_sum_of_responsive_slopes(self):
        s = _surfaces()
        lam = shadow_price_today(s, _plant(s))
        assert lam == pytest.approx(0.5, abs=0.01)  # only RESP responds

    def test_zero_when_nothing_responds(self):
        s = _surfaces()
        s.wells["RESP"].options["12B"] = _opt(300.0, 5000.0, slope=0.0)
        assert shadow_price_today(s, _plant(s)) == pytest.approx(0.0, abs=1e-9)


# ── the one-call summary ────────────────────────────────────────────────────


def test_moves_summary_carries_the_whole_decision():
    s = _surfaces()
    out = moves_summary(s, _plant(s))
    assert out["today"]["pressure"] == P0
    assert out["today"]["oil"] == pytest.approx(405.0)
    assert out["today"]["n_online"] == 3
    assert out["today"]["n_bol_candidates"] == 1
    assert out["plan_gain"] == pytest.approx(180.0)
    assert out["lambda_bopd_per_psi"] == pytest.approx(0.5, abs=0.01)
    assert out["singles"] and out["pairs"] and out["frontier"]
    assert out["n_positive_singles"] >= 2  # BOL1 and SI-PIG


# ── Stage A builder against a fake optimizer ────────────────────────────────


class FakeOptimizer:
    """Perf varies with the constraint pressure so the arrays must vary."""

    instances: list = []

    def __init__(self, well_configs, pf, nozzles, throats, marginal_watercut=1.0):
        self.wells = well_configs
        self.pf = pf
        self.nozzles = nozzles
        self.throats = throats
        type(self).instances.append(self)

    def run_all_batch_simulations(self, max_workers=None):
        self.ran = True

    def get_pump_performance(self, well, nozzle, throat):
        if nozzle == "13":  # the never-converging combo
            return None
        return {
            "oil_rate": 100.0 + self.pf.pressure / 100.0,
            "total_water": 3000.0,
            "lift_water": 2000.0,
            "formation_water": 1000.0,
        }


class TestBuilder:
    def _run(self, monkeypatch):
        import woffl.assembly.network_optimizer as no_mod
        import woffl.gui.scotts_tools._common as common_mod

        from woffl.assembly.network_optimizer import WellConfig
        from woffl.gui.cfp_pad_plant import PLANT

        FakeOptimizer.instances = []
        monkeypatch.setattr(no_mod, "NetworkOptimizer", FakeOptimizer)
        monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)

        pad_configs = {
            "B": [WellConfig(well_name="MPB-28", res_pres=1500, form_temp=70,
                             jpump_tvd=4000)],
            "J": [WellConfig(well_name="MPJ-29", res_pres=1500, form_temp=70,
                             jpump_tvd=4000)],
        }
        seen = []
        surf = cm.build_response_surfaces(
            pad_configs,
            online={"MPB-28": True, "MPJ-29": False},
            current={"MPB-28": ("13", "E")},  # NOT in the candidate lists
            plant_model=PLANT,
            p_grid=[2600.0, 2800.0],
            nozzles=["12"],
            throats=["B"],
            p0=2792.0,
            c_pad_pf_psi=3400.0,
            progress=lambda i, n, p: seen.append((i, n)),
        )
        return surf, seen

    def test_one_batch_per_grid_point_and_progress(self, monkeypatch):
        _surf, seen = self._run(monkeypatch)
        assert len(FakeOptimizer.instances) == 2
        assert seen == [(1, 2), (2, 2)]

    def test_current_size_unioned_into_the_candidates(self, monkeypatch):
        """The baseline must always exist, so the current pump is added to the
        sweep even when the engineer's candidate list omits it."""
        _surf, _ = self._run(monkeypatch)
        opt = FakeOptimizer.instances[0]
        assert "13" in opt.nozzles and "E" in opt.throats

    def test_arrays_vary_with_the_grid_pressure(self, monkeypatch):
        surf, _ = self._run(monkeypatch)
        ws = surf.wells["MPB-28"]
        oil = ws.options["12B"]["oil"]
        assert len(oil) == 2 and oil[0] != oil[1]

    def test_never_converged_option_is_dropped(self, monkeypatch):
        surf, _ = self._run(monkeypatch)
        assert all("13" != o["nozzle"] for o in surf.wells["MPB-28"].options.values())

    def test_online_and_current_recorded(self, monkeypatch):
        surf, _ = self._run(monkeypatch)
        assert surf.wells["MPB-28"].online is True
        assert surf.wells["MPB-28"].current == "13E"
        assert surf.wells["MPJ-29"].online is False
        assert surf.baseline_choices()["MPJ-29"] == OFF


class TestFleetSignature:
    """The response-surface memo key (cfp_pad_page._fleet_signature).

    C-Pad PF was missing from it: editing the "C-Pad booster PF" input left
    the key identical, so the memo served surfaces built at the OLD pressure.
    """

    def _sig(self, **over):
        from woffl.gui.cfp_pad_page import _fleet_signature

        kw = dict(
            pad_configs={},  # empty: keeps store_for out of the test
            online={"MPB-28": True, "MPJ-29": False},
            current={"MPB-28": ("13", "E")},
            nozzles=["12", "13"],
            throats=["A", "B"],
            p0=2792.0,
            c_pad_pf=3400.0,
            measured_pf={"B": 3120.0, "J": 3050.0},
        )
        kw.update(over)
        return _fleet_signature(**kw)

    def test_stable_for_identical_inputs(self):
        assert self._sig() == self._sig()

    def test_c_pad_pf_changes_the_signature(self):
        assert self._sig(c_pad_pf=3425.0) != self._sig()

    def test_measured_pad_pf_changes_the_signature(self):
        """It sets delivered PF per pad (cfp_optimize.delivered_by_pad) and its
        ttl=3600 memo can refresh mid-session."""
        assert self._sig(measured_pf={"B": 3130.0, "J": 3050.0}) != self._sig()
        assert self._sig(measured_pf={"B": 3120.0}) != self._sig()

    def test_insensitive_to_input_ordering(self):
        assert self._sig(
            nozzles=["13", "12"],
            throats=["B", "A"],
            measured_pf={"J": 3050.0, "B": 3120.0},
        ) == self._sig()

    def test_float_noise_does_not_thrash_the_memo(self):
        assert self._sig(c_pad_pf=3400.0 + 1e-9, p0=2792.0 + 1e-9) == self._sig()

    def test_other_physics_inputs_still_key_the_memo(self):
        assert self._sig(p0=2800.0) != self._sig()
        assert self._sig(nozzles=["12"]) != self._sig()
        assert self._sig(throats=["A"]) != self._sig()
        assert self._sig(online={"MPB-28": False, "MPJ-29": False}) != self._sig()
        assert self._sig(current={"MPB-28": ("12", "B")}) != self._sig()

    def test_slope_is_not_an_argument(self):
        """slope only reaches cmv.anchor, which re-runs outside the memo, so it
        must NOT be a parameter here — pinned so it is never added by reflex."""
        import inspect

        from woffl.gui.cfp_pad_page import _fleet_signature

        assert "slope" not in inspect.signature(_fleet_signature).parameters