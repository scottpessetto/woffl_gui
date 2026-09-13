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

from itertools import product

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

    def test_non_converged_edge_is_unavailable_not_held(self):
        """OPT-A1 (review 2026-09-01): a failed solve is an honest gap. The
        old behaviour held the nearest converged value, so a pump that only
        solved at high PF was scored at low PF with its high-PF oil."""
        ws = _surfaces().wells["RESP"]
        ws.options["12B"]["oil"][0] = None  # failed at 2,500
        ws.options["12B"]["water"][0] = None
        assert option_at(ws, "12B", 2500.0) is None
        assert option_at(ws, "12B", 2550.0) is None  # inside the failed bracket
        assert option_at(ws, "12B", 2600.0) == pytest.approx((200.0, 5000.0))
        assert not cm.is_available(ws, "12B", 2500.0)
        assert cm.is_available(ws, "12B", 2650.0)

    def test_interior_gap_is_not_interpolated_across(self):
        ws = _surfaces().wells["RESP"]
        ws.options["12B"]["oil"][2] = None  # failed at 2,700 only
        ws.options["12B"]["water"][2] = None
        assert option_at(ws, "12B", 2650.0) is None
        assert option_at(ws, "12B", 2750.0) is None
        assert option_at(ws, "12B", 2600.0) is not None
        assert option_at(ws, "12B", 2800.0) is not None

    def test_settle_marks_infeasible_states(self):
        """A move that lands the fleet at a pressure where a chosen pump has no
        converged surface is infeasible - never a candidate plan."""
        s = _surfaces()
        plant = _plant(s)
        # BOL1 only converges at >= 2,800; bringing it on (4,000 BPD at
        # 15 psi/kBPD) drops P to 2,740, inside the failed 2,700-2,800 bracket.
        s.wells["BOL1"].options["11B"]["oil"][:3] = [None, None, None]
        s.wells["BOL1"].options["11B"]["water"][:3] = [None, None, None]
        state = settle({**s.baseline_choices(), "BOL1": "11B"}, s, plant)
        assert not state["feasible"]
        assert state["infeasible"] == ["BOL1"]
        assert state["oil"] == float("-inf")
        singles = rank_single_moves(s, plant)
        assert not any(m["well"] == "BOL1" for m in singles)
        assert all(m["fleet_oil_delta"] > float("-inf") for m in singles)

    def test_anchor_refuses_unanchorable_online_well(self):
        """OPT-A9: an online well whose CURRENT pump never converged at P0
        must not be silently treated as idle (its own BOL would then read
        as a gain)."""
        s = _surfaces()
        s.wells["PIG"].options["12B"]["oil"] = [None] * len(GRID)
        s.wells["PIG"].options["12B"]["water"] = [None] * len(GRID)
        with pytest.raises(ValueError, match="PIG"):
            _plant(s)

    def test_option_with_no_converged_points_is_not_a_choice(self):
        ws = _surfaces().wells["RESP"]
        ws.options["10A"]["oil"] = [None] * len(GRID)
        assert "10A" not in ws.labels()
        assert SI in ws.choice_labels()


# ── the anchor: measured state, no exogenous anything ───────────────────────


class TestAnchor:
    def test_missing_current_label_never_becomes_an_idle_baseline(self):
        s = _surfaces()
        del s.wells["RESP"].options["12B"]
        assert s.baseline_choices()["RESP"] == "12B"
        with pytest.raises(ValueError, match="RESP"):
            _plant(s)

    def test_anchor_above_trip_margin_is_rejected_without_moving_baseline(self):
        s = _surfaces()
        s.p0 = 2890.
        s.p_grid = [*GRID, 2900.]
        with pytest.raises(ValueError, match="trip-minus-margin"):
            _plant(s)

    def test_anchor_outside_grid_is_rejected_explicitly(self):
        s = _surfaces()
        s.p0 = 2400.
        with pytest.raises(ValueError, match="inside the response grid"):
            _plant(s)

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
    def test_bring_online_offset_can_be_feasible_when_bring_online_alone_is_not(self):
        s = Surfaces(GRID, P0, {
            "Offset": WellSurface("Offset", "B", True, "A", {"A": _opt(100., 4000.)}),
            "New": WellSurface("New", "B", False, None, {"B": _opt(300., 4000.)}),
        })
        s.wells["New"].options["B"]["oil"][:3] = [None]*3
        s.wells["New"].options["B"]["water"][:3] = [None]*3
        pairs = pair_moves(s, _plant(s))
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["pressure_after"] == P0 and pair["fleet_oil_delta"] == 200.
        assert pair["own_water_delta"] == 0 and pair["own_oil_delta"] == 200.
        assert pair["bring_on"]["standalone_feasible"] is False
        assert pair["bring_on"]["fleet_oil_delta"] is None
        assert pair["bring_on"]["standalone_domain_reason"] == "missing_pump_response"

    def test_pair_search_budget_is_reported(self, monkeypatch):
        monkeypatch.setattr(cm, "MAX_PAIR_EVALUATIONS", 1)
        s = _surfaces()
        out = moves_summary(s, _plant(s))
        assert out["search_scope"]["pair_evaluated"] == 1
        assert out["search_scope"]["pair_combinations"] > 1
        assert out["search_scope"]["pair_search_complete"] is False

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


def test_added_load_below_grid_is_infeasible_with_unclipped_residual():
    s = Surfaces(GRID, P0, {
        "Existing": WellSurface("Existing", "B", True, "A", {"A": _opt(100., 1000.)}),
        "New": WellSurface("New", "B", False, None, {"B": _opt(500., 30000.)}),
    })
    plant = _plant(s)
    assert plant.pressure_at(31000.) == (2350., False)
    state = settle({"Existing": "A", "New": "B"}, s, plant)
    assert not state["feasible"] and not state["converged"]
    assert state["raw_pressure_psi"] == 2350.
    assert state["pressure_residual_psi"] == -150.
    assert state["domain_reason"] == "required_pressure_below_response_grid"
    out = moves_summary(s, plant, required_wells={"New"})
    assert out["plan"] is None and out["plan_gain"] is None
    assert out["plan_status"] == "no_feasible_plan"
    assert out["search_scope"]["rejected_choices_by_reason"]["required_pressure_below_response_grid"] > 0


def test_required_future_stays_on_even_when_gain_is_negative():
    s = Surfaces(GRID, P0, {
        "Existing": WellSurface("Existing", "B", True, "A", {"A": _opt(1000., 1000., 2.)}),
        "New": WellSurface("New", "B", False, None, {"B": _opt(10., 4000.)}),
    })
    out = moves_summary(s, _plant(s), required_wells={"New"})
    assert out["today"]["oil"] == 1000. and out["baseline"]["New"] == OFF
    assert out["plan"]["choices"]["New"] == "B"
    assert out["plan_gain"] == pytest.approx(-110.)
    assert out["baseline_meets_requirements"] is False
    assert moves_summary(s, _plant(s))["plan_gain"] == 0.


def test_required_existing_well_can_resize_but_cannot_shut_in():
    s = _surfaces()
    s.wells["PIG"].options["10A"] = _opt(4., 500.)
    out = moves_summary(s, _plant(s), required_wells={"PIG"})
    assert out["plan"]["choices"]["PIG"] == "10A"
    assert not any(m["well"] == "PIG" and m["to"] == SI for m in out["singles"])
    assert out["baseline_meets_requirements"] is True


def test_unknown_required_well_is_not_silently_dropped():
    s = _surfaces()
    with pytest.raises(ValueError, match="Missing"):
        moves_summary(s, _plant(s), required_wells={"Missing"})


def test_required_well_with_no_pump_response_has_no_plan():
    s = _surfaces()
    s.wells["BOL1"].options = {}
    out = moves_summary(s, _plant(s), required_wells={"BOL1"})
    assert out["plan"] is None and out["today"]["oil"] == 405.
    assert out["search_scope"]["combinations"] == 0
    assert out["search_scope"]["global_optimum_on_surfaces"] is True


def test_bounded_feasibility_seed_keeps_multiple_required_future_wells(monkeypatch):
    monkeypatch.setattr(cm, "MAX_EXACT_COMBINATIONS", 0)
    s = _surfaces()
    s.wells["BOL2"] = WellSurface("BOL2", "B", False, None, {"A": _opt(50., 1000.)})
    out = moves_summary(s, _plant(s), required_wells={"BOL1", "BOL2"})
    assert all(out["plan"]["choices"][w] not in (SI, OFF) for w in ("BOL1", "BOL2"))
    assert out["search_scope"]["global_optimum_on_surfaces"] is False
    assert out["search_scope"]["method"] == "lambda_moves_and_bounded_neighborhood"


def test_required_new_well_can_need_two_offsets_to_fit_the_pressure_domain():
    s = Surfaces(GRID, P0, {
        "Old1": WellSurface("Old1", "B", True, "A", {"A": _opt(100., 10000.)}),
        "Old2": WellSurface("Old2", "B", True, "A", {"A": _opt(100., 10000.)}),
        "New": WellSurface("New", "B", False, None, {"B": _opt(500., 35000.)}),
    })
    out = moves_summary(s, _plant(s), required_wells={"New"})
    assert out["plan"]["choices"] == {"Old1": SI, "Old2": SI, "New": "B"}
    assert out["plan"]["pressure"] == 2575. and out["plan_gain"] == 300.
    assert out["singles"] == [] and out["pairs"] == []


_AUDIT_CASES = {
    "below_do_nothing": [
        [(400, 8000, .2), (525, 17000, .315), (200, 2500, .16)],
        [(675, 18000, .54), (200, 17500, .02), (475, 13000, .285)],
        [(525, 17500, .105), (250, 2000, .1), (175, 4500, .1575)],
    ],
    "worse_than_own_single": [
        [(625, 1500, .5625), (650, 7000, .585), (600, 16000, .12)],
        [(575, 7500, .2875), (550, 2500, 0.), (700, 18000, .63)],
        [(250, 12000, .025), (700, 8000, .14), (425, 14500, .0425)],
    ],
    "missed_two_resize_optimum": [
        [(525, 3000, .2625), (400, 15000, 0.), (275, 10500, .2475)],
        [(500, 17500, 0.), (400, 9000, .32), (700, 17000, .49)],
        [(550, 13500, .11), (400, 1500, 0.), (500, 9000, .5)],
    ],
}


def _audit_surfaces(case):
    return Surfaces(GRID, P0, {
        w: WellSurface(w, "B", True, "current", {
            lab: _opt(*values) for lab, values in zip(("current", "alt1", "alt2"), rows)
        }) for w, rows in zip(("A", "B", "C"), _AUDIT_CASES[case])
    })


def _analytic_best(case):
    """Independent enumeration using the affine equations, not settle()."""
    rows = _AUDIT_CASES[case]
    baseline_water = sum(w[0][1] for w in rows)
    values = []
    for picks in product(*(w+[(0., 0., 0.)] for w in rows)):
        pressure = min(2880., P0+15.*(baseline_water-sum(p[1] for p in picks))/1000.)
        if pressure < min(GRID):
            continue
        values.append(sum(oil+slope*(pressure-P0) for oil, _water, slope in picks))
    return max(values)


@pytest.mark.parametrize("case", list(_AUDIT_CASES))
def test_small_exact_search_matches_independent_enumeration(case):
    s = _audit_surfaces(case)
    out = moves_summary(s, _plant(s))
    assert out["plan"]["oil"] == pytest.approx(_analytic_best(case))
    assert out["plan"]["oil"] >= out["today"]["oil"]
    assert out["plan_gain"] >= max(m["fleet_oil_delta"] for m in out["singles"])-1e-9
    assert out["search_scope"]["global_optimum_on_surfaces"] is True
    assert out["search_scope"]["direct_solver_validated"] is False


@pytest.mark.parametrize("case", ["below_do_nothing", "worse_than_own_single"])
def test_evaluated_baseline_and_singles_cannot_lose_even_without_refinement(monkeypatch, case):
    monkeypatch.setattr(cm, "MAX_EXACT_COMBINATIONS", 0)
    monkeypatch.setattr(cm, "MAX_NEIGHBOR_EVALUATIONS", 0)
    s = _audit_surfaces(case)
    out = moves_summary(s, _plant(s))
    assert out["plan_gain"] >= 0
    assert out["plan_gain"] >= max(m["fleet_oil_delta"] for m in out["singles"])-1e-9
    assert out["search_scope"]["neighborhood_evaluated"] == 0
    assert out["search_scope"]["global_optimum_on_surfaces"] is False
    if case == "worse_than_own_single":
        assert out["plan"]["lam"] is None


def test_bounded_neighborhood_recovers_a_missed_resize_combination(monkeypatch):
    monkeypatch.setattr(cm, "MAX_EXACT_COMBINATIONS", 0)
    s = _audit_surfaces("missed_two_resize_optimum")
    out = moves_summary(s, _plant(s))
    assert out["plan"]["oil"] == pytest.approx(1818.9375)
    assert out["search_scope"]["global_optimum_on_surfaces"] is False
    assert out["search_scope"]["neighborhood_evaluated"] <= cm.MAX_NEIGHBOR_EVALUATIONS


def test_neighborhood_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(cm, "MAX_EXACT_COMBINATIONS", 0)
    monkeypatch.setattr(cm, "MAX_NEIGHBOR_EVALUATIONS", 1)
    s = _surfaces()
    out = moves_summary(s, _plant(s))
    assert out["search_scope"]["neighborhood_evaluated"] == 1
    assert out["search_scope"]["neighborhood_budget_exhausted"] is True


def test_exhausted_choices_do_not_claim_global_optimum_for_multiple_pressure_branches():
    s = _surfaces()
    s.wells["PIG"].options["12B"]["water"][1] = 2100.
    out = moves_summary(s, _plant(s))
    assert out["search_scope"]["all_choices_evaluated"] is True
    assert out["search_scope"]["unique_pressure_response"] is False
    assert out["search_scope"]["global_optimum_on_surfaces"] is False


def test_identical_clean_replacement_keeps_installed_pump_on_a_tie():
    s = Surfaces(GRID, P0, {"Well": WellSurface("Well", "B", True, "13C", {
        "13C (clean)": _opt(100., 1000.), "13C": _opt(100., 1000.),
    })})
    out = moves_summary(s, _plant(s))
    assert out["plan"]["choices"]["Well"] == "13C" and out["plan"]["n_changes"] == 0


def test_action_deltas_use_baseline_and_plan_pressures_when_old_pump_is_unavailable():
    s = Surfaces(GRID, P0, {"Well": WellSurface("Well", "B", True, "A", {
        "A": _opt(100., 1000.), "B": _opt(200., 2000.),
    })})
    s.wells["Well"].options["A"]["oil"][:3] = [None]*3
    s.wells["Well"].options["A"]["water"][:3] = [None]*3
    out = moves_summary(s, _plant(s))
    assert out["plan"]["pressure"] == 2785.
    action = out["plan"]["actions"][0]
    assert action["own_oil_delta"] == 100. and action["own_water_delta"] == 1000.
    move = next(m for m in out["singles"] if m["to"] == "B")
    assert move["own_oil_delta"] == 100. and move["own_water_delta"] == 1000.


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
        import woffl.assembly.parallelism as common_mod

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


