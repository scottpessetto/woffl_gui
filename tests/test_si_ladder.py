"""Shut-in ladder — is buying CFP power-fluid pressure worth the oil it costs?

Protects the parts that decide the recommendation:

* **which water counts** — POPS pads send formation water only, everyone else
  sends the lot (Scott, 2026-07-29). Getting this wrong mis-sizes the plant load
  and so the whole pressure lever.
* **the three rankings genuinely differ** — raw WC condemns high-WC strippers
  that free almost no capacity; volume condemns big-water wells that make real
  oil. Only oil-per-plant-barrel tracks what actually pays.
* **never recommend a shut-in that gains nothing** — ties go to fewer SIs, and
  with no oil response the curve must fall monotonically.
"""

import pytest

from woffl.gui.cfp_pad_plant import PLANT
from woffl.gui.si_ladder import (
    POPS_PADS,
    LadderWell,
    best_rung,
    build_ladder,
    ladder_summary,
    rank_wells,
    wc_threshold,
)

BASELINE_PSI = 2792.0


def _w(name, oil, form, lift=0.0, sens=0.0):
    return LadderWell(
        well=name,
        oil_bopd=oil,
        form_wat_bwpd=form,
        lift_wat_bwpd=lift,
        oil_sens_frac_per_psi=sens,
    )


# ── which water reaches the plant ───────────────────────────────────────────


class TestPlantWater:
    def test_pops_pad_sends_formation_only(self):
        """S/H/I/M/F/E handle lift on-pad."""
        w = _w("MPS-045", 380.0, 2944.0, lift=35000.0)
        assert w.pad == "S" and "S" in POPS_PADS
        assert w.plant_water_bwpd() == pytest.approx(2944.0)
        assert w.total_wat_bwpd == pytest.approx(37944.0)

    def test_non_pops_pad_sends_everything(self):
        w = _w("MPB-028", 283.0, 1500.0, lift=3044.0)
        assert w.pad == "B" and "B" not in POPS_PADS
        assert w.plant_water_bwpd() == pytest.approx(4544.0)

    def test_wc_is_on_the_produced_stream_not_the_plant_stream(self):
        """A POPS well's WC reflects what it produces, even though only part of
        that water reaches the plant."""
        w = _w("MPS-045", 100.0, 400.0, lift=500.0)
        assert w.total_wc == pytest.approx(900.0 / 1000.0)

    def test_zero_water_well_never_ranks_worst(self):
        w = _w("MPX-01", 500.0, 0.0)
        assert w.oil_per_plant_barrel() == float("inf")


# ── rankings disagree, and that is the point ────────────────────────────────


class TestRankings:
    def setup_method(self):
        # A stripper with awful WC that frees almost nothing — the real H-019
        # shape: 94.9% total WC, but H is a POPS pad so only its 187 BWPD of
        # FORMATION water reaches the plant. Its 2,500 BWPD of lift is handled
        # on-pad and is invisible to the plant load.
        self.stripper = _w("MPH-019", 145.0, 187.0, lift=2500.0)
        # a big water maker that still makes real oil
        self.bigwater = _w("MPB-028", 283.0, 4544.0)
        # essentially dead: no oil, lots of water — worst on any sane basis
        self.dead = _w("MPF-093", 1.0, 1980.0)
        self.wells = [self.bigwater, self.stripper, self.dead]

    def test_marginal_puts_the_dead_well_first(self):
        assert rank_wells(self.wells, "marginal")[0].well == "MPF-093"

    def test_wc_ranking_condemns_the_stripper_above_the_big_water_well(self):
        order = [w.well for w in rank_wells(self.wells, "wc")]
        assert order.index("MPH-019") < order.index("MPB-028")

    def test_marginal_ranking_spares_the_stripper(self):
        """It frees only 187 BWPD, so it costs little to keep — the opposite of
        what raw WC says."""
        order = [w.well for w in rank_wells(self.wells, "marginal")]
        assert order.index("MPB-028") < order.index("MPH-019")

    def test_volume_ranking_targets_the_big_water_well_first(self):
        assert rank_wells(self.wells, "volume")[0].well == "MPB-028"

    def test_bad_ranking_rejected(self):
        with pytest.raises(ValueError):
            rank_wells(self.wells, "nonsense")


# ── the ladder ──────────────────────────────────────────────────────────────


def _fleet(sens=0.0):
    return [
        _w("MPF-093", 1.0, 1980.0, sens=sens),      # dead
        _w("MPC-045", 69.0, 1958.0, sens=sens),     # near-dead
        _w("MPB-028", 283.0, 4544.0, sens=sens),    # real oil
        _w("MPJ-029", 700.0, 2000.0, sens=sens),    # good well
    ]


class TestLadder:
    def test_rung_zero_is_the_baseline(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        assert rungs[0].k == 0
        assert rungs[0].shut_in == []
        assert rungs[0].oil_delta_bopd == 0.0
        assert rungs[0].marginal_well is None

    def test_shutting_in_reduces_the_plant_load(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        loads = [r.plant_load_bwpd for r in rungs]
        assert all(a > b for a, b in zip(loads, loads[1:]))

    def test_shedding_water_raises_the_discharge(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        ps = [r.discharge_psi for r in rungs if r.discharge_psi]
        assert all(a <= b for a, b in zip(ps, ps[1:]))

    def test_discharge_never_passes_the_trip(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=0.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        for r in rungs:
            if r.discharge_psi:
                assert r.discharge_psi <= PLANT.max_header_psi

    def test_no_oil_response_means_shutting_in_only_loses(self):
        """The null result, and the right default to argue against: if oil does
        not respond to PF pressure, every shut-in is a straight loss."""
        rungs = build_ladder(
            _fleet(sens=0.0), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        oils = [r.total_oil_bopd for r in rungs]
        assert all(a > b for a, b in zip(oils, oils[1:]))
        assert best_rung(rungs).k == 0
        assert wc_threshold(rungs) is None
        assert ladder_summary(rungs)["recommend_si"] is False

    def test_a_dead_well_is_worth_shutting_in_when_oil_responds(self):
        """MPF-093 makes 1 BOPD and dumps 1,980 BWPD — with any real pressure
        response the field gains more than it loses."""
        rungs = build_ladder(
            _fleet(sens=0.00025), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        best = best_rung(rungs)
        assert best.k >= 1
        assert "MPF-093" in best.shut_in
        assert ladder_summary(rungs)["oil_gain_bopd"] > 0

    def test_ties_prefer_fewer_shut_ins(self):
        """Never shut a well in for nothing."""
        wells = [_w("MPB-01", 100.0, 0.0), _w("MPB-02", 100.0, 0.0)]
        rungs = build_ladder(
            wells, PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        # zero plant water each, so the load never moves and no rung can help
        assert best_rung(rungs).k == 0

    def test_threshold_is_the_wc_of_the_last_well_shut_in(self):
        rungs = build_ladder(
            _fleet(sens=0.00025), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        best = best_rung(rungs)
        assert wc_threshold(rungs) == pytest.approx(best.marginal_well_wc)

    def test_pad_filter_excludes_water_and_oil(self):
        full = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        no_b = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI, pads_included=["F", "C", "J"],
        )
        assert full[0].plant_load_bwpd > no_b[0].plant_load_bwpd
        assert full[0].total_oil_bopd > no_b[0].total_oil_bopd
        assert all("MPB-028" not in r.shut_in for r in no_b)

    def test_over_capacity_rung_is_infeasible_not_a_crash(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=400000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        assert rungs and not any(r.feasible for r in rungs)
        assert all(r.note for r in rungs)
        assert best_rung(rungs) is None
        assert ladder_summary(rungs) == {"feasible": False}

    def test_max_rungs_caps_the_walk(self):
        rungs = build_ladder(
            _fleet(), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI, max_rungs=2,
        )
        assert [r.k for r in rungs] == [0, 1, 2]

    def test_empty_input_gives_no_rungs(self):
        assert build_ladder(
            [], PLANT, exogenous_bwpd=0.0, baseline_discharge_psi=BASELINE_PSI
        ) == []

    def test_summary_reports_water_shed_and_pressure_gain(self):
        rungs = build_ladder(
            _fleet(sens=0.00025), PLANT, exogenous_bwpd=100000.0,
            baseline_discharge_psi=BASELINE_PSI,
        )
        s = ladder_summary(rungs)
        assert s["water_shed_bwpd"] > 0
        assert s["pressure_gain_psi"] > 0
        assert s["best_k"] == len(s["shut_in"])
