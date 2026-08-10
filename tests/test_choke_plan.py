"""run_choke_optimization - the no-JPCO choke / shut-in plan.

Pure allocation tests: ``_model_at_forced_header`` is monkeypatched with
hand-built (oil, pf) curves so no solver runs, and the plant is a stub
free-pressure booster with a controllable PF budget. The physics of pricing
a well at a delivered pressure is covered by the solver suite; these tests
pin the DECISION layer - the equal-slope trim walk, the header sweep, the
test-rate fallback chain, and the meta/rows contract the web app renders.
"""

from types import SimpleNamespace

import pytest

from woffl.gui import pad_optimize

# Ladder used throughout: n_levels=3 over the stub's 2000-3000 psi window.
LEVELS = [2000.0, 2500.0, 3000.0]


class StubPlant:
    """Minimal free-pressure plant: flat budget, fixed window."""

    coupling = "free_pressure"
    max_header_psi = 3500.0
    infeasible_sweep_msg = "no feasible header"

    def __init__(self, budget):
        self.budget = budget

    def pressure_window(self, n_pumps=None):
        return (LEVELS[0], LEVELS[-1])

    def budget_at_pressure(self, pressure, n_pumps=None):
        return self.budget

    def warm_start_psi(self, n_pumps=None):
        return LEVELS[-1]

    def header_at_flow(self, q_total, n_pumps=None):
        return LEVELS[-1]

    def suction_psi(self):
        return 1400.0

    def flow_window(self, n_pumps=None):
        return (1000.0, 60000.0)

    def flags(self, q_total, n_pumps=None):
        return {
            "in_range": True,
            "recirc": q_total < self.flow_window(n_pumps)[0],
            "over_capacity": False,
        }


def _configs(names):
    return [SimpleNamespace(well_name=n) for n in names]


def _patch_grid(monkeypatch, curves, calls=None):
    """curves: {well: {level_psi: (oil, pf[, psu[, sonic]])}}; missing level = no solution."""

    def fake(well_configs, header_psi, current_choices):
        if calls is not None:
            calls.append(header_psi)
        out = {}
        for wc in well_configs:
            out[wc.well_name] = curves.get(wc.well_name, {}).get(header_psi)
        return out

    monkeypatch.setattr(pad_optimize, "_model_at_forced_header", fake)


CHOICES = {"A": ("12", "B"), "B": ("11", "C"), "C": ("13", "B"), "D": ("10", "A")}


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------


def test_slack_budget_runs_everything_full_at_the_best_header(monkeypatch):
    calls = []
    _patch_grid(
        monkeypatch,
        {
            "A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)},
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
        calls,
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "B"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    # one pricing pass per ladder level plus the today anchor
    assert calls == LEVELS + [3000.0]
    assert meta["header_psi"] == 3000.0  # oil rises with pressure
    assert meta["total_oil_bopd"] == pytest.approx(200.0)
    assert meta["total_pf_bpd"] == pytest.approx(2000.0)
    assert meta["pf_slack"] is True
    assert meta["lambda_bopd_per_bpd"] is None  # no trim was needed
    assert len(meta["sweep"]) == 3
    assert all(r["action"] == "full" for r in rows)
    assert all(r["delivered_psi"] == 3000.0 for r in rows)


def test_binding_budget_chokes_the_flattest_well_first(monkeypatch):
    # A's first trim gives up 5 BOPD per 100 BPD (slope 0.05); B's costs
    # 0.20. Freeing 100 BPD must choke A one step and leave B alone. A's
    # curves carry psu so the IPR-landing fields ride through.
    _patch_grid(
        monkeypatch,
        {
            "A": {
                2000.0: (90.0, 800.0, 900.0),
                2500.0: (95.0, 900.0, 850.0),
                3000.0: (100.0, 1000.0, 800.0),
            },
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
    )
    configs = [SimpleNamespace(well_name="A", res_pres=1700.0), SimpleNamespace(well_name="B")]
    rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(1900.0), 2, CHOICES, {}, n_levels=3
    )
    # 3000 psi with A choked (195 BOPD) beats full-open 2500 (175) and 2000 (150)
    assert meta["header_psi"] == 3000.0
    assert meta["total_oil_bopd"] == pytest.approx(195.0)
    assert meta["total_pf_bpd"] == pytest.approx(1900.0)
    assert meta["lambda_bopd_per_bpd"] == pytest.approx(0.05)
    assert meta["n_choked"] == 1 and meta["n_full"] == 1 and meta["n_shut"] == 0

    by_well = {r["well"]: r for r in rows}
    a, b = by_well["A"], by_well["B"]
    assert a["action"] == "choke"
    assert a["delivered_psi"] == 2500.0
    assert a["choke_dp_psi"] == pytest.approx(500.0)
    assert a["d_pf_vs_full"] == pytest.approx(-100.0)
    assert a["d_oil_vs_full"] == pytest.approx(-5.0)
    assert b["action"] == "full"
    # action-first ordering: the choked well leads the table
    assert rows[0]["well"] == "A"
    # psu and the full-open reference ride through for the IPR landing table
    assert a["psu"] == pytest.approx(850.0)
    assert a["psu_full"] == pytest.approx(800.0)
    assert a["res_pres"] == pytest.approx(1700.0)
    assert a["oil_full"] == pytest.approx(100.0)
    assert a["pf_full"] == pytest.approx(1000.0)
    assert a["delivered_full_psi"] == 3000.0
    assert b["psu"] is None  # B's grid carries no psu


def test_low_value_well_is_shut_in_not_pro_rata_choked(monkeypatch):
    # C makes almost no oil for a huge PF draw - the walk must shut it in
    # completely and leave the good well untouched (never pro-rata).
    _patch_grid(
        monkeypatch,
        {
            "A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)},
            "C": {2000.0: (1.0, 1500.0), 2500.0: (1.5, 1750.0), 3000.0: (2.0, 2000.0)},
        },
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "C"]), StubPlant(1000.0), 2, CHOICES, {}, n_levels=3
    )
    by_well = {r["well"]: r for r in rows}
    assert by_well["C"]["action"] == "shut"
    assert by_well["C"]["pf"] == 0.0
    assert by_well["A"]["action"] == "full"
    assert meta["total_oil_bopd"] == pytest.approx(100.0)
    assert rows[0]["well"] == "C"  # shut-ins lead the table


def test_unmodelable_well_holds_measured_rates(monkeypatch):
    # D never solves; with test rates it is HELD at them (and its PF still
    # counts against the budget), projected oil = the measured oil.
    _patch_grid(
        monkeypatch,
        {"A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)}},
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "D"]),
        StubPlant(100000.0),
        2,
        CHOICES,
        {"D": (50.0, 600.0)},
        n_levels=3,
    )
    d = {r["well"]: r for r in rows}["D"]
    assert d["basis"] == "test"
    assert d["action"] == "hold"
    assert d["oil"] == pytest.approx(50.0)
    assert d["pf"] == pytest.approx(600.0)
    assert d["projected_oil"] == pytest.approx(50.0)
    assert meta["total_pf_bpd"] == pytest.approx(1600.0)
    assert meta["n_held"] == 1


def test_unmodelable_well_without_tests_is_excluded(monkeypatch):
    _patch_grid(
        monkeypatch,
        {"A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)}},
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "D"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    d = {r["well"]: r for r in rows}["D"]
    assert d["basis"] == "none"
    assert d["action"] == "excluded"
    assert d["oil"] == 0.0 and d["pf"] == 0.0
    assert meta["n_excluded"] == 1
    assert meta["total_oil_bopd"] == pytest.approx(100.0)


def test_dominant_lower_pressure_point_is_a_choke_with_free_oil(monkeypatch):
    # A's 2500 psi point makes MORE oil for LESS PF than its 3000 psi point
    # (a pump past its sonic knee). The plan must tell the operator to CHOKE
    # to 2500 even with a slack budget, and the deltas - taken against the
    # RAW full-open point - must read as a gain, not zero.
    _patch_grid(
        monkeypatch,
        {
            "A": {2500.0: (105.0, 900.0), 3000.0: (100.0, 1000.0)},
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "B"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert meta["header_psi"] == 3000.0  # B needs the full header
    a = {r["well"]: r for r in rows}["A"]
    assert a["action"] == "choke"
    assert a["delivered_psi"] == 2500.0
    assert a["d_oil_vs_full"] == pytest.approx(+5.0)
    assert a["d_pf_vs_full"] == pytest.approx(-100.0)
    assert meta["n_choked"] == 1


def test_projection_anchors_on_measured_oil(monkeypatch):
    # Model says 100 -> 95 (chosen/today ratio 0.95); the well measures 200
    # BOPD, so the projection is 190 - model bias cancels in the ratio.
    _patch_grid(
        monkeypatch,
        {
            "A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)},
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "B"]),
        StubPlant(1900.0),
        2,
        CHOICES,
        {"A": (200.0, 950.0), "B": (100.0, 950.0)},
        n_levels=3,
    )
    a = {r["well"]: r for r in rows}["A"]
    assert a["action"] == "choke"
    assert a["projected_oil"] == pytest.approx(200.0 * 95.0 / 100.0)
    # meta projected delta = sum(projected - measured) over anchored rows
    assert meta["projected_d_oil_bopd"] == pytest.approx((190.0 - 200.0) + 0.0)


def test_fixed_curve_plant_is_rejected():
    plant = StubPlant(1000.0)
    plant.coupling = "fixed_curve"
    with pytest.raises(ValueError, match="free-pressure"):
        pad_optimize.run_choke_optimization(
            _configs(["A"]), plant, None, CHOICES, {}, n_levels=3
        )


def test_meta_carries_the_chart_contract(monkeypatch):
    _patch_grid(
        monkeypatch,
        {"A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)}},
    )
    _rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A"]), StubPlant(100000.0), 3, CHOICES, {}, n_levels=3
    )
    for key in (
        "mode", "n_pumps", "header_psi", "total_pf_bpd", "total_oil_bopd",
        "frontier_cap_bpd", "pf_slack", "lambda_bopd_per_bpd",
        "header_today_psi", "projected_d_oil_bopd", "suction_psi",
        "min_total_flow", "converged", "history", "sweep",
        "in_range", "recirc", "over_capacity",
    ):
        assert key in meta, key
    assert meta["mode"] == "choke"
    assert meta["n_pumps"] == 3
    assert meta["sweep"][0].keys() == {"header_psi", "total_pf_bpd", "total_oil_bopd"}


# ---------------------------------------------------------------------------
# Staircase pruning
# ---------------------------------------------------------------------------


def test_choke_frontier_drops_dominated_and_duplicate_pf_options():
    pts = [
        (3000.0, 100.0, 1000.0, 800.0),
        (2500.0, 101.0, 900.0, 850.0),  # dominates the 3000 psi point
        (2400.0, 40.0, 900.0, 950.0),  # same PF, less oil: dropped
        (2000.0, 90.0, 950.0, 900.0),  # more PF than 900 for less oil: dropped
        (None, 0.0, 0.0, None),
    ]
    kept = pad_optimize._choke_frontier(pts)
    assert kept == [(2500.0, 101.0, 900.0, 850.0), (None, 0.0, 0.0, None)]
    # staircase invariant: oil and pf strictly decrease down the list
    for (_, o0, f0, _), (_, o1, f1, _) in zip(kept, kept[1:]):
        assert o0 > o1 and f0 > f1



# ---------------------------------------------------------------------------
# IPR landing curve
# ---------------------------------------------------------------------------


def _ipr_config(name, **kw):
    """Well config carrying the Vogel anchor fields the curve builder reads."""
    return SimpleNamespace(well_name=name, **kw)


GRID_A = {"A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)}}


def test_model_row_carries_a_25_point_vogel_curve(monkeypatch):
    _patch_grid(monkeypatch, GRID_A)
    cfg = _ipr_config("A", qwf=1000.0, pwf=1500.0, res_pres=3000.0, form_wc=0.5)
    rows, _meta = pad_optimize.run_choke_optimization(
        [cfg], StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    a = rows[0]
    assert a["basis"] == "model"
    curve = a["ipr_curve"]
    assert isinstance(curve, list) and len(curve) == 25
    assert all(len(pt) == 2 for pt in curve)
    assert curve[0] == [0.0, 3000.0]  # pwf = res_pres, no drawdown
    assert curve[-1][1] == 0.0  # last point at pwf = 0 (oil = vogel qmax)
    # Vogel is monotone: oil strictly rises as pwf falls
    for (o0, p0), (o1, p1) in zip(curve, curve[1:]):
        assert p1 < p0
        assert o1 > o0


def test_non_model_rows_carry_no_curve(monkeypatch):
    # D never solves; held on test rates or excluded - either way no curve,
    # even though its config carries a perfectly usable Vogel anchor.
    _patch_grid(monkeypatch, GRID_A)
    configs = [
        _ipr_config("A", qwf=1000.0, pwf=1500.0, res_pres=3000.0, form_wc=0.5),
        _ipr_config("D", qwf=1000.0, pwf=1500.0, res_pres=3000.0, form_wc=0.5),
    ]
    # basis "test": D has measured rates
    rows, _ = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {"D": (50.0, 600.0)}, n_levels=3
    )
    d = {r["well"]: r for r in rows}["D"]
    assert d["basis"] == "test"
    assert d["ipr_curve"] is None
    # basis "none": no rates either
    rows, _ = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    d = {r["well"]: r for r in rows}["D"]
    assert d["basis"] == "none"
    assert d["ipr_curve"] is None


def test_unusable_anchor_yields_none_curve_without_raising(monkeypatch):
    _patch_grid(monkeypatch, GRID_A)
    # missing res_pres entirely (the bare fixture style used elsewhere)
    rows, _ = pad_optimize.run_choke_optimization(
        _configs(["A"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert rows[0]["basis"] == "model"
    assert rows[0]["ipr_curve"] is None
    # unphysical anchor: pwf >= res_pres (InFlow raises; builder maps to None)
    cfg = _ipr_config("A", qwf=1000.0, pwf=3200.0, res_pres=3000.0, form_wc=0.5)
    rows, _ = pad_optimize.run_choke_optimization(
        [cfg], StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert rows[0]["basis"] == "model"
    assert rows[0]["ipr_curve"] is None


# ---------------------------------------------------------------------------
# cavitation-floor (sonic) flags
# ---------------------------------------------------------------------------


def test_sonic_flags_ride_through_to_the_rows(monkeypatch):
    # 4-tuple grid entries carry the cavitation-floor flag; the row reports
    # it INDEPENDENTLY at the chosen and full-open points. A's oil is flat
    # (sonic) so the frontier collapses its options to the knee at 2000 -
    # the free choke - while B's rising oil pulls the winning header to
    # 3000, exactly the production shape that produced all-CHOKE plans.
    _patch_grid(
        monkeypatch,
        {
            "A": {
                2000.0: (100.0, 800.0, 850.0, True),
                2500.0: (100.0, 900.0, 850.0, True),
                3000.0: (100.0, 1000.0, 850.0, False),
            },
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
    )
    rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "B"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert meta["header_psi"] == 3000.0
    by_well = {r["well"]: r for r in rows}
    a, b = by_well["A"], by_well["B"]
    assert a["action"] == "choke"  # free choke: knee dominates full open
    assert a["delivered_psi"] == 2000.0 and a["delivered_full_psi"] == 3000.0
    assert a["sonic"] is True
    assert a["sonic_full"] is False
    assert b["sonic"] is None and b["sonic_full"] is None  # 2-tuple grid


# ---------------------------------------------------------------------------
# header-drop decision ladder
# ---------------------------------------------------------------------------


def test_decision_ladder_reruns_the_sweep_against_a_degraded_frontier(monkeypatch):
    # Rungs cover every ladder level below the winning header (3000). Each
    # rung scales the frontier so the ALL-RUN header settles at that level,
    # then re-sweeps. With a flat 100k stub frontier the degraded budget is
    # exactly the all-run demand at the rung level:
    #   rung 2500 (drop 500): demand 1800 -> best response holds 3000 by
    #     choking A down two steps (slope 0.05 twice beats B's 0.2):
    #     oil 190 vs 175 run-all -> gain +15.
    #   rung 2000 (drop 1000): demand 1600 -> holding higher headers nets
    #     the same 150 BOPD, so the first (lowest) header wins: no actions,
    #     gain 0.
    _patch_grid(
        monkeypatch,
        {
            "A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)},
            "B": {2000.0: (60.0, 800.0), 2500.0: (80.0, 900.0), 3000.0: (100.0, 1000.0)},
        },
    )
    _rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "B"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert meta["header_psi"] == 3000.0
    ladder = meta["ladder"]
    assert [r["drop_psi"] for r in ladder] == [500.0, 1000.0]

    r5 = ladder[0]
    assert r5["settles_psi"] == 2500.0
    assert r5["run_all_oil_bopd"] == pytest.approx(175.0)
    assert r5["best_header_psi"] == 3000.0
    assert r5["plan_oil_bopd"] == pytest.approx(190.0)
    assert r5["gain_bopd"] == pytest.approx(15.0)
    assert r5["actions"] == [{"well": "A", "action": "choke", "set_psi": 2000.0}]

    r10 = ladder[1]
    assert r10["settles_psi"] == 2000.0
    assert r10["run_all_oil_bopd"] == pytest.approx(150.0)
    assert r10["plan_oil_bopd"] == pytest.approx(150.0)
    assert r10["gain_bopd"] == pytest.approx(0.0)
    assert r10["actions"] == []


def test_ladder_counts_held_wells_but_zeros_dead_levels(monkeypatch):
    # A is modelable everywhere; C is NEVER modelable but tested (held at
    # measured rates, header-independent); at the 2000 rung A additionally
    # has NO solution - a modelable well that cannot lift at that header
    # contributes ZERO to run-all, not its test rates.
    _patch_grid(
        monkeypatch,
        {"A": {2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0)}},
    )
    _rows, meta = pad_optimize.run_choke_optimization(
        _configs(["A", "C"]),
        StubPlant(100000.0),
        2,
        CHOICES,
        {"C": (40.0, 500.0)},
        n_levels=3,
    )
    by_drop = {r["drop_psi"]: r for r in meta["ladder"]}
    # rung 2500: A modeled (95) + C held (40)
    assert by_drop[500.0]["run_all_oil_bopd"] == pytest.approx(135.0)
    # rung 2000: A cannot lift (0) + C held (40)
    assert by_drop[1000.0]["run_all_oil_bopd"] == pytest.approx(40.0)


def test_short_grid_tuples_leave_sonic_none(monkeypatch):
    # 2- and 3-tuple fakes (every other test in this file) predate the flag;
    # the rows must degrade to None, never KeyError/IndexError.
    _patch_grid(
        monkeypatch,
        {"A": {2000.0: (90.0, 800.0), 2500.0: (95.0, 900.0), 3000.0: (100.0, 1000.0, 700.0)}},
    )
    rows, _ = pad_optimize.run_choke_optimization(
        _configs(["A"]), StubPlant(100000.0), 2, CHOICES, {}, n_levels=3
    )
    assert rows[0]["sonic"] is None
    assert rows[0]["sonic_full"] is None