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


# ---------------------------------------------------------------------------
# evidence-corrected suction response
# ---------------------------------------------------------------------------


def _vogel_ratio(psu, psu_ref, res):
    """Expected corrected-oil ratio: vogel factor at psu over psu_ref."""

    def f(p):
        return 1.0 - 0.2 * (p / res) - 0.8 * (p / res) ** 2

    return f(psu) / f(psu_ref)


def _ev_row(**over):
    row = {
        "floor": 350.0,
        "psu_ref": 400.0,
        "beta": 0.1,
        "beta_source": "well",
        "n_days": 30,
        "n_pairs": 8,
        "window": ["2026-01-01", "2026-08-01"],
    }
    row.update(over)
    return row


# A is cavitation-pinned in the model: flat oil, frozen psu, sonic True at
# every level. B rises normally (pulls the winning header to 3000) and has
# no evidence.
GRID_SONIC_FLAT = {
    "A": {
        2000.0: (100.0, 800.0, 500.0, True),
        2500.0: (100.0, 900.0, 500.0, True),
        3000.0: (100.0, 1000.0, 500.0, True),
    },
    "B": {2000.0: (50.0, 800.0), 2500.0: (75.0, 900.0), 3000.0: (100.0, 1000.0)},
}


def _anchored_config(name="A", res_pres=1000.0):
    # oil-basis Vogel anchor: qwf*(1-form_wc)=200 at pwf=300
    return _ipr_config(name, qwf=250.0, pwf=300.0, res_pres=res_pres, form_wc=0.2)


def _strip_evidence_provenance(rows):
    """Pop the two always-populated provenance fields for deep-compares."""
    return [
        (
            {k: v for k, v in r.items() if k not in ("evidence_floor_psi", "floor_violation_psi")},
            r["evidence_floor_psi"],
            r["floor_violation_psi"],
        )
        for r in rows
    ]


def test_evidence_correction_declines_the_staircase_and_charges_chokes(monkeypatch):
    # Model floor 500 psi vs measured floor 350 (violation 150 > 25) on a
    # sonic well: the suction response is replaced by the field data. The
    # staircase now DECLINES (Vogel oil at psu_e = psu_ref + beta*dP), psu
    # rises with choke depth, PF stays the model's, and the binding budget
    # (1900 vs 2000 full-open) chokes A one step at a real oil cost.
    _patch_grid(monkeypatch, GRID_SONIC_FLAT)
    configs = [_anchored_config("A"), _ipr_config("B")]
    rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(1900.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row()},
    )
    assert meta["header_psi"] == 3000.0
    assert meta["n_evidence_corrected"] == 1
    assert meta["n_choked"] == 1
    a = next(r for r in rows if r["well"] == "A")
    b = next(r for r in rows if r["well"] == "B")
    # A: choked to 2500; oil now costs (staircase declines)
    assert a["action"] == "choke" and a["delivered_psi"] == 2500.0
    assert a["oil"] == pytest.approx(100.0 * _vogel_ratio(450.0, 400.0, 1000.0))
    assert a["oil_full"] == pytest.approx(100.0)  # anchored at k*
    assert a["d_oil_vs_full"] < 0.0  # chokes cost oil now
    # psu rises with choke depth (beta * dP off the psu_ref anchor)
    assert a["psu_full"] == pytest.approx(400.0)
    assert a["psu"] == pytest.approx(450.0)
    # PF hydraulics untouched (validated model)
    assert a["pf"] == pytest.approx(900.0)
    assert a["pf_full"] == pytest.approx(1000.0)
    # corrected points are not cavitation-pinned
    assert a["sonic"] is None and a["sonic_full"] is None
    # provenance
    assert a["suction_basis"] == "evidence"
    assert a["evidence_floor_psi"] == pytest.approx(350.0)
    assert a["floor_violation_psi"] == pytest.approx(150.0)  # ORIGINAL model floor
    assert a["response_beta"] == pytest.approx(0.1)
    assert a["beta_source"] == "well"
    assert a["evidence_gate"] == "floor"  # violation is the stronger claim
    # a further trim now has a real, positive marginal cost
    assert a["next_trim_bopd_per_bpd"] is not None and a["next_trim_bopd_per_bpd"] > 0.0
    # B untouched: no evidence, model suction basis, no provenance
    assert b["action"] == "full" and b["suction_basis"] == "model"
    assert b["evidence_floor_psi"] is None and b["floor_violation_psi"] is None
    assert b["response_beta"] is None and b["beta_source"] is None
    assert b["evidence_gate"] is None


def test_evidence_confirming_the_model_changes_nothing_but_provenance(monkeypatch):
    # Floor within 25 psi of the model floor (500 - 490 = 10) AND a measured
    # beta below _EVIDENCE_BETA_MIN: the evidence CONFIRMS the model on both
    # counts and the run is identical to an evidence=None run, except the
    # row now carries the floor and its (small) violation.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A")]
    rows_ev, meta_ev = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(floor=490.0, beta=0.02)},
    )
    rows_none, meta_none = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3, evidence=None,
    )
    assert meta_ev == meta_none
    assert meta_ev["n_evidence_corrected"] == 0
    stripped_ev = _strip_evidence_provenance(rows_ev)
    stripped_none = _strip_evidence_provenance(rows_none)
    assert [s[0] for s in stripped_ev] == [s[0] for s in stripped_none]
    assert stripped_ev[0][1] == pytest.approx(490.0)
    assert stripped_ev[0][2] == pytest.approx(10.0)  # full_raw psu 500 - 490
    assert stripped_none[0][1] is None and stripped_none[0][2] is None
    assert rows_ev[0]["evidence_gate"] is None


def test_measured_response_falsifies_a_confirmed_floor(monkeypatch):
    # MPM-28 shape: the model floor is CONFIRMED (violation 10 <= 25) but a
    # well-measured beta of 0.08 demonstrates the suction response the
    # pinned model denies -> corrected via the RESPONSE gate, and the flat
    # staircase now declines with choke depth.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A")]
    rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(floor=490.0, beta=0.08)},
    )
    assert meta["n_evidence_corrected"] == 1
    a = rows[0]
    assert a["suction_basis"] == "evidence"
    assert a["evidence_gate"] == "response"
    assert a["response_beta"] == pytest.approx(0.08)
    assert a["beta_source"] == "well"
    assert a["evidence_floor_psi"] == pytest.approx(490.0)
    assert a["floor_violation_psi"] == pytest.approx(10.0)
    # the corrected staircase declines: choking costs real oil now
    oil_by_header = {s["header_psi"]: s["total_oil_bopd"] for s in meta["sweep"]}
    assert oil_by_header[3000.0] == pytest.approx(100.0)
    assert (
        oil_by_header[2000.0]
        == pytest.approx(100.0 * _vogel_ratio(400.0 + 0.08 * 1000.0, 400.0, 1000.0))
    )
    assert oil_by_header[2000.0] < oil_by_header[2500.0] < oil_by_header[3000.0]


def test_insensitive_beta_leaves_a_confirmed_floor_alone(monkeypatch):
    # Same confirmed floor but the measured beta (0.02) sits in the
    # insensitive group: no gate fires, the run matches evidence=None.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A")]
    rows_ev, meta_ev = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(floor=490.0, beta=0.02)},
    )
    rows_none, meta_none = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3, evidence=None,
    )
    assert meta_ev == meta_none
    assert meta_ev["n_evidence_corrected"] == 0
    assert rows_ev[0]["suction_basis"] == "model"
    assert rows_ev[0]["evidence_gate"] is None
    assert [s[0] for s in _strip_evidence_provenance(rows_ev)] == [
        s[0] for s in _strip_evidence_provenance(rows_none)
    ]


def test_pad_sourced_beta_never_triggers_the_response_gate(monkeypatch):
    # A responsive-looking beta (0.08) that is only a PAD prior is not
    # measurement: with the floor confirmed the well stays uncorrected.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A")]
    rows_ev, meta_ev = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(floor=490.0, beta=0.08, beta_source="pad")},
    )
    rows_none, meta_none = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3, evidence=None,
    )
    assert meta_ev == meta_none
    assert meta_ev["n_evidence_corrected"] == 0
    assert rows_ev[0]["suction_basis"] == "model"
    assert rows_ev[0]["evidence_gate"] is None
    assert rows_ev[0]["response_beta"] is None and rows_ev[0]["beta_source"] is None
    assert [s[0] for s in _strip_evidence_provenance(rows_ev)] == [
        s[0] for s in _strip_evidence_provenance(rows_none)
    ]


def test_subsonic_well_is_never_corrected(monkeypatch):
    # sonic False at the top solvable level: the model suction is already
    # responsive, so even a huge floor violation leaves the grid untouched
    # (only the floor/violation provenance rides through).
    grid = {
        "A": {
            2000.0: (90.0, 800.0, 500.0, False),
            2500.0: (95.0, 900.0, 500.0, False),
            3000.0: (100.0, 1000.0, 500.0, False),
        }
    }
    _patch_grid(monkeypatch, grid)
    configs = [_anchored_config("A")]
    rows_ev, meta_ev = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row()},
    )
    rows_none, meta_none = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
    )
    assert meta_ev == meta_none
    assert meta_ev["n_evidence_corrected"] == 0
    a = rows_ev[0]
    assert a["suction_basis"] == "model"
    assert a["response_beta"] is None and a["beta_source"] is None
    assert a["evidence_floor_psi"] == pytest.approx(350.0)
    assert a["floor_violation_psi"] == pytest.approx(150.0)
    assert [s[0] for s in _strip_evidence_provenance(rows_ev)] == [
        s[0] for s in _strip_evidence_provenance(rows_none)
    ]


def test_evidence_none_deep_equals_a_no_kwarg_run(monkeypatch):
    _patch_grid(monkeypatch, GRID_SONIC_FLAT)
    configs = [_anchored_config("A"), _ipr_config("B")]
    rows_none, meta_none = pad_optimize.run_choke_optimization(
        configs, StubPlant(1900.0), 2, CHOICES, {"B": (80.0, 950.0)}, n_levels=3,
        evidence=None,
    )
    rows_bare, meta_bare = pad_optimize.run_choke_optimization(
        configs, StubPlant(1900.0), 2, CHOICES, {"B": (80.0, 950.0)}, n_levels=3,
    )
    assert rows_none == rows_bare
    assert meta_none == meta_bare


def test_psu_e_at_or_above_res_pres_zeroes_deep_levels(monkeypatch):
    # Steep beta against a low res_pres: psu_e = 400 + 0.5*dP crosses the
    # 600 psi reservoir at both deeper chokes -> the corrected oil is 0
    # there (the well cannot flow), visible in the header sweep totals.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A", res_pres=600.0)]
    rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(floor=300.0, beta=0.5)},
    )
    assert meta["n_evidence_corrected"] == 1
    oil_by_header = {s["header_psi"]: s["total_oil_bopd"] for s in meta["sweep"]}
    assert oil_by_header[2000.0] == pytest.approx(0.0)
    assert oil_by_header[2500.0] == pytest.approx(0.0)
    assert oil_by_header[3000.0] == pytest.approx(100.0)
    assert rows[0]["oil"] == pytest.approx(100.0)  # full open at the anchor


def test_psu_ref_at_or_above_res_pres_is_unusable(monkeypatch):
    # InFlow cannot anchor a Vogel ratio above res_pres: skip, no correction.
    _patch_grid(monkeypatch, {"A": GRID_SONIC_FLAT["A"]})
    configs = [_anchored_config("A", res_pres=600.0)]
    rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(100000.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row(psu_ref=650.0)},
    )
    assert meta["n_evidence_corrected"] == 0
    assert rows[0]["suction_basis"] == "model"
    assert rows[0]["response_beta"] is None


def test_decision_ladder_charges_oil_for_chokes_on_a_corrected_well(monkeypatch):
    # Pre-correction A's flat staircase collapsed to the knee and chokes
    # were free. Corrected, the 500-psi rung's best response chokes A to
    # 2000 at a real oil cost: plan oil sits BELOW the winning-header total.
    _patch_grid(monkeypatch, GRID_SONIC_FLAT)
    configs = [_anchored_config("A"), _ipr_config("B")]
    _rows, meta = pad_optimize.run_choke_optimization(
        configs, StubPlant(1900.0), 2, CHOICES, {}, n_levels=3,
        evidence={"A": _ev_row()},
    )
    rung = next(r for r in meta["ladder"] if r["drop_psi"] == pytest.approx(500.0))
    oil_2000 = 100.0 * _vogel_ratio(500.0, 400.0, 1000.0)
    oil_2500 = 100.0 * _vogel_ratio(450.0, 400.0, 1000.0)
    # run-all at the rung: A + B priced from the corrected grid at 2500
    assert rung["run_all_oil_bopd"] == pytest.approx(oil_2500 + 75.0)
    # best response holds 3000 and chokes A to 2000 - and PAYS for it
    assert rung["best_header_psi"] == 3000.0
    assert rung["actions"] == [{"well": "A", "action": "choke", "set_psi": 2000.0}]
    assert rung["plan_oil_bopd"] == pytest.approx(oil_2000 + 100.0)
    assert rung["plan_oil_bopd"] < meta["total_oil_bopd"]
    assert rung["gain_bopd"] == pytest.approx((oil_2000 + 100.0) - (oil_2500 + 75.0))