"""CFP Today's Moves regressions from the 2026-09-24 engine review.

Each test reproduces one reviewed defect (docs/cfp_moves_methodology.md
states the contract): required-well studies that shut in nearly the whole
fleet, a singles board emptied by the required-well filter, a pair budget
spent on the first bring-online options, offsets scored across pressures,
catalogs unioned across wells, constant-PF wells re-simulated per grid
point, and the closed-form settle against an independent bisection.
"""

from collections import Counter
import random
from types import SimpleNamespace

import pytest

from woffl.gui import cfp_moves as cm
from woffl.gui.cfp_moves import OFF, SI, Surfaces, WellSurface, anchor, moves_summary, option_at, pair_moves, settle

SIZES = [f"{n}{t}" for n in range(9, 16) for t in "ABCD"]  # a 7 x 4 catalog, 28 sizes
P0 = 2800.0


def _opt(grid, oil, water, oil_slope, water_slope, lo=None):
    """Linear oil/water (BOPD, BPD) around P0; no solve below ``lo`` psi."""
    oil_ray = [oil + oil_slope * (p - P0) for p in grid]
    water_ray = [water + water_slope * (p - P0) for p in grid]
    if lo is not None:
        oil_ray = [v if p >= lo else None for v, p in zip(oil_ray, grid)]
        water_ray = [v if p >= lo else None for v, p in zip(water_ray, grid)]
    return {"_grid": grid, "oil": oil_ray, "water": water_ray}


# ── HIGH: required wells must not shut in the fleet ─────────────────────────

GRID6 = [2500.0, 2600.0, 2700.0, 2800.0, 2850.0, 2880.0]


def _required_above_p0():
    """36 online wells x 28 sizes, a 6,000 BPD low-oil PIG, and a required
    NEW well whose every size solves only at >= 2,850 psi."""
    s = Surfaces(GRID6, P0)
    for i in range(36):
        s.wells[f"ON{i:02d}"] = WellSurface(f"ON{i:02d}", "B", True, "12B", {
            lab: _opt(GRID6, 80 + 2 * k, 1200 + 60 * k, 0.1, 1.0) for k, lab in enumerate(SIZES)})
    s.wells["PIG"] = WellSurface("PIG", "J", True, "12B", {"12B": _opt(GRID6, 5, 6000, 0.1, 1.0)})
    s.wells["NEW"] = WellSurface("NEW", "G", False, None, {
        lab: _opt(GRID6, 300 + 10 * k, 200 + 250 * k, 0.0, 1.0, lo=2850.0) for k, lab in enumerate(SIZES)})
    return s


def _required_with_offsets(new_water, n_pigs):
    """Every NEW size solves everywhere but adds ``new_water`` BPD or more;
    ``n_pigs`` 9,000 BPD low-oil wells are the natural offsets."""
    s = Surfaces(GRID6, P0)
    for i in range(36):
        s.wells[f"ON{i:02d}"] = WellSurface(f"ON{i:02d}", "B", True, "12B", {
            lab: _opt(GRID6, 80 + 2 * k, 1200 + 60 * k, 0.1, 0.2) for k, lab in enumerate(SIZES)})
    for j in range(1, n_pigs + 1):
        s.wells[f"PIG{j}"] = WellSurface(f"PIG{j}", "J", True, "12B", {"12B": _opt(GRID6, 5, 9000, 0.0, 0.2)})
    s.wells["NEW"] = WellSurface("NEW", "G", False, None, {
        lab: _opt(GRID6, 300 + 10 * k, new_water + 200 * k, 0.0, 0.0) for k, lab in enumerate(SIZES)})
    return s


def test_required_well_solving_only_above_p0_gets_the_few_change_plan():
    """Before: +/-0 frontier states, the fallback shed 35 wells, gain -2,967.
    The obvious plan (NEW 9A + shut in PIG) settles at 2,852.7 psi, +484.7."""
    s = _required_above_p0()
    plant = anchor(s, psi_per_kbpd=13.69)
    out = moves_summary(s, plant, required_wells={"NEW"})
    obvious = settle({**s.baseline_choices(), "NEW": "9A", "PIG": SI}, s, plant, tol_psi=1e-6)
    assert obvious["feasible"] and obvious["oil"] - out["today"]["oil"] == pytest.approx(484.7, abs=0.05)
    plan = out["plan"]
    assert plan["choices"]["NEW"] not in (SI, OFF)
    assert out["plan_gain"] >= obvious["oil"] - out["today"]["oil"] - 1e-6
    assert sum(lab == SI for lab in plan["choices"].values()) <= 2 and plan["n_changes"] <= 4
    scope = out["search_scope"]
    assert scope["method"] == "lambda_moves_and_bounded_neighborhood"
    assert scope["seed_finalists"] > 0 and scope["seed_evaluated"] <= cm.MAX_SEED_EVALUATIONS
    assert scope["global_optimum_on_surfaces"] is False
    assert out["frontier"], "the sweep keeps NEW on at a size that solves above P0"


def test_required_well_needing_two_offsets_beats_the_three_change_plan():
    """Before: +74.0 BOPD with 39 changes. NEW 9A + shut in both PIGs: +110.6."""
    s = _required_with_offsets(22000.0, 2)
    plant = anchor(s, psi_per_kbpd=13.69)
    out = moves_summary(s, plant, required_wells={"NEW"})
    three = settle({**s.baseline_choices(), "NEW": "9A", "PIG1": SI, "PIG2": SI}, s, plant, tol_psi=1e-6)
    assert three["oil"] - out["today"]["oil"] == pytest.approx(110.6, abs=0.05)
    assert out["plan"]["choices"]["NEW"] not in (SI, OFF)
    assert out["plan_gain"] >= 110.6 - 0.05
    assert out["plan"]["n_changes"] < 20
    assert out["search_scope"]["seed_descent_evaluated"] > 0


def test_required_well_with_no_single_offset_is_completed_greedily():
    """NEW plus any one offset lands below the grid; the greedy completion
    reaches NEW + two PIG shut-ins instead of shedding every optional well."""
    s = _required_with_offsets(35000.0, 3)
    plant = anchor(s, psi_per_kbpd=13.69)
    base = s.baseline_choices()
    assert not settle({**base, "NEW": "9A", "PIG1": SI}, s, plant, tol_psi=1e-6)["feasible"]
    out = moves_summary(s, plant, required_wells={"NEW"})
    scope = out["search_scope"]
    assert scope["seed_greedy_evaluated"] > 0 and scope["seed_finalists"] > 0
    three_pigs = settle({**base, "NEW": "9A", "PIG1": SI, "PIG2": SI, "PIG3": SI}, s, plant, tol_psi=1e-6)
    assert three_pigs["feasible"]
    assert out["plan_gain"] >= three_pigs["oil"] - out["today"]["oil"] - 1e-6
    plan = out["plan"]
    assert plan["choices"]["NEW"] not in (SI, OFF)
    assert sum(lab == SI for lab in plan["choices"].values()) <= 6


def test_required_sweep_picks_a_size_that_solves_elsewhere_on_the_grid():
    ws = _required_above_p0().wells["NEW"]
    assert cm._best_option(ws, P0, 0.0, required=False) == OFF
    picked = cm._best_option(ws, P0, 0.0, required=True)
    assert picked in ws.labels() and option_at(ws, picked, P0) is None


# ── MEDIUM: boards stay relative to today ───────────────────────────────────


def test_singles_board_stays_relative_to_today_with_a_required_flag():
    """Before: 1,021 rows without the requirement, 11 (no shut-ins) with it."""
    s = _required_with_offsets(22000.0, 2)
    plant = anchor(s, psi_per_kbpd=13.69)
    free = moves_summary(s, plant)
    held = moves_summary(s, plant, required_wells={"NEW"})
    assert len(held["singles"]) == len(free["singles"]) == 1021
    assert sum(m["type"] == cm.MOVE_SHUT_IN for m in held["singles"]) == 38
    assert all(m["meets_required"] is True for m in free["singles"])
    for m in held["singles"]:
        assert m["meets_required"] is (m["well"] == "NEW")
    assert held["n_positive_singles"] == sum(m["fleet_oil_delta"] > 1.0 and m["meets_required"] for m in held["singles"])
    assert held["plan"]["choices"]["NEW"] not in (SI, OFF)
    assert all("meets_required" in p for p in held["pairs"])


# ── MEDIUM: the pair budget reaches every bring-online option ───────────────


def test_pair_budget_rotates_across_bring_online_options():
    """Before: 2,048 evaluations went to BIG's first sizes; no GOOD pair was
    ever tried. GOOD 9A + shut in ON00 is worth +337.6 BOPD."""
    grid = sorted({2500.0 + 63.333 * k for k in range(6)} | {2880.0, P0})
    s = Surfaces(grid, P0)
    for i in range(36):
        s.wells[f"ON{i:02d}"] = WellSurface(f"ON{i:02d}", "B", True, "12B", {
            lab: _opt(grid, 80 + 2 * k, 1500 + 60 * k, 0.2, 2.0) for k, lab in enumerate(SIZES)})
    s.wells["BIG"] = WellSurface("BIG", "G", False, None, {
        lab: _opt(grid, 400 + k, 3000 + 60 * k, 0.2, 2.0, lo=2790.0) for k, lab in enumerate(SIZES)})
    s.wells["GOOD"] = WellSurface("GOOD", "J", False, None, {
        lab: _opt(grid, 390 + k, 1200 + 10 * k, 0.2, 2.0, lo=2790.0) for k, lab in enumerate(SIZES)})
    plant = anchor(s, psi_per_kbpd=13.69)
    diag = {}
    pairs = pair_moves(s, plant, top_n=10**6, _diagnostics=diag)
    assert diag["pair_evaluated"] == cm.MAX_PAIR_EVALUATIONS and diag["pair_search_complete"] is False
    found = [p for p in pairs if (p["bring_on"]["well"], p["bring_on"]["to"], p["offset"]["well"], p["offset"]["to"])
             == ("GOOD", "9A", "ON00", SI)]
    assert len(found) == 1 and found[0]["fleet_oil_delta"] == pytest.approx(337.6, abs=0.05)
    assert found[0]["meets_required"] is True
    # Every GOOD size met the leading offsets.
    assert len({p["bring_on"]["to"] for p in pairs if p["bring_on"]["well"] == "GOOD"}) == len(SIZES)


def test_offsets_are_scored_by_same_pressure_water():
    """Before: an upsize adding 1,000 BPD at every pressure scored +500 (its
    low-pressure water against the current pump's P0 water) and was paired."""
    grid = [2500.0, 2600.0, 2700.0, 2800.0, 2880.0]

    def opt(oil, water_at_p0):
        return {"_grid": grid, "oil": [oil] * 5, "water": [water_at_p0 + 5.0 * (p - P0) for p in grid]}
    s = Surfaces(grid, P0, {
        "ON": WellSurface("ON", "B", True, "12B", {"12B": opt(100, 3000), "13C": opt(130, 4000), "11A": opt(90, 2900)}),
        "NEW": WellSurface("NEW", "G", False, None, {"10A": opt(200, 1000)}),
    })
    scores = {(w, lab): score for w, lab, score in cm._single_changes(s, s.baseline_choices())}
    assert scores[("ON", "13C")] == pytest.approx(-1000.0)
    assert scores[("ON", "11A")] == pytest.approx(100.0)
    assert scores[("ON", SI)] == pytest.approx(3000.0)
    diag = {}
    pairs = pair_moves(s, anchor(s, psi_per_kbpd=13.69), top_n=99, _diagnostics=diag)
    assert diag["pair_combinations"] == 2
    assert not any(p["offset"]["to"] == "13C" for p in pairs)


def test_neighborhood_swaps_spread_across_well_pairs(monkeypatch):
    """Before: the swap loop spent its budget on the first well in name order."""
    grid = [2500.0, 2600.0, 2700.0, 2800.0, 2880.0]
    s = Surfaces(grid, P0)
    for i in range(12):
        s.wells[f"W{i:02d}"] = WellSurface(f"W{i:02d}", "B", True, "cur", {
            "cur": _opt(grid, 100, 1000, 0.0, 0.0), "alt": _opt(grid, 90, 1500, 0.0, 0.0)})
    baseline = s.baseline_choices()
    seen = []
    real = cm.settle

    def spy(choices, *args, **kwargs):
        seen.append(dict(choices))
        return real(choices, *args, **kwargs)
    monkeypatch.setattr(cm, "settle", spy)
    monkeypatch.setattr(cm, "MAX_EXACT_COMBINATIONS", 0)
    monkeypatch.setattr(cm, "MAX_NEIGHBOR_EVALUATIONS", 60)
    out = moves_summary(s, anchor(s, psi_per_kbpd=13.69))
    swaps = [c for c in seen if sum(c[w] != baseline[w] for w in c) == 2]
    assert out["search_scope"]["neighborhood_budget_exhausted"] is True
    assert len({min(w for w in c if c[w] != baseline[w]) for c in swaps}) >= 3


# ── LOW-MED / PERF: Stage A simulation scope ────────────────────────────────


class _Batch(SimpleNamespace):
    pass


def _fake_jobs(monkeypatch, calls=None):
    """Fake the pooled simulation (when ``calls`` is a list) and the reader."""
    import woffl.assembly.network_optimizer as no_mod
    import woffl.assembly.parallelism as par

    def simulate_jobs(jobs, max_workers=1):
        jobs = list(jobs)
        calls.append(jobs)
        return [_Batch(well=w, pressure=p, nozzles=list(n), throats=list(t)) for w, p, n, t in jobs]

    class Reader:
        def __init__(self, wells, pf, nozzles, throats, marginal_watercut=1.0):
            self.batch_results = {}

        def get_pump_performance(self, well, nozzle, throat, pump_state=None):
            batch = self.batch_results[well]
            cfg = batch.well
            if pump_state == "installed":
                ok = (nozzle, throat) == (cfg.installed_nozzle, cfg.installed_throat)
            else:
                ok = nozzle in batch.nozzles and throat in batch.throats
            return {"oil_rate": 100.0, "total_water": cfg.ppf_surf_well} if ok else None

    if calls is not None:
        monkeypatch.setattr(no_mod, "simulate_jobs", simulate_jobs)
    monkeypatch.setattr(no_mod, "NetworkOptimizer", Reader)
    monkeypatch.setattr(par, "worker_ceiling", lambda: 1)


def _scoped(name, pad, installed):
    from woffl.assembly.network_optimizer import WellConfig

    return WellConfig(well_name=name, res_pres=1500, form_temp=70, jpump_tvd=4000, pad=pad,
                      pump_calibration_scoped=True, installed_nozzle=installed[0], installed_throat=installed[1])


def test_each_well_sweeps_the_catalog_plus_only_its_own_pump(monkeypatch):
    """Before: MPB-02's installed 11X gave every well 11B/11X/12X clean
    options and simulated the 2 x 2 union at every grid point."""
    import woffl.gui.cfp_optimize as co

    calls = []
    _fake_jobs(monkeypatch, calls)
    monkeypatch.setattr(co, "delivered_by_pad", lambda *a, **k: ({"B": 2600.0}, []))
    wells = [_scoped("MPB-01", "B", ("12", "B")), _scoped("MPB-02", "B", ("11", "X"))]
    surf = cm.build_response_surfaces({"B": wells}, {"MPB-01": True, "MPB-02": True},
                                      {"MPB-01": ("12", "B"), "MPB-02": ("11", "X")}, object(),
                                      p_grid=[2500.0, 2792.0], nozzles=["12"], throats=["B"], p0=2792.0,
                                      c_pad_pf_psi=3400.0)
    assert [(job[0].well_name, job[2], job[3]) for job in calls[0]] == [
        ("MPB-01", ["12"], ["B"]), ("MPB-02", ["12"], ["B"])]
    assert surf.wells["MPB-01"].labels() == ["12B", "12B (clean)"]
    assert surf.wells["MPB-02"].labels() == ["11X", "12B (clean)"]
    # The current label still names the installed option, so it anchors.
    assert all(ws.current in ws.options for ws in surf.wells.values())


def test_unscoped_or_planned_current_size_joins_its_own_sweep(monkeypatch):
    """A future well's planned pump (no installed identity) outside the
    catalog still becomes a bring-online option, for that well only."""
    import woffl.gui.cfp_optimize as co

    calls = []
    _fake_jobs(monkeypatch, calls)
    monkeypatch.setattr(co, "delivered_by_pad", lambda *a, **k: ({"B": 2600.0}, []))
    wells = [_scoped("MPB-01", "B", ("12", "B")), _scoped("FUT-1", "B", (None, None))]
    surf = cm.build_response_surfaces({"B": wells}, {"MPB-01": True, "FUT-1": False},
                                      {"MPB-01": ("12", "B"), "FUT-1": ("9", "A")}, object(),
                                      p_grid=[2500.0, 2792.0], nozzles=["12"], throats=["B"], p0=2792.0,
                                      c_pad_pf_psi=3400.0)
    grids = Counter((job[0].well_name, tuple(job[2]), tuple(job[3])) for job in calls[0])
    assert grids == Counter({("MPB-01", ("12",), ("B",)): 1, ("FUT-1", ("12",), ("B",)): 1, ("FUT-1", ("9",), ("A",)): 1})
    assert surf.wells["FUT-1"].labels() == ["12B (clean)", "9A (clean)"]
    assert "9A (clean)" not in surf.wells["MPB-01"].options


def test_constant_pf_wells_are_simulated_once_and_share_host_cache_keys(monkeypatch):
    """Before: the C-Pad well (PF held at its own booster) was simulated at
    all 8 grid points, and the host cache key varied with the grid point."""
    from woffl.assembly import compute_runtime
    import woffl.assembly.network_optimizer as no_mod
    from woffl.gui.cfp_pad_plant import PLANT
    from server import pool, surface_cache

    simulated = Counter()

    def fake_sim(well, pressure, nozzles, throats):
        assert pressure == well.ppf_surf_well  # the job pressure is the simulated PF
        simulated[(well.well_name, well.ppf_surf_well)] += 1
        return _Batch(well=well, pressure=pressure, nozzles=list(nozzles), throats=list(throats))

    # Real simulate_jobs -> host run_jobs (exact cache) -> fake single-well solve.
    _fake_jobs(monkeypatch)
    monkeypatch.setattr(no_mod, "_simulate_single_well", fake_sim)
    monkeypatch.setattr(pool, "submit_all", lambda fn, jobs: None)  # no pool: serial path
    monkeypatch.setattr(compute_runtime, "job_runner", surface_cache.run_jobs)
    surface_cache.clear()
    try:
        configs = {"B": [_scoped("MPB-01", "B", ("12", "B"))], "C": [_scoped("MPC-01", "C", ("12", "B"))]}
        grid = [2492.0, 2556.0, 2620.0, 2684.0, 2748.0, 2792.0, 2812.0, 2880.0]
        kwargs = dict(nozzles=["12"], throats=["B"], c_pad_pf_psi=3400.0)
        cm.build_response_surfaces(configs, {"MPB-01": True, "MPC-01": True}, {}, PLANT, p_grid=grid, p0=2792.0, **kwargs)
        assert sum(n for (w, _pf), n in simulated.items() if w == "MPC-01") == 1
        assert sum(n for (w, _pf), n in simulated.items() if w == "MPB-01") == len(grid)
        # A second study on a shifted grid reuses the C-Pad node from the host cache.
        hits = surface_cache.status()["hits"]
        cm.build_response_surfaces(configs, {"MPB-01": True, "MPC-01": True}, {}, PLANT,
                                   p_grid=[p - 10.0 for p in grid], p0=2782.0, **kwargs)
        assert sum(n for (w, _pf), n in simulated.items() if w == "MPC-01") == 1
        assert surface_cache.status()["hits"] > hits
        assert all(wc.ppf_surf_well is None for wells in configs.values() for wc in wells)
    finally:
        surface_cache.clear()


# ── PERF: closed-form settle against an independent bisection ───────────────


def _bisection_root(choices, s, plant):
    """Unique pressure root by plain bisection on every converged interval."""
    grid = sorted(s.p_grid)
    floor, ceiling = grid[0], min(plant.cap, grid[-1])

    def residual(p):
        total = 0.0
        for w, lab in choices.items():
            value = option_at(s.wells[w], lab, p)
            if value is None:
                return None
            total += value[1]
        return plant.pressure_at(total)[0] - p

    knots = sorted({floor, ceiling} | {g for g in grid if floor <= g <= ceiling})
    for a, b in zip(knots, knots[1:]):
        ra, rb = residual(a), residual(b)
        if ra is None or rb is None or not ra >= 0 >= rb:
            continue
        for _ in range(200):
            mid = 0.5 * (a + b)
            a, b = (mid, b) if residual(mid) > 0 else (a, mid)
        return 0.5 * (a + b)
    # Disposal re-trim: the plant holds exactly the cap, even when the
    # interval just below it has a failed solve.
    return ceiling if ceiling == plant.cap and residual(ceiling) == 0 else None


@pytest.mark.parametrize("seed", [3, 7])
def test_closed_form_settle_matches_independent_bisection(seed):
    """Random nondecreasing-water surfaces, some with failed-solve gaps, up to
    40 BPD per 10 psi per well (steep loop gain), both sides of the trip cap."""
    rng = random.Random(seed)
    checked = feasible = 0
    for _case in range(20):
        p0 = rng.choice([2400.0, 2600.0, 2792.0, 2850.0, 2880.0])
        lo = max(p0 - 300.0, 1800.0)
        grid = sorted({lo + (2880.0 - lo) * k / 6 for k in range(7)} | {p0})
        s = Surfaces(grid, p0)
        for i in range(rng.randint(3, 25)):
            options = {}
            for lab in ("cur", "alt"):
                water, acc = [], rng.uniform(500, 8000)
                for k, g in enumerate(grid):
                    acc += rng.uniform(0, 40) * (0 if k == 0 else (g - grid[k - 1]) / 10.0)
                    water.append(acc)
                oil = [rng.uniform(10, 500) for _ in grid]
                if lab == "alt" and rng.random() < 0.3:
                    for k in rng.sample(range(len(grid)), 2):
                        oil[k] = water[k] = None
                options[lab] = {"_grid": grid, "oil": oil, "water": water}
            s.wells[f"W{i}"] = WellSurface(f"W{i}", "B", rng.random() < 0.8, "cur", options)
        plant = anchor(s, psi_per_kbpd=rng.uniform(9, 17.5))
        cache = cm._TableCache(s)
        for _ in range(10):
            choices = {w: rng.choice(ws.choice_labels()) for w, ws in s.wells.items()}
            state = settle(choices, s, plant, tol_psi=1e-6, _cache=cache)
            root = _bisection_root(choices, s, plant)
            checked += 1
            assert state["feasible"] == (root is not None), (choices, state)
            if root is not None:
                feasible += 1
                assert state["pressure"] == pytest.approx(root, abs=1e-6)
                oil = sum(option_at(s.wells[w], lab, root)[0] for w, lab in choices.items())
                assert state["oil"] == pytest.approx(oil, rel=1e-9, abs=1e-6)
    assert checked == 200 and feasible > 50


def test_table_replica_reproduces_option_at_exactly():
    """The fast totals must be bit-identical to option_at, including knots,
    tolerance-near knots, gaps, negative values and the grid edges."""
    grid = [2500.0, 2600.0, 2700.0, 2800.0, 2880.0]
    opt = {"_grid": grid, "oil": [-5.0, 20.0, None, 40.0, 50.0], "water": [100.0, 150.0, None, 170.0, 180.0]}
    ws = WellSurface("W", "B", True, "A", {"A": opt})
    table = cm._prepare(opt)
    assert table is not None and table[3] is False  # negative oil: no closed form
    points = [2400.0, 2500.0, 2500.0 + 5e-10, 2550.0, 2510.0, 2600.0 - 5e-10, 2600.0, 2650.0, 2700.0,
              2750.0, 2800.0, 2800.0 + 1e-10, 2840.0, 2880.0, 2880.0 + 5e-10, 2881.0]
    for p in points:
        assert cm._table_value(table, p) == option_at(ws, "A", p), p


def test_non_monotone_choice_keeps_the_original_search():
    """Two pressure roots: the original fixed point + bracket search runs
    unchanged rather than the closed form picking a branch."""
    grid = [2500.0, 2600.0, 2700.0, 2800.0, 2880.0]
    s = Surfaces(grid, P0, {
        "A": WellSurface("A", "B", True, "cur", {"cur": {"_grid": grid, "oil": [100.0] * 5,
                                                          "water": [9000.0, 1000.0, 9000.0, 1000.0, 9000.0]}}),
    })
    tables = cm._TableCache(s).choice_tables({"A": "cur"})
    eligible, _root = cm._closed_form_root(tables, tuple(grid), anchor(s), 2500.0, 2880.0)
    assert eligible is False
    state = settle({"A": "cur"}, s, anchor(s))
    assert state["feasible"] and state["pressure"] == pytest.approx(P0)
