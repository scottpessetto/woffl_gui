"""Mixed batch jobs: per-well pump grids run in one pooled call
(woffl.assembly.network_optimizer.simulate_jobs / NetworkOptimizer.well_grids,
server.surface_cache.run_jobs)."""

from types import SimpleNamespace

import woffl.assembly.network_optimizer as no


def _fake_sim(calls):
    def sim(well, pressure, nozzles, throats):
        calls.append((well.well_name, pressure, tuple(nozzles), tuple(throats)))
        return SimpleNamespace(well=well.well_name, grid=(tuple(nozzles), tuple(throats)), pressure=pressure)
    return sim


def test_well_grids_run_each_well_on_its_own_pumps_in_one_call(monkeypatch):
    from woffl.assembly import compute_runtime

    batches = []
    monkeypatch.setattr(compute_runtime, "job_runner", lambda jobs: batches.append(list(jobs)) or [
        SimpleNamespace(well=w.well_name, grid=(tuple(n), tuple(t)), pressure=p) for w, p, n, t in jobs])
    wells = [no.WellConfig(well_name=n, res_pres=1500, form_temp=70, jpump_tvd=4000) for n in ("A", "B", "C")]
    opt = no.NetworkOptimizer(wells, no.PowerFluidConstraint(total_rate=1e5, pressure=3000.0, rho_pf=None),
                              ["12", "13"], ["B"], well_grids={"A": (["12"], ["B"]), "B": (["13"], ["C"])})
    opt.run_all_batch_simulations(max_workers=2)
    assert len(batches) == 1 and len(batches[0]) == 3  # one pooled submit
    assert opt.batch_results["A"].grid == (("12",), ("B",))
    assert opt.batch_results["B"].grid == (("13",), ("C",))
    assert opt.batch_results["C"].grid == (("12", "13"), ("B",))  # falls back to the shared grid


def test_simulate_jobs_serial_fallback_keeps_job_order(monkeypatch):
    from woffl.assembly import compute_runtime

    calls = []
    monkeypatch.setattr(compute_runtime, "job_runner", None)
    monkeypatch.setattr(no, "_simulate_single_well", _fake_sim(calls))
    w = lambda n: SimpleNamespace(well_name=n)
    out = no.simulate_jobs([(w("A"), 3000.0, ["12"], ["B"]), (w("A"), 2800.0, ["12"], ["B"]), (w("B"), 3000.0, [], [])])
    assert [(r.well, r.pressure) for r in out] == [("A", 3000.0), ("A", 2800.0), ("B", 3000.0)]


def test_host_run_jobs_caches_each_job(monkeypatch):
    from server import pool, surface_cache

    calls = []
    monkeypatch.setattr(no, "_simulate_single_well", _fake_sim(calls))
    monkeypatch.setattr(pool, "submit_all", lambda fn, jobs: None)  # no pool: serial path
    monkeypatch.setattr(surface_cache, "_key", lambda well, p, n, t: (well.well_name, p, tuple(n), tuple(t)))
    surface_cache.clear()
    w = lambda n: SimpleNamespace(well_name=n)
    jobs = [(w("A"), 3000.0, ["12"], ["B"]), (w("A"), 2800.0, ["12"], ["B"])]
    first = surface_cache.run_jobs(jobs)
    again = surface_cache.run_jobs(jobs)
    assert len(calls) == 2  # the second call is served from the cache
    assert [r.pressure for r in first] == [r.pressure for r in again] == [3000.0, 2800.0]
    surface_cache.clear()
