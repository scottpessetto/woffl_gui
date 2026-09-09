"""Medium-tier caching/scheduling guarantees, independent of wall-clock speed."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event, Lock
from types import SimpleNamespace
import pickle

import pandas as pd
import pytest

from server import pool, surface_cache
from woffl.assembly import compute_runtime, network_optimizer as no


@pytest.fixture
def runtime(monkeypatch):
    from woffl.assembly import parallelism
    monkeypatch.setattr(parallelism, "worker_ceiling", lambda: 2)
    monkeypatch.setattr(pool, "_EXECUTOR_CLS", ThreadPoolExecutor)
    monkeypatch.setattr(pool, "_prime", lambda: True)
    pool.stop()
    pool.start()
    surface_cache.start()
    yield
    surface_cache.stop()
    pool.stop()


def well():
    return no.WellConfig(well_name="Custom", res_pres=1700., form_temp=70., jpump_tvd=4065.)


def test_exact_cache_isolated_and_invalidates_inputs(runtime, monkeypatch, tmp_path):
    calls = []
    def simulate(cfg, pressure, nozzles, throats):
        calls.append((cfg, pressure))
        return SimpleNamespace(df=pd.DataFrame({"oil": [cfg.form_wc * pressure]}))
    monkeypatch.setattr(no, "_simulate_single_well", simulate)
    monkeypatch.setattr(surface_cache, "ROOT", tmp_path)
    cfg = well()
    first = surface_cache.run_batches([cfg], 3000., ["12"], ["B"])
    first["Custom"].df.loc[0, "oil"] = -999
    second = surface_cache.run_batches([cfg], 3000., ["12"], ["B"])
    assert second["Custom"].df.loc[0, "oil"] == 1500
    assert len(calls) == 1
    # Every dataclass input participates, not a hand-selected subset.
    for attr, value in [("form_wc", .974), ("fnz_well", 1.1), ("mach_crit_well", 1.5),
                        ("jpump_md", 4500.), ("jpump_direction", "forward"),
                        ("gas_sg", .7), ("kth_well", .6), ("pwf", 600.), ("rho_pf", 66.)]:
        surface_cache.run_batches([replace(cfg, **{attr: value})], 3000., ["12"], ["B"])
    assert len(calls) == 10
    surface_cache.run_batches([cfg], 3200., ["12"], ["B"])
    surface_cache.run_batches([cfg], 3000., ["13"], ["B"])
    assert len(calls) == 12
    survey = tmp_path / "woffl/jp_data/well_surveys/Custom Deviation Survey.csv"
    survey.parent.mkdir(parents=True)
    survey.write_text("meas_depth,tvd_depth\n0,0\n5000,4000\n")
    surface_cache.run_batches([cfg], 3000., ["12"], ["B"])
    monkeypatch.setattr(surface_cache, "_MODEL", "new-physics")
    surface_cache.run_batches([cfg], 3000., ["12"], ["B"])
    assert len(calls) == 14


def test_cache_byte_limit_and_expiry(monkeypatch):
    surface_cache.clear()
    size = len(pickle.dumps({"payload": "x" * 100}, protocol=5))
    monkeypatch.setattr(surface_cache, "MAX_BYTES", size * 2)
    for i in range(20):
        surface_cache._put(i, {"payload": "x" * 100})
    assert surface_cache.status()["bytes"] <= size * 2
    assert surface_cache.status()["entries"] == 2
    monkeypatch.setattr(surface_cache, "monotonic", lambda: float("inf"))
    assert surface_cache._get(19) is None
    surface_cache.clear()


def test_network_host_hook_avoids_new_executor_and_reuses_nodes(runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(no, "_simulate_single_well", lambda *args: calls.append(args) or {"oil": 80.})
    cfg = well()
    for capacity in [1000., 2000.]:
        opt = no.NetworkOptimizer([cfg], no.PowerFluidConstraint(capacity, 3000.), ["12"], ["B"])
        assert opt.run_all_batch_simulations(max_workers=2) == {"Custom": {"oil": 80.}}
    assert len(calls) == 1  # budget affects allocation, never physics


def test_batch_failure_is_not_cached_and_fallback_completes(runtime, monkeypatch):
    calls = []
    def simulate(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ValueError("synthetic failure")
        return {"oil": 80.}
    monkeypatch.setattr(no, "_simulate_single_well", simulate)
    with pytest.raises(ValueError):
        surface_cache.run_batches([well()], 3000., ["12"], ["B"])
    pool.stop()
    assert surface_cache.run_batches([well()], 3000., ["12"], ["B"])["Custom"]["oil"] == 80.
    assert len(calls) == 2


def test_global_tokens_bound_overlapping_batches_and_allocation(runtime):
    active = maximum = 0
    lock = Lock()
    two_active = Event()
    release = Event()
    def work(i):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                two_active.set()
        assert release.wait(5)
        with lock:
            active -= 1
        return i
    with ThreadPoolExecutor(max_workers=3) as callers:
        a = callers.submit(pool.submit_all, work, [(i,) for i in range(6)])
        b = callers.submit(pool.submit_all, work, [(i,) for i in range(6)])
        def allocation():
            with pool.cpu_slot():
                return work(99)
        c = callers.submit(allocation)
        assert two_active.wait(5)
        release.set()
        assert a.result(timeout=5) == list(range(6))
        assert b.result(timeout=5) == list(range(6))
        assert c.result(timeout=5) == 99
    assert maximum == 2


def test_real_cached_response_matches_uncached():
    surface_cache.start()
    try:
        cfg = well()
        raw = no._simulate_single_well(deepcopy(cfg), 3000., ["12"], ["B"])
        first = surface_cache.run_batches([cfg], 3000., ["12"], ["B"])["Custom"]
        cached = surface_cache.run_batches([cfg], 3000., ["12"], ["B"])["Custom"]
        pd.testing.assert_frame_equal(raw.df, first.df, check_exact=True)
        pd.testing.assert_frame_equal(raw.df, cached.df, check_exact=True)
    finally:
        surface_cache.stop()


def test_performance_endpoint_does_not_query_warehouse(monkeypatch):
    from fastapi.testclient import TestClient
    from server.main import app
    from woffl.assembly import databricks_client
    monkeypatch.setattr(databricks_client, "execute_query", lambda *a: pytest.fail("unexpected SQL"))
    response = TestClient(app).get("/api/meta/performance")
    assert response.status_code == 200
    assert "Server-Timing" in response.headers
    assert response.json()["surface_cache"]["max_bytes"] == 64 * 1024 * 1024
