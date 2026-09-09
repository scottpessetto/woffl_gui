"""Offline Medium-size workload: fresh pools vs shared/cached response nodes.

No warehouse, production data, or writes. Measures Python compute only;
hosted warehouse latency must be read from /api/meta/performance after deploy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("medium-benchmark.json"))
    args = parser.parse_args()
    os.environ["WOFFL_MAX_WORKERS"] = "2"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import pandas as pd
    from server import pool, surface_cache
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint, WellConfig

    wells = [WellConfig(well_name=f"Custom-{i}", res_pres=1700., form_temp=100.,
                        jpump_tvd=4065., form_wc=.8, qwf=500.) for i in range(4)]
    def sweep(pressures=(2800., 3000., 3200.)):
        frames = {}
        for pressure in pressures:
            opt = NetworkOptimizer(wells, PowerFluidConstraint(20000., pressure), ["11", "12", "13"], ["B", "C"])
            for name, batch in opt.run_all_batch_simulations(max_workers=2).items():
                frames[(name, pressure)] = batch.df.copy(deep=True)
        return frames

    timings = {}
    def timed(name, fn):
        start = perf_counter()
        value = fn()
        timings[name] = round(perf_counter()-start, 6)
        return value

    pool.stop()
    surface_cache.stop()
    baseline = timed("fresh_pools_seconds", sweep)
    try:
        timed("shared_pool_start_seconds", pool.start)
        surface_cache.start()
        cold = timed("shared_cold_seconds", sweep)
        warm = timed("cached_seconds", sweep)
        for key in baseline:
            pd.testing.assert_frame_equal(baseline[key], cold[key], check_exact=True)
            pd.testing.assert_frame_equal(baseline[key], warm[key], check_exact=True)

        # Observe a lightweight API during a real cache-miss sweep. No
        # lifespan context: deliberately do not start warehouse warmup.
        from fastapi.testclient import TestClient
        from server.main import app
        samples = []
        client = TestClient(app)  # no lifespan: no warehouse warmup
        with ThreadPoolExecutor(max_workers=1) as callers:
            future = callers.submit(sweep, (2700., 2900., 3100.))
            for _ in range(20):
                start = perf_counter()
                response = client.get("/api/meta/performance")
                assert response.status_code == 200
                samples.append((perf_counter()-start)*1000)
            future.result()
        samples.sort()
        report = dict(workers=pool.workers(), wells=4, pressure_nodes=3,
                      pump_choices=6, timings=timings, bit_identical=True,
                      cache=surface_cache.status(),
                      local_testclient_read_p50_ms=samples[9],
                      local_testclient_read_p95_ms=samples[18],
                      scope="Offline local compute; not a hosted latency guarantee")
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        surface_cache.stop()
        pool.stop()


if __name__ == "__main__":
    main()
