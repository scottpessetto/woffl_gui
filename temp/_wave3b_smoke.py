"""Wave 3b smoke checks: SSE parity, ResMix cache, parallel batch sims,
memory-gauge downsampling, module imports.

Everything runs under the __main__ guard: the ProcessPool test spawns
children that re-import this module on Windows, and unguarded top-level
code would re-execute in each child.
"""

import io
import time

import numpy as np
import pandas as pd


def _scalar_sse(bhp_values, fluid_values, pres):
    """Reference: the pre-3b pure-Python implementation, verbatim."""
    n = len(bhp_values)
    if n < 2:
        return float("inf")
    best_sse = float("inf")
    for anchor_idx in range(n):
        anchor_bhp = bhp_values[anchor_idx]
        anchor_fluid = fluid_values[anchor_idx]
        if anchor_bhp >= pres or anchor_fluid <= 0:
            continue
        ratio = anchor_bhp / pres
        denom = 1.0 - 0.2 * ratio - 0.8 * ratio**2
        if denom <= 0:
            continue
        qmax = anchor_fluid / denom
        sse = 0.0
        for j in range(n):
            if bhp_values[j] >= pres:
                sse += 1e8
                continue
            ratio_j = bhp_values[j] / pres
            predicted = qmax * (1.0 - 0.2 * ratio_j - 0.8 * ratio_j**2)
            sse += (predicted - fluid_values[j]) ** 2
        if sse < best_sse:
            best_sse = sse
    return best_sse


def main() -> None:
    # 1. imports
    import woffl.gui.scotts_tools.header_impact  # noqa: F401
    import woffl.gui.scotts_tools.jp_fric_trend  # noqa: F401
    import woffl.gui.scotts_tools.jp_washout  # noqa: F401
    import woffl.gui.memory_gauge  # noqa: F401
    import woffl.gui.workflow_steps.step3_configure_optimize  # noqa: F401

    print("imports: OK")

    # 2. vectorized SSE parity vs the scalar reference
    from woffl.assembly.ipr_analyzer import _calculate_global_sse

    rng = np.random.default_rng(42)
    cases = 0
    for trial in range(300):
        n = rng.integers(1, 25)
        bhp = rng.uniform(100, 2400, n)
        fluid = rng.uniform(-50, 3000, n)  # include some non-positive rates
        pres = float(rng.uniform(150, 2600))
        a = _scalar_sse(bhp, fluid, pres)
        b = _calculate_global_sse(bhp, fluid, pres)
        if np.isinf(a):
            assert np.isinf(b), (trial, a, b)
        else:
            assert abs(a - b) <= 1e-6 * max(1.0, abs(a)), (trial, a, b)
        cases += 1
    assert np.isinf(_calculate_global_sse(np.array([500.0]), np.array([100.0]), 1500))
    assert np.isinf(
        _calculate_global_sse(np.array([1600.0, 1700.0]), np.array([1.0, 2.0]), 1500)
    )
    print(f"SSE parity: OK ({cases} random cases + edges)")

    bhp = rng.uniform(100, 1400, 50)
    fluid = rng.uniform(100, 3000, 50)
    t0 = time.perf_counter()
    for pres in range(1450, 3450, 5):
        _scalar_sse(bhp, fluid, pres)
    t1 = time.perf_counter()
    for pres in range(1450, 3450, 5):
        _calculate_global_sse(bhp, fluid, pres)
    t2 = time.perf_counter()
    print(
        f"SSE grid (50 tests x 400 candidates): "
        f"scalar={t1 - t0:.2f}s vectorized={t2 - t1:.3f}s"
    )

    # 3. ResMix cache correctness
    from woffl.pvt.blackoil import BlackOil
    from woffl.pvt.formgas import FormGas
    from woffl.pvt.formwat import FormWater
    from woffl.pvt.resmix import ResMix

    mix = ResMix(0.6, 400, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    mix.condition(900, 100)
    v1, r1, c1 = mix.volm_fract(), mix.rho_mix(), mix.cmix()
    assert mix.volm_fract() is v1  # cached object returned
    mix.condition(400, 100)  # cache must invalidate
    v2, r2, c2 = mix.volm_fract(), mix.rho_mix(), mix.cmix()
    assert v1 != v2 and r1 != r2 and c1 != c2
    mix.condition(900, 100)  # back to the first condition -> same values
    v3, r3, c3 = mix.volm_fract(), mix.rho_mix(), mix.cmix()
    assert v3 == v1 and r3 == r1 and c3 == c1
    print("ResMix per-condition cache: OK")

    # 4. parallel run_all_batch_simulations == sequential
    from woffl.assembly.network_optimizer import (
        NetworkOptimizer,
        PowerFluidConstraint,
        WellConfig,
    )

    wells = [
        WellConfig(
            well_name=f"SMOKE-{i}", res_pres=1400, form_temp=80, jpump_tvd=4065,
            form_wc=0.894, form_gor=600, qwf=2300, pwf=1049, use_survey=False,
        )
        for i in (1, 2)
    ]
    pf = PowerFluidConstraint(total_rate=10000, pressure=3168)

    def _run(workers):
        opt = NetworkOptimizer(
            wells=wells, power_fluid=pf,
            nozzle_options=["11", "12"], throat_options=["B", "C"],
        )
        res = opt.run_all_batch_simulations(max_workers=workers)
        return {
            w: bp.df.sort_values(["nozzle", "throat"]).reset_index(drop=True)
            for w, bp in res.items()
        }

    seq = _run(1)
    par = _run(2)
    assert set(seq) == set(par) == {"SMOKE-1", "SMOKE-2"}
    for w in seq:
        pd.testing.assert_frame_equal(
            seq[w][["nozzle", "throat", "qoil_std", "lift_wat", "psu_solv"]],
            par[w][["nozzle", "throat", "qoil_std", "lift_wat", "psu_solv"]],
        )
    print("parallel batch sims == sequential: OK")

    # 5. memory-gauge downsampling preserves the daily median
    from woffl.gui.memory_gauge import MemoryGaugeData, parse_xlsx

    ts = pd.date_range("2026-01-01", periods=5 * 17280, freq="5s")  # 5 days @ 5 s
    press = 800 + 30 * np.sin(np.arange(len(ts)) / 5000.0) + rng.normal(0, 2, len(ts))
    buf = io.BytesIO()
    pd.DataFrame({"Date Time": ts, "Pressure": press}).to_excel(
        buf, sheet_name="Data", index=False
    )
    gf = parse_xlsx(buf.getvalue(), "smoke.xlsx")
    assert gf.sample_count == len(ts)  # raw count preserved for the preview
    assert len(gf.raw_df) <= len(ts) / 11  # actually downsampled
    # preview extremes captured from the RAW samples, not the medians
    assert abs(gf.pressure_min - press.min()) < 1e-9
    assert abs(gf.pressure_max - press.max()) < 1e-9
    assert gf.pressure_max > gf.raw_df["pressure"].max()  # spikes preserved
    gauge = MemoryGaugeData(well_name="SMOKE-1", files=[gf])
    raw_daily = (
        pd.DataFrame({"timestamp": ts, "pressure": press})
        .assign(tag_date=lambda d: d["timestamp"].dt.normalize())
        .groupby("tag_date")["pressure"]
        .median()
    )
    merged = gauge.daily_df.set_index("tag_date")["bhp"]
    diff = (merged - raw_daily).abs().max()
    assert diff < 1.0, f"daily median drifted {diff:.2f} psi"
    print(
        f"memory-gauge downsample: OK (max daily-median drift {diff:.3f} psi, "
        f"{len(ts):,} rows -> {len(gf.raw_df):,})"
    )

    print("all wave 3b smoke checks passed")


if __name__ == "__main__":
    main()
