"""One-off: decompose MPM-64's floor error - how much is the calibrated ken?

1. ken sweep at saved geometry: floor(ken) curve from the library default
   (0.03) up to the calibration rail (0.40).
2. Behavior check: with ken forced to the library default, re-model the two
   November SCADA headers (3429 / 3030 psi). Does the model become responsive
   (subsonic, psu moves) or just pin at a lower floor?

Run:
  cd C:/dev/woffl_gui/woffl_gui
  PYTHONPATH=. WOFFL_MAX_WORKERS=8 venv/Scripts/python.exe scripts/mpm64_ken_decompose.py
"""

from __future__ import annotations

import time

T0 = time.time()
TARGET = "MPM-64"
NOV_BHP = 342.7


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"[{time.time() - T0:5.0f}s] hydrating ...")
    from server.services.optimizer_runs import _build_configs, _current_and_tests
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    import woffl.flow.jetflow as jf
    from woffl.geometry.jetpump import JetPump

    notes: list = []
    prov: dict = {}
    configs = _build_configs(["M"], set(), [], notes, prov)
    current, _tests = _current_and_tests([c.well_name for c in configs])
    wc64 = next(c for c in configs if c.well_name == TARGET)
    cc64 = current[TARGET]

    wellbore, wellprofile, inflow, res_mix, prop_pf = (
        NetworkOptimizer._create_well_objects(wc64)
    )
    tsu = wc64.form_temp
    saved_ken = wc64.ken_well
    jp = JetPump(cc64[0], cc64[1])  # library friction: knz .01 ken .03 kth .3 kdi .4
    log(f"pump {cc64[0]}{cc64[1]}, saved ken_well={saved_ken}, library ken={jp.ken}")
    log()

    # -- 1. floor(ken) curve -------------------------------------------------
    log("ken     -> psu_min floor (psi)   [Nov measured BHP 342.7]")
    for ken in [0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40]:
        try:
            psu_min, _q, _b = jf.psu_minimize(
                tsu=tsu, ken=ken, ate=jp.ate, ipr_su=inflow, prop_su=res_mix
            )
            mark = " <- library default" if ken == 0.03 else (
                " <- calibration rail (saved)" if ken == 0.40 else ""
            )
            log(f"  {ken:5.3f} -> {psu_min:7.1f}{mark}")
        except Exception as e:
            log(f"  {ken:5.3f} -> FAILED: {e}")
    log()

    # -- 2. behavior at Nov headers with library ken --------------------------
    log("re-modeling the Nov SCADA headers with ken forced to library 0.03:")
    wc64.ken_well = 0.03
    from woffl.gui.scotts_tools._common import worker_ceiling

    for hdr, meas_bhp, meas_pf in [(3429.0, 342.7, 5500.8), (3030.0, 377.4, 5231.2)]:
        for c in configs:
            c.ppf_surf_well = hdr
        pf = PowerFluidConstraint(total_rate=200000.0, pressure=hdr, rho_pf=62.4)
        opt = NetworkOptimizer([wc64], pf, [cc64[0]], [cc64[1]], marginal_watercut=0.97)
        opt.run_all_batch_simulations(max_workers=1)
        perf = opt.get_pump_performance(TARGET, cc64[0], cc64[1])
        if perf:
            log(
                f"  hdr {hdr:6.0f} | meas BHP {meas_bhp:6.1f} PF {meas_pf:6.0f} | "
                f"model psu {perf['suction_pressure']:6.1f} "
                f"oil {perf['oil_rate']:7.1f} pf {perf['lift_water']:6.0f} "
                f"{'SONIC' if perf['sonic_status'] else 'subsonic'} "
                f"(mach {perf['mach_te']:.2f})"
            )
        else:
            log(f"  hdr {hdr:6.0f} | model unsolved")
    log()
    log(f"[{time.time() - T0:5.0f}s] DONE")


if __name__ == "__main__":
    main()
