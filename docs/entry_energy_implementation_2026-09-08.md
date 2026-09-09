# Shared throat-entry energy balance — 2026-09-08

Historical v1 record. The subsequent [fluid follow-up](fluid_followup_2026-09-08.md)
documents v2 property changes, new prediction deltas and separate field holdouts.

Implemented locally as **entry-energy-v1**. This replaces the contradictory
choke/operating energy calculations with one unscaled balance. No deployment,
production data update, compute-tier change or extra warehouse query was made.

## Numerical model

`woffl/flow/entry_energy.py` constructs a positive PCHIP specific-volume path
from the configured isothermal PVT, normally sampled at intervals of at most
10 psi. It integrates that interpolant for pressure work and uses the same
density and conserved stream mass to compute velocity and kinetic energy:

    E(p) = (1 + ken) v(p)^2 / 2 + 144 gc integral[psu -> p] dp/rho(p)

The limit is the **first energy minimum reachable from suction**. Analytic
roots of the interpolant derivative locate it within the sampling intervals;
the calculation never jumps to a later, lower minimum. A bracketed suction
solve makes that minimum zero. The operating throat pressure is a root of
the same balance on that reachable branch.

The 50-psig pressure bound is identified separately and never called sonic.
Wood Mach remains a reported diagnostic; it is not forced to equal one at
the energy limit. An inlet with no reachable solution raises the typed
infeasibility error instead of fabricating an entry state.

Both jetflow entry functions, plotting diagnostics and the whole-well solver
use this calculation. The discharge solver retains its existing bracket,
secant, reseed and bisection sequence. Immutable material paths are reused
within a solve or pump batch through a caller-local context; mutable ResMix
objects are copied for path construction. Existing process workers and the
bounded response cache remain in use.

## Calibration transition

- `mach_crit` is retired. Old library/API signatures still accept the value,
  but it cannot alter energy, flow or the limit; nondefault library calls
  issue a deprecation warning. Its API description states that it is ignored.
- Event calibration now optimizes **ken, kth, kdi and nozzle area**. The
  compatibility fifth result coordinate is always 1.0 and is not searched.
  The former Mach-based floor-escape restart is removed; genuine fit errors
  remain visible in BHP/PF/response residuals.
- Hydration normalizes saved Mach overrides to 1.0. Friction and nozzle wear
  remain available as starting values. They are not automatically claimed
  valid under the corrected model.
- The app displays a notice to review saved calibrations. Calibration Apply
  no longer advertises or applies a fitted Mach adjustment. On an explicit
  user save, an old persisted Mach override is cleared to 1.0; no new
  nondefault Mach value is written. This work itself made no production write.
- Meta, Solver responses, pad/CFP results and event-calibration results report
  `entry-energy-v1`. Response cache keys already include a hash of all physics
  source, so the model change invalidates older response nodes on app startup.

No silent legacy physics mode remains in the active solver. Historical
comparison artifacts are preserved. `tools/critical_mach_study.py` refuses to
rerun its pre-change counterfactual edits against the new engine; use
`tools/entry_energy_validation.py` for the current replay comparison.

## Validation

**1,784 Python tests passed; no failures or expected failures.** The former
three energy-mismatch expected failures are now ordinary passing tests.
Warnings are the existing PVT compressibility floor, pandas concatenation,
and the intentional retirement warning in the old-Mach compatibility test.
After adding result-version metadata, 33 relevant API/calibration tests also
passed. Frontend typecheck/production build and both polling tests passed.
Vite retains the pre-existing large chart-bundle warning.

Fifteen production energy tests cover:

- Independent isothermal ideal-gas limiting pressure and velocity for three
  entry-loss coefficients, using absolute pressures in the analytic solution.
- Full-curve agreement between both entry walks and diagnostics, physical
  kinetic energy, mass conservation, and independent PVT pressure integration.
- The distinction between an interior energy limit and a pressure bound.
- The previously failing gas-rich case across all five legacy Mach inputs.
- Pressure-grid refinement and the zero energy/zero slope at the limit.
- Multiple minima: rejecting a later lower minimum beyond the reachable branch.
- Scoped reuse without mutating input fluids.
- Exact Solver/Batch/Network equality at matched physical inputs, including
  clean and worn nozzle cases. The test supplies the same measured pump depth;
  the API template is selected by TVD while Network's profile input is MD.

The fixture replay has **40 cases: six previous solve errors, zero now**.
All 40 currently solve successfully. These are synthetic/reference fixtures,
not new field measurements. In the gas-rich fixture, suction is approximately
327.70 psig and the sonic flag remains true across the retired Mach inputs;
previously the flag changed and the larger settings failed.

The old 9X regression's reported Mach changes from about 0.888 to 0.938
because the state is evaluated at the energy turning point rather than a
discrete pre-Mach-1 sample. Existing oil, water and suction reference
tolerances still pass. Those small reference changes do not establish
accuracy for every previously calibrated well.

## Medium performance

The offline two-worker benchmark covers four synthetic wells, three pressure
nodes and six pump choices (72 solves). With entry-energy-v1:

| Measurement | Seconds |
|---|---:|
| Fresh process pool at each pressure | 2.978 |
| Shared pool startup, paid once | 0.948 |
| Shared pool with cold response cache | 0.159 |
| Exact cached repeat | 0.0033 |

The three execution paths returned identical DataFrames under the new model.
Twenty local diagnostics requests during another cold sweep measured p50
3.02 ms and p95 5.78 ms. These are local Windows measurements, not hosted
latency promises or complete pad workflow timings. **Medium and the two-worker
limit are retained.** No larger compute tier is required for this implementation.

## Remaining qualification

This fixes internal energy consistency. It does not add a slip, finite-rate
gas-release or thermal model, or fix every density/compressibility approximation.
The configured PVT path remains isothermal; constant water density and the
capped-solution-gas oil-property limitation identified in the investigation
remain. Acoustic Wood speed and the material-path derivative can therefore
still differ. The solver no longer conflates them.

Review/refit older well calibrations and evaluate BHP, PF, oil and pressure
response on whole events excluded from fitting before treating changed pump
recommendations as field-qualified. No new field validation is claimed here.
The strict CI qualification report explicitly limits its scope to entry
energy consistency and reports `field_validated: false`.

Artifacts:

- [Current/previous fixture replay](entry_energy_cases_2026-09-08.json)
- [Energy consistency report](entry_energy_qualification_2026-09-08.json)
- [Medium benchmark](entry_energy_medium_benchmark_2026-09-08.json)
- [Original investigation and options](critical_mach_options_2026-09-08.md)

Reproduce from the repository root:

```powershell
$env:WOFFL_MAX_WORKERS='1'
$env:PYTHONPATH='.'
./venv/Scripts/python.exe -m pytest tests/ -q
./venv/Scripts/python.exe tools/physics_qualification.py --strict
./venv/Scripts/python.exe tools/entry_energy_validation.py
```
