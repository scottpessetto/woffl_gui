# Fluid and validation follow-up — 2026-09-08

> **Documentation status, 2026-09-08:** V2 implementation/holdout milestone. Test counts and timings below are from that stage. Later work added the WC GUI and [installed-pump scope](pump_calibration_scope_2026-09-08.md); the [handoff](session_learnings_2026-09-08.md) records the final 1,861 Python / 8 frontend baseline and remaining deployment work.

Implemented locally as **entry-energy-v2**. This completes the code changes
identified after the shared-energy work: independent PF density, low-GOR oil
compression, pressure/temperature-dependent water, separate field holdouts,
deployment dependency cleanup, and misleading progress/calibration labels.
Databricks remains configured for **Medium, two workers**. No deployment or
production property writes were performed. Field validation used one read-only
snapshot; subsequent runs reused it locally.

## Changes and their purpose

**PF density now affects the calculation.** The sidebar, single solver, batch,
pad and CFP optimization paths carry an independent power-fluid density. It is
specified at **0 psig / 60 F**, separate from formation-water SG. Explicit
per-well density overrides the network fallback; pad defaults use plant SG.
The generic default is 63.648 lbm/ft3, matching the formerly used 1.02 SG preset.
Resolved density is part of the response-cache key. Formation and lift water
use separate mutable objects, and the throat mixture conserves both water masses.

**Low-GOR oil no longer stops compressing when its available gas is exhausted.**
The effective bubble point is obtained by inverting the existing solution-GOR
correlation for the available inventory. Above it, Bo integrates the existing
floored Vasquez-Beggs compressibility law, and viscosity uses the above-bubble
correlation. The integral is exact for the piecewise A/P law, including its
floor; finite differences verify `-d(log Bo)/dP = co`. This correction also
improves integration for uncapped oil above the preset bubble point.

**Water properties now change with pressure and temperature.** Density and its
isothermal derivative use IF97 region 1. The standard-density input scales the
pure-water relative response. The liquid domain is checked using the region-4
saturation equation. Published reference volumes are reproduced within 4e-9
relative error. [IAPWS IF97 release, Tables 2–5 and 34–35](https://iapws.org/documents/release/IF97-Rev.download).

Viscosity uses the industrial form of the IAPWS release, checked against three
published liquid-water values to 6e-10 cP absolute tolerance. No new runtime
package is required. SG does **not** infer a brine-viscosity correction.
[IAPWS viscosity release, equations 10–12 and Table 4](https://iapws.org/technical-guidance/release/viscosity.download).

The nozzle integrates pressure work along the PF density path and converts
exit volume to standard rate using mass conservation. The column converts
standard volume to in-situ volume once. A duplicate conversion found in the
final review was removed and is guarded for both tubing and annulus. Throat
secant/bracket searches respect the physical incoming-momentum pressure bound,
preventing excursions outside the water-property domain. The established
fallback chain and Vogel inflow convention remain in use.

**Labels describe what was measured.** S-Pad progress displays pressure rather
than labelling swept flow as psi, including skipped trials with no available
header. Calibration reports BHP *fit RMS error* and *estimated nozzle area*;
it no longer presents RMS as an error bound or fitted area as proven wear.
The API identifies v2. The transition notice was subsequently removed at the
user's request; installation-specific fit-quality warnings remain. Legacy fits
now require [verified pump scope](pump_calibration_scope_2026-09-08.md).

**Deployment uses the vendored physics.** Removed the PyPI `woffl` copy and
unused `databricks-sdk`. Direct runtime dependencies and their resolved
dependencies are pinned through `requirements.txt` and
`requirements-constraints.txt`. The app dependency set requires Python >=3.11.
An isolated Windows/Python 3.13 environment installed successfully, passed
`pip check`, and imported the API using local physics with no site-packages
`woffl` distribution. `tools/deployment_smoke.py` adds this check to CI. Hosted
Linux deployment has not been run. The older multipoint validation script also
now obeys the worker ceiling with a maximum of two instead of hardcoding four.

## Verification and output changes

- Final complete local suite: **1,809 passed**, four existing warnings, 37.56 s.
- Final isolated fresh-install suite: **1,809 passed**, six warnings, 52.76 s.
  Dependency/import checks passed with the final pins. The two additional
  warnings are dependency deprecations in the API test client.
- Frontend TypeScript/Vite build and both polling tests passed; tracked `web/dist`
  was rebuilt. The existing large chart chunk warning remains.
- Strict entry-energy consistency: zero mismatch for every retired Mach input.
  This checks internal agreement, not field accuracy.
- New tests check independent published water values, fluid derivatives,
  standard mass in the column and throat, nozzle pressure-work integration,
  PF-input propagation, cache invalidation, and held-out label isolation.

[Eight-case comparison](fluid_followup_cases_2026-09-08.json), against the saved
v1 results, has no failed solves. BHP changes range from **−0.83 to +3.88 psi**;
oil changes are within 0.27% in these cases. A separate large-pump **16E** fixture
changes from 34.79482 to **32.45012 BOPD** (−6.74%) and 1350.57 to
**1358.31 psi** (+7.74 psi). Its compatibility pins were updated with their
existing tolerances. Small BHP movement does not guarantee small rate movement.

The unchanged independent HYSYS dataset remains an approximate cross-model
comparison. Its assertions now use HYSYS as the reference denominator. Maximum
relative water/gas volume-fraction differences are **4.027% / 6.389%**; tolerances
were explicitly widened from 4% / 6% to **4.1% / 6.5%**, respectively. The gas
mass-fraction tolerance remains 6%. These discrepancies are recorded, not evidence
that the models agree exactly. Strict published-water and conservation checks
are separate. [Measured discrepancies](fluid_followup_hysys_2026-09-08.json).

## Field holdouts: discrepancies remain

The new CLI groups observations by pressure event or time gap, withholds the
latest eligible whole event, leaves a three-day embargo, and removes training
daily records that borrow future test-rate anchors. It fits at most 20 training
points from neutral coefficients. Held-out predictions receive only date, PF
pressure and wellhead pressure, using a fixed IPR/WC/GOR derived from actual
training-period well tests. Derived daily oil is excluded from oil-error scoring.
Failures are counted, including nonfinite predictions. Exact fitted configuration
and training observations are retained for offline replay.

Snapshot captured **2026-09-08 16:51 UTC**. These are retrospective tests using
supplied geometry/reservoir pressure, not prospective field qualification.

| Well | Train / held observations | Training BHP RMS | Held BHP RMS | Held PF RMS | Held oil RMS | Actual held oil tests |
|---|---:|---:|---:|---:|---:|---:|
| MPM-64 | 67 / 31 | 37.83 psi | **48.55 psi** | **4.90%** | **270.88 BOPD** | 1 |
| MPM-28 | 29 / 28 | 43.31 psi | **23.15 psi** | **0.74%** | **108.22 BOPD** | 3 |
| MPM-45 | 47 / 31 | 62.43 psi | **58.96 psi** | **5.04%** | **26.21 BOPD** | 1 |

All 90 held observations solved. All three fits hit the ken bound; MPM-64 also
hit kdi. MPM-45's held pressure-separated pairs have **165.87 psi RMS error in
change of BHP** (28 correlated pairs, not 28 independent events). The other two
holdouts lack 100-psi-separated pairs, so response accuracy is unscored there.

Training-to-held-test median watercut changes from 39.3% to 16.0% on MPM-64,
40.8% to 54.7% on MPM-28, and 41.3% to 62.8% on MPM-45. Changing inflow/composition
is a plausible contributor to the oil errors, **not a demonstrated explanation**.
No model or fit was tuned to these held-out outcomes. Saved production fits
were not changed. [Summary and snapshot fingerprint](fluid_followup_field_holdout_2026-09-08.json).

The next physics priority is explaining those field discrepancies using stable
pump periods, measured PF density/temperature and composition, contemporaneous
IPR evidence, and additional independent pressure steps. More compatibility
pins alone will not resolve them. Current model approximations still include
SG-scaled pure water rather than brine chemistry, a bulk-property PF column,
isothermal flow, and empirical oil/acoustic floors.

## Medium performance

The local offline workload uses four wells, three pressure nodes and six pump
choices with **two workers**. Fresh pools took **2.99 s**, a started shared pool
took **0.203 s** for the cold sweep, and cached repeats took **0.00448 s**.
Pool startup was measured separately at 0.938 s. All result frames matched
exactly. The cache used 184 KB against its 64 MiB cap; a lightweight API read
during a sweep had 4.36 ms local p95. This measures Python compute locally,
not Databricks or warehouse latency. The v1 cold sweep was 0.159 s in its earlier
run; the richer PVT adds some cold compute, while repeat calculations retain
the cache benefit. [Benchmark](fluid_followup_medium_benchmark_2026-09-08.json).

## Reproduce

From the repository root in PowerShell:

```powershell
$env:WOFFL_MAX_WORKERS='1'
$env:PYTHONPATH='.'
./venv/Scripts/python.exe -m pytest tests/ -q
./venv/Scripts/python.exe tools/fluid_followup_validation.py --output build/fluid-cases.json
./venv/Scripts/python.exe tools/physics_qualification.py --strict --output build/fluid-consistency.json
./venv/Scripts/python.exe tools/field_holdout_validation.py
./venv/Scripts/python.exe tools/benchmark_medium.py --output build/fluid-benchmark.json
```

Field validation defaults to the ignored local snapshot. `--live --fetch-only`
explicitly refreshes it using reads. Full field observations and replay inputs
remain in `build/`; only the summary is in this report. The benchmark sets its
worker budget to two. In a newly installed environment, run `python -m pip check`
and `python tools/deployment_smoke.py` with `PYTHONPATH=.`.
