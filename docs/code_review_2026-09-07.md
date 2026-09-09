# Code and physics review — 2026-09-07

> **Documentation status, 2026-09-08:** Historical review. The subsequent [fix record](code_review_2026-09-07_fixes.md), shared-energy/fluid corrections and installed-pump scope supersede pre-fix findings. Use the [September 8 handoff](session_learnings_2026-09-08.md) for the current queue and verified baseline; this report is not a new code audit.

**Follow-up:** R01–R11 and O01–O02 have now been fixed in the working tree.
See [implementation and validation](code_review_2026-09-07_fixes.md).
The review observations below describe the pre-fix state; the separate older
physics limitations remain as identified.

## Outcome

The review found substantive correctness issues despite a passing test suite. The highest priorities are component conservation, watercut precision, consistent treatment of installed pump wear, and physical feasibility of optimization results. These can change predicted production or recommended operating settings; their fleet-wide effect has not been quantified.

This report contains **11 findings beyond the earlier optimization review**, followed by two reproduced optimization issues already documented there and remaining physics debt. P1 means address before relying on the affected calculation or workflow; P2 means a narrower correctness or reliability defect. These are review priorities, not claims of observed field incidents.

This review adds an offline reproduction script and its results; it does **not** change the physics or other application behavior. The optimization fixes made earlier in this session remain in the working tree and are described in [the optimization review](optimization_review_2026-09-07.md).

## Scope and verification

Reviewed the main paths through `woffl/pvt`, `flow`, `geometry`, `assembly`, GUI compute modules, FastAPI services and schemas, React workflow controls and polling, and deployment requirements. Particular attention went to units, conservation, conditioning/mutation, solver convergence, calibration parameter propagation, saved data, capacity constraints, and failure handling. Earlier findings in [the September 1 review](code_review_2026-09-01.md) and the [model trust document](model_trust_2026-08-10.md) were checked to avoid treating known design limitations as newly discovered defects.

Validation on the current working tree:

- **1,684 tests passed**, with two existing warnings, in 28.65 seconds using the project virtual environment and `WOFFL_MAX_WORKERS=1`.
- `npx tsc --noEmit` passed.
- Parsed all 219 Python files under the library, server, tests, tools, and scripts. This is syntax coverage, not a claim that every line was manually audited.
- The targeted pyflakes check found no undefined-name or referenced-before-assignment diagnostics. `pip check` found no broken installed requirements; this does not establish that the declared deployment requirements are complete.
- Offline probes reproduced the numerical and parameter-propagation findings below. [Script](../tools/review_errors_2026_09_07.py) · [Recorded JSON results](code_review_2026-09-07_probes.json).

No live warehouse reads/writes, deployment, or field validation was performed. Mocked service dependencies and synthetic optimization surfaces isolate the failure modes; they do not establish how often each happens in production. No browser end-to-end test or independent petroleum simulator benchmark was performed.

Reproduce from the repository root in PowerShell:

```powershell
$env:PYTHONPATH='.'
./venv/Scripts/python.exe tools/review_errors_2026_09_07.py --output docs/code_review_2026-09-07_probes.json
```

The script records observations, rather than adding tests that bless incorrect behavior. It deliberately simulates missing multipart support and prints FastAPI's expected dependency error. Check the JSON for any `probe_error` entries if rerunning in another environment; the recorded run has none.

## Findings

### R01 — P1: pressure changes the mass flow of a fixed produced mixture

**Location:** `woffl/pvt/resmix.py:409`, especially the standard-oil mass anchor and conversion to total volume in `_static_insitu_volm_flow`; related live-oil density and dissolved-gas fractions in `blackoil.py` and `_owg_mass_fraction`.

The code divides stock-tank oil mass by **live-oil** density and scales all phases using live-mixture volume fractions. Dissolved gas contributes to live-oil mass, but is absent from that anchoring mass. Consequently the flow conversion disagrees with the phase fractions and oil formation volume factor.

**Reproduction:** 100 stock-tank BOPD, 80% watercut, GOR 250, Schrader fluid presets, 100°F. The water model has constant density, so the 400 BPD water component should remain 400 BPD when pressure changes. Instead:

| Pressure, psig | Returned water, BPD | Returned oil, reservoir BPD | Oil from 100 × Bo, reservoir BPD | Total mass, lbm/s |
|---:|---:|---:|---:|---:|
| 0 | 399.874 | 101.647 | 101.692 | 2.04210 |
| 1,000 | 391.431 | 104.431 | 106.725 | 1.99898 |
| 2,000 | 385.318 | 105.863 | 109.900 | 1.96777 |

At 2,000 psig, water is **3.67% below the specified component rate**. This is an internal conservation error, not uncertainty in a field correlation. It affects mixture velocity and calculations using that velocity, including friction and jet-pump energetics. The resulting oil-production bias needs further evaluation and cannot be inferred directly from the water error.

**Correction:** derive phase flows from consistent conserved standard component masses, partition dissolved/free gas, and reconcile live-oil volume with Bo. Verify water and total component mass conservation over pressure, temperature, GOR, and watercut, including bubblepoint transitions and the separate water-mode path. Only then assess solver and calibration changes. Black-oil models explicitly account for gas dissolution and its effect on formation volume factors; see [SINTEF's black-oil example](https://www.sintef.no/projectweb/mrst/modules/ad-core/spe9/).

### R02 — P1: watercut rounding silently changes saved well characterization

**Location:** `server/services/wells.py:443`, `489`, `543`, and `554`.

Context hydration rounds test, locked, and saved watercut values to two decimals. This violates the repository's requirement to retain at least three decimals and is particularly consequential at high watercut.

**Reproduction through `well_context`:** a saved 0.974 watercut and 1,000 BPD liquid become 0.97 and 30 BOPD, versus the saved anchor's 26 BOPD: **15.38% more implied oil**. A valid saved 0.986 becomes 0.99; `_config_from_seeds` then rejects the well at its upper watercut boundary. Neither case is reported in `clamped`.

**Correction:** preserve computational precision through hydration, locking, saves, and optimizer construction; round only display text. Verify a save/load round trip and acceptance of 0.986 in addition to the 0.974 oil-rate invariant.

### R03 — P1: single-well pump sizing applies installed wear to alternative pumps

**Location:** `server/services/solve.py:344`; compare `woffl/assembly/network_optimizer.py:650`.

The single-well batch applies `nozzle_area_factor` to every candidate. The network path restricts it to the installed pump identity when that identity is available. The model trust document says changeout candidates use a clean nozzle factor of 1.0.

**Reproduction:** with installed 12B and area factor 1.2, single-well batch constructs both 12B and alternative 13B at 1.2. Network batch constructs 12B at 1.2 and 13B at 1.0. This compares different physical pumps under the same candidate labels and can change their ranking.

**Correction:** explicitly distinguish the existing installed pump from new replacement candidates and apply that policy consistently in all sizing paths. Include same-size replacement semantics, rather than assuming a matching size always means the old physical pump.

### R04 — P1: saved nozzle wear survives a pump replacement

**Location:** `woffl/gui/ipr_anchor.py:609` and `server/services/wells.py:519`.

Saved friction hydration retains parameter values but drops their individual timestamps. Context then restores nozzle area factor without checking whether the current pump was installed after that calibration. A new pump therefore inherits the previous pump's wear.

**Reproduction:** area factor 1.2 saved January 1, 2025; tracker reports a new 13B set September 1, 2026. Hydration selects the new nozzle and restores factor 1.2.

**Correction:** retain calibration provenance and associate hardware-specific wear with pump identity/installation era. Reset stale nozzle wear at replacement. Do not indiscriminately reset well-specific parameters such as `mach_crit`; the [documented policy](model_trust_2026-08-10.md) distinguishes them.

### R05 — P1: event calibration's single-point fallback fits different physics from the applied result

**Location:** `server/services/event_calibration.py:123`, especially the call to `calibrate_friction_coefs`.

The fallback omits `nozzle_area_factor`, `mach_crit`, `seed_kth`, and `seed_kdi`, although the normal single-point calibration path forwards them. The fallback therefore uses default wear/choking physics. The UI applies the returned friction coefficients without replacing the user's existing wear/choking parameters.

**Reproduction:** a config with area factor 1.2, critical Mach 1.5, kth 0.6, and kdi 0.7 forwards none of those four arguments. Thus a successful match need not remain a match when applied.

**Correction:** share the calibrated model assembly/argument mapping across entry points; verify that applying a fit reproduces its reported BHP under the same configuration. Also review the fallback's test selection: `_latest_test_target` is not restricted to the current pump era, so a young era can use an older pump's test. That age-selection concern was found by code tracing, not independently measured against field histories.

### R06 — P1: CFP can rank a pressure/flow state that has not converged

**Location:** `woffl/gui/cfp_moves.py:227`, `settle`.

After eight fixed-point iterations the function returns `feasible=True` whenever the chosen surfaces contain values, even if plant pressure and well demand are still inconsistent. It reports neither convergence nor the final coupling residual.

**Reproduction:** a synthetic steep two-point surface returns 2,665.52 psi and `feasible=True`; applying the plant curve to the returned water yields 2,500 psi, a **165.52 psi mismatch**. This stress case demonstrates the missing acceptance condition; its frequency on actual fleet curves is unknown.

**Correction:** validate the final pressure residual, use a bracketed fallback where applicable, and exclude unconverged states from candidate ranking. Recompute trip/capacity flags from the accepted final state. Test convergent, oscillating, and out-of-domain demand curves.

### R07 — P1: evidence-derived BHP changes when the search range changes

**Location:** `woffl/gui/pad_optimize.py:1397`, especially the correction around line 1485; `server/services/evidence.py:246`.

The evidence service supplies a recent measured BHP reference, but no paired power-fluid pressure reference. `_apply_suction_evidence` treats the highest modeled sweep level as the reference pressure. That ties a physical prediction to a search setting rather than to the measurement's operating condition.

**Reproduction:** identical evidence (reference BHP 500 psi, beta 0.1) and evaluation pressure 3,000 psi yield corrected BHP 520 psi with a 3,200 psi sweep ceiling, or 550 psi with a 3,500 psi ceiling. Only the search ceiling changed.

**Correction:** preserve a pressure/BHP reference from aligned observations and evaluate the response against that fixed reference. Verify search-range invariance at shared pressure points, then validate any resulting oil projection against the same reference and IPR.

### R08 — P1: declared production dependencies omit required upload support

**Location:** `requirements.txt`; upload routes in `server/routers/gauge.py:28` and `server/routers/tools.py:138`.

The application registers FastAPI `File` routes, but declares neither `python-multipart` nor a FastAPI extra that installs it. The local environment already contains it, concealing the incomplete dependency declaration.

**Reproduction:** blocking multipart imports without uninstalling anything triggers FastAPI's route dependency guard: `Form data requires "python-multipart" to be installed.` This reproduces the missing-module failure, not a complete fresh Databricks deployment.

**Correction:** declare `python-multipart` and validate importing the application from a clean environment installed from the deployment requirements. This dependency is explicitly required in [FastAPI's upload documentation](https://fastapi.tiangolo.com/tutorial/request-files/).

### R09 — P2: API-valid pipe dimensions can create a zero-area annulus

**Location:** `woffl/geometry/pipe.py:106`; associated `SimParams` geometry fields.

The fit check rejects inner-pipe OD greater than outer-pipe ID, but allows equality. Independent API field validation also permits this combination.

**Reproduction:** tubing OD 4.5 inches, casing OD 5.5 inches, casing wall 0.5 inches passes schema and geometry construction, producing annulus area and hydraulic diameter of zero. Downstream velocity/friction calculations cannot use this geometry.

**Correction:** reject zero clearance in `PipeInPipe` and add cross-field schema validation so the user receives a clear geometry error before solving. Cover exact equality and a valid positive clearance. The probe confirms invalid geometry acceptance; it does not assert a particular HTTP error response.

### R10 — P2: connector retry can repeat an already successful INSERT

**Location:** `woffl/assembly/databricks_client.py:157` and `292`.

Reads and writes share a helper that retries the entire operation after connection errors. Cursor cleanup is inside that retry scope. A failure after successful execution can consequently rerun a write, duplicating append-only records.

**Reproduction:** a fake write runner succeeds, then its cursor's first close raises `ConnectionError`. The helper executes the runner twice and returns success. No database or production write gate is involved in this reproduction.

**Correction:** separate read retry policy from write handling. Do not replay writes after an ambiguous execution outcome; use explicit outcome handling or an appropriately designed idempotency mechanism. Verify both a pre-execution connection failure and post-execution cleanup failure.

### R11 — P2: one polling error discards a running job's browser handle

**Location:** `web/src/pages/optimize/RunPanel.tsx:876`, `MatchHealthPanel.tsx:132`, `solver/EventCalibration.tsx:212`, and `sensitivity/CombinePanel.tsx:94`; corresponding query hooks in `web/src/api/hooks.ts`.

These effects clear the persisted job ID on any `job.isError`. The polling hooks disable retries. A temporary network/server error is therefore treated like an expired job; the backend work may continue while the user loses the normal route back to its result.

**Evidence:** traced query error handling and persisted-state clearing. This finding has not been reproduced in an automated browser session.

**Correction:** preserve job IDs on transient failures, retry polling with bounded backoff, and clear only on an explicit terminal/unknown-job response. Test interruption followed by recovery while the original job completes.

## Reproduced optimization issues already documented earlier

### O01 — P1: choke allocation ignores formation water on a total-water plant

**Location:** `woffl/gui/pad_optimize.py:1304` and `1528`.

The main pump optimizer now uses `plant.water_key`, following the earlier fix. The separate choke planner still builds PF-only staircases, trims PF against plant capacity, and anchors today's pressure from PF only. This is wrong for the M-Pad total-machine-water basis. The choke planner currently accepts free-pressure plants, so this specific finding should not be described as an E-Pad choke run.

**Reproduction:** a synthetic total-water plant with a 1,000 BPD capacity accepts 900 BPD PF and 100 BOPD at 90% watercut. Formation water adds 900 BPD, making machine water **1,800 BPD**. The returned allocation exceeds capacity by 80%.

**Correction:** use the plant's water basis throughout feasibility, marginal costs, pressure anchoring, and degraded-header ladders; retain PF as a separate reporting quantity. Held/test-based wells also need a defensible formation-water estimate.

### O02 — P2: automatic water pricing can choose shut-in on an oil-producing tie

**Location:** `woffl/assembly/optimization_algorithms.py:91` and `404`.

For candidate `(water, oil)` rates `(80, 80)` and `(100, 100)` with water capacity 90, automatic pricing derives lambda 1.0. All choices, including shut-in, tie at zero priced objective. The real MILP returns zero oil; lambda zero returns the feasible 80 BOPD choice.

This is a distinction between economic objective semantics and maximizing constrained oil, not a failure of the MILP solver. Clarify automatic-mode intent and implement a deterministic secondary objective if oil production should win priced ties. Verify both slack and constrained cases before changing the policy.

## Remaining physics and performance limitations

- **Critical-Mach energy consistency remains unresolved.** `throat_entry_mach_one` and `throat_entry_zero_tde` do not use the same `mach_crit` closure. This is the previously documented FLOW-4/SOLV-F1 issue. Later solver guards mean old numerical examples should not be assumed unchanged; this review confirms the structural mismatch, without claiming to have reproduced the old percentage errors. A consistent closure requires physics validation across both paths.
- **Water PVT remains simplified.** Constant density/viscosity limits temperature/pressure fidelity. This is distinct from R01: even with those simplifications, conserved water mass must stay conserved. No new field error estimate is assigned here.
- **Calibration cannot establish physical correctness by itself.** A good BHP match can compensate for input or model errors. After R01/R02 are corrected, reassess stored fits and compare against independent BHP, PF rate, and oil observations before treating the changed predictions as improved accuracy.
- **Performance work is still worthwhile**, especially reusing worker pools and avoiding redundant model surfaces across sweeps. This review does not attach a speedup estimate; benchmark representative runs after correctness fixes so faster execution does not conceal invalid candidates.

## Recommended implementation sequence

1. Fix watercut hydration, calibration argument propagation, wear provenance/candidate policy, and dependency declaration. Add narrow regression tests for the reproduced failures.
2. Correct component conservation and implement an agreed consistent choking closure. Follow the repository's shared-library change process, including upstream annotations, `docs/upstream_sync.md`, and meaningful regression coverage. Evaluate stored-calibration impact explicitly.
3. Enforce total-water choke capacity, CFP convergence, and fixed observation references for evidence corrections. Use capacity/residual invariants as acceptance conditions.
4. Repair ambiguous write retry handling and transient polling recovery; settle automatic pricing semantics and tie-breaking.
5. Compare revised results with representative held-out well observations and then benchmark optimization runtime.

Passing the existing suite establishes compatibility with its current assertions. It does not override these reproduced physical and workflow inconsistencies.
