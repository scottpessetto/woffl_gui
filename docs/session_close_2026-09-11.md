# September 11, 2026 — end-of-night handoff

The user asked to document all work and next steps, then stop for the night.
**Resume from this file when the user returns.** No further implementation,
benchmark or background test was started during this documentation pass.
The earlier airport pause has already been resumed; this is the later stopping point.

## Goal and decisions to preserve

The engineering goal is **reliable JP size and pressure choices at the marginal
water cut of the pad, field or other constrained system**. A useful model should
explain a well over its life, transfer across different pump installations and
conditions, and then predict other wells. Matching BHP alone is insufficient:
oil, formation liquid, PF and response to operating changes all matter.

The user explicitly requested investigation beyond friction coefficients,
including incorrect physics, alternative return hydraulics and a **Pump Match
Over Time** screen on or beside the existing production/history plot.

- Tracker nozzle/throat diameters are **nominal specifications, never measured
  dimensions or wear observations**. Disagreements need catalog/manufacturer
  reconciliation; do not automatically use them as effective worn dimensions.
- Gauges are typically within **40 ft of the JP**. Signed vertical offset is
  still unknown per well. This cannot justify an arbitrary hundreds-of-psi
  calibration offset.
- Stay on Databricks **Medium**: one uvicorn process, two process workers,
  one heavy job, native threads at one, shared CPU tokens and the existing
  bounded response cache. No compute-tier increase.
- Preserve `entry-energy-v2`, Vogel inflow and solver fallbacks. Do not restore
  the retired Mach multiplier. Total-liquid IPR inputs exclude returned PF;
  convert to oil once. PF density is independent of formation-water SG.
- Installed-pump fits remain scoped to exact installation, source and physics.
  A clean replacement, including the same size, uses reference coefficients.

## Workspace and stopping state

Actual repository: `C:\dev\woffl_gui\woffl_gui`; the outer directory is only
the workspace. Branch: **main**. HEAD when closing:
`ed85da2fb6b0b5aaf9202b7b404a10484f2c7cb2` (`docs cleanup`).

The recovered fixes, modeling changes, new tests, reports and rebuilt `web/dist`
are **local and uncommitted**. The tree contains modified tracked files and new
untracked files. Old hashed frontend chunks are deleted and new hashed chunks
are present as expected after the build; keep source and built SPA together.
Do not discard the dirty tree, clean generated work indiscriminately or assume
it is all from the final hydraulics block.

No staging, commit, push, deployment or production property/calibration write
was performed. No subagents were started by Codex. All test and benchmark tool
sessions from this work completed; none is awaiting a tool poll. This does not
describe unrelated processes the user may have running.

The stream interruptions were not diagnosed. The saved files and completed
checks were verified afterward; do not invent a cause such as token exhaustion
for the Codex connection interruptions. Fable's earlier session-limit failures
are a separate, documented event.

## Completed work

### Recovered Fable review and application fixes

Recovered nine partial reviewer transcripts and scratchpad material from Claude
session `696f9359-4409-4f2a-96e7-447c48d8998c`, then checked the leads against
source and offline reproductions. This was not a completed review of every
reviewer's assigned scope. See the [recovered review](recovered_review_2026-09-11.md)
for the individual evidence, locations and regressions.

All ten confirmed issues and the additional SQL validation gap were fixed:

1. Calibration saves read fresh enriched tracker data; a boolean cache-refresh
   return or an in-flight refresh cannot stand in for the DataFrame.
2. Context and calibration use exact UTC installation timestamps, including
   same-day changeouts and Anchorage/UTC browser interpretation.
3. Header Impact retains solver errors instead of labeling a failed solve
   as an established lack of response.
4. Washout workers retain measured test-day PF pressure for flagging.
5. Choke trimming skips steps that release no water, avoiding division by zero.
6. Warmup refreshes the longest test-history window first and derives shorter
   windows from the same fresh snapshot.
7. Gauge/OIW workbook parsing runs off the API event loop under the CPU budget,
   with bounded upload reads.
8. OIW logs accept mixed valid US text dates and Excel date cells.
9. Separator coverage uses local calendar boundaries and actual 23/24/25-hour days.
10. Dotenv excludes write-gate keys case-insensitively on Windows.
11. SQL validation accepts the intended append-only parameterized
    `INSERT INTO ... VALUES` grammar and rejects overwrite/replacement variants.

### Optimization feasibility and evidence

The S-Pad fixed-curve search now re-solves each retained hardware selection
against its actual station pressure before ranking it. Pump size and
installed/replacement state stay fixed during that check. Failed hardware,
excess flow and discontinuities that do not close pressure are rejected. The
winning grid is refreshed at the reported pressure, and the UI retains closure
metadata. Duplicate selections reuse bounded results.

The reproduced synthetic pressure inconsistency fell from **400 psi to -2.9 psi**,
inside the default 10 psi tolerance. This is a local consistency check, not
hosted or field validation. The search remains bounded; its allocation checks
do not certify a global optimum. Auto water price retains the originating
search-trial policy. [Before](model_plan_optimization_probes_2026-09-11.json),
[after](model_plan_optimization_probes_after_2026-09-11.json).

Missing match evidence now yields **unknown**. Known contradictions and poor
fits remain visible. The UI calls the former `ok` state **no flags**, which
does not imply independent validation. Complete identifiability, uncertainty
and all-bound diagnostics remain future work.

### Historical model investigation

Reproduced the September 8 frozen BHP plot and ran 14 diagnostic variants
without fitting new loss coefficients. Its 35-well cohort had 34 solved wells,
BHP RMS about 172 psi and mean bias +82 psi; it predates current scoped-fit
hydration. Mesh refinement and small PF-density/temperature perturbations did
not explain the largest errors. The component balances point to pump recovery,
entry behavior and input consistency as well as return hydraulics.

Nominal-spec checks found current conflicts on **MPE-48, MPM-62 and MPF-73**.
Substituting the tracker dimensions in the diagnostic worsened matching and
did not fix F-73. No catalog or wear correction was promoted from that trial.

Corrected historical tenure to **Date Set → next Date Set**, preserving the
original preflight as an older artifact. The corrected preflight has **1,881
eligible tests, 143 installations with at least three tests, and 29 candidate
multi-installation wells**, from 7,541 available tests.

Added a pure chronological benchmark across **MPB-30/37/39, MPF-107/73 and
MPE-42**: 12 target installations and 163 later observations. It uses adjacent
installations, up to ten earlier tests and a three-day training embargo. It
does not bridge an omitted installation. Repeat sizes remain distinct installs.
Earlier tests set oil productivity, WC and GOR; later BHP/oil/measured PF never
enter the prediction inputs. A separate conditional replay uses later measured
WC/GOR while keeping the earlier oil IPR.

### Selectable return hydraulics

**Implemented:** Beggs–Brill + Payne (unchanged default), Hagedorn–Brown +
Griffith and Shi/Pan drift-flux. **Not implemented:** Tulsa unified. Its dropdown
entry is disabled and explicitly unavailable. Complete primary closure
equations were not accessible; public reports did not fill the gaps.

The [implementation record](hydraulics_models_2026-09-11.md) documents the
sources, equations, unit conversions, domains and exact variants. H-B is a
vertical holdup reference with gravity projected for inclination. Shi/Pan is
a steady isothermal gas/liquid closure with oil/water treated as one liquid.
Annular hydraulics still use actual area plus hydraulic diameter. Downhill
return segments and non-subcritical gradients are unsupported by these
alternatives. They are not complete thermal or three-phase simulators.

The selected model reaches the main single solver, pressure profile, batch
sizing, PF sweeps, WC sensitivity, calibration and pad/CFP optimization. The
internal solopump residual and fallback paths carry it; a missed connection to
the return traverse was caught and fixed during this work. Unknown models
cannot silently run BB.

Changing models clears preview pump coefficients. A completed installed-pump
calibration save stores the selected hydraulics and combined physics version
atomically in the existing compact comment record. Old records without a
hydraulics ID mean BB. Save/apply guards, job handles, hydration and cache keys
distinguish models. Optimizer rows show their model. A new installation clears
the pump fit while retaining the well's selected return model.

The selector is a session preview until the fit is saved. **Save well inputs**
does not save hydraulics. **Standalone Header Impact, PF Scenario, washout and
friction-trend tools still build their own BB inputs**; they do not yet share
the main optimizer's saved model/input pipeline.

The pressure-profile plot now uses the solver's conserved formation/PF mixture
and water-mode rate. The previous separate construction could ignore PF density
and omit returned PF volume in dewatering mode. BB operating-point physics
remains unchanged. Shared-library additions are recorded in **upstream patch 44**.

## What the new comparison says

All three available models were evaluated on the same historical challenges
with **no loss-coefficient retuning**. Errors below use the **same 130 observations
solved by every model**; 126 have scoreable PF. Coverage includes all 163 attempts.

| Model | Solved / attempted | Common-case BHP RMS | Median absolute oil error | Median absolute PF error |
|---|---:|---:|---:|---:|
| BB + Payne | 135 / 163 | 130.7 psi | 26.8% | 9.9% |
| H-B + Griffith | 137 / 163 | 127.6 psi | 28.9% | 10.1% |
| Shi/Pan drift-flux | 130 / 163 | 146.5 psi | 34.9% | 7.4% |

**Keep BB as default.** The available evidence does not establish a fleet-wide
replacement. Better BHP alone can give worse sizing guidance:

- B-39's 12B → 10C period has BHP RMS **92.4 / 69.0 / 52.4 psi**, but median
  oil error worsens to **49.5 / 56.2 / 67.0%**, in the table's model order.
- E-42's 11C → 13C forecast stays near **128 psi BHP RMS** for every model.
- F-73 fails **all 26 observations** across both target installations for every model.
- B-30 has **0/7 drift-flux solutions**. H-B solves 7/7 versus BB's 5/7 but
  still matches poorly. Failed cases must remain visible when comparing errors.

These errors use current geometry/PVT/reservoir-pressure priors that have not
been verified historically. Nominal-spec flags, gauge offsets and observational
changeout confounding remain. No model is qualified for sizing by this replay.
Do not mix these results with the original fleet plot or the separate E-42
single-test screenshot; those used different inputs and cohorts.

Final serial replay timings were **18.9 / 16.1 / 83.7 seconds**; earlier runs
were approximately **6.0 / 5.5 / 15.9 seconds** with identical prediction metrics.
The local variability is not a hosted latency measurement. Drift-flux costs
more because of additional PVT/holdup evaluations; Medium limits remain intact.

## Verification at the pause

- Full offline Python suite: **1,962 passed**, four existing warnings, 35.41 s.
- Frontend: **11 Node tests passed**; TypeScript/Vite production build passed.
  The existing chunk-size advisory remains; matching `web/dist` was rebuilt.
- After the final field/fleet report metadata adjustment, **19 focused tests
  passed**. That sandboxed run emitted a pytest-cache permission warning; the
  tests themselves passed. It was not another full-suite run.
- The historical comparison completed with exit code 0. Its recorded source
  hashes were checked against the closing tree and match. Plots were visually
  inspected. `git diff --check` passed before closure documentation.
- New guards cover independent liquid/ideal-gas limits, Griffith slip, Pan's
  reference figure, domain rejection, real model dispatch and discharge closure,
  unchanged BB, single/batch/PF consistency, water-mode plots, calibration scope,
  cache separation and UI reset/restore behavior.
- No new browser QA, live save or deployment was performed for the hydraulics
  block. Earlier browser checks in September 8 documentation belong to that
  earlier implementation state.

This final documentation pass did not rerun simulations or the test suite.
It checked the recorded artifacts/source hashes and documentation links.

## Next steps, in order

1. **Explain the hard misses at component level.** Start with existing E-42 and
   F-73 cases, and use B-39 as a check against optimizing BHP at oil's expense.
   Reconstruct the applicable installation, catalog, circulation, survey,
   reservoir/composition inputs and pressure/rate timing. At measured suction
   and formation rate, report nozzle work, entry limit, throat/diffuser recovery,
   return static/friction/acceleration and discharge residual. Keep measured PF
   separate from a solved nozzle rate. Use existing diagnostics before adding
   parameters. E-42's earlier same-test balance had about a **195 psi** discharge
   deficit; return friction was only about **71 psi**.
2. **Make operating-point qualification explicit.** The earlier diagnostic found
   an L-06 non-sonic feasibility-edge return with about **+61 psi** discharge
   residual. Preserve established solver fallback behavior while exposing regime
   and closure residual; do not present every returned point as a fully closed
   operating balance. For the entry hypothesis, investigate equilibrium versus
   finite-rate/frozen gas release with conserved inventory and independent
   pressure-response evidence. This is an unproven hypothesis, not authorization
   to introduce another fitted Mach multiplier.
3. **Build Pump Match Over Time on the existing history UI.** The PNGs are only
   offline artifacts. Extend the shared production/history view with synchronized
   BHP, oil/formation-liquid and PF panels; show exact installation boundaries,
   hardware/model identity, fit windows, held-out predictions and failed/missing
   periods. Add per-installation scores and a way to inspect a miss. Use existing
   `ChartPanel`/ECharts SVG, shared history/cache and one bounded heavy job. Do not
   apply today's fit across old pumps and label that validation. Link optimizer
   recommendations to this evidence.
4. **Fit a well across installations, then test transfer.** Separate reservoir
   evolution from pump effects; use constrained shared pump behavior and limited
   installation effects. Check which parameters are independently identifiable
   from BHP/oil/PF. Withhold later complete pressure events, then another size,
   then another well. Freeze training inputs and model choice. Set acceptance
   targets from measurement uncertainty and decision size before choosing a winner.
5. **Judge the actual economic decision.** Validate direction and magnitude of
   incremental oil versus incremental constrained water, not only level errors.
   Use `delta_oil - lambda * delta_machine_water`, with
   `lambda=(1-marginal_wc)/marginal_wc`. At 98% marginal WC, another 1,000 BPD
   of constrained water needs about 20.4 BOPD; at 95%, about 52.6 BOPD. Preserve
   the correct water stream: M/E use formation plus lift water; I/S use lift
   water. Include uncertainty and rank stability before treating a small gain
   as an actionable changeout. Benchmark bounded discrete searches separately
   from model accuracy.
6. **Unify current-operation tool inputs.** Header Impact and PF Scenario should
   eventually use verified well/model/installed-pump context consistently with
   the main optimizer. Historical inversions need date-specific inputs. Do not
   copy alternative-model coefficients into a BB diagnostic or transfer current
   wear to a historical installation.
7. **Complete Tulsa only from adequate primary equations.** Obtain the complete
   Zhang unified-model flow-pattern/closure formulation and independent reference
   cases. Then add a distinct versioned option and compare the same cases,
   retaining failures and common-case scores. Keep the existing disabled entry
   until that work is verified. Further thermal/emulsion/annular models should
   follow measured evidence and demonstrated component errors.

Deployment and saving real calibrations are separate future actions. Local
verification does not establish that the hosted app contains these changes.

## Evidence and code map

| Record | Contents |
|---|---|
| [Recovered review](recovered_review_2026-09-11.md) | Fable recovery, reproduced bugs, fixes and earlier checks |
| [Model improvement plan](jp_model_improvement_plan_2026-09-11.md) | Physics hypotheses, validation design, optimization and UI requirements |
| [Resume record](jp_model_resume_2026-09-11.md) | S-Pad/health fixes, nominal specs, corrected tenure and chronological benchmark |
| [Hydraulics implementation](hydraulics_models_2026-09-11.md) | Primary references, exact closures, fit persistence, limitations and comparisons |
| [BHP diagnostic JSON](bhp_model_diagnostic_2026-09-11.json) / [plot](bhp_model_diagnostic_2026-09-11.png) | Fixed-coefficient perturbations and component pressure balances |
| [Preflight v2](pump_lifetime_preflight_v2_2026-09-11.json) | Corrected installation/test eligibility; supersedes v1 counts |
| [Nominal-spec diagnostic](pump_spec_diagnostic_2026-09-11.json) | Conflicting dimensions and substitution results |
| [Chronological reference JSON](pump_history_benchmark_2026-09-11.json) / [plot](pump_history_benchmark_2026-09-11.png) | Earlier-test BB forecast across later installations |
| [Hydraulics comparison JSON](hydraulics_benchmark_2026-09-11.json) / [plot](hydraulics_benchmark_2026-09-11.png) | Same inputs with all available models, including failures and common-case scores |
| [Upstream register](upstream_sync.md) | Shared-library changes through patch 44 and named regression guards |

Key implementation locations:

- `woffl/flow/hydraulics.py`, `outflow.py`, `errors.py`; assembly `solopump.py`,
  `batchpump.py`, `network_optimizer.py`, `sim_factories.py`.
- `server/services/solve.py`, `event_calibration.py`, `pump_calibration.py`,
  `wells.py`, `optimizer_runs.py`, `field_validation.py`, `fleet_validation.py`,
  `pump_history_benchmark.py`; `server/schemas.py` and `woffl/gui/fric_calibration.py`.
- `woffl/gui/pad_optimize.py`, `server/services/match_health.py` and optimizer
  result/health UI for coupled pressure and evidence handling.
- `web/src/layout/Sidebar.tsx`, `web/src/state/params.ts`, `web/src/api/types.ts`,
  solver `EventCalibration.tsx`/`PumpScope.tsx`, optimizer `RunPanel.tsx`.
- `tests/test_hydraulics_models.py`, `test_pump_calibration_scope.py`,
  `test_event_calibration.py`, `test_pump_history_benchmark.py`,
  `test_recovered_review.py`, plus relevant existing modules;
  `web/tests/pumpScope.test.mjs`.

Trusted local replay input: `build/fleet-actuality-snapshot.pkl` (September 8
snapshot; not portable through Git). Do not deserialize an untrusted pickle.
Primary PDFs, text extractions and rendered equation pages are under
`build/hydraulics-research/`; citations and exact filenames are covered by the
hydraulics record. The `wire_solver.py`, `wire_app.py` and `wire_ui.py` files
there were one-time editing scripts—**do not rerun them**. Fable's recovered
scratchpad was under the user's local Temp/claude tree for the session ID above;
the durable findings are in the recovered-review document.

## Commands for a future verification or replay

Run from `C:\dev\woffl_gui\woffl_gui`, after reading [AGENTS.md](../AGENTS.md).
No need to rerun the full suite merely to read this handoff.

```powershell
$env:WOFFL_MAX_WORKERS = '1'
$env:PYTHONPATH = '.'
.\venv\Scripts\python.exe -m pytest tests/ -q

Set-Location web
node --test tests/*.test.mjs
npm run build
Set-Location ..
```

Offline runners are `tools/bhp_model_diagnostic.py`,
`tools/pump_lifetime_preflight.py`, `tools/pump_spec_diagnostic.py --simulate`,
`tools/pump_history_benchmark.py`, `tools/hydraulics_benchmark.py` and
`tools/model_plan_optimization_probes.py`. They write dated report artifacts;
use new output paths when extending an experiment so earlier evidence survives.
For serial performance comparisons set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `MKL_NUM_THREADS` to `1` as well. Avoid merging native stderr into a PowerShell
stdout log: a warning can make the wrapper report failure despite completed
output. Preserve the actual process exit code and inspect errors.

At the next resume, inspect the current diff, read this handoff and the detailed
model plan, then start the component-level E-42/F-73 investigation. Keep the
history-screen request visible. Do not restart the interrupted Fable review or
repeat completed studies without a new question to resolve.
