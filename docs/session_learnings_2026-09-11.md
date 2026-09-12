# September 11, 2026 — airport pause and modeling handoff

Latest pause: [end-of-night handoff](session_close_2026-09-11.md), written after
the airport work resumed and selectable hydraulics was implemented and tested.

**Resumed:** the user returned and clarified the fitting objective, nominal
tracker dimensions and gauge location. Read the
[resumed-work record](jp_model_resume_2026-09-11.md) for the subsequent fixes,
cross-pump benchmark, updated plan and current verification. The rest of this
file preserves the airport-pause state; its open questions and preflight counts
are superseded where the resume record says so.

The user requested consolidation and a pause **by 10:50 a.m. Anchorage time**
to take this laptop to the airport. Resume when the user returns tonight.
This note, the [model improvement plan](jp_model_improvement_plan_2026-09-11.md),
and the new diagnostic artifacts are the starting point. Do not restart the
recovered Fable review or repeat the completed bug fixes.

Actual repository: `C:\dev\woffl_gui\woffl_gui`, one directory below the
workspace. Read [AGENTS.md](../AGENTS.md), this note and the
[September 8 handoff](session_learnings_2026-09-08.md). Keep Medium compute,
one uvicorn, two process workers, one heavy job, native threads limited to one,
the shared CPU tokens and existing bounded cache. No deployment or live write
was performed. Local edits remain uncommitted and unstaged.

## User priorities to preserve

1. Improve the hardest problems: **BHP actual versus model**, optimization,
   and overall jet-pump model quality. Produce and follow a concrete path.
2. Look beyond friction coefficients. If an equation is wrong or an important
   physical effect is missing, investigate and correct it with evidence.
3. Fit a well across its available history with **different jet pumps and
   operating conditions**, and demonstrate that behavior across **several
   different wells**. Matching one point is not enough.
4. Investigate alternatives to **Beggs–Brill**, with selectable hydraulic
   models and a fair comparison of their predictions.
5. **Pump Match Over Time screen:** preferably extend the production plot so
   users see actual versus modeled behavior across pump changes. This is a
   major desired feature for confidence in optimization, not a minor appendix.

The UI request is captured in detail in the plan. `HistoryStrip.tsx` already
provides a shared production/pump timeline in Solver and JP History. Extend
that foundation. Show BHP, oil/formation liquid and PF with pump-change markers,
fit versus held-out periods, source/version information, failures as gaps, and
per-installation error/coverage summaries. Link optimizer recommendations to
this evidence. The screen and alternate hydraulic models are **planned, not
implemented** in this session.

## Completed work before this modeling investigation

Recovered the interrupted Claude/Fable session
`696f9359-4409-4f2a-96e7-447c48d8998c`. Its local nine partial reviewer
transcripts existed; no completed reviewer reports were recovered. The
repository was initially clean at `ed85da2`. The scratchpad was under:

`C:\Users\sc9864\AppData\Local\Temp\claude\C--dev-woffl-gui\696f9359-4409-4f2a-96e7-447c48d8998c\scratchpad`

The user then said “please fix.” Ten verified defects and an additional SQL
validator weakness were fixed locally. Full details and test mappings are in
[recovered review and fixes](recovered_review_2026-09-11.md):

- Pump-fit save now fetches fresh enriched tracker rows; it no longer treats
  the cache-refresh boolean as a DataFrame.
- Exact UTC installation timestamps survive well context, event calibration
  and frontend scope checks. Same-day changeouts remain distinct; displayed
  calendar dates do not truncate identity.
- Header Impact preserves returned solver errors in its result/verdict.
- JP washout calibration retains measured `PfAtTest`.
- Choke budget trimming handles zero-PF fallback rows without dividing by zero.
- Well-test warmup fetches the longest fresh window once and primes all shorter
  windows from that snapshot, respecting version/latch guards.
- Gauge/OIW file parsing runs off the event loop under the shared CPU budget.
- Mixed-format OIW timestamps parse consistently.
- Separator windows use calendar-day DST boundaries and actual day duration.
- `.env` export cannot enable write gates via case variants on Windows.
- SQL write validation allows the supported single `INSERT ... VALUES` shape
  and rejects overwrite/replace/select/expression alternatives.

Verification completed for those changes: **1,887 Python tests passed**, four
existing warnings, **9 frontend tests passed**, and TypeScript/Vite production
build passed. Vite's existing large-chunk advisory remained. Generated
`web/dist` assets were rebuilt and checked; old hashed JS deletions and new
untracked hashed JS are the normal build result. Do not discard them as noise.
No production save, write-gate enablement, deployment, commit or staging occurred.

The modeling work below adds offline tools/documents only. It does not alter
the solver, hydraulics, fitting behavior, optimizer behavior or frontend runtime.

## New evidence from the BHP plot

The plot the user referred to is
[fleet_actuality_2026-09-08.png](fleet_actuality_2026-09-08.png). Its JSON has
90 model configurations, with 35 wells eligible for the displayed latest-test
comparison. It is frozen **before installed-pump-scoped hydration** and must
not be described as a fresh validation of currently saved installed fits.

New artifacts:

- [BHP diagnostic plot](bhp_model_diagnostic_2026-09-11.png).
- [All BHP variants and component pressure balances](bhp_model_diagnostic_2026-09-11.json).
- [BHP diagnostic tool](../tools/bhp_model_diagnostic.py).
- [Historical pump/test preflight](pump_lifetime_preflight_2026-09-11.json).
- [Preflight tool](../tools/pump_lifetime_preflight.py).
- [Optimization probe results](model_plan_optimization_probes_2026-09-11.json).
- [Optimization probe tool](../tools/model_plan_optimization_probes.py).

The 34 successful baseline BHP predictions reproduced exactly; MPF-73 still
failed. There were 490 requested variant solves, with failures retained.
Pump losses and nozzle multipliers were held fixed. No live queries or refits.
The original report and old plots were preserved.

Those frozen losses include legacy fits. Remaining errors are not proof that
the physics alone is wrong. Before the next qualification, establish a separate
baseline under today's installed-pump/model-version scope rules, using only
valid scoped fits or the explicitly identified defaults.

Key results:

- 25/34 modeled BHP values are high. Mean signed error +82.3 psi, median
  absolute error 85.6 psi, RMS 172.0 psi. PF median absolute error 4.6% across
  31 scorable wells, oil 20.0% across 34.
- Test WC/GOR with original oil IPR preserved: BHP median absolute error
  102.3 psi, RMS 177.8 psi. Test oil/BHP anchor with old composition: 81.4/170.2.
  Both updates together: 101.5/160.0. The same-test conditioning is diagnostic
  and circular for validation; do not sell its reduced RMS as improved forecast
  accuracy. Typical error actually worsens.
- With both updates, BHP errors remain +531 psi MPB-35, +308 MPJ-29,
  +243 MPE-48, +138 MPE-42. MPI-22 improves +263 to +97; MPM-62 worsens
  +170 to +282. There is no single uniform input correction.
- Refining return spacing 100 to 25 ft changes BHP at most 6.54 psi.
  Integrating PF density/friction along an isothermal column changes it at
  most 2.25 psi. These are not the main explanations for the large misses.
- A +20% GOR probe moves BHP by median absolute 13.3 psi, maximum 53.5;
  +20°F uniform temperature by median 6.3, maximum 25.2; -2% PF standard
  density by median 4.7, maximum 26.0 psi. These are hypothetical perturbations,
  not measured uncertainty.
- Forcing BB holdup to no-slip moves BHP at most 5.64 psi in this set.
  Removing Payne can move it +98 psi and generally worsens this comparison.
  The no-slip probe retains other BB terms: it is not a validated alternate
  correlation. Do not label either probe “better hydraulics.”
- At measured BHP/oil, MPE-42's pump discharge is about 1,936 psi versus
  2,131 psi required by the return column (195 psi deficit; return friction
  only 71 psi). MPJ-29 deficit 350 psi/friction 23; MPE-48 deficit 398 psi/friction 131.
  Check pressure datum, pump recovery, actual geometry/fluid state before
  blaming pipe friction. PF rate is solved in these traces, not fixed measured PF.
- Fourteen measured BHP/rate/composition states fail the entry-energy balance.
  Thirteen of 34 conditioned predictions sit at the entry limit. This raises
  an important gas-release/entry-model question, but is not proof against
  physics independent of geometry, measurements and datum errors.
- MPL-06's non-sonic feasibility-edge result has about +61 psi discharge
  residual. This is an existing intentional fallback, not an ordinary closed
  root. Expose regime and residual before relying on its pressure response.

The current `entry-energy-v2` fixed historical inconsistent Mach scaling;
do not resurrect that multiplier. The path is isothermal/equilibrium and
homogeneous. Test independent pump benchmarks and frozen/finite-rate gas
release assumptions if the clean field observations contradict it. Existing
conservation tests do not prove field validity.

## Historical benchmark readiness

Trusted local input: `build/fleet-actuality-snapshot.pkl`, about 12 MB, contains
the September 8 source frames, contexts and configs. It is a locally created
pickle: do not load third-party pickles. An additional
`build/field-validation-snapshot.json` exists. Both build artifacts are ignored
and may not exist in a fresh clone.

Available tests span 2024-09-08 to 2026-09-08. The preflight starts from 7,541
tests, finds 1,864 basic eligible tests inside unambiguous installations,
142 intervals with at least 3 tests, and 29 wells with at least 2 supported
distinct nozzle/throat/circulation configurations. This is **not full-life
coverage or a validated fit**. Pump/pull days and ambiguous day-level
installations are excluded conservatively. Historical pressure data alone
cannot establish historical oil response.

Initial candidates: MPB-37, MPB-39, MPF-107, MPM-28, MPE-42, plus the
deliberate failing MPF-73. MPE-42 has 75 screened tests across 3 installations
and 2 pump configurations, but its current 13C interval has only 4 tests with
about 52 psi PF spread. It is not a strong standalone slope-identification set.

Hardware reconciliation is essential. MPE-48's label 14C implies nominal
0.2675/0.5519-inch nozzle/throat; raw tracker fields are 0.2916/0.631. Several
older MPE-42/MPM-28/MPB-37 records also disagree. Raw fields may denote worn
or recovered dimensions, another catalog, or legacy areas. The existing
Guiberson converter already handles some area-versus-diameter ambiguity and
maps to a nearest National equivalent. Do not blindly substitute raw numbers.
The current pump-dictionary/config path carries labels, not those raw diameters.

Gauge metadata in the frozen registry lacks gauge-depth/datum information.
The solver compares pump suction directly with observed BHP. Verify where the
gauge sits and model any intake-to-sandface interval; do not freely fit an
offset simply to remove residuals.

## Plan details and next implementation blocks

The [full plan](jp_model_improvement_plan_2026-09-11.md) gives priorities,
completion criteria, literature links, component code paths and the screen
specification. Recommended order:

1. Reconcile hardware/datum/test timing, build immutable multi-installation
   observations and chronological train/holdout splits for the candidate wells.
2. Extend existing JP History/Solver production plots with modeled history,
   pump boundaries, per-era errors and explicit held-out evidence labels.
3. Qualify pump component balances and an inclination-aware drift-flux
   alternative to Beggs–Brill. Consider Tulsa unified mechanistic modeling as
   a further challenger. Hagedorn–Brown is only a scoped comparison for
   appropriate near-vertical tubing; none is yet demonstrated better here.
4. Separate stable well properties, time-varying reservoir/inflow/composition,
   pump-family geometry/behavior and installation-specific wear. Test later
   events, whole different installations and different wells without using
   held-out BHP/oil to re-anchor the model.
5. Use those results to qualify and rank optimization recommendations; close
   selected station/well hydraulics and show uncertainty and supporting history.

The current multipoint fitter anchors IPR per point on observed oil/BHP and
therefore intentionally does not score oil as independent loss. It fits four
parameters (ken/kth/kdi/fnz), with potential non-identifiability. Its all-pair
pressure-response loss can be confounded by changes in operating conditions.
Avoid extending it by merely giving each historical date more free knobs.
Existing `field_validation.py` already freezes training inputs and excludes
future anchors for whole-event holdout prediction; build on it.

Two new optimization findings are **reproduced but not fixed yet**:

- A synthetic feasible fixed-curve case reports 2500 psi and 10000 BPD draw
  with `converged=True`, while its station curve at that draw is 2900 psi.
  Sweeping a flow budget and allowing unused capacity does not by itself
  close the delivered header. Explicitly model bypass/recirculation if that
  is how the setpoint is achieved, otherwise settle the selected pump plan.
- `match_health._verdict({})` returns `ok`. Unknown evidence should be explicit,
  and optimizer recommendations need the relevant fit/response/transfer status.

Other planned optimization work: benchmark coarse/refined discrete pressure
search against small exhaustive references; retain correct machine-water
streams (M/E total formation+lift, I/S lift); preserve the agreed priced-oil
objective and quantify its difference from pure-oil references; carry measured
load for unsupported wells; rank gains across credible joint uncertainty.
MILP/MCKP agreement at one trial does not establish global or physical validity.

## Reproduction and verification

From the actual repository:

```powershell
$env:WOFFL_MAX_WORKERS='1'
$env:PYTHONPATH='.'
./venv/Scripts/python.exe tools/bhp_model_diagnostic.py
./venv/Scripts/python.exe tools/pump_lifetime_preflight.py
./venv/Scripts/python.exe tools/model_plan_optimization_probes.py
```

These tools write only the new September 11 diagnostic output paths. Use another
`--output` for preserving later experiments where supported. Do not run the
old fleet tool against its default output when trying a new model: it would
overwrite the preserved September 8 report.

The diagnostic plot was opened and visually inspected. Baseline replay,
failure coverage, history interval assignment and JSON serialization succeeded.
The earlier full application suite/build were already green; no application
runtime changes were made during this modeling investigation. Git diff checking
also passed. Use command-local
`git -c safe.directory=C:/dev/woffl_gui/woffl_gui` for Git in this environment.

No subagents were started. All diagnostic subprocesses launched for this
investigation completed. A final process check also found none of the three
diagnostic scripts running. Do not terminate unrelated user/server processes.
The user will explicitly resume the work tonight; do not run a timer or
background task while the laptop is traveling.
