# Session handoff - September 8, 2026

This is the end-of-day handoff for future sessions. Read [AGENTS.md](../AGENTS.md)
for operating rules and [the documentation index](README.md) for topic references.
The implementation was present in local commit `b24fd28` when this documentation
pass began. The repository was clean. These notes do not establish which revision
the hosted Databricks app is running.

## User decisions to preserve

- Keep Databricks **Medium**: cost rules out solving performance by increasing
  the compute tier. Optimize work, caching and concurrency within two CPUs.
- Proceed with already-authorized implementation and fixes without repeatedly
  asking permission. Production writes are not a way to test a save flow.
- Use **one throat-entry energy balance**. The critical-Mach inconsistency has
  been resolved in code; further field discrepancies are not a reason to restore it.
- Distinguish persistent well inputs from fitted properties of the installed
  jet pump. A replacement, including the same catalog size, must not inherit
  fitted pump losses or area from old hardware.
- Show WC sensitivity tastefully: a collapsed panel with oil/BHP ranges and
  transparent assumptions. Milne test/WC uncertainty matters, but does not explain
  every discrepancy by itself.
- The yellow fluid-property/critical-Mach transition banner was removed at the
  user's request. Keep specific fit-quality and installation warnings.

## Current implementation

### Physics and units

The model identifier is **`entry-energy-v2`**. Choking and operating throat entry
share the same unscaled energy balance, density path and conserved mass. The
limit is the first energy minimum reachable from suction; Wood Mach is a
diagnostic and need not equal one there. `mach_crit` survives only as an ignored,
deprecated compatibility input, hydrated to 1.0 and excluded from fitting.

PF density is a separate standard-condition input (0 psig, 60 F), independent
of formation-water SG. Its generic default is 63.648 lbm/ft3; plant SG supplies
pad defaults and explicit well inputs take precedence. Water density and its
derivative use IF97; viscosity uses the industrial IAPWS liquid formulation.
SG scales density, without inferring a brine-viscosity correction. Low-GOR oil
uses its available gas inventory to determine the effective bubble point and
continues compressing above it. Standard/in-situ volume is converted once.

`SimParams.qwf`, `SimulationParams.qwf` and `WellConfig.qwf` are **total formation
liquid**, BLPD excluding returned PF. Convert to oil once at `InFlow` construction.
`ResMix.condition()` mutates children: each independent case needs fresh PVT.
Retain Vogel inflow, typed infeasibility and the bracket/secant/reseed/bisection
fallback sequence. Shared-library changes are inventoried through patch 43 in
[upstream_sync.md](upstream_sync.md).

### Well saves and pump calibration

The operating workflow is documented in [the optimization user guide](optimization_user_guide.md)
and [the calibration implementation](pump_calibration_scope_2026-09-08.md).

1. Edit/review the IPR and supported fluid inputs; **Save well inputs**.
2. **Calibrate to field data** uses those saved inputs plus current-era tests/history.
3. **Apply to inputs** previews a fit in the session. **Save installed-pump
   calibration** separately persists the completed server job's result.
4. New optimization runs hydrate that saved fit only for its verified installation.
   Current-pump operations keep it; replacement candidates use reference hardware.

The well-save endpoint no longer forwards ken/kth/kdi/nozzle-area factor or Mach.
Bubble point and formation temperature are supported changed PVT inputs; this
does not make every sidebar field persistable. PF pressure stays live/run-specific.
As-built identity still comes from the tracker.

Pump fits are bound to well, nozzle, throat, **exact Date Set**, Databricks source
and physics version. Daily fitting windows may normalize the date; persistence
must keep the exact timestamp. A same-size changeout invalidates a prior fit.
Legacy unscoped numeric friction rows remain in history and need refitting and
re-saving; they are not automatically promoted to valid pump calibrations.

A complete record is stored in the existing `mpu.wells.woffl_eng_comment` table
under context `pump_calibration_v1`, in one gated parameterized INSERT. No new
table/property ID/DDL is required. The encoder checks the 500-character limit
before the human-comment writer can truncate it. Coefficients keep full precision;
quality metrics are rounded. Saving accepts a server job ID, checks its kind/status/
well and verifies a fresh Databricks installation. Failed saves do not evict caches.
The latest-record fleet read is cached for five minutes and invalidated on success.

`pump_state` distinguishes installed from replacement through single solves,
batch/PF sweeps, pad/CFP candidates, MILP/MCKP, fixed-scenario lookup and reports.
All replacements use **ken=.03, kth=.30, kdi=.40, fnz=1.0**. Identical same-size
outcomes favor keeping the installed pump. Future wells borrow donor well inputs
without donor loss coefficients, wear or installation identity. Library callers
retain legacy behavior unless they opt into `pump_calibration_scoped`; the app opts in.

### What the MPE-42 fit did and did not show

The user's MPE-42 / 13C screenshot showed a closer single-test comparison:
modeled BHP 705 versus 643 psi, oil 340 versus 388 BOPD and PF 3,791 versus
3,865 BPD. The multi-point fit still reported **76 psi BHP RMS, 61.5% PF RMS,
72 psi delta-BHP RMS**, ken at its .005 lower bound, and modeled response .268
versus measured .062 psi/psi. Nozzle area was estimated 1% above catalog.

That fit is provisional. Better agreement at one test does not establish wear
or validate pressure-response predictions. Saved diagnostics preserve this
distinction. Bound hits, BHP RMS >50 psi, PF RMS >10%, or beta disagreement >.03
flag provisional fits; single-point fallback is always provisional. These are
review flags, not statistical acceptance limits. Clean-reference predictions also
remain model assumptions rather than measured performance of a new pump.

### WC uncertainty

The Solver's collapsed card defaults to **+/-5 percentage points**, with nine
sample offsets clipped to 0-99% and deduplicated. It holds the total-liquid IPR
anchor, GOR, pressures and selected hardware assumptions fixed; operating liquid
can still move when the solver is rerun. Bounds include successful interior
samples, not just endpoints. Failures are counted and disclosed; a failed base
does not keep an old base result. Input changes immediately hide stale output.
Water-pump mode and base WC above 99% are unavailable in this panel.

This is **fixed-GOR sensitivity**, not a confidence interval, a fit, a save or
uncertainty-aware optimization. The separate fleet study also explored fixed
gas per liquid, preserving `GOR * (1 - WC)`; that is not fixed operating gas rate
and is not a second GUI mode. See [GUI details](wc_uncertainty_gui_2026-09-08.md).

### Medium performance and deployment

Keep one uvicorn process, two process workers, native math threads at one,
`WOFFL_MAX_JOBS=1`, shared CPU tokens, and `WOFFL_WARM_INTERVAL_SEC=43200`.
Persistent workers avoid rebuilding a pool at every sweep point. Exact response
caching is bounded to 64 MiB serialized payload / 1,024 entries / one-hour TTL;
keys include all WellConfig fields, pressure, pump grid, survey and model source.
Cached physics may be reused for changed budgets/prices; allocation is recomputed.
These bounds do not cap total app memory. `/api/meta/performance` exposes local
timing, SQL, queue and cache diagnostics without another warehouse query.

The v2 local four-well benchmark measured 2.99 s with fresh pools, 0.203 s for
a warm-pool cold-cache sweep and 0.00448 s for an exact cached repeat. Startup
was 0.938 s. These exclude hydration/allocation and are not hosted latency.

Runtime dependencies are pinned through `requirements.txt` and its constraints.
Do not install the PyPI `woffl` copy over the vendored source. Application Python
is >=3.11; local verification used 3.13. The clean-install smoke passed on Windows;
this does not establish a hosted Linux deployment. `web/dist` is committed and
must accompany frontend changes; Databricks does not build it with npm.

## Evidence and verification, kept separate

| Check | Recorded result | What it establishes |
|---|---|---|
| Offline Python suite after pump scope | 1,861 passed, 4 existing warnings | Code/physics regression coverage at the handoff |
| Frontend | 8 Node tests and TypeScript/Vite build passed | Store contracts, polling and build integrity |
| Browser checks | WC and pump-scope flows passed; pump QA had 0 browser errors and 2 intercepted saves | Fixture UI behavior, no production save verification |
| Strict energy/fluid qualification | Passed documented independent numerical checks | Consistency/conservation/reference correlations, not fleet accuracy |
| Three-well event holdouts | All 90 held observations solved; held BHP RMS 48.55 / 23.15 / 58.96 psi on MPM-64 / 28 / 45 | Separate frozen-training experiment; all three fits hit ken bounds |
| Frozen fleet audit | 35 wells, 9 pads; 34 latest tests solved, 1 failed; median absolute BHP 85.6 psi, PF 4.6%, oil 20.0% | Retrospective reproduction, not independent qualification |

The fleet snapshot was captured with 12 SELECT statements before the new scoped-fit
hydration policy. Its JSON/CSV/plots retain those frozen inputs. They are **not a
rerun of today's final hydration**. Scored tests may have informed saved inputs;
daily/test observations and pressure pairs overlap and are not independent.
Gauge-to-suction datum equivalence was assumed, not independently verified.

The isolated WC study found +/-5-point fixed-GOR BHP swings with median 5.1 psi
and maximum 45.4 psi, while relative oil swings were often larger. WC can explain
part of the oil variation; this does not account for all observed BHP error.

## Next work, not completed or claimed deployed

1. Deploy the verified source and matching built SPA on **Medium**; measure real
   hosted queue/solve/cold-read latency and memory before further performance work.
   No deployment or production calibration write was performed in this session.
2. Review/recalibrate legacy fits with current installation identity, saving well
   inputs before fitting. Do not silently migrate or certify the old coefficients.
3. Reconcile MPB-35's 82,133.56 BPD test PF, MPI-24's circulation conflict, MPF-73's
   lift failure and the largest BHP outliers. Check gauge datum and contemporaneous
   WC/GOR/liquid/PF data before adjusting coefficients to absorb measurement errors.
4. Validate later independent pressure-change events with frozen training inputs
   and an embargo. Distinguish pressure-response error from a good level match.
5. Consider evidence-based WC ranges and coherent WC/GOR/IPR uncertainty, then
   robust low/base/high optimization. Those extensions are proposals, not shipped.

Remaining physics approximations include SG-scaled pure water rather than brine
chemistry, a bulk PF column, isothermal properties and empirical oil/acoustic
correlations. Add tests for concrete conservation/domain/response failures; do
not replace independent checks with pins to current outputs or relax tolerances
merely to improve a plot.

## Reproduction and tooling lessons

Commands and offline guards are in [AGENTS.md](../AGENTS.md). Historical reports
retain their own test counts and timing data; they are not current-suite counts.
Latest local logs are under `build/pump-scope-*.log` (ignored, not portable artifacts).

- Mock scoped calibration reads as well as saved-IPR reads in server fixtures.
  September 11 correction: saves read `datasources.jp_history_fresh()` directly.
  Mock its underlying tracker SELECT in tests, keeping the real enrichment and
  installation selector; `cache_refresh()` returns a boolean, not tracker data.
- Optional Playwright lives under `build/browser-qa`, outside app dependencies.
  `tools/check_pump_scope_ui.py` and `tools/check_wc_uncertainty_ui.py` use local
  fixtures/interception; their MPE-42 screenshots are not new field evidence.
- Do not run a production frontend build concurrently with Vite browser QA.
  HMR can reset state; a bare dynamic import may instantiate a different Zustand
  store from the timestamped module actually loaded in the page.
- Use explicit UTF-8 and `apply_patch`; non-ASCII Python source piped through
  PowerShell can be damaged. Keep browser/test fixtures complete against schemas.
- Default fleet replay uses a trusted local pickle snapshot. Do not load an
  untrusted pickle. Use a new output path for reruns so frozen reports are preserved.
- The old `RESUME_ENG_COMMENT.md` crash note and outer Streamlit plans are archival.
  Do not rerun their DDL, sample writes or outdated resume checklist.
