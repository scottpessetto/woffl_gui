# Medium performance changes — 2026-09-07

> **Documentation status, 2026-09-08:** Implementation and benchmark milestone. FLOW-4 and constant-water limitations listed below were subsequently addressed by [shared entry energy](entry_energy_implementation_2026-09-08.md) and [fluid v2](fluid_followup_2026-09-08.md). The original counts/timings are preserved. Medium remains the cost constraint; the [handoff](session_learnings_2026-09-08.md) records the final baseline and outstanding hosted measurements.

Medium compute is retained. Deployment still uses two process workers and one
uvicorn process. These changes are implemented locally; deployment has not run.

## Changes

- Optimizer response sweeps use the existing primed process pool instead of
  starting a pool at each pressure. Standalone library execution still works.
- A process-local response cache stores exact per-well batch results. Keys
  include every WellConfig input, pressure, pump grid, survey contents and model
  source hash. Budgets and prices are allocated again, so changing those can
  reuse physics without reusing an allocation. Each hit returns an independent
  object. Unreadable surveys bypass caching and defer to the model loader.
- Cache storage is bounded to 64 MiB of serialized payload, 1,024 entries and
  one-hour expiry. This is not a cap on total application memory: active models,
  deserialization and job results also consume memory. Restart clears the cache.
- Worker tasks and synchronous allocation share CPU tokens. Submission windows
  bound pending work; Solver, Batch, pressure-profile and synchronous calibration
  requests also acquire tokens. Native BLAS, HiGHS and CP-SAT threads are capped
  at one. Background studies queue one at a time on the deployed app.
- Pad pressure sweeps retain full optimizer payload only for the best trial;
  other trials retain the scalar values required by refinement and charts.
- HTTP Server-Timing and `/api/meta/performance` expose bounded local timing
  summaries, queue time, SQL-read time, worker count and cache statistics.
  Route labels contain templates rather than well names or query parameters.
  Reading diagnostics performs no warehouse query. Settled job durations stop
  increasing after completion.

No new Databricks service, larger compute tier or extra warmup reads are added.
The existing 12-hour fleet warm cadence remains.

## Offline benchmark

`tools/benchmark_medium.py` uses four synthetic wells, three pressures and six
pump choices per well (72 solves), with two workers on the local Windows host.
All result DataFrames were exactly equal between execution paths.

| Measurement | Seconds |
|---|---:|
| Fresh pool at each pressure | 2.943 |
| Shared pool startup, paid once | 0.887 |
| Shared pool, cold response cache | 0.269 |
| Exact cached repeat | 0.0028 |

Startup plus the first shared sweep was 1.156 seconds. During a separate cold
sweep, 20 local TestClient diagnostics requests measured p50 3.09 ms and p95
4.45 ms. These are synthetic local measurements, not hosted latency or a full
pad optimization benchmark. Data hydration and allocation are outside the
reported response-sweep timing. Small pump grids exercise the existing curve
fit fallback. Raw results: `medium_benchmark_2026-09-07.json`.

## Validation and remaining physics work

The complete offline suite passed: **1,766 passed, 3 expected failures**, with
two warnings (existing oil-compressibility floor and pandas concatenation).
The frontend production build/typecheck and both polling tests passed; Vite
still reports the existing large chart bundle warning.

New tests exercise exact cache equivalence, mutation isolation, invalidation,
expiry and size limits, failed-computation retry, shared CPU limits, hydrostatic
and Bernoulli hand calculations, energy-walk agreement at Mach 1, pressure-step
refinement, and both allocation engines against exhaustive enumeration of
small problems under both water bases and three prices. Existing conservation
and independent HYSYS fixture tests remain part of the full suite.

Three strict expected failures expose FLOW-4: at critical Mach 1.5, 2 and 2.5,
the choking and operating walks disagree in their inlet kinetic-energy term.
`tools/physics_qualification.py --strict` correctly exits 1; its report is saved
in `physics_qualification_2026-09-07.json` and the companion Markdown file.
These discrepancies are not fixed by scheduling or caching. A physically
justified correction and held-out field validation are still required; the
constant water-property approximation also remains. No new field measurements
were fetched or claimed as validation.

`.github/workflows/checks.yml` adds a single offline CI job for tests and the
frontend build, with a 15-minute timeout and cancellation of superseded runs.
It publishes the known physics gap in the job summary and saves qualification
artifacts. It uses no Databricks credentials. The workflow itself has not yet
run on GitHub; checks above were performed locally.
