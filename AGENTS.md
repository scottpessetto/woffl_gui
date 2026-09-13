# AGENTS.md — woffl_gui

Operating rules for coding agents in this repo. Read this before touching anything.
Prose lives in `docs/`; this file is only the rules you will otherwise violate.

Latest optimization implementation: [September 12 capacity repairs](docs/pad_cfp_capacity_delivery_2026-09-12.md).
Default pad runs maximize oil within capacity; manual water pricing is explicit.
Required-online constraints, qualified plant delivery and solver outcomes are
preserved through API/UI. CFP uses a coherent manual reference, unclipped lower
pressure domain and combined candidate search. The delivery distinguishes these
repairs from remaining field qualification and the full new-well comparison study.

Latest well-fit implementation: [September 12 well-fit workflow](docs/well_fit_workflow_delivery_2026-09-12.md).
Solver and JP History have persistent well-save controls, explicit read-only
states and historical previews of supported edits. Save refreshes the database
baseline without losing session edits; new optimization runs load saved values.
The [Pump Match Over Time delivery](docs/pump_match_ui_2026-09-12.md) records the replay.
Latest review: [well fitting, sensitivities and pad decisions](docs/well_fit_pad_workflow_review_2026-09-12.md).
Its fixed-IPR calibration, sensitivity/Apply parity and missing-model coverage
findings are implemented in the delivery record. Common oil-IPR refitting is
explicit, and I/M/E fixed-plan stress cases are available. Read the delivery's
remaining work before treating a saved model as qualified for pad decisions.
The shared production plot defaults to saved-well BHP/oil predictions at every
usable test, with chronological validation modes also available. Shared
multi-installation pump-loss fitting remains unfinished. Every-test replay can
use a saved fit for its exact matching installation/model; other installations
use clean reference losses. No deployment was performed.
Historical replay now uses each test's WC/GOR while preserving one saved oil
IPR across all pumps. Do not introduce automatic IPR shifts; the user wants
one IPR describing the well. See the [fixed-IPR investigation](docs/well_match_diagnostic_2026-09-12.md)
for the measured-composition contract and the interior-lift solver fix.
The [September 11 end-of-night handoff](docs/session_close_2026-09-11.md)
preserves the preceding fixes, experiments, limits and modeling next steps.

Start future sessions with [the 2026-09-08 handoff](docs/session_learnings_2026-09-08.md)
and [the documentation index](docs/README.md). The workspace root is one level
above this Git repository. Run commands here, not in `C:\dev\woffl_gui`.
Respect the user's decision to stay on **Medium** compute and proceed with
already-authorized work without repeatedly asking permission. Do not infer a
deployment or production-data change from passing local checks.

For the September 11 modeling work, also read the
[airport pause handoff](docs/session_learnings_2026-09-11.md). It preserves the
user’s multi-pump/multi-well validation goal, hydraulics options and requested
Pump Match Over Time screen, plus the completed fixes and diagnostic evidence.
After the user resumed, [the resume record](docs/jp_model_resume_2026-09-11.md)
added coupled S-Pad selection checks and a chronological cross-pump benchmark.
The user confirmed tracker diameters are nominal specs, never measured wear,
and gauges are typically within 40 ft of the JP. The primary fit objective is
reliable pump-size decisions at the pad/field marginal WC, not BHP level alone.

The evidence/calibration subsystem (suction-response evidence layer, multi-point
event calibration, installed-pump coefficients, match-health scorecard,
response diagnostic) is documented in `docs/model_trust_2026-08-10.md` - read it
before touching `server/services/evidence.py`, `calibration_points.py`,
`event_calibration.py`, `fric_calibration.py`'s multipoint block, or the
choke-plan evidence gates in `pad_optimize.py`. Tuning knobs and the live
validation harnesses (`scripts/*_validation.py`, `scripts/*_probe.py`) are
inventoried there. Also read [pump calibration scope](docs/pump_calibration_scope_2026-09-08.md)
before changing persistence, hydration, sizing candidates or pump selection.

---

## 1. What this repo is

`woffl` — *Water Optimization For Fluid Lift* — a numerical solver for liquid-powered
jet-pump oil wells (Milne Point Unit), plus a React SPA + FastAPI web app on top of it.

- `origin` = `github.com/scottpessetto/woffl_gui`, a **fork** of `github.com/kwellis/woffl`.
  There is currently **no `upstream` remote configured** (`git remote -v` shows only `origin`).
- Deployed as a **Databricks App**: `app.yaml` → `uvicorn server.main:app` (React SPA
  served from `web/dist`). The Streamlit app it replaced was deleted 2026-08-18.
  Service principal `2013fc45-c30e-40ac-bef0-df0a758faa3c`; SQL warehouse `698745db7da46ba3`.
- `README.md` covers this fork's app setup and links the current guides, followed
  by library examples. The annulus class is `PipeInPipe`, not `Annulus`.

Version lives in two places kept in sync by bumpver: `pyproject.toml:13` and
`woffl/__init__.py`. Never edit one alone. No release/tagging process is documented;
`.github/workflows/checks.yml` runs offline Python/physics tests and the frontend
tests/build on PRs and main/master pushes. Known physics gaps appear in the job
summary. Black/isort are not enforced. Also verify locally before handing off.

---

## 2. Commands

The verified local venv is Python 3.13.7 / pytest 9.0.3. The library metadata
floor is `>=3.10`; the pinned **application** dependencies require `>=3.11`.

```powershell
# Full suite — this exact invocation. PYTHONPATH=. is mandatory: tests/ is a package
# and files do `from tests.asm_helper import make_well`.
$env:WOFFL_MAX_WORKERS = '1'
$env:PYTHONPATH = '.'
.\venv\Scripts\python.exe -m pytest tests/ -q

# Frontend checks (from web/)
Set-Location web
node --test tests/*.test.mjs
npm run build
Set-Location ..

# API + the built SPA; for Vite iteration see web/README.md
.\venv\Scripts\python.exe -m uvicorn server.main:app --port 8000
```

Formatting is **black + isort** (`pyproject.toml:36`), by convention only — nothing enforces it.
Black was not installed in the local venv at this handoff; do not claim it ran.
Use explicit UTF-8 for file edits. PowerShell piping can replace non-ASCII
literals in Python scripts; prefer `apply_patch` or ASCII scripts with Unicode
escapes, rather than line-slicing/reconstructing source through the shell.

(`tests/test_joint_match_sweep.py` was deleted; the old `--deselect` of it is a no-op and was dropped from the command on 2026-09-02.)

Latest recorded green baseline: **2,198 Python tests and 29 frontend tests passed**
(2026-09-12 capacity repairs), plus the TypeScript/Vite production build.
See [the delivery record](docs/pad_cfp_capacity_delivery_2026-09-12.md). The
[recovered review](docs/recovered_review_2026-09-11.md) records the prior fixes.
Earlier counts in dated reports are milestones, not the current baseline.
Live tests are opt-in (`--run-live`); ordinary verification stays offline.
If a solopump test — especially `TestMarginalConvergence` — goes red after
an upstream merge, a local solver patch was dropped (§4).

---

## 3. DANGER — production Databricks writes

There is **one gated SQL write executor**, targeting live production tables.
There is no staging table or database dry-run mode.

```
server / ipr_anchor -> prop_hist_client.push_prop(s) -> execute_write -> mpu.wells.prop_hist
server / ipr_anchor -> prop_hist_client.push_eng_comment -> execute_write -> mpu.wells.woffl_eng_comment
```

Guards, in order:
1. `execute_write` checks `_write_gate_enabled()` **first**, before connecting
   (`databricks_client.py:279`) → `WritesDisabledError`.
2. `_validate_single_insert` (`databricks_client.py:232`) rejects anything that is not a
   single unchained `INSERT` → `UnsafeWriteStatementError`.
3. `push_prop` checks the `prop_xref` whitelist, resolves `enthid` (raises on 0 or >1 match),
   and requires a finite value (`prop_hist_client.py:275-289`).

**Gate semantics:** truthy = `"1"`, `"true"`, `"yes"` (stripped, lowercased). Everything else —
unset, `""`, `"0"`, `"false"` — is false.

### The landmine that used to be here (defused 2026-09-01)

`.env` (gitignored, local-only) may still carry `ALLOW_DATABRICKS_WRITES`, but
`databricks_client._new_connection()` **no longer exports it**: it reads `.env` with
`dotenv_values()` and copies every key EXCEPT the two gates (`_ENV_GATE_KEYS`:
`ALLOW_DATABRICKS_WRITES`, `ALLOW_PROP_HIST_DELETE`) into the environment, and only where
the key is not already set. Before that, `load_dotenv()` exported the gate, so the first
connection any code path opened - the FastAPI warm loop does it seconds after startup,
unprompted - flipped the production write gate ON for the rest of the process (review
2026-09-01, DATA-1). **To write locally you now set the gate in the shell / app
environment explicitly**; `.env` cannot do it for you. Do not "restore" `load_dotenv()`.

**NEVER:**
- Set `ALLOW_DATABRICKS_WRITES` in a shell, test, or conftest to make something pass.
- Remove the `monkeypatch.delenv` / `os.environ.pop` cleanups in
  `test_databricks_client.py:328,418`, `test_ipr_anchor_pin.py:22-28`,
  `test_prop_hist_client.py` and other write-contract fixtures.
- Run `push_prop(s)` / `push_eng_comment` / `save_ipr_values` / `pump_calibration.save_fit`
  against a real connection "to verify".
- Add `UPDATE`/`DELETE`/`MERGE`/DDL, an `execute_update` sibling, or a second connect path.
  `prop_hist` is **append-only**: corrections are new rows; "unset" is a row with SQL `NULL`.
- Un-pin with a negative sentinel — `wt_uid` is signed (≈ −3.6M..+3.1M). Write `NULL`.

Write functions to treat as live: `ipr_anchor.pin_ipr_anchor` / `clear_ipr_pin` /
`save_ipr_values` / `set_prop_lock`, and `server.services.pump_calibration.save_fit`.
The old Streamlit review-persistence modules were deleted; do not resurrect them.
A multi-prop save goes out as ONE statement through `prop_hist_client.push_props`, not a
loop of `push_prop` (the loop cost 6-9 serialized Delta commits and hung the Save button
for seconds; measured 2026-08-08). It shares `push_prop`'s validator, so every row is
still whitelist- and as-built-checked BEFORE anything is sent. Do not reintroduce the
per-prop loop, and keep every bind marker numbered - a repeated parameter name is a
connector-behaviour bet on the one path that cannot be smoke-tested live.
The FastAPI server (`server/`) rides the SAME gate through the same functions: a local
`uvicorn` run with an explicitly enabled shell gate writes REAL rows via `POST
/api/wells/{name}/save-ipr`, `DELETE .../ipr-pin`, `POST .../prop-lock`, and
`POST .../pump-calibration`
(docs/web_port.md "Write safety").

Reads (`execute_query`, `fetch_*`, `load_saved_ipr`) are SELECT-only and need no gate.
`execute_query` has **no parameter binding** — any identifier spliced into read SQL must be
`int()`-coerced or shape-validated (see `_PROP_ID_SHAPE_RE`, `prop_hist_client.py:69`).

Other env vars: `WOFFL_MAX_WORKERS` (unset: 1 when deployed, `min(cores, 8)` locally —
spawn workers re-import the whole app stack, an uncapped default OOMs; explicit values
always clamped to cpu count by `woffl.assembly.parallelism.worker_ceiling()`; `app.yaml`
pins **2** for the 2-vCPU tier — do not raise it unless the tier changes),
`WOFFL_ENTRY_USER` (overrides attribution),
`WOFFL_WARM_INTERVAL_SEC` / `WOFFL_WARM_WORKERS` / `WOFFL_WARM_WELLS`
(the FastAPI fleet cache warmup - `server/warmup.py`; the worker count is a
warehouse-connection cap, NOT a CPU cap, so it is deliberately separate from
`WOFFL_MAX_WORKERS`). The warehouse bills per **wake window**, not per
statement, so the deployed interval is `app.yaml`'s **43200 (12 h)** - two
passes a day (the forced midnight day-roll plus one in the workday) instead of
the code default's five - and each pass warms the fleet's history with
`history.warm_fleet`'s **two** statements rather than `warm_well` x ~90;
`warm_well` remains the per-well fallback and the on-demand path.
`DATABRICKS_CLIENT_ID`/`_SECRET` (presence of both = "deployed"), local lowercase
`bricks_host`/`bricks_token`/`bricks_http`.

Never spawn a pool with a hardcoded `max_workers` — always pass `worker_ceiling()`.
Never delete the `BrokenProcessPool` → serial fallback in `network_optimizer.py:399`.

Medium performance is a standing cost constraint: one uvicorn process, two
process workers, one heavy background job (`WOFFL_MAX_JOBS=1`), native threads
limited to one, and the 12-hour warm cadence. Reuse `server.pool` and its shared
CPU tokens. Response caching is exact and bounded (64 MiB serialized payload,
1,024 entries, one-hour TTL); its key includes the complete WellConfig, pump
grid, pressure, survey contents and physics-source hash. Reallocate for changed
budgets/prices. Never reuse an allocation merely because its physics was cached.
Local synthetic timings are not hosted latency measurements.

---

## 4. The upstream boundary

| Path | Ownership |
|---|---|
| `woffl/pvt/`, `woffl/geometry/`, `woffl/flow/`, `woffl/assembly/` | **Shared with upstream `kwellis/woffl`**, published to PyPI |
| `woffl/gui/` | Fork-only. Free to change. Never upstreamed. |

Exception: the `*_client.py` / `sql_guards.py` / `jp_history.py` / `cfp_plant.py` modules
inside `woffl/assembly/` are fork-only Databricks glue, not upstream physics.

Editing a shared-library file requires **all three**:
1. Tag the site `# [LIBRARY change -> upstream PR to kwellis/woffl]`.
   `rg -n "upstream PR" woffl/` finds the existing tags.
2. Record it in `docs/upstream_sync.md` (numbered through **45** on 2026-09-12).
3. Guard it with a **named regression test** — every documented patch has a `Guarded by:` line.

For robustness/performance patches, preserve already-converging answers;
fallbacks run only after the existing path fails. Intentional physics corrections
(such as entry-energy-v2) require documented before/after deltas, independent
physical checks and a model-version change when saved fits become incompatible.
Do not freeze a known incorrect answer just to preserve a regression number.

Merging upstream: into a branch, never straight onto a release branch. Conflicts concentrate in
`solopump.py` and `jetflow.py`. If `TestMarginalConvergence` (or any solopump test) goes red,
an upstream merge dropped a local fix.

**Permanent intentional divergence:** `jetflow.throat_entry_zero_tde`,
`jetflow.throat_entry_mach_one`, and `jetplot.throat_entry_book` evaluate the IPR on
`method="vogel"`. Upstream uses `"pidx"` and a past sync silently reverted this once. Do not
"reconcile" it. Also re-check the R-10 dead-code deletion list (`docs/upstream_sync.md:336-376`)
before assuming a symbol that reappears after a merge is needed.

---

## 5. Architecture

```
geometry/   pure hardware+wellbore math, imports nothing from woffl
   |            JetPump, Pipe, PipeInPipe, WellProfile
pvt/        fluid property models, imports only pvt
   |            BlackOil, FormGas, FormWater, ResMix
   v
flow/       physics: IPR, Beggs-Brill, jet-pump internals -> pvt, geometry
   |            InFlow, jetflow.*, outflow.*, twophase.*, errors.*
   v
assembly/   whole-well orchestration + (fork-only) Databricks clients
   |            jetpump_solver(), BatchPump, network_optimizer, *_client
   v
gui/        Fork-only pad/CFP plants + optimizers. No UI code (Streamlit deleted).
```

**Never** import `woffl.gui` from `geometry`/`pvt`/`flow`/`assembly`. `geometry` and `pvt` may
not import `flow` or `assembly`.

Composition: `BlackOil + FormWater + FormGas → ResMix`; `Pipe + Pipe → PipeInPipe`;
survey + `jetpump_md` → `WellProfile`; all of it + a `FormWater` power fluid →
`jetpump_solver(...)` (`assembly/solopump.py:339`) returning
`(psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te)`.

- `ResMix.condition()` **mutates and returns self** and cascades to children. A `ResMix` shared
  between two calculations is not independent — clone it.
- Every property getter requires a prior `condition()`.
- Catch the typed family in `flow/errors.py` (`JetPumpError`, `ConvergenceError`,
  `ThroatEntryNoSolution`, `FlowPatternUnknown`), not bare `ValueError`.
- The solve strategy is bracket → secant → BHP re-seed → bisection. Do not "simplify" it.
- All loaders resolve paths from `Path(__file__)`, never cwd. Follow that.

### Units (documented in docstrings, not enforced by types)

Pressure **psig** at every API boundary (differentials **psid**); temperature **degF**; oil
**STBOPD**, water/PF **BPD**; density **lbm/ft3**; viscosity **cP**; velocity **ft/s**; area
**ft2**; insitu volumetric flow **ft3/s**; lengths **ft** but pipe/nozzle diameters **inches**;
GOR **scf/stb**; watercut a fraction 0–1, given to **≥3 decimals**. Standard conditions are
0 psig / 60 degF. Every docstring `Args:`/`Returns:` entry carries `(type)` + unit — match that.

Style: plain classes with `__init__` (not dataclasses) for physics objects; Google-style
docstrings; terse oilfield snake_case (`psu`, `pte`, `ptm`, `pdi`, `qoil_std`, `knz/ken/kth/kdi`);
numpy arrays end in `_ray`.

**Rate convention (normative text: the RATE CONVENTION docstring in `woffl/gui/params.py`).**
`SimParams.qwf` / `SimulationParams.qwf` AND `WellConfig.qwf` are all **TOTAL LIQUID**
(BLPD, excluding returned power fluid) - the measured quantity (`vw_well_test.WtTotalFluid`,
`prop_hist.ipr_qwf_liq`). Oil is DERIVED downward, exactly once, at each `InFlow`
construction site (`params.inflow_rate`, `network_optimizer._create_well_objects`,
`server/services/factories.build_sim_objects`). Never gross a rate up by `1/(1-wc)`.
(An earlier version of this note said the sidebar held OIL and snapshots converted; that
was inverted against the code and cited a deleted module - corrected 2026-09-01.)

---

## 6. GUI conventions

The Streamlit app was **DELETED 2026-08-18**. `woffl/gui/` no longer holds any
page, tab, sidebar or session-state code, `streamlit` is not a dependency, and
nothing in the tree may import it. What survives under `woffl/gui/` is a set of
Streamlit-free modules the FastAPI server depends on: the pad/CFP plants and
optimizers (`pad_plant_base`, `{s,i,m}_pad_plant`, `e_pad_plant`,
`cfp_pad_plant`, `pad_optimize`, `cfp_moves`, `cfp_optimize`),
`e_pad_booster` (the E-Pad booster candidate screen's physics — MPU pump data
+ `woffl/jp_data` loader; `e_pad_plant` is the thin `PadPlant` face on it, so
the physics has ONE home), `params` (the RATE CONVENTION), `ipr_anchor`,
`fric_calibration`, `gaugeless_match`, `pump_identity`, and `memory_gauge` (parse + apply only).
They are fork-only and keep the `gui` package name purely to avoid churn; new
server-facing helpers belong in `server/` or `woffl/assembly/`.

The port-provenance comments of the form `# mirrors woffl/gui/sidebar.py:...`
that `server/` and `web/src` carried were **removed on 2026-09-02** — every one
of them pointed at a module deleted with the Streamlit app. Any `woffl/gui`
reference left in those trees names a **live** module, so treat it as a real
path. Do not reintroduce citation comments for the deleted app; where the
provenance itself is the point (a query copied unchanged, a threshold carried
over), say "ported unchanged from the retired Streamlit app" and give no path.

**E-Pad is a pad run like S/I/M** (`_pad_plant("E")`), but it is the ONE plant
whose configuration is not a measured tag — no E-Pad SCADA point, no motor
nameplate, no piping rating came with the vendor curve sheets. Build,
suction, speed cap, header cap and amp limit are per-run knobs
(`OptimizeRunRequest.e_pad_*` → `_pad_plant_for_run`), and its frontier is
UNIMODAL in flow (the recommended-range floor collapses deliverable pressure
below `ror_lo * hz_max/60`), so its inverses scan before they bisect. Do not
"simplify" them to the monotone I/M shape: that returns 0.0 at every header.

Construction helpers live in `woffl/assembly/sim_factories.py` (Streamlit-free,
fork-only) — `create_pvt_components`, `create_jetpump`, `create_pipes`,
`create_inflow`, `create_reservoir_mix`, `run_jetpump_solver`. That module is
the ONE copy: `network_optimizer` and the server's `services/factories.py` both
use it, replacing the old "faithful copy minus Streamlit" duplication.
`woffl/assembly/parallelism.py` holds `worker_ceiling`/`usable_cpus` for the
same reason — importing it must never drag in a UI framework.

The user-facing app is the React SPA in `web/` on the FastAPI server in
`server/`; prose in `docs/web_port.md`.

### Web app (web/ + server/) charts - one stack, no exceptions

The React port (`web/` SPA, `server/` FastAPI; prose in `docs/web_port.md`) renders every
chart through one stack. The full rule lives in `web/README.md` ("The chart rule"); the
parts you will otherwise violate:

- Mount charts ONLY via `web/src/charts/ChartPanel.tsx` (drag box zoom, ctrl-wheel zoom,
  shift-wheel pan, dbl-click reset, fullscreen). Never a bare div + `useEChartInstance`.
- ECharts **SVG renderer only** - canvas text blurs at Windows 125/150% display scaling.
- Tooltips through `theme.ts` helpers (`axisTooltip`, `ttHeader`/`ttRow`, `nearestByX`).
  The ECharts default tooltip leaks raw epoch-ms datums and drops unaligned time series.
- Nothing zoom-tracked may use custom-series `renderItem` (it does not re-render on
  dataZoom with `filterMode: "none"`); use markArea/markLine carriers like HistoryStrip.

### Well inputs, installed pumps and WC bounds

- **Save well inputs** saves supported IPR/fluid inputs, never ken/kth/kdi/fnz
  or a fitted Mach parameter. PF pressure remains a live/run input. As-built
  hardware identity comes from the tracker, not a property-save payload.
  Keep the top Save bar visible in Solver and JP History; disabled states explain
  read-only access/loading/invalid inputs. Preview edits do not write. Preserve
  saved numeric precision and invalidate canonical well characteristics after
  successful value saves. Context refresh must preserve newer session edits.
- **Calibrate to field data** uses saved well inputs plus in-era history/tests.
  Save edited well inputs before refitting. **Apply to inputs** is a session
  preview; **Save installed-pump calibration** is a separate action using a
  completed server job ID. Never trust client-supplied coefficients or quality.
- A saved pump fit is active only for the same well, nozzle, throat, exact
  tracker Date Set and physics-model version, with Databricks provenance.
  Return hydraulics also belongs to that identity. BB remains default; H-B/
  Griffith and Shi/Pan are selectable. Changing models clears preview pump
  losses; saving the installed-pump fit persists the model for optimization.
  Read [the hydraulics contract](docs/hydraulics_models_2026-09-11.md) before
  modifying this path. Tulsa remains unavailable; do not imply otherwise.
  Same-size changeouts invalidate it. Missing/stale/legacy scope falls back
  visibly to reference coefficients; never silently migrate numeric friction rows.
- The compact `pump_calibration_v1` record in `woffl_eng_comment` must fit the
  500-character limit **before** calling the human-comment writer (which truncates).
  Preserve coefficient precision and commit identity/quality/coefs atomically.
  Read `datasources.jp_history_fresh()` on save. It fetches and enriches tracker
  data for the request, bypassing stale caches and in-flight warm/SWR refreshes.
  `cache_refresh()` returns a boolean, never the tracker frame. Failures do not evict caches.
- Application WellConfig opts into `pump_calibration_scoped`. Carry `pump_state`
  through sizing, lookup, allocations, fixed scenarios, reports and UI keys.
  Keep-installed and clean-same-size are distinct candidates. Every replacement
  uses ken=.03, kth=.30, kdi=.40, fnz=1.0. Only installed hardware retains a fit.
  Future wells borrow donor well inputs without donor fitted pump properties.
- A closer level match is not proof of wear or a reliable pressure response.
  Preserve BHP/PF/delta-BHP RMS, bound hits, measured/model beta and provisional
  labels. The MPE-42 fit had 76 psi BHP RMS, 61.5% PF RMS and a railed ken despite
  its improved single-test comparison. A 1% fitted area change does not identify wear.
- Historical every-test replay preserves the saved **oil** IPR: when applying
  test WC, convert the total-liquid representation with
  `qwf_test = qwf_saved * (1 - wc_saved) / (1 - wc_test)`. Keep pwf and reservoir
  pressure fixed; use the measured test GOR. Never anchor on each test's oil/BHP
  or change saved inputs. Missing composition remains an explained gap. The
  optional chronological comparisons explicitly refit earlier tests and do
  not save those fits. A future multi-pump fitter must default to one well IPR;
  IPR shifts require explicit user intent.
- WC uncertainty is a collapsed sensitivity panel, default +/-5 **percentage
  points**. Hold the total-liquid IPR anchor fixed; keep GOR fixed in the GUI.
  Fresh PVT per sample; include interior samples in extrema. Hide stale results,
  expose failed samples, and withhold the base result when its solve fails.
  It neither saves/fits inputs nor supplies statistical confidence bounds or
  uncertainty-aware optimization. See [the GUI contract](docs/wc_uncertainty_gui_2026-09-08.md).
- Keep the removed yellow fluid-property/critical-Mach transition banner out of
  Topbar. The user explicitly requested its removal; scoped fit warnings remain.

---

## 7. Testing

CI runs the offline suite and frontend checks; the in-app Test Harness adds field
diagnostics. Entry-energy-v2 uses one unscaled balance and derives choking
from its first reachable energy minimum. `mach_crit` is retired; do not restore
the old multiplier or fit it. `test_entry_energy.py` guards this model; CI's
strict consistency report is not field qualification. See
`docs/entry_energy_implementation_2026-09-08.md` before changing the entry model.
The v2 fluid-property changes and separate field holdout results are recorded
in `docs/fluid_followup_2026-09-08.md`. PF density is a standard-condition input
independent of formation-water SG; never convert standard rates to reservoir
volume twice. Application dependencies require Python >=3.11, use the constraints
file, and must not install the PyPI copy of the vendored `woffl` package.

`tests/conftest.py` defines **no fixtures** — only the `live` marker and `--run-live`. All
fixtures are file-local. Do **not** add a `python_files` setting to `pyproject.toml`:
`batch_test.py` / `outflow_test.py` / `e41_test.py` / `jpump_test.py` rely on the default
`*_test.py` collection pattern.

### Writing a server/API test

`fastapi.testclient.TestClient` against `server.main:app`, with the data layer
monkeypatched — no Databricks, no network. Cache-bearing services must be
cleared between tests (`server.cache.clear_all_caches()`), and anything that
touches the process pool patches `woffl.assembly.parallelism.worker_ceiling`
plus `server.pool._EXECUTOR_CLS` (see `tests/test_pf_range_parallel.py`).
Well context, optimizer hydration and pad readiness now read scoped pump records;
mock `pump_calibration.snapshot`/`resolve_current` and fresh tracker reads as
appropriate in each fixture. A mock of old saved-IPR hydration alone is no
longer enough to keep a test offline.

Optional browser QA is in `tools/check_wc_uncertainty_ui.py` and
`tools/check_pump_scope_ui.py`, using isolated Playwright under `build/browser-qa`.
It intercepts production reads/writes and uses fixtures; screenshots do not
validate field physics. Do not build `web/dist` while a Vite QA session is running
(reload resets state). Under Vite HMR, import the actual loaded module URL when
inspecting Zustand; importing a bare URL can create a second store instance.

The old hand-rolled Streamlit patterns (MagicMock `st`, `sys.modules.setdefault`,
plain-dict `session_state`) are gone with the app — do not reintroduce them.

---

## 8. Known open debt — do not "rediscover"

The [dated handoff](docs/session_learnings_2026-09-08.md) is the current queue.
Older review reports and workspace plans are historical evidence, not a live
backlog. Verify a finding against current code before reviving it; several
formerly listed page splits and review-store paths disappeared with Streamlit.

Resolved: FLOW-4's contradictory Mach energy walks, independent PF density
(P1-13), water PVT and low-GOR compression, pinned vendored app dependencies
(P2-1), Medium scheduling/cache fixes, WC sensitivity UI and installed-pump scope.
Do not reintroduce the retired Mach fit or constant-water model.

Remaining as of this handoff:
- Deploy/measure the September 8 changes on Databricks Medium; local builds and
  synthetic timings do not establish hosted deployment or performance.
- Refit/re-save legacy calibrations against verified current installations.
  This session changed no production property records.
- Reconcile fleet outliers (PF allocation, gauge datum, circulation, IPR and
  contemporaneous WC/GOR). Then validate pressure response on independent events.
  The frozen 35-well audit predates scoped-fit hydration and is retrospective.
- Measurement-informed ranges and coupled S/CFP uncertainty remain unfinished.
  I/M/E now compare two fixed plans under bounded WC/GOR/header stress cases;
  these are engineering assumptions, not probabilities or confidence bounds.
- Fluid approximations remain: SG-scaled pure water rather than brine chemistry,
  bulk PF column, isothermal PVT, empirical oil/acoustic correlations. Numerical
  consistency tests are necessary but do not settle field-model accuracy.
- Historical external schema asks in `docs/prop_hist_asks.md` require checking
  their present status before acting; scoped pump saves need no new schema/grant.

Settled decisions — do not relitigate:
- Water-pump mode is keyed on the explicit `ResMix(model_as_water=True)` flag, **never** on
  `wc == 1.0`, and must propagate into **both** throat-mixture construction sites
  (`jetflow.jetpump_base_calcs` and `solopump.discharge_residual`).
- A WC ≥ 0.99 well raises unless `offline=True`. Silently zeroing is the worst option.
- `prop_xref` deliberately excludes pump identity (`jp_nozzle`, `jp_throat_ratio`) and workflow
  state (`well_reviewed`, `well_offline`).
- CFP moves models **deltas off a stated reference anchor only**. Use measured
  conditions when available; label manual scenarios and never mix asynchronous
  pad pressures into them. Never reintroduce an exogenous / bottom-up plant load.
- Pump-at-test-date tenure is **set-to-set** (`Date Set` → next `Date Set`). `Date Pulled` is
  never consulted.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| MPU | Milne Point Unit — the field |
| pad | Surface production pad; single letters B, C, E, F, G, H, I, J, M, S |
| S/I/M/E pad | The four pads with booster-plant models the pad optimizer can run. E joined 2026-08-27 and is the only one whose plant configuration is per-run rather than measured |
| POPS | Pad with on-pad production separation (E/F/H/I/M/S) — handles its own lift water, so only formation water reaches the plant |
| CFP / PW | The produced-water plant whose discharge pressure the CFP pages optimize |
| JP | Jet pump, sized nozzle-number + throat letter, e.g. `12B`, `9X` |
| JPCO | Jet pump changeout |
| PF | Power fluid — high-pressure water driving the nozzle; `ppf_surf` = its surface pressure, psig |
| IPR | Inflow Performance Relationship — deliverability vs flowing BHP |
| vogel / pidx | Curved IPR vs straight-line productivity-index IPR |
| qwf / pwf / pres | IPR anchor rate, flowing BHP at that rate, reservoir pressure |
| psu / pte / ptm / pdi / pni | Suction, throat-entry, throat-mixture, diffuser-discharge, nozzle-inlet pressures |
| knz / ken / kth / kdi | Jet-pump friction coefficients: nozzle, entrance, throat, diffuser |
| wc / GOR / FGOR | Water cut (fraction) / gas-oil ratio / formation GOR |
| form-WC vs total-WC | Formation-water cut vs total (formation + lift water) cut — mixing bases over-recommends bring-online |
| marginal WC | Legacy economics gate (water cut above which a well stops paying for its water). Since 2026-09 it is only a LABEL: the optimizers price water with λ, and a gate w maps to λ = (1 − w) / w |
| λ / water price | BOPD given up per BPD of machine water in the pad objective oil − λ·water. Default λ=0 maximizes oil within capacity; manual prices change the objective. Concave-frontier λ is a separate capacity-value diagnostic (`docs/pad_cfp_capacity_delivery_2026-09-12.md`). |
| SI / BOL / LTSI | Shut in / bring on line / long-term shut-in (mechanical, out of Triage scope) |
| joint match | Solve IPR + PF pressure + friction coefs so the installed pump reproduces a test's oil AND PF |
| backmatch | Oil-only inverse: infer the `pwf` at which the installed pump makes the test's oil rate |
| gaugeless match | `woffl/gui/gaugeless_match.py` / `POST /match-test`: for wells with no downhole gauge, the test's PF rate through the nozzle stands in for the BHP measurement; fits (pwf, kth, kdi) so the installed pump reproduces the test's oil AND PF. Reports `pf_reachable=False` (BHP not identified) when the catalog nozzle cannot pass the test's PF at any BHP |
| washout | Nozzle/throat erosion, flagged when required PF pressure exceeds the pad threshold |
| si_ladder | Ranks shut-in candidates by water contribution to plant load, applying the POPS rule |
| Header Impact / HPI | Tool that re-solves wells at a candidate header pressure and reports the response |
| XV / ProdXV / PFXV | Production / power-fluid safety valves, 1=open 0=closed; usually empty on the hosted app |
| MCKP / MILP | Multiple-choice knapsack (CP-SAT) / mixed-integer LP — the two optimizer paths. Since 2026-09 they solve the SAME priced problem over the same candidate set; a pad run with MCKP re-solves the winner with MILP and reports `solver_agreement` |
| equal-slope / λ | At the optimum, marginal oil per unit shared resource is equal across wells; λ_today = d(fleet oil)/dP |
| prop_hist / prop_xref | `mpu.wells` tables: append-only per-well property history + the valid-`prop_id` whitelist |
| ipr_wt_uid | prop_hist key pinning a well's chosen IPR anchor well-test; SQL NULL = un-pinned |
| enthid | Well entity id, FK into `vw_well_header` |
| pump scope | Installed hardware identity = well + nozzle + throat + exact Date Set + physics model; a same-size clean replacement is a separate candidate |
| bit-identical | Required for unchanged physics paths in robustness/performance work; intentional model corrections have documented prediction deltas |
