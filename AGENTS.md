# AGENTS.md — woffl_gui

Operating rules for coding agents in this repo. Read this before touching anything.
Prose lives in `docs/`; this file is only the rules you will otherwise violate.

The evidence/calibration subsystem (suction-response evidence layer, multi-point
event calibration, `mach_crit`/`nozzle_area_factor`, match-health scorecard,
response diagnostic) is documented in `docs/model_trust_2026-08-10.md` - read it
before touching `server/services/evidence.py`, `calibration_points.py`,
`event_calibration.py`, `fric_calibration.py`'s multipoint block, or the
choke-plan evidence gates in `pad_optimize.py`. Tuning knobs and the live
validation harnesses (`scripts/*_validation.py`, `scripts/*_probe.py`) are
inventoried there.

---

## 1. What this repo is

`woffl` — *Water Optimization For Fluid Lift* — a numerical solver for liquid-powered
jet-pump oil wells (Milne Point Unit), plus a React SPA + FastAPI web app on top of it.

- `origin` = `github.com/scottpessetto/woffl_gui`, a **fork** of `github.com/kwellis/woffl`.
  There is currently **no `upstream` remote configured** (`git remote -v` shows only `origin`).
- Deployed as a **Databricks App**: `app.yaml` → `uvicorn server.main:app` (React SPA
  served from `web/dist`). The Streamlit app it replaced was deleted 2026-08-18.
  Service principal `2013fc45-c30e-40ac-bef0-df0a758faa3c`; SQL warehouse `698745db7da46ba3`.
- `README.md` is the **upstream library README** and is stale for this fork. It documents no
  GUI, no Databricks, no pads — and it references `from woffl.geometry import Annulus`, which
  does not exist (`geometry/__init__.py:9-11` exports `JetPump, Pipe, PipeInPipe, WellProfile`).

Version lives in two places kept in sync by bumpver: `pyproject.toml:13` and
`woffl/__init__.py`. Never edit one alone. No release/tagging process is documented;
`.github/workflows/` is deliberately **empty** — there is **no CI**. Nothing runs pytest,
black, or isort on push. You must verify locally.

---

## 2. Commands

The venv is Python 3.13.7 / pytest 9.0.3 (the package floor is `>=3.10`).

```bash
# Full suite — this exact invocation. PYTHONPATH=. is mandatory: tests/ is a package
# and files do `from tests.asm_helper import make_well`.
WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m pytest tests/ -q

# Opt-in tests that hit the real well-properties source
... -m pytest tests/ -q --run-live

# Run the app
uvicorn server.main:app --port 8000        # API + the built SPA
cd web && npm run dev                       # Vite dev server, proxies /api to :8000
```

Formatting is **black + isort** (`pyproject.toml:36`), by convention only — nothing enforces it.

(`tests/test_joint_match_sweep.py` was deleted; the old `--deselect` of it is a no-op and was dropped from the command on 2026-09-02.)

Green baseline: **1,667 passed** in ~26 s (2026-09-02; was 1,686 + 1 skipped in ~52 s on 2026-08-03).
If a solopump test — especially `TestMarginalConvergence` — goes red after
an upstream merge, a local solver patch was dropped (§4).

---

## 3. DANGER — production Databricks writes

There is **one** write path in the entire codebase and it targets a **live production
Unity Catalog table**. There is no staging table and no dry-run mode.

```
gui  ->  prop_hist_client.push_prop()  ->  databricks_client.execute_write()  ->  mpu.wells.prop_hist
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
  `test_prop_hist_client.py:70-71`, `test_step_review_wells_pin.py:57-58`.
- Run `push_prop` / `sync_pad` / `save_ipr_values` against a real connection "to verify".
- Add `UPDATE`/`DELETE`/`MERGE`/DDL, an `execute_update` sibling, or a second connect path.
  `prop_hist` is **append-only**: corrections are new rows; "unset" is a row with SQL `NULL`.
- Un-pin with a negative sentinel — `wt_uid` is signed (≈ −3.6M..+3.1M). Write `NULL`.

Write functions to treat as live: `ipr_anchor.pin_ipr_anchor` / `clear_ipr_pin` /
`save_ipr_values` / `set_prop_lock`, `review_persistence.sync_pad` (runs on **every pad-page
rerun**), `workflow_steps/step_review_wells._maybe_pin_saved_ipr`.
A multi-prop save goes out as ONE statement through `prop_hist_client.push_props`, not a
loop of `push_prop` (the loop cost 6-9 serialized Delta commits and hung the Save button
for seconds; measured 2026-08-08). It shares `push_prop`'s validator, so every row is
still whitelist- and as-built-checked BEFORE anything is sent. Do not reintroduce the
per-prop loop, and keep every bind marker numbered - a repeated parameter name is a
connector-behaviour bet on the one path that cannot be smoke-tested live.
The FastAPI server (`server/`) rides the SAME gate through the same functions: a local
`uvicorn` run with `.env` present writes REAL prop_hist rows via `POST
/api/wells/{name}/save-ipr`, `DELETE .../ipr-pin`, and `POST .../prop-lock`
(docs/web_port.md "Write safety").

Reads (`execute_query`, `fetch_*`, `load_saved_ipr`) are SELECT-only and need no gate.
`execute_query` has **no parameter binding** — any identifier spliced into read SQL must be
`int()`-coerced or shape-validated (see `_PROP_ID_SHAPE_RE`, `prop_hist_client.py:69`).

Other env vars: `WOFFL_MAX_WORKERS` (unset: 1 when deployed, `min(cores, 8)` locally —
spawn workers re-import the whole app stack, an uncapped default OOMs; explicit values
always clamped to cpu count by `scotts_tools/_common.worker_ceiling()`; `app.yaml`
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
   `grep -rn "upstream PR" woffl/` finds all ~50 existing tags.
2. Record it in `docs/upstream_sync.md` (15 patches inventoried there).
3. Guard it with a **named regression test** — every documented patch has a `Guarded by:` line.

House style for library patches: **additive, explicitly gated, bit-identical** — a well that
already converged must return the same `psu`/oil after your change. Fallbacks run only *after*
the existing path fails.

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
nothing in the tree may import it. What survives under `woffl/gui/` is 16
Streamlit-free modules the FastAPI server depends on: the pad/CFP plants and
optimizers (`pad_plant_base`, `{s,i,m}_pad_plant`, `e_pad_plant`,
`cfp_pad_plant`, `pad_optimize`, `cfp_moves`, `cfp_optimize`),
`e_pad_booster` (the E-Pad booster candidate screen's physics — MPU pump data
+ `woffl/jp_data` loader; `e_pad_plant` is the thin `PadPlant` face on it, so
the physics has ONE home), `params` (the RATE CONVENTION), `ipr_anchor`,
`fric_calibration`, `pump_identity`, and `memory_gauge` (parse + apply only).
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

---

## 7. Testing

There is no CI. The suite plus the in-app Test Harness page is the whole safety net.

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

The old hand-rolled Streamlit patterns (MagicMock `st`, `sys.modules.setdefault`,
plain-dict `session_state`) are gone with the app — do not reintroduce them.

---

## 8. Known open debt — do not "rediscover"

Read the two status banners at `docs/code_review_2026-07-01.md:3-27` for current state; the
table in `review_status_2026-07-06.md` is the **pre-fix** snapshot and will mislead you.

Those two files are dated point-in-time review artifacts and still cite a `CLAUDE.md` that never
existed — left as written on purpose. Every *live* reference in `woffl/`, `tests/`, `tools/`, and
`docs/upstream_sync.md` now points at this file.

Still open:
- **P1-13 (behavior half)** — `pad_optimize.py:194,273,532,701,801` hardcode `rho_pf=62.4`
  against the I/M plants' real PF SG ≈1.03–1.04. Wire-or-remove decision unmade.
- **R-1** — the three pad pages are ~75–80% triplicated across ~2,900 lines.
- **R-2..R-5** — file splits: `header_impact.py` (3,257 lines), `jetpump_solver.py` (2,783),
  `well_sort.py`, `utils.py`, `batch_run.py`, `pdf_export.py`. Use the Python cutter pattern,
  **not** PowerShell line-slicing (mojibake lesson).
- **P2-1** — `requirements.txt:8` still pins bare `woffl` (Databricks installs *unpatched
  upstream* woffl into site-packages beside the vendored patched tree; it works by path-precedence
  luck) and `:10` still lists the unused `databricks-sdk`.
- Dead out-of-range check in `wellprofile._depth_interp` (`is False` on a numpy bool);
  orphaned `databricks_client.get_tags_for_wells`; zero-caller `WellTestProcessor`,
  `assembly/calibration.py` (+ `NetworkOptimizer.set_calibration`), `pf_calibration.robust_bracket`,
  `cfp_optimize.run_joint_optimization`. (The "unguarded tag-list f-string" note is obsolete -
  that path no longer exists.) Full inventory: `docs/code_review_2026-09-01.md` §5.
- External asks (`docs/prop_hist_asks.md`): `manual_well_tests` table, NULL un-pin confirmation,
  MODIFY on `woffl_active`.
- `pwf` auto-match seed is clamped to (100, 2500) psi; PF surface pressure floats freely.
- **Stale comment, actively misleading:** `pad_page.py:1594-1596` says the review store is
  autosaved to `mpu.wells.woffl_review_store`. No such table exists anywhere in the code —
  `sync_pad` writes `mpu.wells.prop_hist`. `woffl_active` / `woffl_review_store` appear only in
  that comment and in `docs/prop_hist_asks.md` ask (f), which is undelivered.
- Independent copies of the `ALLOW_DATABRICKS_WRITES` truthy check exist —
  `databricks_client.py` (`_write_gate_enabled`), `ipr_anchor.py`, `server/config.py`
  (`writes_enabled`), and the same shape for the delete gate in `prop_hist_client`. The
  `ipr_anchor` copy is deliberate (the UI must hide controls before attempting a push); keep
  them in sync if you touch the semantics.

Settled decisions — do not relitigate:
- Water-pump mode is keyed on the explicit `ResMix(model_as_water=True)` flag, **never** on
  `wc == 1.0`, and must propagate into **both** throat-mixture construction sites
  (`jetflow.jetpump_base_calcs` and `solopump.discharge_residual`).
- A WC ≥ 0.99 well raises unless `offline=True`. Silently zeroing is the worst option.
- `prop_xref` deliberately excludes pump identity (`jp_nozzle`, `jp_throat_ratio`) and workflow
  state (`well_reviewed`, `well_offline`).
- CFP moves models **deltas off a measured anchor only**. Never reintroduce an exogenous /
  bottom-up plant water load.
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
| λ / water price | BOPD given up per BPD of lift water in the pad objective oil − λ·water; one λ for every engine (`docs/optimization_redesign_2026-09.md`). auto = the plant budget's own shadow price off the pooled pump frontier |
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
| review store | Per-pad persisted well-review state (`well_review_store.py`) feeding the pad optimizer |
| bit-identical | The acceptance bar for a library patch: the already-working path returns exactly the same numbers |
