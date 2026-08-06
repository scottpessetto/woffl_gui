# AGENTS.md — woffl_gui

Operating rules for coding agents in this repo. Read this before touching anything.
Prose lives in `docs/`; this file is only the rules you will otherwise violate.

---

## 1. What this repo is

`woffl` — *Water Optimization For Fluid Lift* — a numerical solver for liquid-powered
jet-pump oil wells (Milne Point Unit), plus a Streamlit GUI on top of it.

- `origin` = `github.com/scottpessetto/woffl_gui`, a **fork** of `github.com/kwellis/woffl`.
  There is currently **no `upstream` remote configured** (`git remote -v` shows only `origin`).
- Deployed as a **Databricks App**: `app.yaml` → `streamlit run woffl/gui/app.py`.
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

The venv is Python 3.13.7 / streamlit 1.57 / pytest 9.0.3 (the package floor is `>=3.10`).

```bash
# Full suite — this exact invocation. PYTHONPATH=. is mandatory: tests/ is a package
# and files do `from tests.asm_helper import make_well`.
WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m pytest tests/ -q

# Opt-in tests that hit the real well-properties source
... -m pytest tests/ -q --run-live

# Run the app
streamlit run woffl/gui/app.py
```

Formatting is **black + isort** (`pyproject.toml:36`), by convention only — nothing enforces it.

Known-slow, skip during iteration: `tests/test_joint_match.py`,
`tests/test_joint_match_sweep.py` (72-combination sweep),
`tests/test_utils.py::TestDataFlowIntegration::test_full_solver_flow` (real end-to-end solve).

Green baseline: **1686 passed, 1 skipped** in ~52 s (2026-08-03). The one skip is the
`live`-marked class. If a solopump test — especially `TestMarginalConvergence` — goes red after
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

### The single worst landmine in this repo

`.env` (gitignored, local-only) sets `ALLOW_DATABRICKS_WRITES` to a **truthy** value, and
`databricks_client._new_connection()` calls `load_dotenv()` (`databricks_client.py:120`).
So **any local code path that opens a Databricks connection silently turns the production
write gate ON for the rest of the process.** Tests document this exact leak
(`tests/test_ipr_saved_values.py:73-77`).

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

Reads (`execute_query`, `fetch_*`, `load_saved_ipr`) are SELECT-only and need no gate.
`execute_query` has **no parameter binding** — any identifier spliced into read SQL must be
`int()`-coerced or shape-validated (see `_PROP_ID_SHAPE_RE`, `prop_hist_client.py:69`).

Other env vars: `WOFFL_MAX_WORKERS` (default 1, clamped to cpu count by
`scotts_tools/_common.worker_ceiling()`; `app.yaml` pins **2** for the 2-vCPU tier — do not
raise it unless the tier changes), `WOFFL_ENTRY_USER` (overrides attribution),
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
gui/        Streamlit. Nothing below may import this.
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

**Unit gotcha, do not simplify away** (`workflow_steps/well_review_store.py:20-31`):
sidebar/Solver `params.qwf` is **OIL** (BOPD); `WellConfig.qwf` is **TOTAL LIQUID** (BLPD).
Snapshots convert `qwf/(1-wc)`; the optimizer converts back.

---

## 6. GUI conventions

Entrypoint `woffl/gui/app.py`. Routing is a **manual `st.radio`** with `key="app_mode_radio"`
(`app.py:334`) plus an if/elif dispatch with lazy per-branch imports. There is no
`st.navigation`, no `st.Page`, no `pages/` dir. Renaming a mode requires a pre-render
session_state migration or live sessions silently reset (two already exist, `app.py:303-323`).

Import-time side effects in `app.py` you must preserve: the `sys.path` insert,
`set_entry_user_provider(_streamlit_forwarded_user)` (`app.py:53-55` — hosted attribution comes
from the `X-Forwarded-Email` header; without it every write is stamped with the service
principal), and the module-level `@st.cache_data` jp-history fetcher.

### Pads: two layers, do not conflate

- **Physics** — `PadPlant` (`pad_plant_base.py:66`) with 9 `NotImplementedError` overrides and
  class attrs `coupling` / `n_pump_options` / `max_header_psi`. Subclasses `SPadPlant`,
  `IPadPlant`, `MPadPlant`, `FixedHeaderPlant` all live **in that same file**.
  `s_/i_/m_pad_plant.py` are delegation shims exposing `PLANT` — put physics in the class.
  Every pressure handed to the optimizer must be clamped into **[1000, 5000] psi**.
- **Page** — frozen `PadSpec` dataclass (`pad_page.py:70`) consumed by the single
  `run_pad_page(spec)` render path (`pad_page.py:1590`). A `*_pad_page.py` owns only its `SPEC`,
  a no-arg `run_*_pad_page()`, and pad-specific plot/warning hooks. Never fork `run_pad_page`.

To add a pad: new `x_pad_page.py` with `SPEC` + no-arg entry function, unique `pad` and
`prefix`, a plant (`FixedHeaderPlant(3200.0)` is a complete working pad), then register in
**both** `pad_hub._SPEC_PADS` and `pad_hub._spec_for`. Nothing in `app.py` changes — the note at
`pad_page.py:35-36` saying otherwise is stale. Add a spec test to `tests/test_pad_page_specs.py`.

CFP (`cfp_pad_page.py`) is deliberately **not** a `PadSpec` — four pads share one plant with
different delivered PF. Don't force it through `run_pad_page`.

### The Solver renders two different layouts

`tabs/jetpump_solver.render_tab` serves both the Single-Well page and the pad-review step, and
they look nothing alike. The switch is one line:

```python
_standalone = hero_container is None   # True  = Single-Well page
                                       # False = embedded in pad review
```

- **Standalone** is the workbench: verdict line → rate/pump trend → IPR curve `|` control panel
  (50/50) → well tests. `app.py` also drops the `WOFFL Haus` header on this view, keyed off
  `app_mode_radio` + `sw_active_view` from the *previous* run's widget state.
- **Embedded** keeps the original stacked ordering, because the pad page injects
  `hero_container` / `anchor_container` / `jp_strip_container` and positions them itself.

Both branches are live — a change to one is not a change to the other. Zone containers are all
declared up front (Streamlit paints in declaration order) and existing blocks are filled into
the slots, so reordering never means moving code.

Layout facts learned the hard way; don't re-derive them:
- A column row is as tall as its tallest cell, so **columns only compress when cells balance**.
  A 2/5 action panel (422px) wrapped its text into a 1,942px column and made the page *taller*;
  a 350px table beside the ~1,100px IPR group was pure white space. 50/50 at ~600px works.
- `st.metric` needs ~180px per card for a 5-digit value. Four across a half-width column clip to
  `1,665 …`. Use text or fewer cards below full width.
- `st.dataframe` already caps height and scrolls internally — capping it yourself buys nothing.
- Streamlit's grid renders a float column's nulls as a literal `None` and **ignores a Styler's
  `na_rep`** (the Styler is fine; its HTML has the em dash). Pre-format to strings and let the
  Styler carry only colour. It also ignores `text-align` from a Styler.

### Streamlit rules that break live sessions when violated

- `@st.cache_data` only. There is zero `@st.cache_resource` and zero `@st.fragment` in the repo.
- **Cached fetcher raises; uncached wrapper soft-fails.** Failures must never be cached
  (`utils.py:1773` vs `utils.py:284`).
- Pass cache-key args **explicitly** — `st.cache_data` keys on args as passed, so a no-arg call
  caches under a different key and reruns the query.
- Streamlit **forbids writing a widget's session_state key after that widget renders**. All
  programmatic seeds run first — this is why `sync_pad` is called before any widget
  (`pad_page.py:1599`).
- A seed outside a widget's min/max is silently reset to the **minimum**. Adding a seeded
  number input means adding its bounds to `sidebar.SEED_BOUNDS`.
- `st.checkbox(value=...)` must read `st.session_state.get(key, default)`, never a literal.
- Widget state is GC'd when its view isn't rendered — mirror persisted selections in a
  **non-widget shadow key** (`pad_hub.py:29`).
- Session key naming: `sw_*` single-well; `_leading_underscore` internal flags; pad pages use
  `f"{spec.prefix}_..."` (`sp_`/`ip_`/`mp_`) — changing a prefix strands live sessions and is
  test-pinned. Sidebar uses two tiers: logical `k` + widget `f"{k}_input"`.
- Background cache warming = daemon `threading.Thread` + `add_script_run_ctx(thread)`, guarded
  by a once-per-session flag, with per-fetch exceptions swallowed.
- Downloads use `components/download.py:autodownload(...)`. **Never** a two-step
  Generate-then-Download.
- Never `print()` for warnings — invisible on Databricks Apps and useless inside `@st.cache_data`.
- Never `d.get(k) or default` on numeric Databricks fields — NaN is truthy and reaches the solver.
- Wrap every per-well batch loop iteration in `try/except`. One bad well must never blank a page.

### Reuse, don't reinvent

`utils.get_well_tests_for_well()` is the single source of truth for per-well tests (applies the
memory-gauge BHP override, extended windows, manual tests, count cap). `utils.create_*` are the
object factories. `tab_helpers.py` and `pad_helpers.py` are deliberately Streamlit-free and
unit-tested. `params.SimulationParams` is the one container threaded sidebar → page → tabs.
`explainers.render_kcoef_explainer()` is the one help block.

### Add-here-too registries

| Adding | Touch |
|---|---|
| App mode | `app.py:294-301` list **and** `app.py:348-379` dispatch (+ migration if renaming) |
| Pad | `pad_hub._SPEC_PADS` **and** `pad_hub._spec_for` |
| Single-well tab | `single_well_page._VIEWS` **and** `_render_view` **and** the import block |
| Scott's Tools tab | `scotts_tools/page.py` import block, `tab_labels`, **and** the `with tabs[i]` body |
| A `SimulationParams` field affecting sweep physics | `tab_helpers.physical_sweep_signature` — **or caches silently serve results computed under the old value** |
| A seeded number input | `sidebar.SEED_BOUNDS` |
| A stored per-well review field | the right tuple in `workflow_steps/well_review_store.py:46-121` (lands in the CSV schema) |
| A GUI module | `tests/test_gui_smoke.py` `PAGE_MODULES` |

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

Do not move or rename `tests/harness_cases.py` — the **deployed app** imports it via
`woffl/gui/scotts_tools/test_harness.py`.

### Writing a GUI test

There is no `streamlit.testing.AppTest` here. Three hand-rolled patterns, in preference order:

1. **Function doesn't touch `st`** → mock nothing streamlit-related; monkeypatch only the data
   functions. Reference: `tests/test_batch_automatch_inputs.py`.
2. **Real streamlit available** → `monkeypatch.setattr(<module>, "st", fake)` where `fake` is a
   small class with a dict `session_state`. Reference: `tests/test_well_chars_loading.py`.
   If the real `st.cache_data` is live, clear the cache around the test.
3. **Module must import before streamlit exists** → `MagicMock` + a passthrough `cache_data` +
   `sys.modules.setdefault("streamlit", mock)`, then import the page module.
   Reference: `tests/test_solver_ipr_sync.py:19-34`.

⚠ Pattern 3 **poisons the process**: a later `import streamlit.components.v1` fails against a
mock, making `test_gui_smoke` order-dependently flaky. Prefer 1 or 2.

Always patch the **module's** `st` attribute, not the `streamlit` package.

A plain-dict `session_state` cannot reproduce Streamlit's real widget-state GC or a full
tab-switch. Fixes in that area must additionally be click-tested in the running app.

Databricks in tests: monkeypatch/`patch` `execute_query`, `_new_connection`,
`fetch_well_props_enriched`, `push_prop`, `resolve_entry_user`, and — critically —
`load_saved_ipr`, or you hit the real client and trip the `.env` gate leak (§3).
No test may reach the network unless marked `@pytest.mark.live`.

Zero coverage today, per `docs/code_review_2026-07-01.md:445-450`: pad plant models, pad pages,
all `workflow_steps`, `well_review_store`, Triage logic, and the `pf_scenario` / `jp_calibration`
/ `jp_fric_trend` / `jp_washout` compute cores.

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
  orphaned `databricks_client.get_tags_for_wells`; unguarded tag-list f-string in
  `databricks_client`; zero-caller `WellTestProcessor`.
- External asks (`docs/prop_hist_asks.md`): `manual_well_tests` table, NULL un-pin confirmation,
  MODIFY on `woffl_active`.
- `pwf` auto-match seed is clamped to (100, 2500) psi; PF surface pressure floats freely.
- **Stale comment, actively misleading:** `pad_page.py:1594-1596` says the review store is
  autosaved to `mpu.wells.woffl_review_store`. No such table exists anywhere in the code —
  `sync_pad` writes `mpu.wells.prop_hist`. `woffl_active` / `woffl_review_store` appear only in
  that comment and in `docs/prop_hist_asks.md` ask (f), which is undelivered.
- Three independent copies of the `ALLOW_DATABRICKS_WRITES` truthy check exist —
  `databricks_client.py:225`, `ipr_anchor.py:188`, `scotts_tools/jp_calibration.py:930`. The
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
| S/I/M pad | The three pads with dedicated optimization pages and booster-plant models |
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
| marginal WC | Water cut above which a well stops paying for its water handling |
| SI / BOL / LTSI | Shut in / bring on line / long-term shut-in (mechanical, out of Triage scope) |
| joint match | Solve IPR + PF pressure + friction coefs so the installed pump reproduces a test's oil AND PF |
| backmatch | Oil-only inverse: infer the `pwf` at which the installed pump makes the test's oil rate |
| washout | Nozzle/throat erosion, flagged when required PF pressure exceeds the pad threshold |
| si_ladder | Ranks shut-in candidates by water contribution to plant load, applying the POPS rule |
| Header Impact / HPI | Tool that re-solves wells at a candidate header pressure and reports the response |
| XV / ProdXV / PFXV | Production / power-fluid safety valves, 1=open 0=closed; usually empty on the hosted app |
| MCKP / MILP | Multiple-choice knapsack / mixed-integer LP — the two optimizer paths, which currently solve different problems |
| equal-slope / λ | At the optimum, marginal oil per unit shared resource is equal across wells; λ_today = d(fleet oil)/dP |
| prop_hist / prop_xref | `mpu.wells` tables: append-only per-well property history + the valid-`prop_id` whitelist |
| ipr_wt_uid | prop_hist key pinning a well's chosen IPR anchor well-test; SQL NULL = un-pinned |
| enthid | Well entity id, FK into `vw_well_header` |
| review store | Per-pad persisted well-review state (`well_review_store.py`) feeding the pad optimizer |
| bit-identical | The acceptance bar for a library patch: the already-working path returns exactly the same numbers |
