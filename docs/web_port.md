# WOFFL web port (React + FastAPI)

Status: v1 shipped 2026-08; CUT OVER to production 2026-08-06 - the root
`app.yaml` now runs this app. The Streamlit config is preserved in
`app-streamlit.yaml` for rollback; the Streamlit GUI still runs locally for
the flows not yet ported.

## Why this architecture

The ask was "a real Node.js app": professional, faster, wide-screen, sidebar
that gets out of the way. The physics (`woffl/flow|pvt|geometry|assembly`,
1686 tests, bracket -> secant -> BHP re-seed -> bisection solve strategy) is
Python and does not move. So:

- **Frontend = the Node.js part.** TypeScript + React 19 SPA built with Vite;
  Tailwind v4; TanStack Query for server state; Zustand for sim inputs;
  ECharts (SVG renderer - crisp at fractional Windows display scaling; see
  web/README.md "The chart rule"). Fully client-rendered: interactions never
  re-execute a server script (the Streamlit model) - only real compute makes
  a network call.
- **Backend = thin FastAPI** (`server/`) importing the existing assembly
  clients and solver wrappers. Databricks Apps' own reference pattern
  (React + FastAPI) and its default env includes fastapi/uvicorn.
- One process, one runtime on the 2 vCPU tier. uvicorn binds via
  UVICORN_HOST/UVICORN_PORT which the Apps runtime sets automatically.

## What died with Streamlit

- Session-state two-tier keys (`k`/`k_input`), widget GC mirrors, seed
  clamping on rerun: the client owns form state in one Zustand store.
- Server-rendered charts: the client renders; the Vogel curve is mirrored in
  `web/src/lib/vogel.ts` and redraws live as reservoir pressure changes.
- Whole-script reruns: auto-solve is a debounced (400 ms) POST /api/solve
  keyed by a stable params hash, cached client-side (instant back/forward).

## API surface (v1)

All under `/api` (OpenAPI at `/api/docs`). Contract: `server/schemas.py`
mirrored by `web/src/api/types.ts`.

| Endpoint | Purpose |
|---|---|
| GET /meta | version, user (X-Forwarded-Email), writes_enabled, warehouse |
| GET /wells | universe from vw_prop_mech/vw_prop_resvr (+ csv fallback) |
| GET /wells/{name}/context | server replay of the sidebar seeding pipeline: chars, pump-from-history, IPR auto-populate, saved-IPR overlay + locks, live PF; returns seeds + as-built locks + provenance |
| GET /wells/{name}/tests | windowed well tests (months, cap) |
| GET /wells/{name}/profile | survey MD/VD/HD + filtered profile + JP marker |
| GET /wells/{name}/jp-history | installs + extended tests + daily BHP |
| GET /wells/{name}/ipr-pin | saved anchor pin status |
| POST /wells/{name}/save-ipr | WRITE: pin the anchor test + save the sidebar's IPR/fluid values (+ calibrated friction) to prop_hist |
| DELETE /wells/{name}/ipr-pin | WRITE: un-pin (appends the cleared-marker row) |
| POST /wells/{name}/prop-lock | WRITE: toggle a WC/GOR/ResP field lock (locking pins the sent value in the same click) |
| POST /solve | single solve -> psu, qoil_std, fwat, qnz, mach, sonic |
| POST /ipr/fit | Vogel fit (recent / median / specific anchor) + seeds |
| POST /batch | nozzle x throat sweep + recommender + exp fit curve |
| POST /pf-range | oil vs PF-pressure sweep |
| POST /pressure-profile | surface -> suction traverse, both strings |
| GET /database/wells, /database/aging-pumps, /database/prop-history/{well} | Well Database page |
| GET /well-sort/tables | online / offline / LTSI tables + POPs config echo |
| GET /well-sort/events | 30-day shut-in events (down-day threshold walk) |
| GET /well-sort/marginal-wc | field marginal WC (cumulative-water walk) |
| GET /well-sort/pad-marginal-wc | per-POPs-pad marginal WC + pump headroom |
| GET /well-sort/triage | keep / SI / BOL decisions vs the marginal line |
| GET /well-sort/bench.xlsx | 3-sheet MPU_Well_Bench workbook |
| POST /well-sort/refresh | clears the Well Sort fetch caches (read-only op) |

Server caching mirrors the old `@st.cache_data` TTLs (`server/config.py`):
tests 24 h, chars/PF/profiles 1 h, saved IPR / prop history 5 min. Failures
are never cached. Startup warms chars/PF/jp-history/tests in daemon threads.

## Write safety

The server has exactly THREE write endpoints - the Solver's "Save as well
default", its un-pin, and the sidebar's WC/GOR/ResP lock toggles (ported
2026-08-06). All replicate the Streamlit gate chain in full:

1. Router pre-check: `writes_enabled()` off -> 403, and the UI hides the
   save controls entirely on `/api/meta.writes_enabled` (the read-only
   badge shows instead).
2. All mechanics go through `woffl.gui.ipr_anchor.pin_ipr_anchor` /
   `save_ipr_values` / `clear_ipr_pin` - the SAME functions the Streamlit
   button calls - so the rules can never diverge: `push_prop` re-checks the
   gate, enforces the prop_xref whitelist, and REJECTS as-built physical
   properties outright (`AS_BUILT_PROP_IDS`); WC caps at 0.99; friction
   coefficients ride along only when calibrated (never materialized
   defaults); one batch stamp per save with the engineer comment joined on
   it; prop_hist stays append-only.
3. Attribution: `server/identity.py` binds X-Forwarded-Email per request
   into a ContextVar consumed by `resolve_entry_user`'s provider hook - the
   FastAPI equivalent of the Streamlit `set_entry_user_provider`
   registration. Without the header (local dev) it falls back to the SQL
   session's `current_user()`.

The root `app.yaml` sets ALLOW_DATABRICKS_WRITES=true; the app's service
principal has held MODIFY on mpu.wells.prop_hist since 2026-07-30 (same app
as the Streamlit era - no new grants). The .env local-write landmine
(AGENTS.md section 3) now applies to this app too: local runs with the .env
gate on write REAL rows. Contract tests: tests/test_web_save_ipr.py.
`sync_pad` (pad review) remains unported; `execute_write` is reachable only
through `push_prop`.

## v1 scope

Complete: app shell (collapsible auto-hide sidebar, '[' shortcut, overlay on
narrow screens, localStorage persistence, wide layout), well selection +
full 37-field parameter sidebar with as-built and saved-prop locks, Solver
(verdict, model-vs-actual, IPR chart with anchor modes + rate calculator,
tests table, WC-washout detection), Batch Run (sweep, performance chart,
recommender, CSV), PF Range, Pressure Profile, Well Profile, Pump
Equivalents, JP History (strip chart + installs), Well Database (chars,
aging pumps, prop history audit), Well Sort (Wells / Triage / Marginal WC
views, shared POPs config, bench xlsx + CSV exports; decision + marginal
math single-sourced in `woffl/assembly/well_sort_engine.py`, shared with
Streamlit).

Not ported yet (Streamlit remains the tool for these): pad optimization
(S/I/M/CFP), Scott's Tools, memory-gauge upload, manual test entry,
calibration/auto-match actions, pad-review sync writes, PDF export.

Layout convention (single-well analysis pages): the pump-history strip
renders just above the historical-tests table, never above the page's
primary chart - the top of the page stays chart-first while history loads.

## Deploy

Production deploys straight from the repo (same flow as the old Streamlit
app): the root `app.yaml` runs `uvicorn server.main:app`, and `web/dist` is
COMMITTED because Databricks Apps never runs npm.

1. Commit as usual. The pre-commit hook (`scripts/git-hooks/pre-commit`)
   rebuilds and stages `web/dist` automatically whenever web sources are in
   the commit. Fresh clones wire it once with
   `git config core.hooksPath scripts/git-hooks`. A `--no-verify` commit
   skips the rebuild and would deploy a stale UI.
2. Push, then pull the repo in the workspace Git folder.
3. App page -> Deploy (source path unchanged).

Rollback: copy `app-streamlit.yaml` over `app.yaml` (or `git revert` the
cutover commit), pull, Deploy.

For a throwaway test instance beside prod, `scripts/stage_web_app.py` stages
a minimal tree for `databricks sync` + `databricks apps deploy` under a
separate app name.

## Local dev

**Quick test before committing: double-click `run_local.bat`** (repo root)
or run `run_local` from a terminal there. It builds the SPA, serves the app
at http://127.0.0.1:8000 exactly as a deploy would, and opens the browser
(`run_local nobrowser` skips that). Ctrl+C stops it.

For frontend iteration with hot reload, see `web/README.md`. API:
`uvicorn server.main:app --reload --port 8000` (uses .env
`bricks_host`/`bricks_token` for SELECT-only reads). SPA: `npm run dev`
(proxies /api). Without Databricks creds the app serves the jp_chars.csv
fallback universe and bundled JP history.
