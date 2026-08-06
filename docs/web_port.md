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
| GET /wells/{name}/ipr-pin | saved anchor pin (read-only) |
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

## Write safety (v1 = read-only)

The server has **no write endpoints**. It never calls `execute_write` /
`push_prop` / `sync_pad`. The root `app.yaml` deliberately omits
ALLOW_DATABRICKS_WRITES; `/api/meta.writes_enabled` is display-only and the
UI shows a read-only badge. The .env local-write landmine documented in
AGENTS.md section 3 therefore cannot fire through this app. When write flows
(IPR pin/save, pad review sync) are ported, they must replicate the full gate
chain and entry-user attribution.

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
calibration/auto-match actions, IPR pin/save writes, PDF export.

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
