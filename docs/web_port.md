# WOFFL web port (React + FastAPI)

Status: v1 shipped 2026-08. Owner docs for the Node.js-era WOFFL app that
replaces the Streamlit GUI page by page. The Streamlit app remains deployed
from the root `app.yaml` until cutover; both apps share this repo.

## Why this architecture

The ask was "a real Node.js app": professional, faster, wide-screen, sidebar
that gets out of the way. The physics (`woffl/flow|pvt|geometry|assembly`,
1686 tests, bracket -> secant -> BHP re-seed -> bisection solve strategy) is
Python and does not move. So:

- **Frontend = the Node.js part.** TypeScript + React 19 SPA built with Vite;
  Tailwind v4; TanStack Query for server state; Zustand for sim inputs;
  ECharts (canvas) for charts. Fully client-rendered: interactions never
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
| GET /pumps/equivalents | cross-brand pump match |
| GET /database/wells, /database/aging-pumps, /database/prop-history/{well} | Well Database page |

Server caching mirrors the old `@st.cache_data` TTLs (`server/config.py`):
tests 24 h, chars/PF/profiles 1 h, saved IPR / prop history 5 min. Failures
are never cached. Startup warms chars/PF/jp-history/tests in daemon threads.

## Write safety (v1 = read-only)

The server has **no write endpoints**. It never calls `execute_write` /
`push_prop` / `sync_pad`. `app-web.yaml` deliberately omits
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
aging pumps, prop history audit).

Not ported yet (Streamlit remains the tool for these): pad optimization
(S/I/M/CFP), Well Sort, Scott's Tools, memory-gauge upload, manual test
entry, calibration/auto-match actions, IPR pin/save writes, PDF export.

## Deploy

```bash
python scripts/stage_web_app.py          # builds SPA + stages build/webapp_stage
databricks sync ./build/webapp_stage /Workspace/Users/<you>/woffl-web --full
databricks apps deploy woffl-web --source-code-path /Workspace/Users/<you>/woffl-web
```

Staging exists because Databricks Apps requires `app.yaml` at the source
root and the repo root's `app.yaml` belongs to the production Streamlit app.
`app-web.yaml` is copied to `app.yaml` inside the stage. Cutover later =
replace the root app.yaml command and add web/dist to the main deploy.

## Local dev

See `web/README.md`. API: `uvicorn server.main:app --reload --port 8000`
(uses .env `bricks_host`/`bricks_token` for SELECT-only reads). SPA:
`npm run dev` (proxies /api). Without Databricks creds the app serves the
jp_chars.csv fallback universe and bundled JP history.
