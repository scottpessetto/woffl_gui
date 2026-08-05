# WOFFL web frontend

React 19 + TypeScript SPA (Vite, Tailwind v4, TanStack Query, Zustand, ECharts).
The backend is `server/` (FastAPI) which imports the woffl physics unchanged.

## Develop

Two terminals from the repo root:

```bash
# 1. API on :8000 (needs bricks_host/bricks_token in .env for live Databricks reads)
./venv/Scripts/python.exe -m uvicorn server.main:app --reload --port 8000

# 2. SPA on :5173, /api proxied to :8000
cd web
npm install
npm run dev
```

## Build

```bash
cd web
npm run build        # typechecks (tsc -b) then emits web/dist
```

`web/dist` is served by FastAPI in production (same origin, no CORS).

## Conventions

- API types in `src/api/types.ts` mirror `server/schemas.py` field-for-field.
  Change them together or not at all.
- All fetching goes through `src/api/hooks.ts` (TanStack Query). No raw fetch
  in components.
- Simulation inputs live in one Zustand store (`src/state/params.ts`). Well
  selection triggers a server-side seeding replay (`GET /wells/{name}/context`)
  and `applyContext` lays the seeds over defaults exactly once per selection.
- Vogel math in `src/lib/vogel.ts` is a line-for-line mirror of
  `woffl/gui/vogel.py` so IPR curves redraw client-side with zero latency.
  Keep them in lockstep.
- Charts: import from `src/charts/echarts.ts` (tree-shaken registry), mount
  with `useEChart(option)`, style with helpers from `src/charts/theme.ts`.
- Keyboard-typable characters only (no em dashes, curly quotes, ellipsis).
