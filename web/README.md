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
- Server-state caching (the snappiness contract - don't regress it):
  `main.tsx` sets a 60 s default `staleTime`; expensive stable reads pin
  their own windows (`MIN_30`, or `Infinity` + `gcTime` for snapshot-keyed
  sweeps like Batch, where identical inputs give identical physics).
  Background JOB pollers must set `refetchIntervalInBackground: true` -
  TanStack pauses interval refetches in unfocused windows by default and
  a run monitor that freezes when the engineer alt-tabs is a bug. Writes
  invalidate exactly the queries they change (see `api/hooks.ts`
  `invalidateSavedIpr`). Server-side TTLs + stale-while-revalidate live in
  `server/cache.py`; the browser never needs to compensate for them.
- Charts: follow "The chart rule" below. No exceptions, including one-offs.
- Keyboard-typable characters only (no em dashes, curly quotes, ellipsis).

## The chart rule

Every chart in this app is built the same way. A chart built any other way
is a regression, even if it happens to render.

1. **Mount through `src/charts/ChartPanel.tsx`** - never a raw div + hook.
   ChartPanel owns the interaction contract on every chart:
   - drag = box zoom (axes declared per chart via the `zoom` prop)
   - ctrl + wheel = zoom at the cursor; shift + wheel = pan along x
   - double-click or the corner button = reset; second button = fullscreen
   - plain wheel is never captured - the page must keep scrolling
2. **SVG renderer only** (`src/charts/useEChart.ts` inits with
   `renderer: "svg"`). Canvas text blurs at Windows fractional display
   scaling (125/150% -> devicePixelRatio 1.25/1.5); SVG is vector-crisp at
   any DPI and any browser zoom - the "plotly look". Our data volumes
   (<= a few thousand points per chart) are well inside SVG's range. Do not
   reintroduce `CanvasRenderer` for a single chart.
3. **Register modules in `src/charts/echarts.ts`** (tree-shaken registry)
   and import `echarts` only from there.
4. **Style with `src/charts/theme.ts`**: `houseOption`, `axis()`, the house
   palettes, `baseTooltip`. Axis names carry units.
5. **Tooltips never show raw datums.** Use `axisTooltip({unit, ...})` for
   single-quantity axis tooltips; compose `ttHeader` + `ttRow` (+
   `nearestByX` for mixed-frequency time series) for bespoke ones; item
   tooltips get explicit formatters. The ECharts default leaks epoch-ms
   values from custom/band series and silently drops series whose x grid
   does not match the snapped axis value.
6. **No custom-series `renderItem` for anything tied to a zoomable axis.**
   Custom series do not re-render on dataZoom when `filterMode: "none"`
   leaves their data untouched, so their pixels drift from the axes (the
   old "bands misalign after zoom" bug). Use `markArea`/`markLine` on an
   unnamed, silent carrier series instead - see `HistoryStrip.tsx`.
7. **Multi-grid time charts**: list every x axis in `zoom.xAxisIndex`
   (e.g. `[0, 1]`). ECharts links dataZoom components that share an axis,
   which keeps the grids window-synced natively - no pixel mirroring.
