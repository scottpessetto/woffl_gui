# WOFFL web frontend

React 19 + TypeScript SPA (Vite, Tailwind v4, TanStack Query, Zustand, ECharts).
The backend is `server/` (FastAPI), importing this repository's vendored physics.
Current behavior and verification are in the [session handoff](../docs/session_learnings_2026-09-08.md)
and [documentation index](../docs/README.md).

The sidebar's return-hydraulics selector offers BB + Payne, Hagedorn–Brown +
Griffith and Shi/Pan drift-flux. Tulsa remains unavailable. Model changes reset
pump coefficients; completed installed-pump calibration saves carry the model
into optimization. See [the implementation and comparison](../docs/hydraulics_models_2026-09-11.md).

## Develop

Two terminals from the repo root:

```bash
# 1. API on :8000 (needs bricks_host/bricks_token in .env for live Databricks reads)
./venv/Scripts/python.exe -m uvicorn server.main:app --reload --port 8000

# 2. SPA on :5173, /api proxied to :8000
cd web
npm ci
npm run dev
```

## Build

```bash
cd web
npm run build        # typechecks (tsc -b) then emits web/dist
node --test tests/*.test.mjs  # polling + pump-scope store contracts
```

`web/dist` is served by FastAPI in production (same origin, no CORS), and is
committed for Databricks deployment. The latest recorded baseline is 18 frontend
tests and a successful production build on September 12; that does not deploy it.

## Solver state and calibration

- **Show model match** in Solver's pump history and JP History defaults to
  **Every test (well fit)**: one fixed oil IPR, each test's measured WC/GOR,
  historical pump and recorded pressures at every usable test. **Well inputs**
  selects saved database values or a preview of supported sidebar edits. Optional chronological modes
  fit earlier tests within a pump or across changes. It overlays BHP/oil,
  adds PF detail, and separates historical-model scores from held-out scores.
  Missing inputs and solver failures retain explanations. Pump losses stay at
  clean reference; see [the workflow and limits](../docs/pump_match_ui_2026-09-12.md).
- **Save well inputs** and **Save installed-pump calibration** are separate.
  Well Save stays visible at the top of Solver and JP History, with a disabled
  state and explanation for read-only access or invalid inputs. Preview never
  writes. New optimization runs use saved inputs; previous results are snapshots.
  Well saves exclude pump losses/area. Pump saves submit a completed server job
  ID; installation/model identity and quality are verified on the server.
- `params.ts` owns installed/replacement state. Size edits and explicit clean
  selection reset all four pump coefficients to reference values. Same-size
  replacement is distinct from keeping installed hardware. Context refreshes
  update saved baselines and invalidate old installation scope while preserving
  unrelated well edits, including edits made during an in-flight save.
- Event calibration uses saved well inputs; the UI tells the user to save
  edits before refitting. Apply previews a fit; optimization uses the saved fit.
- WC uncertainty uses the current effective params, debounces 400 ms and caches
  full requests for 30 minutes. Immediately hide stale output on input changes;
  never use previous results as the new bounds. Report failed samples. It is a
  sensitivity panel, not a save/fit/optimizer action or confidence interval.
- Preserve provisional/railed/response diagnostics. Keep the removed global
  yellow physics-transition banner out of Topbar.

Optional browser harnesses include `tools/check_wc_uncertainty_ui.py`,
`tools/check_pump_scope_ui.py`, `tools/check_pump_match_ui.py` and
`tools/check_well_input_save_ui.py`, using isolated Playwright in `build/browser-qa`.
They use fixtures and intercept saves; no production writes are needed. Do not
run `npm run build` during Vite browser QA because reload can reset the scenario.
When inspecting Zustand under HMR, import the actual loaded module URL (including
its timestamp) to avoid creating a second store. See the harness source for setup.

## Conventions

- API types in `src/api/types.ts` mirror `server/schemas.py` field-for-field.
  Change them together or not at all.
- All fetching goes through `src/api/hooks.ts` (TanStack Query). No raw fetch
  in components.
- Simulation inputs live in one Zustand store (`src/state/params.ts`). Well
  selection triggers a server-side seeding replay (`GET /wells/{name}/context`)
  and `applyContext` lays the seeds over defaults exactly once per selection.
- Vogel math in `src/lib/vogel.ts` follows `woffl/flow/inflow.py`, so IPR curves
  redraw client-side without a solve request. Keep the math and total-liquid/oil
  conversion consistent; there is no current `woffl/gui/vogel.py`.
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
