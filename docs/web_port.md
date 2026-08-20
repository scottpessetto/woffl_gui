# WOFFL web port (React + FastAPI)

Status: v1 shipped 2026-08; CUT OVER to production 2026-08-06 - the root
`app.yaml` now runs this app. The Streamlit config is preserved in
`app-streamlit.yaml` for rollback; the Streamlit GUI still runs locally for
the flows not yet ported.

## Why this architecture

The ask was "a real Node.js app": professional, faster, wide-screen, sidebar
that gets out of the way. The physics (`woffl/flow|pvt|geometry|assembly`,
bracket -> secant -> BHP re-seed -> bisection solve strategy) is
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
| GET /wells/{name}/depth | MD <-> TVD along the survey by minimum curvature (SPE 84246); exactly one of `md`/`tvd`, returns every MD crossing a TVD plus hole angle and DLS |
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
| POST /calibrate | BHP friction calibration: fit ken/kth/kdi to the test's measured BHP (Nelder-Mead multi-start in woffl.gui.fric_calibration; as-built geometry never varied; result applied to the sidebar, saved only via Save-as-default) |
| GET /optimize/pad-status | Optimization pad board: per-well saved-fit readiness (+ donor wells for planned future wells) |
| POST /optimize/run | start an S/I/M pad or CFP optimization run as a background job over saved fits (engines: woffl.gui.pad_optimize / cfp_moves - the Streamlit pages' own compute cores) |
| GET /optimize/run/{job_id} | poll a run job: status, live progress, result |
| GET /database/wells, /database/aging-pumps, /database/prop-history/{well} | Well Database page |
| GET /well-sort/tables | online / offline / LTSI tables + POPs config echo |
| GET /well-sort/events | 30-day shut-in events (down-day threshold walk) |
| GET /well-sort/marginal-wc | field marginal WC (cumulative-water walk) |
| GET /well-sort/pad-marginal-wc | per-POPs-pad marginal WC + pump headroom |
| GET /well-sort/triage | keep / SI / BOL decisions vs the marginal line |
| GET /well-sort/bench.xlsx | 3-sheet MPU_Well_Bench workbook |
| POST /well-sort/refresh | clears the Well Sort fetch caches (read-only op) |
| GET /meta/warmup | fleet cache warmup progress (passes, per-well counts, failures) |
| POST /gauge/parse | memory-gauge XLSX parse + multi-file combine (stateless; client holds gauge state per session, math in woffl.gui.memory_gauge) |
| GET /tools/sep-oil-loss | Separator Oil Loss: oil leaving with the first-stage water leg, as a bounded band (see below) |
| POST /tools/sep-oil-loss/samples | operator OIW grab-sample XLSX parse -> daily sampled loss overlay (stateless; page holds it, see below) |

Server caching mirrors the old `@st.cache_data` TTLs (`server/config.py`):
tests 24 h, chars/PF/profiles 1 h, saved IPR / prop history / historian 5 min.
Failures
are never cached. Beyond Streamlit, `server/cache.py` is stale-while-
revalidate: an expired entry serves instantly while a persistent background
thread refreshes it (single-flight per key, stale grace = one extra TTL),
so TTL expiry never lands on a user request - only the first-ever read per
process blocks. `clear()` bumps a version so a fetch that started before a
write's cache-clear can never store pre-write data (read-your-writes).
`server/warmup.py` owns ONE warm loop for the whole process, started from the
app's `lifespan`. Each pass warms (a) every fleet-wide frame - chars, PF,
jp-history, the 6- AND 12-month test windows, 365-day fleet pressure, 365-day
fleet PF volume, the saved-IPR snapshot, prop-write metadata, and Well Sort's
own five pulls (`well_sort.warm_targets`) - then (b) EVERY well, via
`history.warm_well` (the app's only genuinely per-well warehouse queries) plus
its deviation-survey CSV. Two things make it matter: nothing per-well was warmed
before, so the first engineer to open a well paid its two queries; and a
one-shot warm decays, because `ttl_cache` deletes an entry past 2 x TTL and the
next reader then blocks - a server up for days was cold again for the 1 h tier.

Every target is a `cache.refresher` - `fn.cache_refresh(...)`, a FORCED
re-query that overwrites the entry. A plain call returns a fresh entry and
queries nothing, so the old loop's real refresh clock was the TTL, not the
interval, and the interval had to sit under the shortest TTL it protected.
`cache_refresh` stores with a retention floor (`cache.set_warm_retention`,
= 2 x the interval), so a warmed entry stays SERVABLE across passes even if a
pass fails outright; reads past the TTL still get the stale value plus a
background SWR refresh, they just never get a blocking cold query. That
decoupling is what lets the cadence be `WOFFL_WARM_INTERVAL_SEC` = 21600
(6 h) by default (`0` = one pass, no loop), on `WOFFL_WARM_WORKERS` threads
(default 3 - each holds one THREAD-LOCAL warehouse connection, so this is a
warehouse-concurrency knob, not a CPU one). `WOFFL_WARM_WELLS=0` skips the
per-well pass for local `uvicorn --reload`. Failures are logged and counted,
never raised. Two exceptions to the "everything is forced" rule, both
deliberate: `_xv_status` (5 min live safety-valve state - a 6 h-old reading
would be worse than none) and `prop_hist_client`'s write metadata (its own 1 h
TTL dict, paid inline by the first save, never on a read path).

The loop never sleeps past local midnight (plus 2 min): `extended_tests` /
`bhp_daily` keys carry today's date, so after the roll every per-well entry is
a key nobody has filled and no retention floor can help. Per-well cache
`maxsize` is sized for two days x the fleet for the same reason.
`GET /meta/warmup` reports `interval_sec` and `retention_sec` alongside the
pass counters.

Static assets: /assets/* are content-hashed and served
`immutable` (1 y); index.html is `no-cache` so redeploys pick up new asset
hashes immediately. API JSON is gzipped. On Databricks Apps this matters
doubly: every request rides the Apps proxy, and warehouse queries carry
0.5-1 s overhead - the SWR + immutable-asset combination keeps both off
the interactive path.

Client side: TanStack Query defaults to a 60 s staleTime floor
(`web/src/main.tsx`); heavy immutable reads pin longer windows per hook,
snapshot-keyed sweeps (Batch) hold for the whole session, and background
job pollers set `refetchIntervalInBackground: true` so a run keeps
updating while the engineer alt-tabs.

### Separator Oil Loss (Scott's Tools)

`GET /api/tools/sep-oil-loss` reads three SCADA tags off the first-stage
separator through `woffl/assembly/historian_client.py` and integrates the oil
leaving with the water leg. The engine and the full derivation live in
`server/services/tools/sep_oil_loss.py`; the parts you will otherwise get
wrong:

- `reporting.historian.vw_mpu_measurements` is **exception reported**, not
  fixed interval. Every average must be time weighted and every cross-tag
  merge must be an as-of step-hold. A plain `mean()` over-weights whichever
  tag was moving fastest.
- The Red Eye analyzer (`MPU_AI_5317`) **films over** and stops reaching 100%.
  Readings are therefore referenced against the analyzer's own trailing 24 h
  p95 plateau, and only departures below that plateau are charged. On the
  validation window the film was 27,333 bbl of a 138,807 bbl raw integral.
- Validity gates on **flow** (`MPU_FI_5365` > 1,000 BPD), never on the water
  cut. A deep water-cut drop is real oil carry-under: the analyzer sweeps
  continuously through 90/60/30/5/0 over minutes. Filtering on the analyzer
  value would delete exactly the events the tool exists to find.
- The level channel is `MPU_LIC_5365CV1`, the **controlled** indication the
  loop acts on, with `MPU_LC5365SP1` as its setpoint. It is NOT `MPU_LI_5365A`
  - that transmitter sits a mean 14.53 points from setpoint (corr 0.35) while
  the LIC sits 3.98 (corr 0.71). The choice decides the diagnosis: on
  LI_5365A two thirds of upsets looked like lost level, and on the controlled
  channel 63% of the barrels go out with the level held AT setpoint, which
  makes them a separation problem rather than a level-control one.
- The meter-implied rate can exceed total field oil (Milne sells
  50,000-65,000 BOPD), so the answer is always a **band**: `bbl_upper` is the
  analyzer as read with the instantaneous rate capped at field production,
  `bbl_lower` caps the oil fraction of the leg at `max_oil_frac`, which
  defaults to 0.10 - a leg running more than a tenth oil is already an upset,
  and a looser cap stops being a floor and just tracks the as-read meter. Both
  carry a percent-of-field column so an implausible number announces itself.
  Never quote one end of the band alone.
- `periods` are ROLLING look-backs from the last sample; `daily` is one row
  per **Alaska calendar** day the window touches, so night-shift upsets land
  on the day the crew would name. The two do not sum to each other and are
  not meant to: the daily bars cover clipped end days the rolling cut drops.
  A day flagged `partial` is clipped by the window or cut by downtime, and
  `pct_field_*` is withheld below 1 h of runtime because the denominator goes
  to zero and turns a handful of barrels into a headline percentage.

`POST /api/tools/sep-oil-loss/samples` takes an upload of the operators' CFP
grab-sample workbook and returns the SAMPLED loss per Alaska calendar day, to
overlay on those same daily bars (`server/services/tools/oiw_samples.py`).
Stateless, exactly like `POST /gauge/parse`: it parses and returns, the page
holds the result, nothing is stored on disk or in Databricks. What matters
here:

- `oil_bopd = ppm x water_rate_bpd / 1e6`, with `water_rate_bpd` a REQUEST
  parameter echoed in the response. The workbook's own `(BOPD)` column is a
  hardcoded 95,000 BWPD basis, blank on most recent rows, and is never read.
- A day is the plain UNWEIGHTED mean of its samples' rates. These are a
  handful of irregular manual grabs; time weighting them would invent a duty
  cycle the log does not carry.
- The log is hand kept: blank rows, text in numeric cells and a stray year
  2107 date all live in it. A row survives only with a parseable, plausible
  date AND a finite positive ppm; the count dropped comes back in `notes`.
  Case variants of a tap ("P-5417C" / "p-5417c") are one location; anything
  further apart is left alone.
- **The two series are not the same stream.** `P-5417C`, the only tap still
  being sampled, sits DOWNSTREAM of the deoilers (V-5419 / V-5421 / V-5422 /
  V-5425) while the calculated band is the first-stage water leg upstream of
  them. Its 1,000-2,500 ppm baseline on 71,000 BPD is ~106 BOPD, far below the
  band's lower bound, and that gap is deoiler RECOVERY, not measurement error.
  `V-5317` is the one location that samples the band's own stream. Every
  response carries that caveat in `notes` and the page renders it under the
  chart; do not present the sampled marks as a check on the band.

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
(verdict, model-vs-actual with Auto-match BHP friction calibration +
coefficient explainer, IPR chart with anchor modes; the fitted seeds
auto-apply once per well on load - the web equivalent of Streamlit's
open-time anchor sync - so the curve and the solve agree from the first
paint, EXCEPT over locked fields, over a reviewed save (`ipr_source` =
"saved"), and over anything the engineer set by hand or applied from a
sensitivity permutation, which the store tracks as `manualFields` and the
IPR card offers to release; a Total liquid / Oil only toggle on the chart
swaps the x axis between total fluid and oil - the curve scales by the
sidebar water cut, the tests plot their MEASURED oil and the model point
plots the solved `qoil_std`, and both rates stay in every tooltip; rate
calculator sits directly under the IPR chart, comparison + anchor controls
in the right column; tests table, WC-washout detection), Batch Run (sweep,
performance chart, recommender, CSV; the submitted snapshot lives in a
session store so results survive page detours without re-running, and the
marginal-WC cutoff is editable on the run card), PF Range, Pressure
Profile, Well Profile, Pump Equivalents, JP History (strip chart +
installs), Well Database (chars, aging pumps, prop history audit), Well
Sort (Wells / Triage / Marginal WC views, shared POPs config, bench xlsx +
CSV exports; decision + marginal math single-sourced in
`woffl/assembly/well_sort_engine.py`, shared with Streamlit),
memory-gauge upload (Solver IPR card: gauge daily medians
override test BHP for the chart, tests table, pump-history strip and the
server-side Vogel fit via `bhp_overrides`; session-only like Streamlit;
the divergence-based "disregard Databricks BHP" flag is NOT ported yet).

Optimization pad board (/optimize - REDESIGNED from the Streamlit pad
pages): engineers match + save fits on the Single Well solver; the board
shows per-pad readiness (saved IPR date/author, calibrated friction,
missing parts), per-well offline flags, and planned FUTURE wells that
match an existing well's saved fit (donor may be on any pad). Well names
link to the Single Well solver (same selection flow as the sidebar
picker), and each S/I/M run tab carries the same readiness board scoped
to its pad, so check-offline -> run happens without leaving the tab.
Offline + future config persists in localStorage per browser; fit status
is live from prop_hist.

The board also reads the field's daily downtime log (`/well-sort/tables`,
the same cached fetch the Well Sort page uses) and badges every well that is
currently down with its shut-in date and down code. Only LONG-TERM shut-in
(T-coded: T01 mech, T02 reservoir, T03 convert, T05 P&A) pre-ticks offline -
those wells have no business in a pad plan. An ordinary short-term shut-in is
advisory only, because the log can lag a restart by a day and because on a
bad day most of a pad is logged down; auto-excluding all of it would empty
the run. Unticking an auto-ticked well is stored as an explicit `keepOnline`
entry so the override survives reloads instead of being re-applied every
visit, and the auto-tick itself is derived, never written. Test recency is
deliberately NOT a source: the repo already computes a 60-day `StaleTest` and
already reads it as "no representative test, do not judge" (Triage's
`verify_stale`), not as "well is down" - a producing well with an overdue
test would be silently dropped from the plan, and a well shut in last week
with a test from the week before would not be flagged at all. If
`/well-sort/tables` is unavailable the board degrades to manual ticks alone.

Optimization runs (S/I/M pad + CFP) execute server-side as background jobs
over the board's config: offline wells excluded, future wells modeled on
their donor's saved fit, per-run constraint knobs mirroring the Streamlit
Configure stages. CFP runs take a pad subset (cfp_pads chips, default the
four CFP pads B/G/C/J). Any non-POPs pad in the universe (L, R, ...) may
join: its produced water rides the CFP machines; PF for pads beyond B/G/J
is modeled as boosted on-pad at the C-Pad booster knob, and the run notes
say so. POPs pads separate water on-pad and are rejected with the reason
(422). Wells whose saved fit violates the physics invariants (pwf >= ResP,
no rate) are skipped with a note instead of killing the run.
Engines are imported unchanged (MILP/MCKP allocation
with plant coupling for S/I/M; anchored-delta moves with the equal-slope
frontier for CFP). CFP results render three charts: the modeled-oil-vs-
discharge efficiency frontier (today + plan markers, 2,900 psi trip line),
a today-to-plan waterfall bridge decomposed by action with the plant
pressure-feedback residual shown honestly as its own bar, and a per-well
today-vs-plan dumbbell sorted by delta (solid dot = today, hollow ring =
plan). Per-well chart rows are read off the same response surfaces at the
settled pressures, so they sum exactly to the fleet totals. "Modeled oil"
is deliberately labeled so (not "fleet oil"): the modeled total across the
run's wells, the optimizer's baseline - not field production. The shut-in /
bring-online ladder prices every on/off move per well (best pump option
only): own oil cost, PW freed/added (server-enriched own_water_delta on
every single move), net fleet oil after pressure feedback, and whether the
move made the best plan.

Every pad-run row carries where its inflow curve came from, because a pump
recommendation is only as good as the IPR it was chosen against and the run
table otherwise presents all of them with equal confidence. The Fit column
reads `saved` (an engineer-reviewed IPR from prop_hist), `auto R2 <n>` (an
automatic Vogel fit over recent tests, amber under 0.5 and crimson at or
below 0 where the curve tracks worse than a flat line), `1 test`, or
`defaults` - the last meaning the well had no usable tests and was modeled on
qwf 750 / pwf 500 / ResP 1700, which is why several such wells return the
same predicted oil. `nofric` marks a well running library friction
coefficients rather than a saved BHP calibration. The provenance rides
`well_context` (`ipr_source` / `ipr_r2`, set where the seeds are), is
collected per well by `_build_configs` and lands on each row - nothing is
recomputed for it.

Pad runs (S/I/M) render the booster plant as an industry pump-curve sheet,
served static and cached from `GET /api/optimize/pump-curve?pad=&n_pumps=`
(`PadPlant.curve_report`, `server/services/pad_curves.py`), so the plant is
on screen while the engineer configures the run and the duty point drops
onto it when the run lands. Panel 1 is the station curve: the delivered
header vs total PF family (1/2/3-pump curves for the fixed-speed S station,
iso-speed lines for the I and M VFD trains), the amp-limited capability
frontier on I-Pad, the 3,500 psi discharge cap and the recirc/min-continuous
flow line, BEP, and the preferred (70-120 percent of BEP) and vendor
allowable operating bands. POR is NOT always inside AOR - Summit's I-Pad
range is 80-120 percent, so the preferred band hangs out to its left; the
bands are drawn as reported, never clipped to look tidy. Panel 2 is the
vendor curve sheet per machine (two for I-Pad's series train): head, BHP and
efficiency vs flow per pump on three axes, with M-Pad's as-new head shown
against its 0.91 field-derated curve. Efficiency is `Q_gpm * H_ft * SG /
(3960 * BHP)` at the SG the BHP curve was FIT at - 1.0 for S and I, the
Schlumberger design 1.05 for M, which is the difference between reproducing
the datasheet's 78.2 percent and being two points low. Panel 3 is the
optimizer's own trace: oil and total PF vs header pressure across the sweep
for the free-pressure pads, or the header fixed-point convergence for
S-Pad's curve coupling.

Not ported yet (Streamlit remains the tool for these): the fixed-pump /
existing-baseline scenario comparators and Base-vs-Future, the CFP
dashboard (tradeoff verdict + match-check gate),
Scott's Tools, manual test entry, the joint oil+PF auto-match, pad-review
sync writes, the gauge "disregard Databricks BHP" flag, PDF export.

Layout convention (single-well analysis pages): the pump-history strip
renders just above the historical-tests table, never above the page's
primary chart - the top of the page stays chart-first while history loads.

Layout convention (optimization run tabs): the page runs edge to edge (no
max-width, no gutter of its own - Layout's `main` already insets), pump
curves and that pad's readiness board split it 50/50, and metrics plus the
plan table sit full width underneath. The curve panels pair up on container
queries (`@4xl`), not viewport ones, so they respond to the half they are
actually given rather than the width of the window.

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
