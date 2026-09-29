# Header page: handoff (2026-09-29)

This is the current state of the production-header impact page, written for
the next session. It is built and tested locally only: not committed and not
deployed. The [delivery log](header_impact_delivery_2026-09-29.md) keeps the
chronological record of the five rounds, with measurements. Parts of it
(first-round numbers, the J/q reservoir groups) are superseded by this
document. The [first plan](header_impact_plan_2026-09-29.md) is design
history.

"Header" always means the **production** header (wellhead back-pressure).
It is never the power-fluid header the pad optimizer sweeps.

## What the user asked for, and decided

- **Priority 1:** "The header jumped (or will jump) X psi. What is the oil
  impact?" One step, per pad, with a range.
- **Priority 2:** every well has a saved WHP→BHP relation or a correlation,
  and a saved or assumed IPR. Engineers review and adjust wells as time
  allows; saves become everyone's defaults.
- **Decisions to preserve:**
  - **Page layout:** its own page (`/header`), not an Optimize tab. Two tabs:
    **Impact** and **Wells**. The Wells tab loads in the background when pads
    are picked; there is no Load button.
  - **Gaugeless ESPs:** they borrow measured relations from gauged wells with
    good WHP, assignable by lift type (+ reservoir). A 1:1 ESP assumption was
    rejected.
  - **Reservoir pressure:** every well has its own ResP — the latest
    `resvr_press` saved in prop_hist, else the documented default. Defaults
    are **Schrader 1,800** and **Kuparuk 3,000** (both confirmed by the user).
  - **Reservoir pressure is never set by:** groups, gauge-test fits, or
    shut-in gauges (these are shown as evidence only).
  - **R-Pad:** Schrader, header tag `MPU_PI_4661`, no jet pumps today.
  - **L-Pad jet pumps:** use live PF.
  - **ESP pump-curve data:** not yet; the user is "not ready to feed in more
    pumps".
  - **Event cause-finding:** low priority. POP suggestions from the downtime
    log are enough.
  - **UI wording:**
    - The gauge control reads "Bad gauge" (tick if bad).
    - "Default" names what it resolves to.
    - Menus never list the same number twice.
- **Field facts learned:**
  - MPL-20's "BHP" tag reads PF pressure at the pump entry (a dead gauge). Its
    PF reached the header on 2026-09-28 at about 12:00; the downtime log calls
    it up on 09-29.
  - The 2026-09-20 +28 psi step is pigging (FLR → M → C). Pressure stays high
    for days, then trends down.

## Model

For a well that is not a jet pump:

```
dWHP = r × dHeader       r = within-day WHP~Header slope (1.0 if not measured)
dBHP = s × dWHP          s = closed-loop within-day BHP~WHP slope
dLiq = Vogel(BHP + dBHP) − Vogel(BHP)
dOil = dLiq × (1 − WC)   WC from the latest test; oil derived once
```

- **The slope is closed-loop.** `s` is measured while the well flows, so it
  already includes the rate falling as BHP rises. Never couple it to the IPR
  again.
- **ESPs:** at fixed speed `s ≈ 1/(1 + PI·k)`. High-rate ESPs barely move BHP
  yet lose the most liquid. Schrader ESPs read about 0.05–0.27; Kuparuk about
  0.8.
- **Jet pumps:** hydrated exactly like an optimization run
  (`optimizer_runs._build_configs`: saved IPR, scoped pump fit, hydraulics).
  Each is solved at the model WHP and at WHP + dWHP with PF held
  (`pad_optimize._model_at_forced_header`). A sonic pump gives about 0.
  - If the pump model fails, the well falls back to its own measured
    relation on the pump model's IPR.
  - If there is no measured relation either, it gets no estimate.

**Relation ladder ("Default"):**
1. The saved relation.
2. The well's own measured slope. This needs a working gauge, and at least 5
   days of r² ≥ 0.5 that make up at least 25% of the days the WHP moved. No
   slope band is applied.
3. The correlation for its lift type + reservoir at its test rate. A group is
   formed once it has 4 measured wells: Theil-Sen `s = a + b ln q`, or the
   median with fewer than 5 wells or less than 2× rate spread. The lift-only
   group is the fallback.
4. None.

**IPR ladder ("Default"):**
1. The saved IPR.
2. The well's own gauge data (user, 2026-09-29: "default to the gauge data if
   it exists"). This is a **usable** fit of its gauged tests: at least 4
   tests, 100 psi of BHP spread, not pinned at the cap, and under 25% rate
   scatter. The fit's pseudo-ResP is used.
3. The well's own ResP (saved, else default) at its latest test rate and BHP:
   - with a working gauge, BHP is the gauge's 72 h median;
   - otherwise BHP = ResP × the median BHP/ResP of gauged, flowing wells in
     the same pad + reservoir (at least 3 wells), else the reservoir-wide
     median, else 0.35.
4. None, if ResP − BHP is under 300 psi. Such wells show in coverage; they
   are never counted as zero.

A **flagged** gauge fit (pinned at the cap, or too little BHP spread, which
is most of R-Pad) is offered as a choice but never used by default. Saving
any fit sets that well's ResP.

**Range (low/high):**
- **Slope:** the measured IQR, or the correlation's ± residual MAD.
- **ResP:** a default ResP gets ±20%. A saved ResP gets none.
- **Totals:** every well at the same end at once. This is an envelope, not a
  confidence interval.

**Gauge check:** a saved verdict (`hdr_gauge_bad`) wins; without one, the
automatic check decides. A session tick overrides either until it is saved.
The automatic check flags:
- BHP now above `GAUGE_MAX` (Schrader 3,200, Kuparuk 4,500). This catches
  MPL-20's PF reading and MPR-110's stuck 4,188.
- Or under 1 psi of movement in 72 h (MPF-62).
- A pump intake below WHP is not a fault.
- A gauge far above the latest test (>300 psi and >1.5×) means the well
  "looks shut in", and it defaults to offline.

**Online default:** the latest test is at most 45 days old and the well
doesn't look shut in.

**Firm vs conditional** (user decision, 2026-09-29): a well is **firm** when
it rests on its own data or on an engineer's saved review
(`header_model.is_firm`, mirrored in `model.ts`).
- **Relation:** the pump model, its own measured slope, or **any saved**
  relation.
- **IPR:** the jet pump's Solver IPR, **any saved** IPR, a **usable** fit of
  its own gauged tests, or its own **saved** ResP.
- **Conditional:** unsaved correlations, default ResP (1,800 / 3,000),
  flagged gauge fits, and manual values.

"Conditional" therefore means "not reviewed yet", and it clears as wells are
saved on the Wells tab. The same rule drives the amber colouring and the
"needs review" filter.

**Status chip:**
- **Firm:** every online well is firm.
- **Conditional:** every online well has an estimate, but some are unreviewed.
  The note lists them by reason.
- **Incomplete:** some online well has no estimate.

**Event mode:**
- Header change = the median of hourly header pressure over a window before
  vs after the event time (defaults: 72 h before, 24 h after, skipping 6 h
  either side).
- Validation compares predicted with measured dBHP on gauged wells. A gauged
  well whose error exceeds max(15 psi, 3× dHeader, 3× prediction) is flagged
  as having its own event in the window, and excluded.
- POP suggestions come from `vw_shut_in`: a fully-down day followed by a
  partial day is "came on"; an up day followed by a down day is "went down".

## prop_hist

Seven ids were added to `mpu.wells.prop_xref` on 2026-09-29 (authorized;
gated `execute_write`, gate set in-process only; `hdr_gauge_bad` came later
the same day):

| prop_id | Value |
|---|---|
| `hdr_bhp_whp_slope` | closed-loop dBHP/dWHP |
| `hdr_whp_hdr_slope` | dWHP/dHeader |
| `hdr_fit_r2` | mean day r² (0 if not a gauge fit) |
| `hdr_fit_days` | fit days (0 if not a gauge fit) |
| `hdr_rel_source` | 1 measured, 2 correlation, 3 manual |
| `hdr_ipr_source` | 1 gauge fit, 2 "assumed" = the well-ResP option, 3 manual |
| `hdr_gauge_bad` | engineer's gauge verdict: 1 bad, 0 good |

- **Where the IPR goes:** the existing `ipr_qwf_liq`, `ipr_pwf` and
  `resvr_press` (one IPR per well).
- **How Save writes:** one `push_props` per well. Values come from the
  board/run job on the server, resolved by the same `effective()` as the run.
  Only manual numbers come from the client.
- **What Save refuses:** jet-pump IPRs (those stay in Solver) and a weak
  measured relation.
- **First live saves, 2026-09-29 21:09 and 21:11, by the user, read back.**
  - MPF-01: measured 0.620; gauge-fit IPR, ResP 1,365.
  - MPF-05: measured 0.493; manual IPR, ResP 1,500.

  Each is 9 rows in one statement with one timestamp and the user stamp.
- **The gauge verdict is saved** (`hdr_gauge_bad`). The latest row wins and
  overrides the automatic check for every user until someone saves the other
  value. Unsaved ticks are session choices. A gauge-only save is allowed,
  jet pumps included. "Online" remains a session choice.
- **Saved values on F/L/R today:**
  - MPL-06: full IPR, ResP 2,597 (user, 09-16).
  - MPF-73, MPF-107, MPL-20, MPR-111: `resvr_press` 1,800 only, from the
    ka9612 bulk load on 04-16. For the Kuparuk ones this is suspicious; L-20's
    buildup reached about 3,490.
  - Every ESP: nothing saved.

## Code map

**Server:**

| File | Contents |
|---|---|
| `server/services/header_model.py` | Pure math. Relation summary; correlation fit and prediction; Vogel rate and PI; pseudo-Pr fit wrapper; `nonjp_delta`; `range_delta`; `gauge_problem`; event windows and medians; `run_status`; defaults and caps. |
| `server/services/header_study.py` | The board: overview, tests, historian, saved props; per-well rows with `pres_well` / `pres_basis`; correlations; BHP-ratio groups; `_attach_options`. `effective()` resolves choices for run and save. Also `run_impact` (with range, curve and validation), `_jp_solve`, `_pop_candidates`, `well_detail`, `save`, and the jobs. |
| `server/routers/header.py` | `GET /api/header/pads`; `POST /board` and `POST /run` (jobs); `GET/DELETE /job/{id}`; `GET /well/{well}?pads=&fit_days=` (reuses the board's cached historian pull); `POST /save` (403 unless `ALLOW_DATABRICKS_WRITES`). |
| `server/schemas.py` (end of file) | `HeaderBoardRequest`, `HeaderWellChoice` (online, gauge_bad, relation, corr_group, slope, ipr, ipr_group, manual qwf/pwf/pres), `HeaderRunRequest`, `HeaderSaveWell`, `HeaderSaveRequest`, `HeaderJobStatus`. |
| `server/services/tools/header_trend.py` | R header tag `MPU_PI_4661` added. |
| `server/main.py` | Router registered. |

**Web:**

| File | Contents |
|---|---|
| `web/src/pages/HeaderPage.tsx` | Tabs. Impact: pads, mode, per-pad psi or event, Estimate. |
| `web/src/pages/header/ResultPanel.tsx` | Answer card, range, response curve, pads, coverage, POP suggestions, per-well chart and table, validation. |
| `web/src/pages/header/WellsTab.tsx` | Background board, pad picker, cards/table toggle, per-well Save, correlation and ResP tables. |
| `web/src/pages/header/WellCards.tsx` | Per-well cards: controls, impact line, review status (not saved / saved / drifted), three lazy charts. |
| `web/src/pages/header/BoardTable.tsx` | Shared controls: `GaugeCell`, `RelationCell`, `IprCell`, and the compact table. |
| `web/src/pages/header/model.ts` | TypeScript mirror of `effective()`, request builders, save plan, `wellImpact`, Vogel curve, review/drift, labels. **Keep in step with the server.** |
| `web/src/pages/header/charts.ts` | Response curve, impact bars, validation, correlation scatter, card charts. |
| `web/src/pages/header/JobStatus.tsx` | Job progress and error lines. |
| `web/src/state/header.ts` | Zustand store (localStorage `woffl.header`): form, per-well choices, job ids, Wells-tab pads. |
| `web/src/api/{types,hooks}.ts` | Types and hooks for the header endpoints. |

Route and nav: `web/src/App.tsx`, `layout/Topbar.tsx` ("Header"), and
`layout/Layout.tsx` (no sidebar).

**Tests:** `tests/test_header_study.py` (34) and
`web/tests/headerModel.test.mjs` (10).

## Latest live numbers (read-only, F/L/R, 120-day fits)

- **Wells:** 50 producers (45 ESP, 5 JP).
- **Auto-flagged gauges:** MPL-20, MPR-110, MPF-62.
- **+15 psi on F, L and R:** −41.9 BOPD (range −24.6 to −79.4). F −13.1,
  L −5.9, R −22.9.
- **No estimate:**
  - MPF-73: its pump model fails and it has no measured relation.
  - MPL-46: its plausible gauge reads 2,618 psi, above the 1,800 default.
- **MPL-20 event** (72 h before / 24 h after, measured with the earlier
  rules): header +5.5 / +5.7 / +5.4 psi, about −24 BOPD. Predicted vs
  measured dBHP had a median error of 1.7 psi on 29 wells.
- **Board speed:** 13–30 s cold, under 1 s cached; runs 3–9 s. The Wells
  tab's single-pad R board loaded in 15 s.

## Verification at handoff

- Python: 2,418 passed (full suite, run after the last backend change).
- Frontend: 62 node tests, `tsc` clean, production build clean.
- Chrome (Playwright, read-only local server): no console errors across:
  - an Estimate from a cold page;
  - an event run;
  - marking a gauge bad and re-estimating;
  - the review cards and filters;
  - the Wells tab background load (R, then L+R);
  - the reworked menus.
- **Live saves:** verified on two wells.
- **Fixed after the first saves:** the Wells tab keeps the last board on
  screen while it reloads, saves against that board's job, and clears the
  saved well's dropdown overrides.
- **Not verified:** hosted deployment and Medium-tier timings.

## Open items (in suggested order)

2. **Well ResP review:**
   - MPF-73 and MPL-20 carry a bulk-loaded 1,800 on Kuparuk;
   - MPL-46's gauge conflicts with the 1,800 default;
   - MPF-73's pump model does not solve (check in Solver).
3. **Correlation scope.** Groups are built from the pads on the board, so the
   Impact and Wells tabs can differ for unsaved wells when their pad sets
   differ. Saved values are unaffected. Fleet-wide groups would remove this,
   at a historian-cost trade-off.
4. **Commit and deploy.** The working tree also holds the uncommitted
   2026-09-24 optimization work, so separate or commit it deliberately.
   Rebuild `web/dist`, deploy, and measure on Medium.
5. **Old Tools > Header Impact.** Retire it or redirect it here. Its probe
   (`tools/hpi_backtest_probe.py`) imports deleted modules.
6. **Later (user-deferred):**
   - ESP pump curves: Δq = −(1 − s)/k × ΔWHP, with no IPR needed;
   - shut-in-derived ResP;
   - fleet-wide event finding.

Local helper scripts (census, smoke runs, Playwright drives, the prop_xref
insert) were in the session scratchpad and are not in the repo.
