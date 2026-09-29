# Header page: production-header pressure impact (delivery, 2026-09-29)

Request: after MPL-20 was brought on line, the production header at F, L and R
rose about 10 psi. Estimate the cost per well and pad in an Optimize-style
workflow. Every well has a saved or correlated WHP-to-BHP relation and a saved
or assumed IPR. ESPs must be modeled from wells with BHP gauges and good WHP,
and those correlations must be assignable to same-lift wells without a gauge.

Built locally the same day. Not deployed. The [plan](header_impact_plan_2026-09-29.md)
holds the first design; this record supersedes it where they differ.

## What exists now

A new top-level **Header** page (`/header`, `web/src/pages/HeaderPage.tsx`)
works in three steps:

1. **Load wells.** A board job (`POST /api/header/board`) loads every producer
   on the chosen pads. For each well it reports:
   - lift type, reservoir and latest test;
   - WHP and BHP now (historian median over the last 72 h);
   - the measured BHP~WHP relation and the WHP~Header relation;
   - saved values from `prop_hist`;
   - a gauge-test Vogel fit, an assumed IPR, and its lift-group correlation.
2. **Run.** A run job (`POST /api/header/run`) takes one of two inputs:
   - **Scenario:** a change in psi per pad.
   - **Observed event:** an event time plus hours before and after. The server
     measures each pad's header change and excludes the event well.

   The answer comes first: a status chip, a headline, the net against the
   event well's oil, and the largest changes. Below it are:
   - per-pad metrics (BOPD per 10 psi);
   - a coverage note and a breakdown of wells on borrowed or assumed inputs;
   - an oil-change bar chart, a per-well table and a CSV export;
   - for events, predicted vs measured BHP change on gauged wells.
3. **Save selected.** Writes relations and IPRs to `prop_hist`
   (`POST /api/header/save`, gated).

The server code is in three places:
- `server/services/header_model.py`: the pure math.
- `server/services/header_study.py`: data, jobs and save.
- `server/routers/header.py`.

The frontend helpers are in `web/src/pages/header/`; the page's state is in
`web/src/state/header.ts`.

The Tools > Header Impact page is unchanged and still has the gaps listed in
the plan. Retire it or point it here when convenient.

## Model

For a well that is not a jet pump:

```
dWHP = r * dHeader     r = within-day WHP~Header slope (1.0 when not measured)
dBHP = s * dWHP        s = CLOSED-LOOP within-day BHP~WHP slope
dLiq = Vogel(BHP_now + dBHP) - Vogel(BHP_now)
dOil = dLiq * (1 - WC_latest_test)
```

- **Closed-loop slope.** `s` is measured while the well produces, so it already
  contains the rate response. It is never coupled to the IPR a second time.
- **What `s` means for an ESP.** At fixed speed, `s = 1/(1 + PI*k)`, where `k`
  is the steepness of the pump curve. A high-PI ESP barely moves BHP but loses
  the most liquid.
- **Measured status.** A relation is *measured* with at least 5 days of
  r² ≥ 0.5, and those days must be at least 25% of the days the WHP moved.
  The old tool also required each day's slope to fall in [0.2, 1.5]. That band
  wrongly called every high-rate ESP "slugging", so it is gone here.
- **Correlations.** Groups are lift type + reservoir once a group has 4
  measured wells; otherwise the lift-type group is the fallback. Each group
  fits `s = a + b ln(q_liq)` by Theil-Sen, or uses the group median when there
  are fewer than 5 wells or less than 2x rate spread.
- **IPR ladder.** The run uses the first of these that exists:
  1. Saved (`ipr_qwf_liq`, `ipr_pwf`, `resvr_press`).
  2. Gauge-test fit: a pseudo-Pr Vogel on the latest test, used only with 4 or
     more tests, 100 psi of BHP spread, the Pr off the cap, and under 25% rate
     scatter.
  3. Assumed: ResP from `vw_prop_resvr`, else Schrader 1800 / Kuparuk 3000.
     BHP comes from the gauge or from the group's median BHP/ResP ratio.
  4. None. A well with less than 300 psi of drawdown gets no IPR and shows up
     in coverage.
- **Jet pumps.** Hydrated exactly like an optimization run
  (`optimizer_runs._build_configs`: saved IPR, scoped pump fit, hydraulics),
  then solved at the model WHP and WHP + dWHP with PF held
  (`pad_optimize._model_at_forced_header`). A sonic pump shows about 0.
  - If the pump model fails, the row falls back to the well's own
    *measured* relation on the pump model's IPR.
  - If there is no measured relation either, the well has no estimate.
- **Online default.** A well starts online only when its latest test is at
  most 45 days old. A gauge that has built more than 300 psi (and 1.5x) above
  the latest test marks the well as looking shut in.
- **Status chip.**
  - Complete: every online well uses a measured relation or the pump model,
    plus a fit or Solver IPR.
  - Conditional: some well uses a correlation, a weak or manual relation, or
    an assumed or manual IPR.
  - Incomplete: some online well has no estimate.

## prop_hist additions (production write, authorized by Scott 2026-09-29)

Six rows were added to `mpu.wells.prop_xref`. This was one parameterized
INSERT through the gated `execute_write`, with the gate enabled only in that
script's process. The script was idempotent and was verified by read-back.

| prop_id | meaning | units |
|---|---|---|
| `hdr_bhp_whp_slope` | closed-loop dBHP/dWHP | psi/psi |
| `hdr_whp_hdr_slope` | dWHP/dHeader | psi/psi |
| `hdr_fit_r2` | mean within-day r² (0 when not a gauge fit) | unitless |
| `hdr_fit_days` | fit days (0 when not a gauge fit) | days |
| `hdr_rel_source` | 1 own gauge fit, 2 lift-group correlation, 3 manual | code |
| `hdr_ipr_source` | 1 gauge-test fit, 2 assumed, 3 manual | code |

No `prop_hist` rows have been written yet.

- **The IPR is the existing curve.** It is stored in `ipr_qwf_liq`,
  `ipr_pwf` and `resvr_press`, keeping one IPR per well.
- **Jet-pump IPRs stay in Solver.** Save refuses them, and it saves no
  relation for jet pumps.
- **How Save builds its values.**
  - Each well is one `push_props` statement.
  - Values for measured, correlation, fit and assumed choices are read from
    the completed board job on the server.
  - Only manual entries carry client numbers, and those are validated.
  - A weak measured relation cannot be saved.
- **Effect on the well list.** An ESP well that gets a saved `resvr_press`
  then shows it in `vw_prop_resvr`. The app's well list is a
  `vw_prop_mech LEFT JOIN vw_prop_resvr`, so this adds no well to the JP list.

## Live results (read-only, 2026-09-29, F/L/R, 120-day fits)

- **Wells.** 50 producers: 45 ESP, 5 JP (MPR-111's last test is April; it and
  MPL-20 start offline). No gas-lift or flowing wells. The R header tag is
  `MPU_PI_4661`; it tracks F/L at +8 to +10 psi. WHP~Header slopes are
  0.90–1.0.
- **Relations.**
  - 33 ESPs have measured relations.
  - Kuparuk ESPs have a median slope of about 0.8, and within Kuparuk the
    trend with rate is essentially flat.
  - Schrader ESPs (all of R, plus L-54/56/57) sit at 0.05–0.27.
  - A pooled line would have given gaugeless R wells about 0.42; the Schrader
    group gives about 0.2.
- **Scenario, +10 psi on F, L and R:** −29.9 BOPD (−83 BLPD) across 42 wells.
  - F −10.6, L −5.4, R −13.9 BOPD. R's high-rate Schrader ESPs lose the most
    oil per psi.
  - Status is *incomplete*. MPF-73's pump model does not solve and it has no
    measured relation. MPL-46 and MPR-110 have BHP within 300 psi of their
    IPR's ResP; MPR-110's gauge reads about 4,190 psi.
- **Observed event: MPL-20 on line 2026-09-28 ~12:00.** Header rise was
  measured with a 72 h window before and 18 h after, skipping 6 h each side:
  F +5.5, L +5.7, R +5.4 psi.
  - Estimated cost: −20.9 BOPD.
  - The change was about half the 10 psi seen at first. Daily means ran
    402 → 410 psi, but the window medians are the defensible number.
- **Validation on gauged wells.** 30 of 41 were usable; 11 had a well event in
  the window. Mean error +0.9 psi, median |error| 2.1 psi, 24 within 5 psi.
  MPF-107's jet-pump model predicted +2.8 psi and +2.8 psi was measured.
- **Timings.** Board 13–28 s (historian cold vs warm), run 2–7 s.
- **A separate event.** A +28 psi header step on 2026-09-20 is not the MPL-20
  bring-online; MPL-20's gauge was still building while shut in until Sep 28.

## Verification

- Python: 2,407 passed (23 new in `tests/test_header_study.py`; writer
  mocked, gate never set).
- Frontend: 58 node tests passed (6 new in `web/tests/headerModel.test.mjs`),
  `tsc` clean, production build clean.
- Local server against live Databricks with writes off:
  - board, scenario run and event run all completed over HTTP;
  - save returned 403;
  - a Chrome (Playwright) pass had no console errors.

## Limits and next steps

- **Many IPRs are assumed.** 29 Kuparuk/Schrader ESPs use the assumed ResP
  because their pseudo-Pr fits pin at the cap. Oil deltas scale with the PI
  that follows from that ResP. Saving reviewed IPRs or better reservoir
  pressures moves the answer more than any slope refinement.
- **The ESP model is empirical.** There is no ESP pump-curve model. The
  measured closed-loop slope carries the pump's behavior, and a speed change
  invalidates it.
- **Event windows are short after a recent event.** Right after an event the
  post window is short (14 h here); rerun later for a firmer header change.
- **MPF-109's slope is negative.** Its within-day slope is about −0.96 (weak),
  so it uses the correlation. Worth a look at its gauge or controls.
- **Deployment is not done.** Nothing here has been deployed or measured on
  Databricks Medium.

## Second round (2026-09-29, later): one-step answer, ranges, assignment

The user asked for this priority: "header pressure jumped 15 psi, or we know
it will — what's the impact to oil rate". Also requested:
- assigning a WHP/BHP correlation to an online well without a gauge;
- assigning a reservoir IPR correlation to such a well;
- suggested POPs as a nice-to-have.

Deferred for now: ESP pump curves, and reservoir pressures from shut-ins.

What changed:

- **Estimate in one step.** Pick pads and the psi change, then press
  Estimate. The run builds (or reuses) the wells board itself and returns it,
  so the wells table fills from the Estimate. "Load wells" is optional. The
  board cache is now 15 minutes, cleared on save. A cold Estimate took about
  20 s in the browser; repeats take seconds. Save accepts either a board or a
  run job id.
- **Range.** Each well gets a low/high estimate:
  - Relation spread: the measured slope's IQR, or the correlation's scatter
    (± residual MAD).
  - Reservoir IPR spread: the group's J/q IQR.
  - The largest loss pairs the high slope with the low ResP. Both ends keep
    the 300 psi drawdown rule.
  - The pad and total ranges put every well at the same end together. That is
    an envelope, not a confidence interval, and the page says so.
- **Response curve.** Oil change against a uniform header change from −30 to
  +40 psi on every selected pad: total with its band, one line per pad, and a
  marker at the requested change.
  - Non-JP wells are exact at every point.
  - Jet pumps are solved at −30/−10/+10/+20/+40 and interpolated through zero.
- **Gauge quality.**
  - Automatic flag: BHP above the reservoir's pressure cap (MPL-20 reads PF
    pressure at the pump entry; also MPR-110, MPL-46, MPR-111), or under
    1 psi of movement in 72 h (MPF-62).
  - A per-well **Gauge** checkbox overrides either way.
  - A bad gauge takes the well to its correlations and stops the "looks shut
    in" test from dropping it.
  - An intake below WHP is not a fault for a pumped well (MPF-73), so that
    rule was removed.
- **BHP~WHP correlation assignment.** Every well carries its options: each
  lift-type + reservoir group and each lift-type group, at its own test rate.
  The table's relation menu lists them ("Corr ESP schrader 0.21"), same lift
  type first.
- **Reservoir IPR correlation assignment ("Res. corr" in the IPR menu).**
  - **The quantity:** fractional productivity
    J/q = (0.2 + 1.6r) / (Pr (1 − 0.2r − 0.8r²)), with r = pwf/Pr. That is
    the share of the well's liquid lost per psi of BHP.
  - **Where it comes from:** same-reservoir wells of any lift type that have
    a saved IPR or a usable gauge-test fit.
  - **Groups:** pad + reservoir with at least 4 contributors, else reservoir
    with at least 3. Below that, the documented ResP ±20%, labelled.
  - **How a well uses it:** it keeps its own test rate. With a working gauge
    it solves for the ResP that gives the group's J/q at its own BHP; without
    one it takes the group's BHP/ResP ratio.
  - **Why J/q:** 17 Kuparuk ESP fits pin at the 4,200 psi cap, so a median
    ResP taken from fits would be biased. The group note counts those pinned
    fits and says the loss would sit nearer the low end if they are right.
  - **Saving:** the resolved anchor goes into the existing IPR props with
    `hdr_ipr_source` = 2. The prop_xref text says "assumed"; it now means the
    reservoir correlation.
- **One resolver.** `header_study.effective(row, choice)` gives the gauge
  state, relation, IPR and ranges for the run and the save.
  `web/src/pages/header/model.ts:effective` mirrors it for the table. Choices
  carry `gauge_bad`, `corr_group` and `ipr_group`.
- **POP suggestions (event runs).** Taken from the daily downtime log
  (`vw_shut_in`), a day either side of the event:
  - a partial day after a fully-down day = "came on";
  - a down day after an up day = "went down".

  Suggestions only. MPL-20's power fluid reached the header on 09-28 about
  12:00, while the log shows it down all of 09-28 and 8.2 h up on 09-29.

Live results (read-only, F/L/R, 120-day fits):

- **Reservoir IPR groups.**
  - Kuparuk: 10 contributors, J/q median 0.69% of liquid per 10 psi
    (IQR 0.23–0.80%); BHP/ResP 0.38.
  - L-Kuparuk: 7 contributors.
  - Schrader: only 2 firm IPRs (MPF-109, MPR-102), so it uses the documented
    1,800 psi ±20%.
- **+15 psi on F, L and R: −65 BOPD (−254 BLPD) across 44 wells, range −36 to
  −93.** F −29.7, L −12.8, R −22.5. The curve gives about −4.3 BOPD/psi.
  - This is higher than the first round's −30 BOPD per 10 psi, because the
    Kuparuk reservoir correlation replaced the flat 3,000 psi default
    (steeper IPRs).
  - The pinned fits argue for the low end; saving reviewed IPRs will settle
    it.
- **MPL-20 event (72 h before / 24 h after):** header +5.5 / +5.7 / +5.4 psi,
  −23.8 BOPD (range −12.8 to −33.8).
- **Event validation:** 29 of 40 gauged wells usable, median |error|
  1.7 psi, 24 within 5 psi.
- **Only MPF-73 remains without an estimate:** its pump model fails and it
  has no measured relation.

Verification:
- 2,413 Python tests passed, 29 of them in `test_header_study.py`.
- 60 frontend tests passed.
- `tsc` and the production build are clean.
- In a Chrome pass there were no console errors. It covered the Estimate from
  a cold page, marking F-05's gauge bad, the stale-result warning, and
  re-estimating: F-05 then ran on its ESP-Kuparuk correlation and the Kuparuk
  reservoir IPR.
- Save is still exercised only with mocks.

## Third round: review cards, R-Pad / Schrader evidence

The user noted that R-Pad is Schrader and has BHP gauges. They asked for one
scrolling panel for quick per-well review and edits, modelled on their
well-surveillance screen: well, the two fits, controls and Save.

**Why R-Pad was on the documented default.** R's gauges are good. But from
test to test the rate moves with ESP speed and water cut, not with BHP:
- R-104: rate and BHP are uncorrelated (r = −0.02).
- R-108: rate *rises* with BHP (r = +0.56).

So pseudo-Pr fits pin at the cap:
- Only R-102 fits cleanly (Pr 1,622; about 0.8% of liquid per 10 psi).
- R-144 is clean too (about 1.2% per 10 psi), but its test BHPs span only
  67 psi, so it is flagged.
- R-110's test BHPs are stuck at 1,861 psi since March, and its live tag
  flaps between 2,700 and 4,200 psi.

**Shut-in gauges point to higher Schrader pressure.** MPR-111 (offline) reads
about 2,443 psi and MPL-46 about 2,618 psi. Both are above the 2,200 psi fit
cap and the 1,800 psi documented default.

Changes:

- **Gauge plausibility limit.** `header_model.GAUGE_MAX` is Schrader 3,200 and
  Kuparuk 4,500. It is separate from the fit cap, so shut-in Schrader gauges
  are no longer called broken. MPL-20 (PF pressure, about 4,630) and MPR-110
  (4,188) stay flagged.
- **Shut-in readings per reservoir group.** Plausible gauges on wells that
  look shut in or have no recent test are listed on each group as "ResP at
  least". They are evidence only.
- **Flagged fits.** `ipr_fit_any` keeps the gauge-test fit even when flagged.
  An engineer can choose it explicitly ("Gauge fit (flagged)") and save it.
  Once saved it counts toward the reservoir group, so groups improve as wells
  are reviewed. The default ladder still uses only usable fits.
- **Review cards** (`web/src/pages/header/WellCards.tsx`). This is the default
  view; the compact table is a toggle. One card per well:
  - **Left:** test, WHP/BHP, online and gauge toggles, relation and IPR menus
    (the table's controls), the well's impact at the typed or measured change
    with its range, the saved state, and a per-well Save.
  - **Right, three compact charts:**
    - pressures (BHP; WHP and header on a second axis), where dead gauges are
      obvious;
    - each day's BHP~WHP slope, with measured, correlation, saved and manual
      reference lines;
    - the IPR: tests shaded by age, today's operating point, and the gauge
      fit, reservoir correlation, saved and manual curves. The curve in use is
      drawn thicker.
  - **Filters:** pad, lift type, well search, and "needs review" (bad gauge,
    borrowed or weak inputs, no estimate): 37 of 50 wells on F/L/R.
  - **Loading:** cards fetch `GET /api/header/well/{well}?pads=...` only when
    they scroll into view. With `pads` the endpoint reads the board's cached
    pad-wide historian pull, so scrolling costs no extra historian queries.
  - **Impact line:** the same chain as the server
    (`model.ts:wellImpact`). A node test checks it against the Python case.

Verification:
- 2,418 Python tests passed (34 header tests).
- 61 frontend tests passed.
- `tsc` and the build are clean.
- A Chrome pass against live data (read-only) had no console errors. It
  covered the Estimate, the R-Pad cards and the needs-review filter.

### Review status and drift (ESP / gas-lift review)

The review cards are the workflow for ESP, gas-lift and flowing wells:
1. Check the three charts.
2. Choose the relation and IPR (or mark the gauge bad).
3. Save the well.

Saved values become everyone's defaults on the next load. Jet pumps are
reviewed in Solver.

Each non-JP card shows a status (`model.ts:reviewState`):
- **not saved**;
- **saved**, with date and user;
- **saved - drifted**, in two cases:
  - the saved slope differs from today's measured slope by more than
    max(0.15, the IQR) on a working gauge (a speed change or pump swap);
  - the test rate has moved more than 25% from the saved IPR anchor.

Filters:
- lift type: Non-JP / ESP / gas-lift / flowing / JP;
- status: any / needs review (now including drifted) / not saved / saved /
  drifted.

Save has still been verified only with mocks; a first live save needs the
write gate on.

### Per-well reservoir pressure (user decision, 2026-09-29)

"Each well will have a unique RP. Keep 1,800 as default unless the user has
saved a different value in prop_hist."

- **ResP per well.** Every well runs on its own ResP: the latest `resvr_press`
  in prop_hist (a Solver save, a header save or a bulk load), else the
  documented default for its reservoir (Schrader 1,800; Kuparuk 3,000 pending
  confirmation) ±20%. A saved ResP carries no range. No group, fit or
  shut-in reading sets it.
- **Default IPR** (no saved IPR): that ResP at the latest test rate and the
  gauge BHP.
- **Without a working gauge,** BHP = ResP × the median BHP/ResP of gauged,
  flowing wells in the same pad + reservoir (at least 3 wells, else the
  reservoir).
- **Gauge-test fits** back out a pseudo-ResP. They are offered but never the
  default; choosing and saving one sets that well's ResP.
- **Removed:** the J/q (fractional-productivity) groups from the second
  round.
- **Shut-in gauges** stay listed per group as evidence only.

**Live (F/L/R):**
- Five wells have a saved ResP: MPL-06 at 2,597, and 1,800 on MPF-73,
  MPF-107, MPL-20 and MPR-111 (bulk load).
- +15 psi gives −41.9 BOPD (range −24.6 to −79.4): F −13.1, L −5.9,
  R −22.9.
- MPL-46 now has no estimate. Its plausible gauge (2,618) exceeds the 1,800
  default: either it is down or it needs a saved ResP.

### Impact / Wells tabs

`/header` now has two tabs; `?tab=wells` opens the second.

- **Impact:** the Estimate form and its answer.
- **Wells:** its own pad picker.
  - Picking or unpicking pads starts the board in the background (0.6 s
    debounce). There is no Load button.
  - A job for pads no longer wanted is cancelled.
  - Boards are cached for 15 minutes, so switching back is under a second.
  - A Reload button is there for after someone else saves.
  - The tab holds the review cards / compact table, per-well Save, and the
    correlation and ResP/BHP-ratio group tables.

Both tabs share per-well choices, so an edit on Wells is what the next
Estimate uses. On a first visit the Wells tab starts from the Impact tab's
pads.

Menu clean-up after the user's review:
- The gauge control reads **Bad gauge** (tick if bad), with
  "(auto-flagged)", "(marked by you)" or "(auto flag overridden)".
- **Default** names what it resolves to: "Default: measured 0.22", or
  "Default: ResP 1,800 (default) @ gauge BHP".
- The relation menu lists only the well's own lift-type groups, most specific
  first. A group giving the same slope as a more specific one is hidden.
- The IPR menu offers one "well ResP @ gauge BHP" option when the gauge works.
  Without one it offers each same-reservoir BHP-ratio group
  ("BHP = 0.39 × ResP (R schrader)"), deduplicated the same way.
- A group already selected always stays listed.
