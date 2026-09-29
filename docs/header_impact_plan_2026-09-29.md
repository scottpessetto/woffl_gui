# Pad header pressure impact: plan (2026-09-29)

User request: after bringing MPL-20 online (BOL), the production header at
F, L and R pads rose about 10 psi. Estimate what that costs each well, in the
style of the Optimize workflow. Every well should either have a saved
WHP-to-BHP relation or use a correlation when BHP is not known, and either a
saved IPR or an assumed one.

**Status:** built the same day - see the [delivery record](header_impact_delivery_2026-09-29.md),
which supersedes this plan where they differ. The user then chose its own page,
new prop_hist ids for relations and lift-group correlations (not VLP) for
gaugeless ESPs; R's header tag is MPU_PI_4661 and R has no jet pumps today.
This plan is kept as the design rationale.

"Header" here means the **production** header (wellhead back-pressure,
`WellConfig.surf_pres` / BatchPump `pwh`). It does not mean the PF header that
the pad optimizer and `plan_robustness` call "header" (`ppf_surf_well`). Keep
those two names apart in code and UI.

## 1. What exists today

### Header Pressure Impact tool (Tools tab)

Request: `POST /tools/header-impact/run`, handled by
`server/services/tools/runs.py:165`. It runs as a job (`tool_header_impact`).

- **Input:** one `delta_p` applied to every selected pad
  (`schemas.HeaderImpactRequest`).
  - There is no per-pad delta.
  - There is no "observed event" mode.
  - Nothing can be edited per well (`useHeaderImpactInputs` exists, but no
    component calls it).
- **JP wells** (`header_impact.solve_jp_row`) run BatchPump at WHP and at
  WHP + delta, holding PF fixed. The sonic case is tagged. The model is built
  from `_common.build_well_config`, **not** from the optimizer's hydration:
  - it does not load the saved prop_hist IPR (`get_vogel_for_wells` refits it
    from tests on every run);
  - it does not use the scoped installed-pump fit;
  - it always uses Beggs-Brill, never the saved hydraulics model;
  - a gaugeless JP falls back to hard-coded values: qwf 750, pwf 500,
    WC 0.5, GOR 250.
- **Non-JP wells** (ESP, gas lift, flowing):
  - The slope is a measured median of within-day Theil-Sen BHP~WHP fits on
    hourly historian data (`header_trend.fit_well`).
  - The IPR is a generic Vogel through test oil, the recent median BHP and
    ResP (ResP falls back to 1800).
  - Gaugeless wells get **no number** ("gaugeless - use Analog").
  - Their estimate is `Emp ΔOil`, which is **excluded from the total and
    the chart** (`runs.py:237` sums only `DeltaOil`).
- **Unused code:**
  - donor/group analogs (`estimate_header_impacts`, `_analog_doil`,
    `_resolve_corr_fits`);
  - gaugeless JP IPR back-calculation (`_estimate_gaugeless_ipr`);
  - per-pad sensitivity (`summarize_sensitivity`, BOPD per 100 psi);
  - response curves, backtest, PDF report (`header_report.py`).
- **Broken:**
  - `tools/hpi_backtest_probe.py` imports deleted modules;
  - `tests/test_header_engine.py` and `test_header_impact.py` survive only as
    `.pyc`;
  - `test_tools_layer.py` only checks source text.
- **Pads:**
  - `PAD_HEADER_TAGS` (`header_trend.py:49`) has F and L but **no R**.
  - `_PAD_PF_DEFAULTS` has no L or R; both fall back to 3400.
  - The formation of a non-JP well with no pump depth defaults to Kuparuk.

### Optimize workflow (the template)

- `optimizer_runs._build_configs` hydrates every pad well from:
  - saved well inputs (`wells.well_context`: saved IPR, locks, WC/GOR/ResP,
    `surf_press`);
  - the scoped pump calibration (`pump_calibration.resolve`);
  - the saved hydraulics model.
- It keeps a **coverage ledger**:
  - statuses: `modeled`, `missing_inputs`, `failed_model`, `held_measured`
    and others;
  - `_qualify_coverage` labels the result complete, conditional or
    exploratory.
- It is already reused by match_health, pump_decision and event_calibration.
- The UI has three parts:
  - **Readiness board:** per-well IPR saved/by, pump-fit chips, offline and
    required checkboxes.
  - **RunPanel:** the form is persisted per tab; blockers are shown before
    submit; the job id is stored; it polls every 2.5 s; progress and ETA;
    cancel; a stale-inputs warning.
  - **Results:** the recommendation card first, then the coverage notice,
    then metrics, table and charts.
- `pad_optimize._model_at_forced_header` / `pf_pressure_what_if` already solve
  current pumps at forced per-well pressures. The same pattern applies with
  `surf_pres` in place of `ppf_surf_well`.

### Physics and data available

- `outflow.production_top_down_press` walks a pressure from WHP down the
  survey using BB, H-B/Griffith or Shi/Pan. Only the JP discharge uses it.
  - No code runs a WHP-to-BHP tubing curve (VLP) for non-JP wells.
  - Lift gas is not added to the fluid.
- Test WHP comes from `vw_well_test.whp`. Forward-circulation wells produce
  up the annulus: MPL-20 and several F wells use `inn_ann_prs`.
- Hourly BHP comes from `vw_bhp_tags`. The WHP tag is derived as
  `MPU_PI_<pad#>2<well#>`.
- Saved IPR props: `ipr_qwf_liq`, `ipr_pwf`, `resvr_press`, `form_wc`,
  `form_gor`, `surf_press`, locks and pin. No prop stores a WHP/BHP relation.
- Local surveys exist only for MPF-107, MPF-73, MPL-06, MPL-20 (flagged
  corrupt TVD step) and MPR-111. Other wells use the template profile.

## 2. Model

Each online well on the selected pads gets:

```
ΔWHP_i  = r_i · ΔHeader_pad          r_i: WHP~Header slope, default 1.0
ΔBHP_i  = relation_i(ΔWHP_i)         closed-loop (includes rate response)
ΔLiq_i  = IPR_i(BHP_0 + ΔBHP_i) − IPR_i(BHP_0)
ΔOil_i  = ΔLiq_i · (1 − WC_i)        total-liquid IPR, oil derived once
```

### WHP to BHP relation: first available source wins, and is labeled

1. **JP wells: WOFFL physics.**
   - Hydrate with `_build_configs`: saved IPR, scoped pump fit, saved
     hydraulics.
   - Solve the installed pump at `surf_pres` and `surf_pres + ΔWHP`, with PF
     held at its live value.
   - The result is ΔBHP and ΔOil together. A sonic pump is labeled
     "decoupled", not zero-by-accident.
   - The measured slope, where it exists, is shown alongside as a check.
2. **Saved measured relation.** The slope, r², days and window that the
   user reviewed and saved (phase 4).
3. **Live measured relation.** The `header_trend.fit_well` Theil-Sen slope,
   when the well classifies as responsive. This slope is *closed-loop*: it
   already includes the rate falling as BHP rises. Do not apply the IPR
   coupling to it again.
4. **Correlation (VLP), for wells without usable BHP data.**
   - Walk `production_top_down_press` from WHP and from WHP + Δ, at the test
     rate, WC and GOR (plus lift gas for GL wells), down to the gauge or
     perforation depth, with the well's tubing and survey.
   - This gives the fixed-rate slope `s_q`.
   - Couple it to the IPR in closed form:
     `ΔBHP = s_q·ΔWHP / (1 + PI·∂VLP/∂q)`, or do one secant nodal solve.
   - Label it "correlation". ESP: treat the pump as a fixed-speed head adder
     and set `s_q ≈ 1` (the default; the pump curve is not modeled). Say so.
5. **Analog.** The median measured slope of pad and formation/lift peers
   (reconnect `_resolve_corr_fits`). Label it "analog".
6. **Assumed 1:1.** The last resort. It is labeled, and it is counted as
   *assumed* in the coverage.

### IPR: first available source wins, and is labeled

1. Saved prop_hist IPR (`load_saved_ipr` or the `well_context` seeds).
2. Vogel derived from gauge BHP and a test (`get_vogel_for_wells`).
3. Generic Vogel anchored at the latest test rate and the BHP from the
   relation source. ResP comes from `vw_prop_resvr`, then a documented
   formation default. **Reconcile the conflicting defaults first:**
   `ipr_analyzer` uses 1800/3000 while `header_engine` uses 2200/4200.
4. Refuse when ResP − BHP < 300 psi (existing EVID-F17 rule). The coverage
   ledger reports the well as `missing_inputs`. It is never silently zero.

### Why a 10 psi change is easy numerically

A 10 psi change is small. The linear results (ΔBHP from the slope, ΔQ from
the local IPR derivative) should agree with the full solves to within
rounding. Tests should assert this.

Beyond about ±100 psi, JP wells can cross sonic or turn infeasible. The full
solve handles that, and the rows say so.

## 3. Workflow and UI ("Header" tab on Optimize)

Mirror the Optimize page. Put it as a new tab next to the pads so it reuses
the page's form persistence and job UX. The Tools > Header Impact page
redirects to it (phase 5).

1. **Readiness board.** One row per pad well:
   - lift type;
   - relation source (physics / saved / measured / correlation / analog /
     assumed) with r² and days;
   - IPR source and date;
   - pump-fit chip (JP);
   - WHP now;
   - an Offline tick.
   Well names link to Solver, as they do today.
2. **Run form.** Two modes:
   - **Scenario:** a per-pad Δ in psi, defaulting to the same value for all
     selected pads.
   - **Observed event:** pick a date, for example the MPL-20 BOL. The server
     computes each pad's header Δ from the `PAD_HEADER_TAGS` series using
     pre and post windows (default 3 days each, excluding the transition
     day). It shows the Δ it measured, so the user can override it.
   - **Optional "include the event well":** add the BOL well's post-event
     rate, so the card can report the net gain.
3. **Result card, first.** Example layout:

   > "+10 psi at F/L/R: −N BOPD across M wells (JP −a, ESP −b, GL −c).
   > Largest: MPL-xx −9, MPF-yy −7, …
   > Net of L-20's +Q BOPD: +Q−N."

   Then:
   - a status chip (**complete / conditional / exploratory**, from coverage,
     where "assumed" and "analog" wells make a run conditional);
   - the coverage notice;
   - a per-pad table (ΔOil, BOPD per 10 psi, wells by source);
   - a per-well table (all sources in one ΔOil column, source chip, ΔBHP,
     sonic flag);
   - a bar chart through `ChartPanel`;
   - a CSV export.
4. **Observed-event check.** Only in observed-event mode. For gauge wells,
   show measured ΔBHP (pre vs post) beside predicted ΔBHP, plus any
   well-test oil pairs that straddle the date. This is the direct validation
   of the model on the event you just saw. Tests are sparse; label n.

## 4. Phases

### Phase 0: read-only census (no code, SELECTs only)

- For F/L/R online producers, list:
  - lift type;
  - gauge coverage;
  - responsive/slugging from `header_trend`;
  - saved IPR present;
  - survey/tubing present.
- Find R-pad's header tag.
- Measure the F/L/R header Δ around the L-20 BOL date.

The census decides how many wells actually need the VLP path. If almost
every well is JP or gauged, phase 2's correlation can be smaller.

### Phase 1: engine and API (pure functions, tests first)

- Create `server/services/header_runs.py`. It owns the hydration and the job.
- Create a pure module for the relation/IPR ladders and the linear-vs-solve
  math.
- Reuse:
  - `_build_configs` for JP wells;
  - a new non-JP hydrator (test rate/WC/GOR, lift gas, tubing, BHP series,
    saved IPR);
  - `header_trend` for slopes.
- Job kind `header_impact` through `server/jobs.py`, with pool work through
  `server.pool` (`worker_ceiling`, Medium: 2 workers). Add progress and
  cancel.
- Schema: `HeaderRunRequest {pads, mode, delta_by_pad, event_date,
  windows, offline, include_event_well}`, and a result with rows, per-pad
  totals, coverage and assumptions.
- Add R to `PAD_HEADER_TAGS` once the tag is known. Add L and R to the PF
  defaults, or require a live PF for them.
- Tests:
  - an offline fixture pad with one well per relation source;
  - totals include every source;
  - linear ≈ solve at 10 psi;
  - a sonic JP reports decoupled;
  - gaugeless without VLP inputs lands in coverage, not in the total;
  - the WC conversion is applied once.

### Phase 2: correlation (VLP) path for non-JP wells without usable BHP

- The walk goes from WHP to gauge/perf depth, using `production_top_down_press`
  with the saved hydraulics model (default BB).
- Include lift gas in the fluid for GL wells. Check the mixture path handles
  the added free gas; if it needs a shared-library change, follow the §4
  upstream rule (tag, register, named test).
- Calibrate where possible: on gauge wells, compare the VLP-predicted BHP at
  the test with the measured value, and report the offset. Use the offset
  only as a diagnostic, not as a silent shift.

### Phase 3: UI tab

- Readiness board, run form (both modes), result card, tables, chart,
  observed-event check.
- Reuse the RunPanel patterns: per-tab stored form, blockers, stored job id,
  stale-inputs warning, CancelJobButton.
- Add node tests for the summary/status helpers and run the production build.
- Browser QA uses the Playwright fixture pattern (`tools/check_*_ui.py`).

### Phase 4: save the WHP/BHP relation per well (write path; confirm before building)

- Use the existing `woffl_eng_comment` ledger under a new context,
  `whp_bhp_relation_v1`, the same way `pump_calibration_v1` does:
  - one gated parameterized INSERT through `push_eng_comment`;
  - encode within 500 characters before calling the writer;
  - no new table, prop id or DDL.
- The record holds: slope, WHP~Header ratio, r², n_days, window, source,
  hydraulics model (for VLP), fitted_by/at.
- Saving accepts a completed server job id. The server never trusts
  client-supplied numbers, matching the pump-fit save.
- The alternative is a new `prop_hist` prop. It needs a `prop_xref`
  whitelist addition (an external ask), so it is not preferred.
- Verify with mocks only. Never test against a live connection.

### Phase 5: cleanup

- Point Tools > Header Impact at the new tab, or retire it.
- Delete or repair `tools/hpi_backtest_probe.py`.
- Restore real engine tests.
- Remove dead code that the new path does not reuse.
- Write a delivery doc; update `docs/README.md` and AGENTS.md pointers.

## 5. Open decisions for the user

1. **Where it lives.** Recommended: an Optimize tab. The alternative is to
   rebuild the Tools page.
2. **Relation storage.** Recommended: the `woffl_eng_comment` ledger, which
   needs no new schema. The alternative is a new prop_hist prop, which needs
   a whitelist ask.
3. **R-pad header tag.** If the user knows it, phase 0 is shorter.
4. **ESP treatment without a gauge.** Recommended for now: a fixed-head
   assumption (`s ≈ 1`) with a label. Pump-curve modeling is out of scope
   unless ESP wells dominate F/L/R.
5. **PF on L and R.** Live PF readings when available, otherwise a required
   input. Do not use a silent 3400 default.

## 6. Risks and limits

- **Closed-loop measured slopes vs open-loop VLP slopes.** Mixing them is
  the easiest way to double-count. Each row states which kind it used.
- **Header sensitivity is second order for choked JPs.** Most of the pad
  total may come from a few responsive wells. The card ranks them.
- **The observed 10 psi Δ is small** relative to within-day WHP noise.
  - Event-mode Δ needs multi-day windows.
  - The BHP validation will be noisy for slugging wells, which are flagged
    and excluded from the comparison.
- **This is an estimate on current inputs, not field qualification.** The
  status chip and coverage say so.
