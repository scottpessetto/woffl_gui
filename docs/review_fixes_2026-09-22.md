# Save workflow and S-Pad optimization fixes (2026-09-22)

From a review of the single-well save workflow and the S-Pad optimization
tab (two read-only code reviews plus live timings), then user requests made
the same afternoon. Local verification only; nothing was deployed and no
production rows were written.

## Save workflow

- **Saved inputs now reload after a Match test.** A Match test fits kth/kdi
  (ken held) together with the IPR anchor. Save well inputs stored the anchor
  but never the coefficients, so a reopened well used reference losses
  against the matched BHP and looked reset. `/match-test` now keeps a
  saveable fit server-side for an hour and returns a `save_token`.
  `POST /wells/{well}/match-calibration` saves it as the installed pump's
  calibration (`pump_calibration.save_match_fit`): same record as event
  calibration, one test so always provisional. The save is refused unless a
  fresh tracker read shows the pump the match ran on and the fresh saved
  well inputs equal the matched inputs. The Match test block shows a
  before/after table of every value it sets, and where each is saved, plus
  **Save this match** (well inputs pinned to the test, then the pump fit).
- **Atomic save.** The anchor pin (or un-pin marker) now rides in the values'
  single INSERT (`save_ipr_values(pin_value=)`). As two statements, a failed
  values write left a newer pin that silently replaced the previous save.
- **Canonical context window.** The well context is always fetched at 6
  months, no cap: the window calibration, pump-fit save and replay use. With
  the lookback in the request, the save-bar baseline and the pump-fit
  identity depended on whichever lookback was set at fetch time. The
  window's IPR fit still applies over the seeds.
- **No double submit.** Saves wait for the pin and context refetch before
  re-enabling. The lock toggle shares the well-input write key. The pump-fit
  save is one per fit.
- **Lock toggle confirms** when it would save an unsaved sidebar value.
- **Calibration** is blocked (with the reason shown) on geometry conflicts
  and session-only edits, hydrates only the target well, and names that
  well's failure instead of "no usable saved fit".

## IPR anchor and bad tests (later requests)

- **Anchor auto-applies.** Changing the IPR anchor applies that anchor's fit
  to the sidebar as soon as it lands (SolverPage `anchorPicked`), replacing
  hand edits of the seeded fields like the old button did; field locks still
  hold. The "Apply IPR to inputs" button is gone. The first-load rule is
  unchanged: saved values outrank the fit on open.
- **The save bar says what it saves:** the IPR, the test it is anchored on
  (or a manual point / common curve), rate, BHP, ResP, WC, GOR, WHP and any
  changed temperature/bubble point. It flags an anchor that differs from
  the saved pin. The anchor was already saved: a test anchor pins its
  resolved test, and a manual point clears the pin. The well reopens on
  that test ("Specific test") or Manual.
- **Verified on live data** (read-only): MPS-03, MPS-05 and MPS-54 reopen
  with exactly their saved rate, BHP, ResP, WC, GOR and WHP.
- **Exclude a bad test.** The Solver's well-test table has an Exclude box.
  An excluded test leaves the anchor and comparison dropdowns, the chart and
  the IPR fit (`IprFitRequest.exclude_wt_uids`, so no anchor mode can pick
  it). The table keeps it, greyed, to undo. Exclusions are per well in the
  browser (localStorage). The optimizer, calibration and history replay
  still read every test.

## Optimization tab

- **Cancel** for pad/CFP runs, match health, event calibration and the PF
  cost panel (`DELETE /optimize/run/{id}`). Progress updates go through
  `jobs.set_progress`, which honors a pending cancel; the event-calibration
  pool wait keeps plain writes so a cancel lands after the uninterruptible
  worker. With `WOFFL_MAX_JOBS=1`, a mistaken run no longer blocks every
  other panel for minutes.
- **Default offline** (user request): LTSI, wells currently shut in under
  the plain SI down code, and named non-producers (`DEFAULT_OFFLINE`,
  currently MPS-29 as a recycle well) are pre-ticked on every pad. An untick
  still wins and persists.
- Runs wait for the downtime log before starting, and warn if it failed.
  Match health now honors the offline set.
- **Batched simulation.** `NetworkOptimizer(well_grids=...)` and
  `simulate_jobs` run wells with different pump grids or headers in one
  pooled submit (the S-Pad settle loop and the PF cost panel use them). The
  PF panel's modeled range now reaches -800 psi, so large upsizes are not
  priced on held-flat rates.

## Measured

Live S-Pad run at 2 workers, committed code against this change, same
script: 117.8 s / 103.5 s before, 115.9 s after. No measurable speedup. The
header sweep dominates, and each trial already fills both workers. On
identical data the plans matched exactly (13,107 BOPD, 40,621 BPD). An
earlier 85 BOPD gap was live data changing between runs, shown by two
committed-code runs that also differed.

Cold S-Pad hydration is 6.4 s (18 one-per-well saved-IPR queries); warm is
0.2 s. The PF cost panel takes 29 s.

## Not done

- Sharing one hydration across panels, and replacing the per-well saved-IPR
  query with one fleet query.
- Stale-result warnings when offline/fit/pump settings change after a run.
- Making "today" consistent between the pad run, match health and the PF
  panel.
- Speeding up the header sweep itself (fewer trials or reuse between trials).
