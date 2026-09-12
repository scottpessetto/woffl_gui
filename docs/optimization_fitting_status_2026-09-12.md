# Optimization and multi-pump fitting status — September 12, 2026

Later the same day, the user authorized implementation. The first app replay,
production-plot toggle and optimizer history links are now implemented locally.
Following screenshot feedback, the default now applies the saved well fit to
every usable test, with chronological validation available as separate modes;
see [the delivery record](pump_match_ui_2026-09-12.md). The review below preserves
the state before that work. The shared multi-pump loss fitter remains unfinished.

Reviewed the current local source, September 11 handoffs and recorded benchmark
artifacts in response to the user's request to see what remains. The requested
production-plot toggle and a fitted well model spanning multiple installations
are both unfinished. This review changes documentation only.

The user reiterated the desired view: modeled BHP over actual BHP on the
production plot, with predicted oil rate as well, spanning pump changes and
time. Its purpose is to judge whether a fit can support optimization and oil
forecasts. The [September 11 closing record](session_close_2026-09-11.md)
retains the detailed engineering investigations and implementation history.

## Current state checked against source

| Area | Present locally | Still needed |
|---|---|---|
| Production/history plot | Shared `HistoryStrip` in Solver and JP History; measured oil, formation water, BHP, optional PF pressure, and pump installation bands | Modeled BHP/oil series, a Show model match toggle, fit/prediction windows and per-installation scores |
| Installed-pump fitting | Multi-point BHP/PF/pressure-response calibration, exact installation/model identity, separate apply/save actions | A constrained fit spanning several installations, independent oil forecasts and parameter-identifiability checks |
| Historical prediction | Offline event holdouts and adjacent-installation forecasts; the six-well benchmark contains 12 target installations and 163 later tests | An app replay job/API, historical input assembly, selectable fit windows and reusable evidence results |
| Hydraulics | BB/Payne, H-B/Griffith and Shi/Pan reach the main solve/calibration/optimizer path | Explain remaining component-level misses; reconcile standalone tool inputs; Tulsa remains unavailable |
| Optimization | S-Pad selections settle against actual station pressure; missing match evidence is unknown; installed and replacement pumps have distinct assumptions | Link recommendations to historical evidence, validate predicted gains, assess uncertainty/rank stability and benchmark the bounded discrete search |

The existing [offline time plot](pump_history_benchmark_2026-09-11.png) already
shows actual versus predicted BHP, oil and PF. It is a reference forecast with
clean pump losses and earlier-test inflow, not the output of a multi-pump fit.

The recorded frozen-composition comparison gives BB **135/163 solved**. On
the **130 observations solved by every model**, its BHP RMS is **130.7 psi**,
median absolute oil error **26.8%**, and median absolute PF error **9.9%**
(126 scoreable PF observations). H-B improves BHP slightly while worsening
oil; Shi/Pan does not establish a better overall model. Historical geometry,
PVT and reservoir-pressure priors remain unverified. These results do not
qualify sizing recommendations. See the
[comparison data](hydraulics_benchmark_2026-09-11.json).

## The main fitting gap

`calibrate_multipoint` fits the current installation's pump losses and nozzle
area. Each observation supplies its own measured oil/BHP inflow anchor. The
loss deliberately excludes oil because it would be circular under that
anchoring. Drawing oil from those same anchors would show reproduction of
training data, not an independent test of oil prediction.

The chronological benchmark solves the prediction part differently: it freezes
inflow from earlier tests and withholds later oil, BHP and measured PF from
prediction inputs. However, it keeps clean-reference pump losses fixed and
does not learn shared behavior across several pumps. The remaining model work
must join these capabilities while separating changes in reservoir productivity
and composition from changes in pump condition. Another installation, including
a replacement of the same size, must not inherit the previous pump's wear.

## Next product deliverable: Show model match

Extend the shared production/history view so both Solver and JP History have
the same control and results:

1. Add a **Show model match** toggle, initially off. Show modeled BHP and oil
   against the corresponding actual series on the same date axis, using clear
   line styles and point markers. Offer synchronized detail panels when stacked
   water makes oil differences difficult to read; retain the pump timeline.
2. Label **Fitted history** and **Held-out prediction** explicitly. The latter
   means a prediction made with earlier fitting inputs frozen. Display the
   training window, cutoff, hydraulic model and applicable installation.
   A historical replay using measured later WC/GOR must disclose that condition.
3. Include an optional PF **rate** comparison and formation-liquid detail.
   PF pressure is an operating input and the existing PF-pressure checkbox
   does not compare predicted power-fluid consumption.
4. Keep exact installation identities, same-size changeouts and missing periods
   visible. Break model lines at changes, failed solves and unsupported spans.
   Start predictions at actual test dates; add daily predictions only where
   controls and input history support them. Score oil against measured tests,
   not daily oil inferred from measured BHP.
5. Add compact installation scores: test/solve coverage, BHP bias/RMS, oil error
   in BOPD and percent, PF error, pressure range, and fit/holdout status. Selecting
   a miss should reveal its hardware, operating inputs and failure or pressure
   balance. Preserve actuals even when a model job fails.

The chart currently receives measured history only. Its extended test payload
lacks some replay inputs, including GOR, WHP and stable test IDs. Reuse the
existing fuller test/history sources and cache to assemble immutable,
date-specific replay inputs; a frontend checkbox alone is insufficient.
Adapt the existing benchmark and event-holdout services into one bounded heavy
job with progress/cancellation and versioned results. Keep Medium limits and
hide stale results when the well, model, training window or source inputs change.

An initial app delivery can expose the reference and event-holdout predictions
with their actual labels while the shared multi-pump fitter is developed.
Do not label the reference benchmark as a completed fitted well model.

## Remaining engineering sequence

1. **Build the replay-backed toggle.** Reuse `HistoryStrip`, `ChartPanel`, SVG
   charts and shared history. Establish the per-date input/result contract
   before drawing prediction lines. The historical evidence should remain
   useful even when it shows a poor match.
2. **Resolve hard model/input misses before adding fitting freedom.** Continue
   E-42/F-73 component balances and use B-39 to check whether better BHP harms
   oil prediction. Reconcile nominal geometry, contemporaneous measurements
   and gauge offsets. Expose operating regime and discharge-closure residual
   for returned feasibility-edge points.
3. **Fit across installations and test transfer.** Use constrained shared
   behavior with supported reservoir evolution and limited installation
   effects. Check which parameters the measurements can distinguish. Withhold
   later whole pressure events, then a complete different-size installation,
   then a different well; keep model selection outside scored holdouts.
4. **Use that evidence in optimization.** Link each proposed change to the
   supported pump/pressure range. Validate incremental oil against incremental
   constrained water using the existing priced objective, then show gain ranges
   and rank stability. Set acceptance targets from measurement uncertainty and
   decision size. Separately compare bounded searches with small exhaustive
   references. Preserve M/E total formation-plus-lift water and I/S lift water.

Unifying Header Impact/PF Scenario with verified current model context remains
useful follow-up work. Further hydraulic models should follow demonstrated
component errors. The existing [model improvement plan](jp_model_improvement_plan_2026-09-11.md)
and closing record retain these secondary tasks and their limitations.

## Verification and code map

This review ran the focused pump-history benchmark, match-health and pad
optimization suites: **95 passed**. Pytest reported a cache-write permission
warning after the tests. No full-suite run, new field replay, frontend build,
runtime implementation, deployment or production-data write was performed.
Existing local source/build changes from September 11 remain in place.

- Chart and pages: `web/src/components/HistoryStrip.tsx`,
  `web/src/pages/JpHistoryPage.tsx`, `web/src/pages/SolverPage.tsx`.
- History contract: `server/services/history.py`, `server/routers/history.py`,
  `web/src/api/types.ts` (`JpHistoryResponse`).
- Current fit: `server/services/event_calibration.py`,
  `woffl/gui/fric_calibration.py` (`calibrate_multipoint`).
- Prediction foundations: `server/services/field_validation.py`,
  `server/services/pump_history_benchmark.py`.
- Optimization/evidence: `woffl/gui/pad_optimize.py`,
  `server/services/match_health.py`, optimizer result/health components.
