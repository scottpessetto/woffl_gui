# Pump Match Over Time: first app delivery — September 12, 2026

The shared production plot has a **Show model match** control in Solver and
JP History. The default **Every test (well fit)** comparison applies
one saved oil IPR to every usable historical test, with the pump installed
then and that test's measured WC, GOR and PF/WHP pressures. It overlays modeled BHP and oil on actual
history, with synchronized oil and optional PF-rate detail panels. Optimizer
well names link to the same history view with the controls open.

Later September 12 update: **Well inputs** now selects **Saved in database**
or **Current edits (preview)**. The preview applies the sidebar's supported
well inputs to the same fixed-oil-IPR replay; it never saves or automatically
refits them. A persistent **Save well inputs** bar is available in both pages.
See [the workflow and verification](well_input_save_workflow_2026-09-12.md).

This implements the first product block from the
[remaining-work review](optimization_fitting_status_2026-09-12.md). The shared
multi-installation pump-loss fitter and qualification of optimization gains
remain separate engineering work. All three comparison modes use clean-reference
pump losses, as disclosed beside the run controls. The two chronological modes
fit inflow from earlier tests; the every-test mode applies the saved well fit
retrospectively and does not claim held-out validation.

## Use the comparison

1. Open **JP History**, or expand **Pump history** in Solver. Turn on
   **Show model match** and choose the comparison and history window. The
   sidebar's selected hydraulic model is used.
2. **Every test (well fit)** is the initial selection and API default.
   It retains the saved/server-loaded oil IPR across all installations,
   using each test's measured WC/GOR without changing that oil IPR,
   including early tests and installations with only one test. No new fit or
   training embargo is applied. Training-count controls are hidden. Dashed
   **BHP model** / **Oil model** lines and filled circles show model estimates.
3. **Refit earlier tests: next pump** fits oil productivity and WC/GOR from up to the
   selected number of last usable tests on the immediately preceding pump.
   Those inputs stay frozen while the next installation is predicted.
4. **Refit earlier tests: same pump** fits the first usable tests on each installation.
   Fitted observations use diamonds/dotted lines; predictions after the training
   embargo use circles/dashed lines. Their scores remain separate.
5. Click **Run comparison**. Progress and cancellation use the shared background
   job system. Toggle **Oil detail** and **PF rate detail** for clearer rate
   comparisons. Turning the overlay off retains the ordinary production view.
6. Select an installation in the score table to show its fixed well inputs
   and hardware flags, plus training details for chronological comparisons. Select a
   plotted test marker or use **Inspect a test or model miss** for actual/model
   BHP, oil, PF, formation liquid, operating pressures and solver failure text.

The initial history window is 24 months; 6, 12 and 60 months are available.
Chronological training counts are 3, 5, 10 or 20. Those two modes require at
least three distinct eligible training dates. Shorter display windows can use
the existing 24-month training history; a 60-month display also uses 60 months
for training. Narrowing the display no longer discards earlier training tests.
Run controls wait for the selected well's context. Changed
well, hydraulic model, settings, displayed source history or saved context hide
the old result immediately. A stale in-flight start is cancelled rather than
being attached to another selection. Expired jobs can be run again.

## Prediction contract

- Historical hardware uses exact UTC Date Set timestamps; tenure ends at the
  next Date Set. Same-size replacements are distinct installations. Unknown,
  ambiguous or undated tracker records remain visible and block unsupported
  attribution/transfer. A missing intermediate installation is never bridged.
- Test-day pressure observations are daily. Installation days cannot identify
  the operating pump and are excluded, with no installation assigned to them.
  The chronological modes separate training and prediction by a three-day embargo.
- Every-test replay keeps one saved/server-loaded Vogel oil IPR fixed. Each
  test's WC/GOR defines its water/gas mixture. Since WellConfig stores total
  liquid, the worker's equivalent total anchor is `saved_qwf * (1-saved_wc) /
  (1-test_wc)`, keeping the oil curve, anchor BHP and reservoir pressure fixed.
  Missing/invalid composition is an explained gap, never a present-day fallback.
  Test oil and BHP never re-anchor that inflow point by point. Its rows
  use phase/status `replay` and separate `replay_scores`, not held-out scores.
- The chronological modes reuse the existing frozen-training Vogel inflow
  calculation and keep earlier WC/GOR. Their predictions do not use current
  saved IPR anchors. All modes use clean `ken=.03`, `kth=.30`, `kdi=.40`, `fnz=1`;
  fitted pump wear never transfers to another installation.
- A worker receives the frozen well configuration plus test-day PF pressure
  and WHP. Held-out measured oil, BHP, PF rate and composition never enter its
  inputs. Missing held-out outcomes do not prevent a solve when controls exist;
  the unavailable comparisons are omitted from scores.
- Only actual well tests score oil. Failed solves have null predictions and
  count in coverage. Missing/excluded rows and gaps over 45 days break model
  lines. Every installation gets separate lines. PF above 20,000 BPD remains
  visible but is excluded from percentage scores, matching the earlier benchmark.
- Saved/current geometry, reservoir pressure, PVT and survey are historical
  priors whose past values have not been verified. Signed gauge offsets remain
  unverified. Nominal catalog conflicts are flagged; tracker diameters are not
  treated as wear measurements. Uploaded gauge previews are not replay inputs.
- Result records retain training test IDs, frozen configurations, actuals,
  controls, model identity, source, capture time and a snapshot identifier.
  Results explicitly set `validated_for_sizing=false`.

## Backend and resource limits

`POST /api/wells/{name}/pump-match` starts the job;
`GET /api/pump-match/{job_id}` polls it;
`DELETE /api/pump-match/{job_id}` requests cooperative cancellation.
Request/model bounds and job kinds are checked. Cancellation does not expose
an optimizer or calibration job through the history endpoint.

Data comes from the existing cached full well-test source, enriched tracker
and saved well context. One well's replay is capped at 1,000 tests and solves
in chunks of eight through the existing process pool/CPU tokens. A cancellation
waits for the current chunk, discards its result and releases the job slot;
a cancelled queued job reads no data. No new compute tier, worker pool, warmup
query or database persistence path was added.

Immutable replay results share the existing 64 MiB / 1,024-entry / one-hour
cache budget. Keys include observations, exact installations, requests,
configurations, replay source, physics source and survey contents. Copies are
independent. Historical results never substitute for an optimizer allocation.

## Verification and remaining work

User screenshots exposed why the first comparison often showed only a short
recent line. Across-pump validation could not train on the immediately preceding
installation (for example, B-28's short 13B run and B-30's October 2025 12C run).
The tooltip only showed `missing`, concealing the reason. It now says **No
prediction** or **Solve failed** with the explanation; days without a modeled
test are identified explicitly. Unsupported counts are grouped by reason, and
insufficient training retains its candidate count, dates and source installation.
Solver/source messages are escaped when rendered in an HTML tooltip.

The user then explicitly requested predictions at every test point. The new
every-test default removes training-history eligibility from that comparison;
missing controls, ambiguous installation days and failed solves remain explicit
gaps. It does not fill failures with zero or carry another date's prediction
into the tooltip.

`tests/test_pump_match.py` covers chronological separation, outcome leakage,
missing observations, ambiguous/same-size installations, circulation conflicts,
failed solves, immutable caching, model forwarding, API bounds and cancellation.
`web/tests/historyMatch.test.mjs` covers gaps, phase/installation boundaries,
UTC timestamps and date-specific hover values.

Initial every-test local checks: **1,982 Python tests passed**, with four existing warnings;
**17 frontend tests passed**; and the TypeScript/Vite production build passed.
`web/dist` was rebuilt with the matching source. The build retains the existing
large chart-chunk advisory. The first full Python run encountered Windows temp
directory permissions in the sandbox; authorized full-suite runs passed. The
final every-test run completed in 41.89 s.

`tools/check_pump_match_ui.py` is fixture-only Playwright QA. It verifies the
history entry link, controls, four synchronized SVG axes, training-window
inspection, explained gaps/failures, all three comparison modes, the default
every-test coverage, separate model/held-out legends, stale-result removal and
cancellation. It passed with zero browser errors at desktop and narrower widths.
Screenshots under `build/pump-match-*.png` are synthetic examples.

`tools/pump_match_replay.py` runs the actual solver on the trusted September 8
local snapshot. The final run attempted **166** predictions across six wells:
**138 solved, 28 failed**. All **121** successful predictions with the same
training tests as the preserved September 11 benchmark reproduced exactly.
The wider held-out eligibility differs from that benchmark; these are not a
replacement set of validation scores. F-73 still fails all 26 target tests.
This is adapter verification on frozen data, not independent field validation
or a hosted latency measurement. Earlier benchmark artifacts were preserved.
The chronological check after adding every-test mode is
`build/pump-match-benchmark-after-all-tests-2026-09-12.json`; it records the input
snapshot and source hashes and completed in 3.0 s locally. It still reproduces
all 121 comparable benchmark predictions exactly.

Every-test saved-snapshot checks are in
`build/pump-match-all-tests-b28-2026-09-12.json` (24 months: **69 attempted and
solved**, one test lacks usable controls) and
`build/pump-match-all-tests-b30-2026-09-12.json` (12 months: **44 attempted,
21 solved, 23 failed**). These are the September 8 saved inputs, not a read of
the user's current model. The B-30 failures and the larger B-28 oil errors show
the limits of applying a fixed present-day inflow/composition across history;
every-test coverage does not imply a successful or accurate physical prediction.
These initial saved-composition results are preserved historical milestones.
The subsequent user-requested measured-WC/GOR replay and numerical solver fix
are documented in the [fixed-IPR investigation](well_match_diagnostic_2026-09-12.md).
That revision passed **1,997 Python tests**, **17 frontend tests**, the
TypeScript/Vite build and fixture browser QA; `web/dist` matches source. The
tooltip and test inspector display the WC/GOR actually used, and the installation
inspector displays the fixed oil IPR in BOPD. Explicitly labeled optional refit
modes remain read-only and cannot change the saved IPR.
The earlier coverage diagnostic remains in `build/pump-match-coverage-2026-09-12.json`.

Remaining work: constrained fitting across installations; component-level
E-42/F-73 diagnosis; operating-point closure/regime diagnostics; parameter
identifiability; blind pressure-event/size/well holdouts; and optimization gain
uncertainty/rank stability. The new view and links expose evidence but do not
apply a new recommendation gate or change the economic objective.

No deployment or production-data write was performed. The local source and
matching frontend build must be deployed together when that action is requested.
