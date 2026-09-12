# Edit, preview and save well inputs for optimization

September 12, 2026. The user needed to edit the well fit, compare BHP/oil across
the production history, and save the result for optimization. Save controls
sometimes disappeared. This change completes that workflow in the local app;
the source and matching `web/dist` have not been deployed.

## User workflow

1. Select a well in Solver or JP History and edit the sidebar inputs.
2. On the production plot choose **Show model match**, **Every test (well fit)**,
   then **Well inputs: Current edits (preview)**. Set the lookback and run the
   comparison. Review BHP, oil, coverage and explained failures by installation.
3. Use the **Well inputs** bar at the top to review values, optionally add a note,
   and **Save well inputs**. The bar stays visible while the main page scrolls.
4. Start a new optimization run after the save confirmation. The optimizer loads
   the saved well context; existing results remain snapshots of their run inputs.

The bar displays the count of supported values differing from the loaded
database. Save remains visible but disabled with a reason in read-only mode,
while context/access is loading, or for an invalid oil IPR. Editing and preview
remain available in read-only mode. A failed save cannot show the successful
optimization confirmation. Edits made while a save is in flight are preserved.

Well inputs and installed-pump calibration remain separate. Save the well inputs
first, then **Calibrate to field data**. A saveable completed result offers
**Save installed-pump calibration**, which persists the server fit and its
installation/model identity. Apply only previews it in the session. Refer to
the [user guide](optimization_user_guide.md) and
[pump-scope contract](pump_calibration_scope_2026-09-08.md).

## Model and persistence contract

- Historical preview accepts only supported well-save values: total-liquid IPR
  rate, anchor BHP, reservoir pressure, WC, GOR, WHP, and supported changed
  formation temperature/bubble point. The frontend shares one payload builder
  for preview and save. Unknown preview fields and invalid domains are rejected.
- Preview applies one explicitly edited oil IPR across the entire history.
  Each test supplies its measured WC/GOR and PF/WHP. Historical hardware comes
  from the tracker; losses remain clean reference. Observed oil/BHP/PF rate do
  not anchor the model separately at each test. Changing inputs hides stale
  results immediately. No automatic IPR shifts, fitting or writes are added.
- Optional chronological refit modes keep their prior contracts and reject
  edited-input payloads. Their fitting remains explicit and read-only.
- The preview response/cache records the requested inputs and effective well
  configuration; edited and saved comparisons cannot share the wrong result.
  Existing job admission, pool, CPU tokens and bounded cache remain in use.
- Save uses the existing gated, append-only property-history path. It writes
  neither pump losses nor tracker hardware. PF pressure and unsupported sidebar
  settings remain session/run inputs. The hydraulic model is persisted through
  the separate installed-pump calibration record.
- Solver anchor pin/unpin behavior is retained; JP History saves values without
  changing the pin. Save and Clear saved IPR share pending-write state so the
  now-separated controls cannot issue conflicting operations concurrently.

## Defects corrected

The old IPR controls removed the whole well-save block when write access was
false or had not loaded. `SaveWellInputs.tsx` provides one persistent action on
each page with explicit availability, review and result text. The history
comparison previously read saved server inputs only; `edited_inputs` now permits
a read-only preview before committing the supported changes.

Successful saves invalidate well context, saved-IPR metadata, property history,
Well Database and pad readiness. Context refresh updates the saved seeds and
provenance while preserving session/manual ownership and newer edits. It also
retains existing pump installation invalidation and restoration behavior.

The canonical well-characteristics cache previously survived value saves.
Reservoir pressure, temperature or bubble point could therefore remain stale
in Well Database or a newly constructed optimization configuration. Successful
value saves now invalidate that fleet cache; a successful reservoir-pressure
lock does too. Failed writes and pin-only changes do not. The next consumer
refreshes one shared fleet read rather than issuing a read per well.

Saved qwf, pwf, reservoir pressure, GOR and WHP also lost fractional precision
through integer casts during hydration. Those overlays and formation-temperature
hydration now retain saved numeric precision. Existing domain limits still apply.

## Verification and limits

- Full offline suite: **2,006 Python tests passed**, four existing warnings,
  40.65 seconds. The authorized run needed Windows temp-directory access.
- **18 frontend tests passed**, TypeScript check and production build passed.
  `web/dist` matches source. The existing chart-chunk advisory remains.
- `tests/test_web_save_ipr.py` follows an in-memory save through recorded property
  rows, saved-IPR assembly, fresh well context and the real optimizer config
  builder. It verifies fractional IPR/fluid values, changed temperature/bubble
  point, cache refresh and failed-save behavior without a database connection.
- `tests/test_pump_match.py` verifies edited preview inputs, actual test WC/GOR
  and controls, historical hardware, unchanged saved context, separate caching,
  invalid-domain/unknown-field rejection and explicit-mode restrictions.
- `web/tests/pumpScope.test.mjs` checks refreshed saved baselines while preserving
  edits made during a save and ignoring a different well's stale context.
- `tools/check_well_input_save_ui.py` passed: two preview runs, four intercepted
  saves, one intercepted anchor clear, and zero browser errors. It covers
  read-only access, dirty state, stale preview removal, successful/failed saves,
  concurrent edits, Save/Clear exclusion, client navigation and desktop/narrow
  sticky controls. Screenshots are under `build/well-input-save-*.png`.
- `tools/check_pump_match_ui.py` and `tools/check_pump_scope_ui.py` also passed
  with zero browser errors. Plot axes, tooltips, cancellation, separate saves
  and installed/clean restoration remain covered. All API data/saves are fixtures;
  pump-scope single solves use the local API with a Custom well and no lifespan.

No production write, write-gate change, deployment, compute-tier change or new
physics equation was made for this workflow. The prior numerical replay evidence
in the [fixed-IPR investigation](well_match_diagnostic_2026-09-12.md) was preserved.
Shared multi-installation pump-loss fitting and independent qualification of
optimization gains remain unfinished; the comparison is retrospective evidence.
