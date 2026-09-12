# Recovered Fable review — September 11, 2026

**Implementation update:** the ten numbered findings below and the additional
append-only SQL validator gap have now been fixed locally. The original review
and baseline below describe the pre-fix state. Final validation and implementation
details are recorded at the end of this document.

Reviewed revision: `ed85da2` (`docs cleanup`). The Git working tree was clean
before the initial review. That review changed only this report; implementation
followed the user's subsequent authorization. No deployment or production writes
were performed in either phase.

Fable's September 11 session and nine reviewer transcripts survived in local
Claude session `696f9359-4409-4f2a-96e7-447c48d8998c`. Six original reviewers and
three frontend subreviewers stopped at the session limit without final reports.
Their scratchpad contained reproduction scripts and the 1,861-pass baseline log.
The findings below were checked against current source and reproduced offline
or traced through their callers. Partial reviewer suspicions are not treated
as completed review findings. This is recovery and verification of those leads,
not a claim to have completed every reviewer's entire assigned scope.

## Fix first

### 1. [P1] Read the refreshed tracker data when saving a calibration

Location: [pump_calibration.py](../server/services/pump_calibration.py), lines 110–114.

`save_fit` assigns the return value of `cache_refresh()` to `df` and passes it
to `get_current_pump`. The real decorator returns a boolean, not the fetched
DataFrame. A valid completed fit therefore raises
`TypeError: 'bool' object is not subscriptable` before it can save.

Reproduced with the real cache decorator and tracker selector, a mocked tracker
SELECT, and write functions blocked. The existing `save_case` fixture in
`tests/test_pump_calibration_scope.py` masks this by making `cache_refresh`
return a DataFrame and mocking the tracker selector too.

Fix: provide a synchronous fresh-data read with a defined concurrency contract.
Do not simply treat `False` from a refresh already in flight as fresh data.
Test with the real cache wrapper and selector while mocking only storage boundaries.

### 2. [P1] Preserve and consistently serialize installation timestamps

Locations: [wells.py](../server/services/wells.py), line 401;
[EventCalibration.tsx](../web/src/pages/solver/EventCalibration.tsx), lines 32–35;
[params.ts](../web/src/state/params.ts), lines 195–197.

Well context formats the tracker timestamp through `frames.json_value`, which
reduces it to `YYYY-MM-DD`. Calibration results preserve a naive ISO datetime
from `calibration_points.points_for_well`. Both the button guard and store action
compare JavaScript timestamps. In Anchorage, `2026-08-10` parses as midnight
UTC, while `2026-08-10T00:00:00` parses eight hours later. The same installation
fails verification: Apply and Save are disabled, and the store rejects Apply.
A non-midnight Date Set also loses its time even in a UTC browser.

Reproduced under Node with `TZ=America/Anchorage` using the exact emitted string
forms. Fix: serialize the exact installation identity consistently at both API
boundaries, including an explicit agreed timezone. Do not fix this by comparing
calendar dates; same-day, same-size changeouts must remain distinct. Add tests
for Anchorage, UTC, and non-midnight tracker timestamps.

### 3. [P1] Keep solver failures in Header Impact verdicts

Location: [header_impact.py](../server/services/tools/header_impact.py), lines 979–989.

`_solve_at_whp` returns NaN rates and an error string for failed batch solves.
`solve_jp_row` discards that error, calls `_verdict` without its `error` argument,
and emits `Error=""`. A failed prediction is consequently labeled
`no response`, which is an engineering conclusion the calculation did not establish.

Reproduced through `solve_jp_row` with failed-solver result fixtures:
`Verdict='no response', Error='', DeltaOil=NaN`. Fix: combine the two error
strings with the existing `solver_error_note` helper, pass the result to
`_verdict`, and preserve it in the output row. Test this worker call path,
not only `_verdict` in isolation.

### 4. [P1] Carry test-day PF pressure through the washout worker

Location: [jp_washout.py](../server/services/tools/jp_washout.py), lines 139–152.

The scan builder includes `PfAtTest`, but `calibrate_one` omits it from the
returned row. `scan` therefore always uses the infrastructure cap fallback,
even when measured test-day PF is available.

Reproduced through the actual worker and `scan` with a stubbed physics result:
required PF 3,200 psi, measured PF 2,600 psi, cap 3,400 psi yields
`Flagged=False` and `FlagBasis='vs limit (no test-day PF)'`. The intended measured
ratio is 1.231, above the 1.15 flag threshold. Fix: preserve `PfAtTest` in the
worker's shared base row and test both measured-PF and missing-PF branches.

## Other confirmed application defects

### 5. [P2] Skip trims that free no water

Location: [pad_optimize.py](../woffl/gui/pad_optimize.py), line 1352;
fallback construction at line 1739.

An unmodelable well with positive measured oil and zero/missing measured PF
gets hold and shut-in options with identical zero water use. If any other well
puts the pad over budget, `_trim_to_budget` divides by zero while evaluating
this pair and crashes the whole choke plan. Server hydration can turn a missing
test PF median into zero, so this is reachable through normal input preparation.

Reproduced both directly and through `run_choke_optimization`: modeled well
4,000 BPD PF, budget 3,500 BPD, plus an unmodeled 50 BOPD/0 BPD PF well.
Fix: reject nonpositive water-release steps or normalize fallback options
through the same dominance logic. Preserve the unknown-measurement distinction
in reporting rather than pretending a missing PF measurement is evidence of zero use.

### 6. [P2] Refresh the longest test-history window before deriving shorter windows

Locations: [warmup.py](../server/warmup.py), lines 266–267;
[tests.py](../server/services/tests.py), lines 46–51.

The warm pass submits the 6-, 12-, and 24-month refreshes concurrently. On a
cold cache this issues three fleet queries instead of one. On a warm cache,
the short windows can copy the old 24-month entry and receive a fresh retention
period before the new 24-month query completes. Default shorter-window callers
continue seeing old tests until another refresh, despite a successful warm pass.

Reproduced with a versioned fake SELECT: after warm refresh, cached generations
were `{6: 1, 12: 1, 24: 2}`. The cold pass issued three queries; longest-first
serial refresh issued one. Fix: await the longest refresh before deriving and
priming the shorter windows from that same new snapshot. Test changed source
data, not merely query counts.

### 7. [P2] Move workbook parsing off the API event loop

Locations: [gauge.py](../server/routers/gauge.py), lines 28–57;
[tools.py](../server/routers/tools.py), lines 137–160.

Both upload handlers are `async def`, but after reading the bytes they call
synchronous openpyxl/pandas parsing and aggregation directly. During that work,
the application's single uvicorn event loop cannot handle unrelated requests
or job polls. Reading the upload asynchronously does not offload its parser.

Verified by tracing both handlers into the synchronous parsers. An offline
synthetic gauge-workbook probe measured 0.38 s parsing 17,280 rows (0.7 MB)
and 4.24 s parsing 172,800 rows (7.4 MB). These are local timings, not hosted
performance measurements. Fix: offload parsing and
combination through a bounded worker path, preserving Medium concurrency limits.
Verify another request or an event-loop heartbeat progresses during parsing.

### 8. [P2] Accept valid mixed text-date formats in sample logs

Location: [oiw_samples.py](../server/services/tools/oiw_samples.py), line 163.

The date-column conversion infers one format for the series and coerces other
valid formats to NaT. A hand-entered log containing `05/03/2026`, `2026-05-04`,
`May 5 2026`, and `5/6/26` retains only the first row and reports the other three
as unparseable. This drops legitimate observations from the daily comparison.

Reproduced with the installed pandas version through `_clean`. Fix: explicitly
support mixed formats with the intended US month/day convention, while retaining
the existing plausible-date filters. Test the complete workbook upload as well
as the cleaner. Ordinary Excel datetime-cell handling passed the recovery probe.

### 9. [P3] Use calendar-day boundaries for separator coverage

Location: [sep_oil_loss.py](../server/services/tools/sep_oil_loss.py), lines 381 and 403.

The daily rollup advances midnight by a fixed 24-hour Timedelta and tests
coverage against 23.5 hours. An Alaska spring-forward day has 23 hours; a complete
day is reported as 24 covered hours and marked partial. The fall-back day has
25 hours and also needs its actual local boundary.

Reproduced with uninterrupted minute data spanning March 8, 2026 in Anchorage:
`hours=23.0, covered_hours=24.0, partial=True`. Fix: use the next local calendar
midnight and compare against that day's actual duration. Preserve elapsed-time
barrel integration rather than forcing all days to 24 hours.

## Write-guard findings

### 10. [P1] Exclude dotenv gate names case-insensitively on Windows

Location: [databricks_client.py](../woffl/assembly/databricks_client.py), lines 133–136.

`_ENV_GATE_KEYS` is uppercase, but dotenv preserves key spelling and the filter
is case-sensitive. Windows `os.environ` is case-insensitive. Consequently a
lowercase or mixed-case gate entry in `.env` passes the exclusion and can enable
the real gate on the first local connection, when no explicit shell value is set.

Verified using an unrelated probe environment key for Windows case folding and
a synthetic dotenv file for exclusion behavior. Neither real gate was enabled
and no connection was opened. This is a conditional defect; the user's actual
credential file was not inspected. Fix: compare normalized key names against
the gate set, and test mixed-case entries with fake environment/connection objects.

The SQL validator also accepts `INSERT OVERWRITE` and `INSERT ... REPLACE WHERE`
because it checks only the leading INSERT keyword and statement chaining.
This was reproduced by calling the validator only, without SQL execution.
Current application callers use fixed append-only INSERT templates, so this is
an additional defensive-hardening gap, not a demonstrated destructive UI path.
Restrict the executor's accepted grammar to its intended append-only forms.

## Discarded lead and verification limits

Fable's low-flow E-Pad probe returned the setpoint where the raw frontier is
absent. `PadPlant.delivered_header` explicitly models bypass/recirculation there;
that observation alone is not a defect and is not included in the fix list.

No new core-physics defect was established by this recovery. The interrupted
physics review remains incomplete; a green regression suite does not certify
field-model accuracy or complete a whole-codebase review.

Validation on September 11:

- Full offline Python suite: **1,861 passed**, four warnings, 44.28 s.
- Frontend Node tests: **8 passed**.
- `npm run typecheck`: passed.
- Offline probes reproduced the application findings above using synthetic
  data and mocked storage/solver boundaries where applicable. The save probe
  retained the actual cache decorator and tracker selector.
- The restricted Python run encountered errors and stalled; it was interrupted.
  The unrestricted offline rerun passed. Node's initial sandbox run failed to
  spawn test processes (`EPERM`); its approved rerun passed. These initial runs
  are not counted as additional application regressions.
- No frontend build was needed for this documentation-only review.

The existing green tests do not cover these integration and edge cases. No
bug fixes or new regression tests were applied during the initial review.

## Implementation following the review

The user authorized fixes after the review. All ten numbered issues were addressed:

- Saves now call `datasources.jp_history_fresh()` for a synchronous, enriched
  tracker read that bypasses cached data and in-flight refreshes. Tests retain
  the real cache wrapper, tracker selector and enrichment, mocking the SELECT
  and write boundaries. A failed fresh read cannot fall back to an old snapshot.
- Context and calibration responses use the ledger's exact UTC timestamp
  serialization. Naive tracker timestamps retain the existing UTC identity
  convention. Same-day replacements remain distinct. The sidebar still shows
  a compact calendar date; the full timestamp remains in the identity data.
- Header Impact preserves failures from either solve and uses the existing
  failure verdict. Washout worker rows preserve measured test-day PF pressure.
- Choke trimming ignores choices that free no water, avoiding the zero-division
  crash without shutting in a well for no water benefit.
- One warm target fetches the longest test window and primes every configured
  shorter window from that new snapshot. The existing cache invalidation/version
  and in-flight prime guards remain in force. Failed fetches preserve old entries.
- Both XLSX endpoints run as synchronous FastAPI handlers in its thread pool,
  with parsing under the shared `pool.cpu_slot()` budget. Reads are capped at
  the permitted byte count plus one. Tests verify the API event loop can progress
  while each parser is still running.
- OIW parsing accepts mixed US date formats and Excel datetime cells while
  keeping plausible-date and valid-PPM checks.
- Separator coverage uses local calendar midnights and actual 23/24/25-hour
  durations; elapsed-time barrel integration is unchanged.
- Dotenv gate exclusions compare uppercase names. The SQL executor accepts only
  append-only `INSERT INTO ... VALUES` templates with bind markers, numbers or
  NULL, rejecting overwrite/replacement, SELECT sources and expressions. Existing
  application writes remain parameterized. Tests use fake environments/connections.

Core physics and compute sizing are unchanged. No deployment or production
property/calibration writes were performed. The regression additions live in
`tests/test_recovered_review.py`, the relevant existing Python test modules and
`web/tests/pumpScope.test.mjs`.

Final local verification after the fixes:

- **1,887 Python tests passed** (26 added cases), four existing warnings, 67.72 s.
- **9 frontend tests passed**, including exact installation identity under
  Anchorage and UTC, with same-day replacement rejection.
- TypeScript/Vite production build passed; matching `web/dist` artifacts rebuilt.
  The existing large-chunk advisory remains. Entrypoint and relative JavaScript
  chunk references resolve to files in the built output.
- `git diff --check` passed. No live data access was needed for verification.
