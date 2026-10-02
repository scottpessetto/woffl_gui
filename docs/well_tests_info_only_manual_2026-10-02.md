# Info-only well tests and the manual (LRS) test - 2026-10-02

Built and checked locally on 2026-10-02, then committed (12e81a0) and deployed by
the user the same day. Every check below ran against a local read-only server:
hosted behaviour and timings were not verified in-session, and no production
property record was written while building it. Not exercised with a real
write: Save well inputs with the engineer's own (LRS) test as the anchor.

## Why

The Solver's test list for MPE-48 stopped at 2026-09-10 while FDC showed eight
newer tests. Databricks was current. The fleet query in
`woffl/assembly/well_test_client.py` kept `allocated = True` rows only, and
allocation is a monthly accounting pass: every MPE-48 test after 9/10 was still
info-only (FDC "Info Only = Yes"). In the 24-month window 7,641 of 24,072
distinct tests are allocated.

## What `vw_well_test` holds (probe 2026-10-02, 24 months, MPU)

- An allocated test is a COPY of an info-only SCADA row: new `wt_uid`, stamped
  at midnight of the same `wt_date` day, identical rates. 7,484 of 7,647
  allocated tests had a same-day twin; 159 had none (typed in by hand).
- SCADA repeats rows: 2,025 info-only rows duplicated another on the same day.
- About a fifth of info-only rows share a well-day with a DIFFERENT test, so a
  date does not identify a test.
- `wt_date` is often +1 day from the test (FDC 9/29 is `wt_date` 9/30).

## Info-only tests

- The query returns every test with the `allocated` flag.
  `_collapse_duplicate_tests` keeps one row per (well, day, oil, water, PF
  rate): the allocated copy when there is one, else the latest. Folded
  `wt_uid`s ride on the survivor as `dup_wt_uids`.
- `tests.fetch_all_well_tests` caches all of them. `tests.tests_for_well` and
  `tests_json` return ALLOCATED tests unless `include_info=True`.
- Info-only tests reach two places: `GET /wells/{name}/tests` (the Solver's
  list, anchor picker, chart and Sensitivity target) and the saved-pin lookup.
- Everything that picks tests on its own stays on allocated tests: recent and
  median anchors, well-context seeding, calibration points, evidence, match
  health, optimizer hydration, pump match, common oil IPR, header study, JP
  washout and the tools frames. Direct readers of the fleet frame wrap it in
  `tests.allocated_only`.
- A fit runs on the allocated tests plus an info-only test only when it is the
  specific anchor (`ipr._fit_tests`). The reservoir-pressure fit never sees the
  other info-only tests.
- **Show info-only tests** (IPR chart header, on by default, chart-only):
  hides or draws the info-only triangles. It is forced on while the fit uses
  them, so the curve never leans on points that are not on screen.
- **Toggle** "Use info-only tests in the IPR fit" (IPR chart header, off by
  default, per well in localStorage beside the exclusions) sends
  `include_info_only`. `ipr._fit_tests` then returns every test: info-only
  tests set the fitted reservoir pressure and Most recent / the medians may
  anchor on them. Excluded tests still stay out. Flipping it applies the new
  fit to the sidebar like an anchor change. It is off by default because the
  optimizer and well-context seeding stay on allocated tests. MPE-48 live:
  off = 9 tests, ResP 787 psi, anchor 9/10; on = 71 tests, ResP 717 psi,
  anchor 9/30, and the fit reports weak (R2 -0.50).
  Under a Manual point no fit runs; the chart says so when the box is ticked.
- Fixed alongside: switching Manual point -> Most recent did not apply the
  fit to the sidebar (a Manual point shares the "recent" query key, so the
  cached fit object never changed and the apply effect never re-ran). The
  effect now also depends on the anchor.
- The specific anchor is identified by `anchor_wt_uid` (request and
  `coeffs.anchor_wt_uid`), with `anchor_date` as the fallback.
- A pin saved against an info-only `wt_uid` still reads "applied" after FDC
  allocates that test: `ipr.pin` finds the allocated copy through
  `dup_wt_uids` and reports its `wt_uid`.
- UI: Type column (Allocated / Info only / Manual), "info only" in picker
  labels, triangles on the IPR chart, and "Most recent allocated" as the mode
  label when info-only tests are listed.

Not changed: `server/services/history.py` (pump-history strip, report card)
still reads allocated tests only.

## The engineer's own test (LRS sheet or typed in)

A panel in the Solver's IPR Anchor card, shown for every named well.

- The test holds its OWN numbers (`state/manualTest.ts`: date, oil, water,
  BHP, GOR, WHP, PF rate, PF pressure, source). It is never the sidebar's
  values by reference. Session-only: it has no Databricks row.
- It is a test like any other: a "Manual" row in the table, a square on the
  IPR chart, "... | your test" in the Specific test picker, the comparison
  target, the Sensitivity target, and excludable.
- It rides in the fit request (`manual_test`, `anchor_manual`). The server
  appends it to the fit frame as an allocated-equivalent row
  (`ipr._with_manual_test`), so it shapes the reservoir pressure and Most
  recent / the medians may anchor on it. The response reports
  `coeffs.anchor_manual`; the in-frame marker uid never leaves the server. A
  test with no GOR does not seed the sidebar GOR.
- **Load LRS test sheet** posts the workbook to `POST /api/lrs/parse`
  (`server/services/lrs_test.py`). Cells are found by label, not address.
  Read: well, test date, duration, location, WHP, PF rate, PF pressure,
  corrected formation oil / water / fluid rate, corrected water cut, GOR.
  "AVERAGE SPIN OUT W/C" (total cut with power fluid) is never used. A GOR
  outside 20-10,000 scf/stb (the sheet reports 1.8 when no gas rate is
  measured) is dropped from the test.
- Loading a sheet makes the test the IPR anchor when it has a BHP: the curve
  runs through it and the reservoir pressure is fitted through it. The fit's
  seeds, not the loader, write the sidebar.
- The sheet has no BHP. The test takes the well's daily gauge reading on the
  test date, else the latest within three days before; otherwise the BHP is
  left blank and the test cannot anchor until one is entered.
- **Enter by hand** starts a test from the sidebar's inflow point; **Use as
  IPR anchor** anchors a typed test. Editing an anchored test re-applies the
  fit.
- A sheet naming another well loads only after an explicit confirmation.
- Saving with this anchor cannot pin (no FDC `wt_uid`): the values are saved,
  an existing pin is cleared, and the save note defaults to the test's
  provenance. The well reopens as a manual point. **Manual point** itself is
  unchanged: the sidebar's own qwf / pwf, no fit.

## Previewing an anchor and reverting

Changing the anchor is a session preview: the fit is laid over the sidebar
and nothing is written until Save well inputs. **Revert to saved** in the save
bar puts back the loaded values of the saved well inputs plus PF pressure and
circulation direction (`params.revertToLoaded`), and the Solver returns the
anchor to what the well opened on (pin, manual point, or most recent). The
pump on the bench and other session settings are left alone.

Limits: `.xlsx` / `.xlsm` only (no `.xls` reader is installed). The parser was
written from a screenshot of the MPE-48 sheet of 2026-10-02 and checked
against a workbook built to that layout. The user then loaded a real LRS sheet
in the local build on 2026-10-02 and confirmed it read correctly (one sheet).

## Checks

- 2,461 Python tests, 75 frontend tests, production build.
- Live read-only: MPE-48 lists 71 tests (62 info-only), newest 2026-09-30;
  recent anchor stays on the 9/10 allocated test; the 9/30 info-only test
  anchors when picked. The 24-month fleet query plus slices took 12 s.
- Headless Chrome against a local read-only server on MPE-48: every anchor
  mode moves the sidebar; Revert returns 1,784 BLPD / 598 psi / ResP 757 and
  the Manual point; the LRS sheet anchors at 1,785 BLPD / 597 psi with ResP
  fitted to 687; a corrected test BHP re-anchors; no write request, no page
  error.
