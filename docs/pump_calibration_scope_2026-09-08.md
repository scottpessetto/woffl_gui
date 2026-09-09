# Well inputs and installed-pump calibration

The Solver now saves well inputs independently from fitted jet-pump properties.
The user requested this after the MPE-42 / 13C fit: an improved BHP match must
not silently characterize every replacement pump as having the same fitted losses.

## Using it

1. **Save well inputs**, under IPR Anchor, saves the IPR anchor, reservoir pressure,
   WC, GOR, wellhead pressure and supported changed PVT inputs. It does not save
   ken/kth/kdi/nozzle-area factor.
2. **Calibrate to field data** fits the installed pump using saved well inputs
   and the installation's history/test data. Save edited well inputs before refitting.
3. **Apply to inputs** previews the fitted coefficients in the current session.
   **Save installed-pump calibration** saves that particular server fit, with its
   installation and quality diagnostics. Saving alone does not overwrite unrelated
   session edits. New optimization runs hydrate the saved fit automatically.
4. **Try clean replacement** resets ken/kth/kdi to 0.03/0.30/0.40 and nozzle-area
   factor to 1.0, keeping well inputs. **Restore installed pump** restores the
   current installation's saved fit. Unsaved fits can be applied again from their
   result card. Selecting a different catalog size also resets pump coefficients.

The Solver displays installed/replacement scope, save status and session changes.
Read-only apps display a disabled pump-save button with an explanation. Expired
server jobs must be refitted before saving (the existing job retention applies).

## Identity and persistence

A fit is bound to well, tracker nozzle/throat, exact **Date Set**, and physics
model version. The calibration result retains the exact tracker timestamp even
though daily-history gating uses the calendar date. A fresh Databricks tracker
read is required on save; missing data or the bundled spreadsheet cannot certify
an installation. A same-size changeout therefore invalidates the previous fit.

The existing `mpu.wells.woffl_eng_comment` ledger stores one self-contained,
versioned JSON record under context `pump_calibration_v1`. No new table, DDL or
numeric property IDs are required. Coefficients, identity and quality are committed
in **one parameterized INSERT** through the existing gated write path and request
identity. The encoder rejects records exceeding the existing 500-character limit
before calling the comment writer, which otherwise truncates long human notes.
Coefficient precision is preserved; diagnostic metrics are rounded for storage.
A failed write does not claim success or evict caches.

Compact record keys: `v` schema version, `n/t` catalog nozzle/throat, `i` installation
timestamp, `m` physics model, `k` ordered `[ken,kth,kdi,fnz]`, and `q` quality.
Quality retains BHP/PF/delta-BHP RMS, point count, parameter bounds and modeled/
measured response slopes. Fits with a bound hit, BHP RMS >50 psi, PF RMS >10%, or
response mismatch >0.03 psi/psi are displayed as provisional. These are review
flags, not statistical confidence intervals or an acceptance certification.
Single-point fallback fits remain provisional and preserve their input nozzle
area; one BHP observation cannot identify area.

A cached fleet SELECT reads the latest record per well. Only a matching current
installation and physics model supplies coefficients. Malformed, stale or
unavailable records fall back visibly to reference coefficients. Legacy `jpfric_*`
numeric rows remain in history but are not automatically activated, because they
do not establish installation identity. Refit/re-save them through the new action.
The older low-level numeric save API remains for compatibility; the web well-save
endpoint no longer forwards pump coefficients.

## Optimization behavior

Application WellConfig opts into scoped pump candidates. Resize optimizers,
single-well batches, PF-pressure sweeps and CFP response surfaces distinguish
**keep installed** from **clean replacement**, including the same size. Replacements
use reference coefficients and catalog nozzle area. Fixed-current operations use
the installed candidate. Future wells inherit donor well inputs without donor
pump losses, wear or installation identity.

Candidate identity survives performance lookup, MILP, MCKP, parsimony and fixed
scenario scoring. A replacement cannot supply its label while the installed row
supplies its numbers. Identical same-size modeled outcomes favor keeping the
installed pump. The response cache already keys every WellConfig field, so the
scope flag and changed coefficients invalidate prior nodes. Legacy library callers
retain previous behavior unless they opt into scoped candidates (upstream patch 43).

No energy equations or host sizing changed. Reference clean-pump predictions
remain model estimates: calibration can absorb WC/test error, inflow error and
imperfect physics. A fitted area 1% above catalog is not evidence of measured wear.
Old fleet audit artifacts still describe their frozen inputs; this hydration
change is not a new field validation of their predictions.

## Verification

- `tests/test_pump_calibration_scope.py`: installation/model binding, corrupt
  records, fresh-save checks, write gating/attribution/failure, full-precision
  persistence, both allocation engines, fixed scenarios and CFP same-size choices.
- Real installed/clean single solves agree with batch and PF-pressure results.
- Existing well-hydration/save tests now enforce the separate persistence paths.
- `web/tests/pumpScope.test.mjs`: same-size replacement, size edits, stale fits,
  installation changes and restoration after saving, without losing WC edits.
- `tools/check_pump_scope_ui.py`: optional browser check; all saves are intercepted
  fixtures. Verifies both payloads, applied/saved status, replacement/restore,
  desktop/narrow rendering and absence of browser errors.

Verification uses no production writes. Browser screenshots under `build/` use
fixture data and are not another MPE-42 field validation.

Final local checks: **1,861 Python tests passed** (4 existing warnings),
**8 frontend tests passed**, TypeScript/Vite production build passed, browser
checks passed with 0 browser errors and both saves intercepted. The saved-fit
readiness board also uses verified scope, and future rows explicitly show clean
pump assumptions. Logs: `build/pump-scope-tests.log`, `pump-scope-web-build.log`,
`pump-scope-ui.log`. Changes are local; no deployment was performed.
