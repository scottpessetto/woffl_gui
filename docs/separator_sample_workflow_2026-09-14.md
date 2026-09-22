# Separator carryover and field samples - September 14, 2026

## Shutdown recovery

The separator work survived the computer shutdown as uncommitted source,
tests and built frontend files on `main`, based on `63af4c8`. This record was
missing even though the recovered services referenced it. The recovery pass
reviewed those changes, corrected remaining descriptions of the scenario as
raw meter output or a proven bound, retained dates in comparison exports,
updated the architecture/index documentation and rebuilt the frontend.

This is a local implementation. No deployment, production-data write, meter
calibration or operating-setting change was performed. The earlier local
`build/check_separator_live.py` script exists, but its intended output directory
contained no saved snapshot or summary at recovery. No live-result claim can
be recovered from that directory.

## Current workflow

Open Scott's Tools > Separator Oil Loss (`/tools/sep-oil-loss`). The page has
historian scenarios and a separate **Validate with field samples** section.

1. Import a CSV/XLSX log or enter samples manually. Preserve Date, Time
   (Alaska), Location, PPM/concentration, Sampler, Method and Notes. The CSV
   template uses these headers. XLSX accepts the legacy second-row header
   or an ordinary first-row header on the selected worksheet.
2. Select the sample point and confirm the lab units. Unknown units retain
   concentrations but withhold oil fractions and rates. `ppmv` means oil
   volume divided by total sample liquid volume. `mg/L` requires oil density
   in kg/m3 on the sample basis. Mass ppm (mg/kg) is not interchangeable.
   One units selection applies to the imported file; use separate imports
   for different lab bases.
3. Enter the meter-to-tap transport delay if known. Comparison time is sample
   time minus that delay. Only V-5317 is treated as the same upstream stream
   as Red Eye. P-5417C is downstream of the deoilers; other taps remain
   unverified. Neither can validate the upstream meter in this tool.
4. Compare samples. Each upstream grab needs an unambiguous local time,
   confirmed units, and actual preceding flow/WC reports no more than
   15 minutes old. Future reports are never used. Invalid readings and flow
   at or below 1,000 BPD remain unpaired. Missing, ambiguous or nonexistent
   DST times remain visible but unpaired. Date-only samples are retained.
5. Review meter-minus-sample oil concentration in percentage points. Positive
   means the meter indicates more oil. Matched rates use that pair's measured
   flow, assumed total liquid. The preceding two-minute WC range flags changes
   above five points; those pairs remain visible but are excluded from the
   reported median. This median is a diagnostic, not a calibration or an
   independent accuracy result.
6. Download the manual log and/or comparisons. Data stays in page memory and
   is not persisted on the server. Changes to comparison settings hide previous
   output immediately; an older in-flight response cannot restore it. Historian
   failure retains the parsed samples with an explanatory note.

Numeric zero is retained as a reported zero. Text such as `<5` is excluded from
arithmetic and should stay in the source log. Invalid rows are counted. Missing
samples at the selected location produce an empty result and the available taps.

## Calculation contract

The historian service uses four tags: `MPU_FI_5365` flow, `MPU_AI_5317` WC,
`MPU_LIC_5365CV1` controlled level and `MPU_LC5365SP1` setpoint. It step-holds
every tag transition on the union timeline between the first and last flow
reports, splitting intervals at Alaska midnight and flow-report expiry.
Intervals require finite flow above 1,000 BPD, finite WC in 0-100%, and flow
report age below 15 minutes. A valid 0% WC reading is retained. WC itself is
exception-held without an age gate in the integral; no quality tags establish
that a held reading is healthy. Sample pairing uses the stricter two-tag age
criterion described above.

With `Q` as total liquid BPD and `WC` in percent:

| Output | Calculation and interpretation |
|---|---|
| Raw indicated rate | `Q * (1 - WC/100)`; meter indication, not proven loss |
| Excursion reference | Causal trailing 24-hour p95 WC on a fixed one-minute clock, clipped to 80-100%; no future backfill |
| Uncapped excursion | `Q * max((reference - WC)/100, 0)` |
| Reference subtraction | Raw indication minus uncapped excursion; may remove meter bias or sustained real oil carryover |
| Field-capped scenario (`upper`) | Excursion capped at the entered field oil rate |
| Fraction-capped scenario (`lower`) | Also capped at `Q * max_oil_frac`; always at or below the field-capped scenario |
| Barrels | Sum of interval rate times valid hours divided by 24 |

The caps are scenario assumptions. They are not confidence bounds or established
physical bounds during vessel inventory changes. Oil in the first-stage water leg
can still be recovered downstream. A low plateau alone cannot distinguish meter
bias from sustained carryover; the raw volume and reference subtraction therefore
remain visible in period cards and daily tooltips.

Rolling periods end at the latest flow report; calendar-day rows use Alaska
midnights and actual 23/24/25-hour day lengths. Partial coverage and excluded
intervals are disclosed. Daily field shares are withheld below one valid hour.
Charts are downsampled, while volume integration uses all valid intervals.
Event labels describe indicated level relative to setpoint, with `unknown` for
incomplete evidence. They do not identify root causes. Daily event counts use
event start dates, while daily integrals split at midnight.

For samples, `f = ppmv / 1e6` or `f = (mg/L) / (oil_density_kgm3 * 1000)`.
Fallback sample-day rate is `Q * f` for total-liquid flow or `Qw * f / (1-f)`
for water-only flow. These are unweighted means over collected grabs, without
an invented duty cycle. Daily sample `bbl` is always null. The workbook's own
fixed-flow BOPD column is ignored. Paired rates use the historian flow instead
of this fallback.

## Verification and remaining work

- Focused separator/sample/recovered-review tests: **76 passed**.
- Full offline Python suite: **2,217 passed**, with three existing physics/
  deprecation warnings. The initial sandbox run could not access Windows
  temporary folders or start the process-pool tests; the approved unrestricted
  rerun passed without changing the tests.
- Frontend: **29 Node tests passed**; TypeScript/Vite production build passed.
  Vite retains its existing large-chart-chunk advisory. `web/dist` was rebuilt.
- Browser fixture check: **passed with zero browser errors**, using
  `tools/check_separator_samples_ui.py` against the built SPA and intercepted
  API requests. Covered manual entry, unknown units, measured-flow pairing,
  comparison CSV/date export, the mg/L density gate and rejection of a stale
  response after an input edit. Local screenshot:
  `build/separator-recovery-2026-09-14/sample-comparison.png`.

Named regressions in `tests/test_separator_sample_validation.py` cover an
excursion between flow reports (two minutes at 50% oil and 72,000 BPD gives
50 barrels), sustained raw indication despite zero excursion (96% WC gives
2,880 indicated barrels over 24 hours at 72,000 BPD), ordered caps, causal
reference, midnight splitting, invalid WC versus valid zero, unknown level,
unit/density conversion, DST times, separate streams, water-only flow basis,
backward pairing with measured flow, and retained logs after historian failure.

Field qualification remains open: confirm the physical tap/tag mapping, total-
liquid versus water-only flow and reference-condition bases, lab method/density,
transport delay and instrument quality/cleaning history. Obtain representative
same-stream grabs, including steady operation and excursions, before interpreting
meter bias or carryover totals. This implementation does not automatically fit a
correction, prove downstream recovery or quantify daily lost sales oil.
