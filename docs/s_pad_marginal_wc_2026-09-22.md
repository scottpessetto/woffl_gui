# S-Pad cost of power fluid and single-well pump decision (2026-09-22)

User request, as clarified: S-Pad's boosters always run at 60 Hz, so the
question is not a flow cap. It is: if a well takes another 1,000 BPD of PF,
or gives 1,000 BPD back, how many barrels does that cost (or give) all the
OTHER wells as the PF pressure drops (or rises)? Use that to size one well's
pump, for a JP replacement or a new well, on the S-Pad optimization screen.
Headline units: PF water cut plus BOPD per MBPD.

A first version assumed the station sat at the curve's recommended maximum
flow and priced choking other wells. The user corrected the premise the same
day; that version was replaced and is not in the code.

## What was built

- `woffl/gui/pad_marginal.py` - pure compute over per-well response curves.
- `server/services/pump_decision.py` - background job; `POST /optimize/pump-decision`,
  polled on `GET /optimize/run/{job_id}` (kind `pump_decision`).
- `web/src/pages/optimize/PfCostChart.tsx` - the ±PF cost curve.
- `web/src/pages/optimize/PumpDecisionPanel.tsx` - S-Pad tab only, between the
  run panel and match health.
- `tests/test_pad_marginal.py`.

## Method

1. **Operating point.** Every producing well on its installed pump (saved
   fit) is modeled at six headers across the curve's range. Today's header
   `H0` is where `header_at_flow(sum PF_i(H)) = H`. The root is unique
   because more pressure draws more PF and the curve answers with less head.
2. **Header response.** The same wells are re-modeled at `H0` -400, -200,
   -100, 0, +100 and +200 psi, then interpolated linearly.
3. **PF sensitivity.** An extra ±`delta_pf_bpd` (default 1,000) of draw
   resettles the header. The other wells' oil change is the cost or gain.
   They also shed or take PF, so the station's flow changes by less than
   the step. `lambda` = oil lost per BPD added, and marginal PFWC =
   `1 / (1 + lambda)`.
4. **Candidates.** Each catalog size for the target uses clean reference
   coefficients, plus the installed pump with its saved fit. Each is
   modeled at the fine headers and resettled on the curve with its own
   header-dependent PF. Net pad oil = the target's change plus every other
   well's change at the new header, with no linearization. The "vs marginal"
   chip is the linear screen: the target's incremental PFWC against the pad
   marginal.
5. **Chart.** `pf_sweep` resettles the curve for extra draw from -5,000 to
   +5,000 BPD (or ±5 × the step) in 21 points. `PfCostChart` plots the other
   wells' oil change against it, with the ±step points labeled. The target's
   pump sizes sit on the same axes at their own extra PF, so it's visible
   which sizes ride the curve's steep end. The tooltip adds the header change
   and the wells that respond. ECharts now registers `MarkPointComponent` and
   the `LabelLayout` feature for it.

PFWC = PF / (PF + oil), the stream S-Pad's boosters handle (the same basis
Well Sort uses for S-Pad).

## Limits

- Model rates, not test-scaled. Review match health first. The coupled
  header is only as good as each well's PF-vs-pressure fit.
- Wells with no identified installed pump, or whose pump does not solve
  across the range, are left out of the curve demand (listed in notes). The
  modeled header is then too high.
- Candidate headers beyond the modeled range are marked `*`; rates are held
  at the nearest modeled point.
- Only S-Pad has a fixed-speed curve. I/M/E have a free header.
- Local verification only; no deployment.
