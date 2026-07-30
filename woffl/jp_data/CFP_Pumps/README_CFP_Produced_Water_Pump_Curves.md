# CFP Produced-Water Pump Curves — provisional, and how to replace them

Three machines (A / B / C) in parallel at a common discharge pressure, feeding
power fluid to the CFP-side pads (B / G / J) plus produced-water injection and a
disposal well. Consumed by `woffl/assembly/cfp_plant.py` and wrapped for the
optimizer by `woffl/gui/cfp_pad_plant.py`.

## Read this first: the numbers are wrong, the interface is right

Validated against live SCADA on 2026-07-29. The **structure** held up; the
**magnitudes** did not.

| | Curve says | Plant actually does |
|---|---|---|
| Flow at 2,792 psi discharge | ~95,000 BWPD | **~112,327 BWPD** |
| Discharge at 112,327 BWPD | ~2,496 psi | **~2,792 psi** |
| Throughput→pressure slope | −17.5 psi / 1,000 BWPD | **−1.8** (r²=0.03) |

So the fit **under-predicts capacity by ~17,000 BWPD** and will cap the model's
feasible discharge roughly 300 psi below where the plant really runs. Every run
carries `provisional_curve=True` in its meta and the page badges it.

### Acceptance test for replacement coefficients

> **The curve must pass ~112,000 BWPD at ~2,790 psi.**

That single point needs only pump data — no downstream water balance. Sources:

- discharge — `MPU_PIC_5418`
- produced water — `MPU_MOD 54_ProdWaterAvgFlowRate_Calc` ("MOD 54" is the CFP)

both in `reporting.historian.vw_mpu_measurements`. Note the hosted app's service
principal cannot read that catalog (see `docs/prop_hist_asks.md` and the
reporting-catalog grant), so pull it locally.

## Swap contract

Replacing the curve is a **data change only** — the interface is frozen.

1. Edit `machine_coeffs` here **and** `MACHINE_COEFFS` in
   `woffl/assembly/cfp_plant.py` (the module still hardcodes them; this file is
   the record and the loader target).
2. Update `measured_operating_point_2026_07_29` if you re-measure.
3. Run `pytest tests/test_cfp_plant.py` — **its pinned values WILL fail, and that
   is correct.** They pin the *provisional* spreadsheet anchors (125,901 BWPD @
   2,200 psi; 101,428 @ 2,700). Re-pin them to the new curve's own anchors, and
   add the acceptance point above as a new pin.
4. Run `pytest tests/test_cfp_pad_plant.py tests/test_cfp_optimize.py` — these
   should pass unchanged except the four `budget_at_pressure` value pins. If
   anything else fails, the interface moved and that needs a look.
5. Set `validated: true` here and drop the provisional badge (`provisional_curve`
   in `cfp_optimize.run_joint_optimization`'s meta).

If the incoming curve is **not** three quadratics — per-pump lookup tables, a
different polynomial order, VFD speed as a variable — then `machine_flow` needs a
new evaluator, but `plant_flow` / `plant_pressure` / `CFPPlant` and everything
above them are unaffected. Scott confirmed 2026-07-29 that the form is unchanged
(same three quadratic machines, new coefficients).

## Two data problems that are NOT the curve

**Machine C's fit is non-physical.** Its `q=0` intercept is 2,151 psi — below the
entire operating window — and the parabola *rises* to a 3,015 psi vertex at
22,198 BWPD before falling. Inside 2,200–2,700 psi the totals are monotone and
match the spreadsheet exactly, so today's three-machine numbers are usable. But
per-machine extrapolation is meaningless, which is why
`machine_curve_validated: false` makes `CFPPlant` **refuse** 1- and 2-machine
cases instead of emitting numbers. Validate each machine individually before
flipping it.

**All three `pad_line_dp_psi` values are over-stated.** Measured 2026-07-29
against live per-well PF from `vw_pressure_daily`, at 2,792 psi discharge:

| Pad | table | real | error | evidence |
|---|---|---|---|---|
| B | 272 | **~169** | −103 | 5 wells clustered 2,619–2,630 psi (11 psi band) |
| G | 293 | **~44** | −249 | only MPG-18 (2,748) is on the header |
| J | 251 | **~110** | −141 | MPJ-27/29/32 clustered 2,678–2,687 |

So the table under-delivers PF by 100–250 psi, biasing every CFP jet-pump
simulation pessimistic. **Prefer the measured anchor** —
`CFPPlant.delivered_pf_for_pad(pad, discharge, measured_pad_pf=...)` holds the
pad's own gauge reading and moves it by the *change* in discharge. The table is
the fallback for a pad with no live reading.

⚠️ When gathering that measured PF, take the **high cluster, not the pad
median**. B/G/J/C are ESP-heavy, so only a minority of wells sit on the JP PF
header — G's two-well "median" of 2,084 is just the midpoint of 2,748 and 1,420,
and C-Pad's median of 1,208 hides its real ~3,404 psi booster pressure.

**Is G-Pad still on plant PF?** Its implied 44 psi line loss rests on a single
well, and this table already carried one stale entry — H-Pad, dropped once it
took its own POPS install. Worth confirming before the model assumes G rides the
plant discharge.

## The mechanism, for whoever reads this next

Discharge pressure is **not** a passive consequence of production. Operators set
it by opening or closing the disposal well, which moves total flow through the
pumps and so where they ride on the curve (Scott, 2026-07-29). That is why the
metered throughput↔pressure correlation is nearly zero, and why the optimizer
treats discharge as a **decision variable** (`coupling="free_pressure"`, like the
I-Pad VFD train) rather than iterating a fixed point.

The real trade is **pressure versus volume**: riding up for more PF pressure
necessarily passes less total water (+500 psi costs ~28,800 BWPD), and the pumps
must still move everything arriving. Cutting controllable water at B/G/C/J lowers
the required flow and buys discharge pressure, hence lift.
