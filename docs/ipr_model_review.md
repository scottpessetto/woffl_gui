# Is Vogel the right IPR for these wells, and what else is there

2026-08-08. Question from Scott: if Vogel really is not fitting, are there better
options in the literature, especially for modeling jet pumps.

Short answer: **the functional form is not where the error is.** On this fleet,
every alternative the literature offers moves the predicted rate at jet-pump
suction by 0 to 17 percent, and most of them by under 1 percent. The anchor
choice moves it by up to 100 percent, and 24 of the 32 automatic Vogel fits on
the fleet have an R2 at or below zero, meaning the fitted curve tracks the tests
worse than a flat line would. Swapping in a curve with more parameters would fit
that noise better and extrapolate worse.

Everything below is either measured against the live warehouse or cited to a
source with a link. Three background research passes ran in parallel; where they
produced a general result I re-ran it against our own wells, and in two cases
that changed the conclusion.

---

## 1. What we measured on our own fleet

All runs 2026-08-08, 90 wells, 6 month test window, using each well's own
context seeds (the anchor the app actually solves on) and each well's own
`resvr_bubb`, `resvr_press` and `form_wc` from the pivots. "psu" is the jet pump
suction pressure, which is where woffl evaluates the IPR
(`jetflow.throat_entry_zero_tde` and `throat_entry_mach_one`).

### 1.1 The fits themselves are the problem

Automatic Vogel fits carrying an R2 (n = 32):

```
min -8.71   p25 -2.64   median -1.58   p75 +0.05   max +0.84
R2 <= 0 (worse than a flat line):  24 of 32
R2 < 0.5 (weak):                   27 of 32
worst: MPS-204 -8.71, MPH-23 -7.05, MPM-64 -4.36, MPM-18 -3.79, MPM-16 -3.50
```

A negative R2 here does not mean Vogel is the wrong shape. It means the test
cloud carries no usable curvature signal at all: the tests sit at essentially
one drawdown, scattered by water-cut drift, pump changes between tests, and
measurement noise. No two-parameter curve can be fitted to a one-point cloud,
and no three-parameter curve can be fitted to it either.

### 1.2 Anchor choice dominates everything else

Already measured earlier today: of the 31 wells carrying saved IPR values, 12
have an automatic fit that disagrees with the saved anchor. MPB-35 saved
668 BLPD at 222 psi where the fit writes 322 at 152, a factor of two on the
anchor rate. That is the size of the real error, and it is a choice about
*which point* the curve passes through, not about the shape of the curve.

### 1.3 Composite (Standing) form versus Vogel: under 1 percent here

The jet-pump research pass came back recommending the composite form (straight
line above the bubble point, Vogel below) as its top item, because that is what
the reference design software actually computes. I tested it on our anchors:

```
psu = 400 psi (n=64):  min -0.0%  median +0.0%  max +0.5%
psu = 200 psi (n=64):  min -0.1%  median +0.0%  max +0.4%
```

Essentially zero, and the reason is specific to this field: **our bubble points
sit on top of our reservoir pressures.**

```
90 wells with both resvr_press and resvr_bubb
  undersaturated (pb < pres):  59
  saturated or above:          31
  pb/pres:  min 0.74   median 0.90   max 3.45
  median pres 1800 psi, median pb 1572 psi
```

With pb/pres at 0.90 the single-phase segment is the top tenth of the pressure
range and the well is in Vogel territory everywhere a jet pump operates. For the
31 wells at or above saturation the composite form *is* Vogel, identically. The
literature recommendation is sound in general and inert here.

### 1.4 Linear water plus Vogel oil (Brown composite): 0 percent median, up to 17 percent

The multiphase pass found, on a synthetic well (pr 2000, pb 1300, water cut
0.95), that applying Vogel curvature to the water fraction under-predicts total
liquid by 10 to 38 percent, and argued that a Brown composite (water linear in
drawdown, oil on Vogel, summed) is the correct construction. This is a real
structural criticism of what woffl does: one Vogel curve is fitted to total
liquid, so the water gets solution-gas curvature it should not have.

Re-run on our anchors with each well's own water cut and bubble point:

```
psu = 400 psi (n=64):  min -3%  p25 -0%  median +0%  p75 +3%  max +15%
psu = 200 psi (n=64):  min -3%  p25 -0%  median +0%  p75 +4%  max +17%
```

Same reason as 1.3: their synthetic well had pb/pr = 0.65, ours run at 0.90.
The effect is real, it is just small here, and it is concentrated in the most
undersaturated high-water-cut wells (the +15 to +17 percent tail). Worth
knowing; not worth reworking the inflow model for.

### 1.5 The one model choice that does matter, and we already made it

Straight-line PI against Vogel, same anchors:

```
psu = 400 psi:  min -11%  p25  +5%  median +10%  p75 +19%  max +35%
psu = 200 psi:  min  -4%  p25 +17%  median +24%  p75 +35%  max +53%
```

Median 24 percent at 200 psi suction, up to 53 percent. This is the choice
`docs/upstream_sync.md` section 15 records being made deliberately in 2026-03
(commit ee3886e, "change woffl to solve on ipr not straightline PI, t'isnt
right") and silently reverted once by an upstream sync. The numbers above are
why that revert mattered and why the guard test exists.

Structurally: both curves pass through the anchor and through (0 rate, pres).
Between the anchor and pres, Vogel sits above the chord. **Below** the anchor,
which is where a jet pump lives, Vogel sits below it. So the PI form is the
optimistic one exactly where we operate.

---

## 2. What the literature actually offers

### 2.1 The whole family is one equation with one knob

Ilk, Camacho-Velazquez and Blasingame (SPE 110821) show the Vogel family
collapses to a single quadratic with one free curvature parameter:

```
qo / qo_max = 1 - nu * (pwf/pr) - (1 - nu) * (pwf/pr)^2
```

- `nu = 1`   gives the straight line, `q = J (pr - pwf)`
- `nu = 0.2` gives Vogel 1968 exactly. That is the entire content of the
  "0.2 / 0.8" coefficients: Vogel picked nu = 0.2 off his simulator cases.
- `nu = 0`   gives Fetkovich with n = 1.

https://blasingame.engr.tamu.edu/0_TAB_Public/TAB_Publications/SPE_110821_(Ilk)_IPR_for_Sol_Gas_Drive_Res_Analytical_Considerations_(wPres).pdf

That reframing is the most useful thing in the whole review. "Should we use
Vogel or Fetkovich or a PI" is really "what value of nu", and section 1.5 is the
measured answer to how much nu is worth. Nothing in our data can identify nu, so
picking it from physics (Vogel, nu = 0.2) rather than from a regression is the
defensible move.

### 2.2 The catalogue, and why each one is or is not for us

| Model | What it adds | Data it needs | Verdict here |
|---|---|---|---|
| Straight-line PI | nothing, nu = 1 | 1 test + pres | Optimistic by 24 percent median at 200 psi suction. No. |
| Vogel 1968 (SPE-1476-PA) | nu = 0.2 from simulation | 1 test + pres | What we use. Fits a single anchor by construction. |
| Standing 1970 | flow efficiency / skin correction on Vogel | 1 test + a skin estimate | We have no reliable skin per well. |
| Composite / Standing | linear above pb, Vogel below | 1 test + pres + pb | Measured 0.5 percent here (1.3). |
| Brown composite | linear water + Vogel oil, summed | 1 test + pres + pb + WC | Median 0 percent, tail +17 percent (1.4). |
| Fetkovich 1973 | nu = 0 plus an exponent n | **multi-rate / isochronal test** | We cannot get a multi-rate test out of monthly production tests. |
| Jones, Blount, Glaze 1976 | non-Darcy `pr-pwf = aq + bq^2` | multi-rate | Same blocker. |
| Klins and Majcher 1992 | exponent varies with pb and depletion | multi-rate | Same blocker. |
| Wiggins 1993 | genuine three-phase oil and water IPR | multi-rate, three-phase | More parameters than we have information. |
| Sukarno 1995 | skin varying with rate | multi-rate | Same. |
| Cheng 1990 / Bendakhlia-Aziz 1989 | Vogel coefficients versus deviation | deviation survey | Real for our deviated wells, but it perturbs nu, and nu is worth 24 percent at most while the anchor is worth 100. |
| Transient / RTA IPR | time-varying deliverability | high-frequency rate and pressure | For unconventionals. Schrader Bluff and Kuparuk are pseudo-steady waterfloods. |
| ML / data-driven IPR | flexibility | hundreds of labelled examples per well | 90 wells, under 20 tests each, most without a BHP. No. |

The pattern is blunt: **everything more sophisticated than Vogel needs a
multi-rate test, and we do not have one.** Vogel's real virtue was never
accuracy, it is that one test point plus a reservoir pressure determines the
whole curve. That is why it is everywhere in artificial lift design.

---

## 3. What the jet-pump literature says specifically

This was the interesting one, because the answer is mostly silence, and the
silence is quantified. OnePetro full-text counts, run 2026-08-08:

```
"jet pump"                                     1851
"jet pump" "inflow performance relationship"    115
"jet pump" "Vogel"                               48
"jet pump" "Fetkovich"                           19
"jet pump" "Wiggins"                             16
"jet pump" "composite IPR"                        2
"jet pump" "IPR uncertainty"                      0
"jet pump" "IPR" "extrapolat"                     0
"jet pump" "update the IPR"                       0
```

- The device papers (Cunningham 1970/1974/1995, Grupping 1988, Jiao 1990,
  Hatzlavramidis 1991, Noronha 1998, Corteville 1987, Kurkjian 2019) **do not
  discuss the IPR at all.** They take suction pressure and suction rate as
  boundary conditions and solve the pump. woffl's Cunningham groups and its
  throat-entry / throat-mixture / diffuser chain are direct implementations of
  that lineage, and Noronha 1998 is the closest published model to woffl's
  architecture.
- The design references (Petrie via SPE PEH ch.6, Guo ch.18.6, the SNAP manual)
  name the IPR curve as an input and stop. Guo's procedure is a one-way handoff:
  pick a rate off the IPR, convert to a required intake pressure, ask whether a
  catalog pump can deliver it. No coupled solve.
- **No published study anywhere varies the IPR form or its uncertainty and
  reports the effect on a jet pump design.** That hole is why section 1 of this
  document had to be measured rather than cited.

Two findings from the research pass that are worth keeping:

**The reference software uses one test for slope and PVT for curvature.** The
published SNAP sample report (Ryder Scott, funded by ConocoPhillips Alaska,
algorithm credited to Petrie) was reverse-fitted from its own output columns:
its deliverability column is a composite IPR with J = 0.500 STB/D/psi and
pb = 429 psi, RMSE 0.34 STB/D over 25 points from 1000 psig down to 1 psig.
J = 0.500 is exactly its single well test (50 bpd at 100 psi drawdown), and the
bubble point is not an input, so it came from the PVT. Pure Vogel anchored on
pres fits that column at RMSE 19.6, a straight PI at 30.6. So the industry
reference does what we do: one anchor test, curvature from physics.
https://nationsconsultingllc.com/software/snap/SNAPHELP/jet_pump_design_basics.htm

**The cavitation limit can cap the well far below what the IPR promises.**
Overlaying that same report's deliverability and cavitation columns: the
achievable rate tops out near 270 B/D at about 460 psig intake, which is
33 percent below the 405 B/D AOF the IPR predicts, and the location of the cap
is set entirely by IPR curvature between 400 and 500 psi. The only well test in
that dataset was at 900 psig. Nothing measured constrained the curve where the
answer was decided.

---

## 4. Where our error actually is

In descending order of size, measured:

1. **The anchor** (up to 100 percent). Which test, whether its BHP is real, and
   whether the engineer's chosen point survives to the next session. This is
   what today's work addressed: saved values now outrank the fit, a hand-set
   point is held against it, and a curve with no test behind it is labelled a
   manual point everywhere it feeds a decision.
2. **Fit quality** (24 of 32 fits worse than a flat line). The automatic Vogel
   regression is fitting noise on most wells. `ipr_r2` already grades this
   amber under 0.5 and crimson at or below 0; the grading is right and it is
   telling us the regression should usually lose to a chosen anchor.
3. **The curvature parameter nu** (24 percent median at 200 psi suction, PI
   versus Vogel). Already decided, deliberately, in favour of Vogel.
4. **Water on the wrong curvature** (0 percent median, +17 percent tail).
5. **Composite versus Vogel** (under 1 percent).

The jet pump amplifies items 1 and 2 rather than item 3, because it evaluates
the IPR below every tested drawdown. Christ and Petrie (SPE-15177-PA) advertise
running jet pumps to under 10 percent submergence, which is to say deliberately
into the region where no test has ever been taken.

---

## 5. Recommendations

**Do not change the IPR form.** Measured payoff 0 to 1 percent for the composite
variants; the alternatives that could pay more all need a multi-rate test we
cannot get, and every extra fitted parameter makes the negative-R2 problem
worse.

**Keep Vogel, keep nu fixed at 0.2, and keep solving on Vogel rather than pidx.**
The guard test in `test_asm_solopump.py::TestSolverUsesVogelIPR` is worth 24
percent median at jet-pump suction. Section 1.5 is the number to put next to it.

Worth doing, in order:

1. **Make the anchor the reviewed object, not the fit.** Largely done today.
   The remaining piece is to stop presenting an automatic fit with R2 <= 0 as if
   it were a curve: on those 24 wells the fit should be offered as "one recent
   test plus a physics curvature", not as a regression, and the Solver should
   say so.
2. **Add the production-cavitation limit as a plotted constraint.** We already
   model the *flow* ceiling correctly and better than the correlations do
   (`throat_entry_mach_one` walks the mixture sound speed to Mach 1, and
   `sonic_status` surfaces it). What we do not have is the *damage* limit: the
   suction-annulus velocity limit that erodes throats. The published form, with
   the free-gas term, verified against commercial software output to within
   1 B/D:

   ```
   Qs <= As / [ (1/691)*sqrt(GradSuc/Pps) + ((1-WC)*GOR)/(24650*Pps) ]

   Qs      max produced liquid before cavitation, B/D
   As      suction annular area = A_throat - A_nozzle, in^2
   GradSuc produced fluid gradient at pump depth, psi/ft
   Pps     pump intake pressure, psig
   WC      water cut, fraction
   GOR     producing gas/oil ratio, scf/STB
   ```
   Source: Nunez-Pino 2019, SWPSC 2019031,
   https://www.swpshortcourse.org/sites/default/files/papers/31.pdf
   Value: it shows the engineer when the IPR's low-pressure tail stops mattering
   because the pump cannot get there anyway.
3. **Use the nozzle as a BHP gauge and trend the residual.** The nozzle is a
   calibrated orifice, so intake pressure falls out of
   `p3 = p1 - gamma1 * (q1 / (1214.5 * Aj))^2` with no downhole gauge. The
   commercial real-time jet pump controllers do exactly this (US 11,466,704,
   US 11,078,766). The matching re-fit trigger from Nunez-Pino: at constant
   injection pressure, a constant power fluid injection rate means a constant
   bottomhole producing pressure. That is a free, continuous, surface-measured
   residual, and it beats calendar re-fitting on a fleet where the pump changes
   between tests.
4. **If you want a better curve, get a better test, not a better equation.** The
   jet pump can produce a genuine multi-rate drawdown test by stepping power
   fluid pressure, which is the one thing that would let a two-parameter form be
   identified honestly. There is a Russian patent family and a field paper doing
   exactly this (Arbatskii et al. 2021,
   https://doi.org/10.24887/0028-2448-2021-7-90-93), and their stated motivation
   is ours: a method that requires assuming the reservoir pressure "can lead to
   significant errors".

---

## 6. Sources

Primary, read in full by the research passes:
- Vogel 1968, SPE-1476-PA, https://doi.org/10.2118/1476-PA
- Ilk, Camacho-Velazquez, Blasingame, SPE 110821 (the nu formulation),
  https://blasingame.engr.tamu.edu/0_TAB_Public/TAB_Publications/SPE_110821_(Ilk)_IPR_for_Sol_Gas_Drive_Res_Analytical_Considerations_(wPres).pdf
- Nunez-Pino 2019, SWPSC 2019031 (cavitation, with the free-gas term),
  https://www.swpshortcourse.org/sites/default/files/papers/31.pdf
- Guo, Petroleum Production Engineering, ch. 18.6 (jet pump design procedure),
  https://irmat-ucan.com/library/admin/books_pdf/pdf_67a784bcb596f7.00260478.pdf
- SNAP jet pump design basics and sample output report,
  https://nationsconsultingllc.com/software/snap/SNAPHELP/jet_pump_design_basics.htm

Jet pump device and design lineage (abstracts read):
- Cunningham and Hansen 1970, cavitation index, https://doi.org/10.1115/1.3425040
- Cunningham 1974, https://doi.org/10.1115/1.3447143 and 1995, https://doi.org/10.1115/1.2817147
- Grupping, Coppes, Groot 1988 (intake versus throat-entrance pressure, which is
  woffl's `psu` versus `pte`), https://doi.org/10.2118/15670-PA
- Jiao, Blais, Schmidt 1990 (single-phase models always overpredict pressure
  recovery with gas present; the two-phase correction cuts standard error to
  18 percent of its previous value),
  https://onepetro.org/PO/article-abstract/5/04/361/168379/Efficiency-and-Pressure-Recovery-in-Hydraulic-Jet
- Hatzlavramidis 1991, https://doi.org/10.2118/19713-PA
- Noronha, Franca, Alhanati 1998, https://doi.org/10.2118/50940-PA
- Christ and Petrie 1989, low BHP with jet pumps, https://doi.org/10.2118/15177-PA
- Kurkjian 2019, https://onepetro.org/PO/article/34/02/373/206931/Optimizing-Jet-Pump-Production-in-the-Presence-of
- Ellis and Awoleke 2025, SPE-224132-MS (woffl's own upstream paper),
  https://doi.org/10.2118/224132-MS

Alaska and heavy oil context:
- Peirce et al. 2008, formation-powered jet pumps at Kuparuk, https://doi.org/10.2118/114912-MS
- Bidinger et al. 2005, West Sak heavy oil, https://doi.org/10.2118/97856-MS
- Singh et al. 2013, large-scale jet pump optimisation in a viscous oil field
  with hot water power fluid, https://doi.org/10.2118/166077-MS
- **No published jet pump work on Milne Point or Schrader Bluff exists.**

Measurement and back-calculation:
- Bruijnen 2016, nodal analysis from pump intake and discharge gauges,
  https://doi.org/10.2118/178433-PA
- Arbatskii, Shchurenko, Zatsepin 2021, jet pump as a well-testing device,
  https://doi.org/10.24887/0028-2448-2021-7-90-93
- US 11,466,704 and US 11,078,766 (jet pump controllers that infer intake
  pressure from the nozzle; both take a straight-line productivity index as
  their inflow input)

Full research transcripts, with every equation, coefficient table and
UNVERIFIED marker, are in the session's agent artifacts: the jet pump pass
(75 KB), the classical IPR pass (91 KB), and the multiphase pass with its
Wiggins, deviated-well, depletion, fitting-practice and ML sub-reports.
