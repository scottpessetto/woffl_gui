# CFP "Today's Moves" — formulation, literature grounding, and algorithm

*2026-07-30. The decision engine behind the PW Pressure Optimization page's
Configure/Results stages (`woffl/gui/cfp_moves.py`).*

## The question, in Scott's words

> "All I want to know is: today, at a given pressure, I have knobs to turn by
> changing JP size, and shut in or bring online wells. … I need to know today
> should I resize JPs to increase PF, SI a well, BOL a well and drop PF, BOL
> wells that drop PF pressure but offset by downsizing a jet pump."

And the constraint on the modeling itself:

> "What we need to do is just know current PF discharge pressure and what is
> online. … The summing of the tests to the pumps is irrelevant."

## 1. The state is measured, and everything is a delta

Earlier CFP modeling tried to build the plant load bottom-up (well tests +
exogenous water). Scott rejected that, and he is right to: the split between
power fluid, injection support (E/S-Pad take PW for reservoir support — not
oil-impactful), and disposal is operator-set and unmeasured well-by-well.

The reframe that fixes it: **anchor at today's measured operating point and
model only deltas.**

* `P₀` — today's PW discharge (`MPU_PIC_5418`), measured.
* `x⁰` — today's configuration: which wells are online, each at its current
  jet pump size.
* `W⁰` — the model's own estimate of the water those wells put through the
  machines at `P₀` (power-fluid draw + formation water; the returned PF *is*
  machine throughput on each pass).

For any candidate configuration `x`:

```
ΔW(x, P) = W_model(x, P) − W⁰
P(x)     = min( P₀ − s · ΔW/1000 ,  P_trip − margin )
```

Every unknown — injection volumes, disposal valve position, other pads'
carryover, even most model bias (both sides of the delta use the same
IPR/friction per well) — **cancels in the subtraction**. No exogenous water
number exists anywhere in this formulation.

`s` is the machine-curve slope in psi per 1,000 BPD. It is bracketed by three
independent estimates: the 120-day regression of measured discharge on the
**real** per-machine flow tags (`MPU_FIC_5419S/5420S/5421S`), −13.69 at
r²=0.54; the fitted pump curve, −17.5; and the plant's own Mar→Jul operating
trend, −12.2. Default 13.69, slider-bounded [9, 17.5].

The `min(·, trip)` is the disposal re-trim: shed more water than the trip
allows and operators open disposal, capping the pressure gain — beyond that
point shutting things in is pure oil loss (the kink from `cfp_tradeoff`).

## 2. The decision variables

Each CFP-fed well `w` (pads B/G/J responsive to discharge; C-Pad wells carry
water but hold their own boosted PF) picks one option:

* online well: `{SI} ∪ {jet pump size k}` — staying at the current size is one
  of the options;
* offline (bring-on-line candidate): `{stay OFF} ∪ {size k}`.

Per option, WOFFL provides the physics: `oil_w(k, P_del)` and
`water_w(k, P_del)` (total water = PF draw + formation), with
`P_del = measured pad PF + (P − P₀)` for B/G/J and constant for C-Pad.

Objective: **maximize total oil**, with `P` determined by the choices through
the anchor equation — a fixed point, since PF draw itself depends weakly on
`P`. The loop gain is ≈ `dW/dP · s/1000` ≈ 0.1 here, so simple iteration
converges in a few passes.

## 3. What the literature says this problem is

This is the **lift-resource allocation problem** with a shared, endogenous
back-pressure — the structure the gas-lift literature has worked on for
40 years, with jet-pump power fluid in place of lift gas:

* **Kanu, Mach & Brown (1981)** — the *equal-slope* method: at the optimum,
  the marginal oil per unit of shared resource is equal across wells; with a
  limited supply, allocate until every well sits at the same slope. This is
  exactly a Lagrangian dual: price the resource at λ and let each well
  independently maximize `oil − λ·resource`.
* **Rashid, Bailey & Couët (2012)** — the survey of gas-lift optimization:
  frames the field's standard architecture as per-well *performance curves*
  (proxy models sampled from a simulator) plus an allocation optimizer, with
  Newton/equal-slope, heuristic (Buitrago's for nonconvex curves), and MILP
  families. Rashid (2010) adds the *iterative offline–online* scheme: build
  proxies offline, optimize, re-simulate, repeat — which is our
  build-surfaces / optimize split with a fixed-point re-evaluation.
* **Gunnerud & Foss (2010)** and the piecewise-linear MILP line (Codas,
  Camponogara; SOS2 formulations) — when well response is nonlinear and the
  network couples wells through pressure, sample the simulator on a grid and
  optimize over the piecewise-linear tables, optionally with decomposition.
  Justifies the response-surface grid and its linear interpolation.
* **SPE WRM 2025, "Optimizing Power Fluid in Jet Pump Oil Wells"** — the
  WOFFL lineage itself: continuous PF allocation across a jet-pump network
  under a shared surface pump, per-well efficient frontiers + reduced-Newton.
  Our problem extends it in two ways: the *pressure is endogenous* (set by
  what the machines pass, not a fixed constraint), and the knobs include
  *discrete* SI / bring-online / size-change moves, which the continuous
  formulation cannot express.

The discrete per-well choice under one shared constraint is a
**multiple-choice knapsack** (each well = a class, options = its sizes/SI);
the codebase already solves MCKP elsewhere (`optimization_algorithms`,
CP-SAT). We use the Lagrangian route instead because it *is* the equal-slope
method, needs no solver dependency in the pure engine, produces the whole
pressure–oil frontier and the shadow price as by-products, and its known
weakness (small duality gaps on non-concave per-well curves) is immaterial at
this scale. CP-SAT polish can be added later if a gap ever matters.

## 4. The algorithm

**Stage A — response surfaces (expensive, cached).** For each discharge `P`
on a grid (≈7 points spanning `[P₀ − 300, trip]`): assign every well its
pad's delivered PF, run one `NetworkOptimizer` batch over all wells × all
candidate sizes (the existing machinery; one batch covers every size), store
`oil` and `total_water` per (well, size, P). ~7 batch runs total,
process-pooled, cached on the store signature.

**Stage B — pure optimization (instant, fully tested).**

1. *Anchor*: `W⁰`, `oil⁰` from the surfaces at current choices and `P₀`.
2. *Fixed point* `settle(choices)`: iterate `W → P → W` to convergence
   (loop gain ≈ 0.1); linear interpolation in `P` on the surfaces.
3. *Frontier*: sweep λ over [0, ~1] oil-bbl/water-bbl. At each λ, each well
   independently picks `argmax oil − λ·water` (equal-slope), then settle.
   Collect `(λ, P, oil, choices)` — the oil-vs-pressure frontier.
4. *Plan*: the frontier point with max oil (ties → fewest changes from
   today). Its diff vs `x⁰` is the action list.
5. *Single moves*: for every well × alternative option, settle just that one
   change; report Δoil (fleet), ΔP, classified as **Resize / Shut in /
   Bring online**. This is the knob board.
6. *Pairs*: top bring-on moves × top pressure-raising moves (SI/downsize),
   settled jointly — "BOL a well, offset with a downsize" made explicit.
7. *Shadow price*: `λ_today = d(fleet oil)/dP` at current choices — "one psi
   of discharge is worth X BOPD today", the number that prices every knob.

## 5. What to trust, and what not to

* The **ranking and signs** of moves are the robust output: both sides of
  every delta share the same well model, so IPR/friction bias largely
  cancels. The stage-0 model-accuracy gate (`cfp_match_check`) still applies
  to the absolute BOPD figures.
* `s` is bracketed 9–17.5; the UI exposes it. Conclusions that flip inside
  that bracket are flagged by re-running at both ends.
* Non-converged WOFFL points (a size that can't run at low PF) are honest
  gaps in the surface — options with no valid points are dropped.
* The equal-slope sweep can miss interior combinations on non-concave
  curves (duality gap); at ~25 wells × ≤10 options the practical impact is
  small, and the single-move/pair boards are exhaustive regardless.

## Sources

- [Rashid, Bailey & Couët, "A Survey of Methods for Gas-Lift Optimization" (2012)](https://onlinelibrary.wiley.com/doi/10.1155/2012/516807)
- [Rashid, "Optimal Allocation Procedure for Gas-Lift Optimization" (2010)](https://pubs.acs.org/doi/10.1021/ie900867r)
- [Kanu, Mach & Brown, "Economic Approach to Oil Production and Gas Allocation in Continuous Gas Lift" (1981)](https://onepetro.org/JPT/article-abstract/33/10/1887/68703/Economic-Approach-to-Oil-Production-and-Gas)
- [Gunnerud & Foss, "Oil production optimization — a piecewise linear model, solved with two decomposition strategies" (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0098135409002701)
- [Codas & Camponogara, "Mixed-integer linear optimization for optimal lift-gas allocation with well-separator routing" (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0377221711007983)
- [Silva & Camponogara, "A computational analysis of multidimensional piecewise-linear models with applications to oil production optimization" (2014)](https://www.sciencedirect.com/science/article/abs/pii/S0377221713006425)
- ["Optimizing Power Fluid in Jet Pump Oil Wells", SPE Western Regional Meeting (2025)](https://onepetro.org/SPEWRM/proceedings-abstract/25WRM/25WRM/D031S003R005/656730)
