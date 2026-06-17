# Draft PR for upstream `kwellis/woffl`

Draft to offer the solver-robustness patches upstream (see `docs/upstream_sync.md`
for the patches themselves). Tone is deliberately soft / peer-to-peer. Tweak
freely before sending.

---

**Title:** Solver: converge on marginal wells (small throat / high water cut) instead of aborting

**Body:**

When trying to optimize S-pad several wells (S-17, S-204, S-03) hit a convergence issue. Worked out a fix that's held up well; offering it here in case it's useful upstream. You know this solver better than I do, so take whatever's helpful and leave the rest.

**What I saw.** A number of wells that produce fine in the field raise `ConvergenceError: throat mixture did not converge` (and sometimes the suction-pressure variant). It clusters on marginal pumps — small throat ratios with high water cut. The odd tell was that nudging the IPR toward a higher oil rate would *break* convergence, which seemed backwards since the well obviously flows at that rate.

**Why, as far as I can tell.** Two spots:

1. `jetpump_solver` brackets the discharge-residual root by evaluating it at `psu_min` and `psu_max`. For these marginal pumps the inner throat solve has no solution right *at* `psu_min` — but it does just inside the range; the residual crosses zero a little above it. So `discharge_residual(psu_min, …)` raises and the whole solve gives up, even though a valid root is sitting right there.
2. In `throat_discharge`, the throat-pressure secant is capped at 15 iterations and can oscillate around a real root on the compressible mixture near the bubble point.

**What I changed** — three small fallbacks, each of which only runs *after* the existing path fails, so any well that already converges follows the same path and returns the same psu/oil:

- `throat_discharge`: at the iteration cap, bracket the momentum balance and finish with Brent; only raise when there's genuinely no sign change.
- `jetpump_solver`: if a bracket endpoint is infeasible, step the suction inward to the nearest feasible point so the bracket stays valid (the first probe is the endpoint itself, so feasible endpoints are untouched).
- `jetpump_solver`: if the discharge-residual secant stalls, re-seed it from the measured flowing BHP (`ipr_su.pwf`) and fall back to bisection on the bracket.

**Testing.** The existing solver tests pass unchanged. I also swept ~6,000 nozzle/throat/WC/GOR/pressure combinations: every case whose residual actually crosses zero now solves, and the only remaining failures are physically infeasible geometries (tiny throat at ~99% WC demanding thousands of bbl/d), which should fail. Added a small regression test covering the marginal cases.

