# Pad and CFP capacity repairs — September 12, 2026

The confirmed numerical and allocation defects from the
[capacity review](pad_cfp_capacity_review_2026-09-12.md) are repaired. The app
also exposes required-online constraints, planned future hardware for choke
runs, and CFP bring-online/offset pairs. This is a local implementation;
no deployment, production property writes or commit were performed.

The saved single oil IPR, actual modeled composition, selected return
hydraulics and installed-versus-clean replacement rules are preserved. This
work changes search, feasibility and reporting; it does not tune field data
or alter physical equations to improve a production match.

## Allocation and capacity

Default pad runs maximize oil subject to their machine-water limit. A manual
water price still deliberately changes the objective to oil minus price
times water. Automatic frontier lambda is now a separate concave-relaxation
diagnostic. It cannot suppress a feasible oil-producing choice merely because
the same capacity limit was also converted to a price. Actual unused capacity
and pooled-frontier slack have separate fields.

Choke allocation solves the discrete sampled settings using MILP, retaining
the installed hardware and the original plant water basis. The reproduced
5,000-BPD example now retains 500 BOPD instead of the greedy result's 150.
Its displayed reduction tradeoff is an average sacrifice diagnostic, not a
certified marginal value or an objective charge.

Both resize allocators share valid candidate tables and required-online
constraints. Unavailable required models and infeasible mandatory combinations
fail explicitly. An optimal all-shut-in allocation is distinguished from
infeasible, unknown, unsupported and failed solver outcomes. Status, objective,
bound/gap and precision refinements travel with the results. Candidates with
nonfinite or negative rates are excluded. Installed pumps are included even
when their size is outside the replacement grid; replacements still use clean
reference hardware.

Fractional-resource CP requests use original-unit MILP instead of rejecting
an exact-fit choice through per-candidate rounding. CP integer scaling and
final verification use a declared 1e-8 BPD arithmetic tolerance. This is not an
operating capacity margin. Successful duplicate installed/clean responses are
counted as successful simulations even when deduplicated for allocation.
See [upstream patch 46](upstream_sync.md).

## Plant operating points

E-Pad must deliver the proposed header at the selected total machine flow,
within the same speed, amp and cap assumptions used by its envelope. The
4,000-BPD/3,500-psi counterexample is rejected. Suction at or above the header
cap is rejected before running; reported speed and amps use the same derating
as the pressure frontier.

S-Pad searches a bounded set of interior pressure samples when an outer
evaluation fails. It recovers the reproduced 3,130.60-psi intersection despite
a failed remote sample. Hydraulic closure remains required, and a failed
pressure evaluation is never an economic shut-in choice.

Hydraulically closed plans below a recommended minimum flow remain visible as
conditional results with `feasible=false`, their operating assumptions and
range warnings. Fully qualified plans rank above conditional ones. No recycle
flow is invented. Hardware gains are withheld for unqualified pad plans;
current-model rates remain available as conditional comparisons. Choke results
also show the limits and withhold their projected headline gain when infeasible.

## CFP reference and search

CFP remains a delta model: modeled changes in water demand act on a reference
discharge through the stated local machine slope and upper disposal-control
cap. The form labels P0 as a manual reference. Optional B/G/J pad PF pressures
must describe the same conditions; omitted values use fixed line-loss
assumptions. The service no longer combines manual P0 with asynchronous latest
pad pressure clusters. P0 must be at or below the 2,880-psi margin cap and is
explicitly included in the response grid.

Online baseline wells must have their current pump response at P0. A missing
current option stops anchoring instead of becoming zero production. Settling
retains the raw pressure equation, residual and domain reason. A result below
the sampled pressure floor is unsupported, never pressure-clipped into a
feasible result. Before/after oil and water deltas use their own reference and
settled pressures, including when the old pump cannot run at the new pressure.

Finalists include the baseline, every evaluated single, raw bring-online/offset
pairs and the lambda frontier. Pairs can be evaluated when bringing a well
online alone fails. Required wells constrain the final plan, while the unchanged
configuration remains the comparison baseline. A required new well can
therefore produce a negative net gain without being silently dropped.

Small studies enumerate combinations under a 4,096-combination/250,000-work
threshold. Larger studies use a bounded neighborhood with explicit search
coverage. Global optimality on the response tables is reported only when all
choices were evaluated and nondecreasing water response establishes a unique
settled pressure. Displayed pair results are capped at eight; search and
display limits are disclosed separately. The winning plan cannot be worse
than another admissible evaluated candidate under the same objective.

These rates are interpolated response-table predictions. Metadata and UI
explicitly disclose that direct final-pressure well re-solves have not been
performed. The result is conditional on modeled pressure response and the
assumed disposal/control regime.

## User workflow

On Pad review, add a unique future name and donor. New future rows default to
**Required online**. Existing wells can also be protected from shut-in using
that column. **Offline** remains a baseline selection: excluded for pad runs,
or a bring-online candidate for CFP. A pad request cannot require a well that
it also excludes. A hold-pumps choke run needs an explicit planned nozzle and
throat for each future well. Duplicate names across pads and collisions with
existing wells are rejected.

Run the existing-well case and then the required-new-well case with consistent
inputs and operating assumptions. CFP shows combined bring-online/offset moves.
The pad hardware comparison remains current and proposed hardware at the
same proposed header; it is not an independently settled before/after pad
comparison. The complete frozen multi-scenario impact table specified in the
review remains a separate feature.

## Verification and remaining work

The integrated Python suite passed **2,198 tests** (91.87 seconds), with the
existing low-GOR correlation and retired-Mach deprecation warnings. All
**29 frontend tests** passed. TypeScript and the production build passed;
Vite retained its existing large-chart-chunk warning. Named regression suites
cover allocation precision/status, required
wells, pressure-domain and anchor invariance, E deliverability, S interior
closure, default/manual objectives, exact choke selection, CFP combination
oracles, API forwarding and conditional reporting. The review's original
probe scripts and JSON are unchanged historical evidence, not post-fix outputs.

`tools/check_capacity_optimization_ui.py` intercepts every API request to
exercise required wells, future hardware, duplicate names, objective defaults,
CFP reference pressures/pairs and operating-limit warnings in the built SPA.
It passed against the final production build: four mocked runs, zero browser
errors and zero unexpected write requests. Desktop and 960-pixel layouts were
inspected, including a separate CFP pair-card capture. The existing
`tools/check_pad_decision_ui.py` also passed: three stress starts, cancellation,
and no browser errors. QA screenshots are local artifacts under `build/`;
they validate interface behavior, not field physics. The preview server was
stopped after verification. `git diff --check` passed with the repository's
Windows newline normalization.

Medium worker/job/native-thread limits and shared compute/cache remain in use.
The new choke MILP has a five-second solve limit; CFP enumeration/neighborhood
work is bounded. Resize MILP and CP-SAT retain their existing lack of a
per-solve wall-clock limit. A hard runtime bound for those solves remains work.

Field qualification remains necessary: matched operating events, measured
pressure response, well-addition/downsize outcomes, routing, recycle/minimum
flow, actual E hardware and M suction/control assumptions. Broader facility
constraints, the complete new-well comparison study, S/CFP coupled uncertainty,
adaptive grids and final direct physics validation are not implemented here.
No claim of field prediction accuracy follows from passing synthetic tests.
