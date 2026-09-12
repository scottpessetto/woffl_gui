# Pad decision accounting and hardware comparison

The optimization workflow now records every expected online, explicitly offline,
and requested future well. Input failures and unsupported oil models stay visible.
An absent allocation is a modeled shut-in only when the winning candidate grid
contains viable pump choices. Failed simulations and missing inputs receive
separate outcomes, with unknown rates rather than assumed zeros.

S/I/M/E runs with unaccounted online loads are **incomplete exploratory runs**.
Their modeled subset may still be inspected, but whole-pad feasibility is withheld.
This implementation does not invent fixed measured loads or pressure responses.
The returned coverage manifest records each requested well and its reason. Choke
plans also disclose incomplete coverage, including a measured fallback with unknown
PF. CFP keeps its measured-pressure anchored delta formulation; omitted wells are
identified as missing response and possible decisions, without introducing a
bottom-up plant load.

JPCO results show installed hardware at the **same proposed header and saved well
inputs** alongside proposed hardware. This isolates the modeled hardware decision.
Recent test medians remain separate context and never scale the prediction.
The installed hardware counterfactual is not itself certified as a feasible
whole-pad operating plan at that header. Aggregate hardware gain is withheld when
either coverage or current-hardware comparison is incomplete. A no-change hardware
case has zero modeled gain. Future wells have the explicit offline baseline.

The comparison first reuses the winning response grid. Missing installed
candidates use one batch through the existing shared pool and exact cache. No new
pool, compute tier, persisted well input, IPR shift, or production write is involved.
Readiness now labels saved inputs and pump fits as saved, avoiding the implication
that persistence establishes field qualification.

Verification includes input failure and invalid-IPR coverage, economic shut-in
versus model failure, no-change gain despite biased measured rates, unknown PF,
and cached/fresh installed-candidate comparisons preserving the oil IPR and GOR.
The frontend outcome regression verifies that a missing or failed model can never
display a shut-in recommendation. Full validation is recorded in the session's
delivery record. No deployment was performed by this change.

## Fixed-plan stress cases (I/M/E)

Completed I/M/E JPCO runs with full coverage and known current hardware capture
an immutable configuration/plan snapshot in the existing server job. An optional
**Stress-test current and proposed plans** panel compares those same two plans.
The request supplies only the source job ID and bounded ranges; it cannot replace
the server's configurations or chosen hardware. Source/plant/survey fingerprints
reject a changed model before replay. A restart/expired job requires a new run.

The default seven deterministic cases are base, all-well WC +/-3 percentage
points, all-well GOR +/-20 percent, and shared delivered header +/-100 psi. Users
can edit these engineering assumptions. Two optional joint corners combine
higher WC/GOR with lower header and the reverse. Limits are nine cases, 100
wells, +/-10 WC points, +/-50 percent GOR and +/-250 psi header. Zero ranges
remove duplicate cases. These are stress cases, not measurement-informed
probabilities or uncertainty bounds.

Each case preserves every well's oil IPR while changing the stated fluid inputs.
Both fixed plans use the same case. Only their selected catalog choices are
requested, grouped into grids of at most two nozzle and two throat sizes per
well through the existing shared pool/cache. There is no substitution, measured
rate scaling, input save or reallocation. Explicit offline/shut-in choices are
zero; missing plan entries and failed selected solves are unknown. Invalid
composition cases remain visible rather than being clipped.

Each plan is checked at the case's controlled header against the correct machine
water budget, operating pressure limits and actual delivered-header balance.
The last check matters on E-Pad's low-flow frontier: being below a maximum flow
budget alone does not establish deliverable pressure. Failed and infeasible
cases remain in the result. Numeric oil-gain ranges and preference/regret use
only cases where both plans are feasible. Regret uses the source run's water
price and compares these two plans only; case counts are not probabilities.

Frontend controls are explicitly user-triggered, hide stale results when ranges
change, expose cancellation and remain unavailable for unsupported or incomplete
source runs. Fixture browser QA checked missing-model versus shut-in labels,
saved-input readiness wording, zero hardware gain despite biased measured oil,
stress requests, joint corners, stale output, cancel, desktop/narrow rendering,
and zero browser errors. It intercepted every API request and performed no
production reads or writes.

Remaining work includes a common-date measured pad anchor for total operating
uplift, independent response and cross-pump qualification, and coupled S-Pad/CFP
fixed-plan scenarios. Those engines are not exposed by the controlled-header
stress panel. No global optimum or field-qualified robust recommendation is
implied by this comparison.
