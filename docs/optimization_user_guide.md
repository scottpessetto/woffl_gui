# Saving a well fit for optimization

Current behavior: September 12, 2026. See [the well-fit workflow delivery](well_fit_workflow_delivery_2026-09-12.md)
for verification and deployment status. The app is React/FastAPI; older Streamlit
tab/CSV instructions do not describe the current workflow.

## Prepare the well

Open the well in Solver. Check the tracker pump and installation date, wellbore
geometry, circulation, IPR anchor, reservoir pressure, WC, GOR and pressures.
The IPR rate is **formation liquid**, excluding returned power fluid. Use the
comparison test and gauge evidence to assess whether the current inputs make sense.

## Edit, compare and save

1. Edit the well's IPR and fluid inputs in the sidebar in Solver or JP History.
   The **Well inputs** bar at the top shows how many supported values differ
   from the loaded database. It stays visible while the main page scrolls.
2. On the production plot, turn on **Show model match**, choose **Every test
   (well fit)**, then set **Well inputs** to **Current edits (preview)**.
   Choose the history window and click **Run comparison**. Inspect modeled
   BHP and oil against actual tests across the installations.
3. Review the values and optional save note in the top bar, then click
   **Save well inputs**. Wait for the confirmation that new optimization runs
   will load them. An unsuccessful save keeps the edits and shows its error.
4. Start a new optimization run. Existing results retain their original inputs.

The preview holds one explicitly selected oil IPR across all pumps and dates.
It uses each test's measured WC, GOR, PF pressure and WHP. Actual oil/BHP do not
re-anchor the IPR at each test. Changed inputs hide the old preview until the
comparison is rerun. **Saved in database** compares the loaded well fit instead;
the plot explains when sidebar edits are excluded. **Saved fit where valid**
uses the installed-pump calibration only on its exact matching installation and
well model; other pumps use clean reference losses. The table names the actual
loss assumption. These comparisons are retrospective, not independent
qualification of sizing gains.

To propose a different common oil curve, expand **Fit one oil IPR across pump
history**. Select the window and later dates held out, then fit a candidate.
Reservoir pressure stays fixed. Review measured oil/BHP, exclusions and training
versus holdout oil errors. **Apply candidate IPR** changes session inputs only;
run the forward history preview and explicitly save when satisfied. The holdout
curve score uses measured BHP and does not establish operating-rate accuracy.

Save retains the displayed total-liquid IPR anchor/BHP, reservoir pressure,
WC, GOR and WHP, plus supported changed bubble point/formation temperature.
Saved values refresh the well context, Well Database and new optimization
configurations without replacing edits made while the save was in progress.
PF pressure remains a live/run input, and tracker hardware is not rewritten by
this save. Pump loss coefficients, nozzle area and the selected hydraulic model
have a separate installed-pump calibration save. Other sidebar settings remain
session inputs.

The Save button remains visible when unavailable, with an explanation: read-only
access, a well still loading, or an invalid oil IPR. A read-only app still permits
editing and previewing; it cannot persist changes. The Solver retains its anchor
pin/clear controls beside the IPR selector. Saving from JP History preserves the
existing anchor pin.

## Fit the installed pump

Select **Calibrate to field data** after saving edited well inputs. It uses saved
well inputs and the current installation's history/tests. Read the full fit:
BHP, PF and delta-BHP RMS, pressure-response agreement and parameter bound hits.
An improved match to one test can coexist with a poor fit across the history.

**Apply to inputs** previews the result. **Save installed-pump calibration**
persists that completed fit and its diagnostics against the exact installation
and model version. Read-only apps disable saving. If the job has expired or the
installation changed, refit before saving. Provisional fits stay visibly provisional.
Fitted losses or a small area increase do not prove physical wear.

## Compare keeping the pump with replacing it

**Try clean replacement** retains the well inputs and resets pump losses to the
reference values (ken .03, kth .30, kdi .40) and catalog area (factor 1.0).
This also works for a same-size replacement. Selecting another nozzle/throat
uses those reference assumptions. **Restore installed pump** restores the current
installation's saved fit; an unsaved fit can be reapplied from its result card.

The clean prediction is an engineering scenario, not a measured new-pump test.
Installed and clean rows can share a catalog size but have different performance.

## Run optimization

Open Optimize, select the pad/CFP workflow and review each well's saved-fit
readiness. Start a new run after saving changes; an existing result is a snapshot
of its run inputs. Unsaved Solver edits do not become optimizer defaults.

Current-pump operations retain the verified installed fit. Sizing runs include
clean replacements, including the same size when included in the pump grid.
Legacy/unverified fits use visible reference assumptions until refitted and saved.
Future wells borrow donor well properties with clean pump assumptions.

Pad optimization defaults to **Maximize oil within capacity**. Uncheck it to
enter a deliberate water price; that changes the objective to oil minus price
times machine water. E/M machines handle total water; I/S use lift water. A
separate frontier estimate describes the value of capacity, without charging
for the constraint twice. Both allocation engines share valid candidates;
installed pumps remain options even outside the replacement grid.

For a new-well study, add a unique future-well name and its donor on Pad review.
New rows default to **Required online**. Leave that checked to require the
proposed well in the result. Existing wells also have this constraint when an
engineer wants to prevent shutting them in. Missing required models or an
unserviceable required set fail explicitly. **Offline** describes baseline
status: excluded from a pad run, or available to bring online in CFP. It is
different from a constraint on the proposed plan.

Resize runs search the selected replacement grid. A hold-pumps choke run needs
an explicit planned nozzle/throat on every future row; it holds that hardware
while allocating lower-pressure settings and permitted shut-ins. Choke is
available on I/M/E; S-Pad currently requires its coupled resize workflow.
Duplicate future names and names colliding with existing wells are rejected.

CFP uses a **manual reference discharge**, which must describe the same online
configuration as the board and remain at or below 2,880 psi. Enter B/G/J PF
pressures under **Pad PF at the same reference conditions** if contemporaneous
measurements are available. Blank entries use line-loss assumptions. Unrelated
latest measurements are not mixed into this reference. CFP models changes in
water demand from its reference; the displayed modeled run-well water is not
measured total plant throughput. Bring-online/offset pairs appear together,
including combinations that only operate when both actions are taken.

Results disclose allocation/search status, unsupported pressure ranges and
minimum-flow conditions. A hydraulically closed plan below a recommended
operating range is conditional, not a qualified operating recommendation.
No recycle flow is assumed without a verified recycle model. See the
[capacity delivery](pad_cfp_capacity_delivery_2026-09-12.md) for validation and
remaining limitations.

## Explore watercut sensitivity

Expand **WC uncertainty** to view sampled oil and suction-BHP ranges, initially
+/-5 percentage points. By default the oil IPR and GOR remain fixed while the
liquid-anchor representation adjusts with WC. **Anchor measurement** is a
separate explicit basis that changes inferred oil deliverability. Failed
samples are disclosed; results refresh when inputs change. The range explores
the selected installed/clean scenario and does not save inputs or change the
optimizer. It is a sensitivity range, not a statistical confidence interval.

Combined Match Sensitivities results preserve the original inputs and comparison
targets. Apply is disabled when that study no longer describes the current well,
hardware, inputs or comparison. A sampled envelope is not a proof that an
unsampled case cannot match.

## Review pad coverage and stress cases

Every expected well has an explicit outcome. Missing inputs or failed physics
do not mean the optimizer recommends shutting that well in. Incomplete online
coverage makes the run exploratory and withholds whole-pad feasibility.

**Current pumps at plan header** and **Modeled hardware gain** compare current
and proposed hardware at the same pressure and well inputs. Measured test
production is separate; differences from a biased baseline model are not counted
as a hardware benefit.

For a complete I/M/E JPCO run, expand **Stress-test current and proposed plans**.
Choose WC/GOR/header ranges supported by your data. Optional joint cases state
their assumed co-movement. The same two fixed plans are checked against plant
capacity in each case. Unknown and infeasible cases stay visible. Gain ranges
cover only cases where both plans are feasible; preference counts are not
probabilities, and regret is relative only to the other plan. S-Pad and CFP
require a separate coupled study and are not supported by this new control.
