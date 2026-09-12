# Saving a well fit for optimization

Current behavior: September 12, 2026. See [the edit/preview/save delivery](well_input_save_workflow_2026-09-12.md)
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
the plot explains when sidebar edits are excluded. These comparisons use clean
pump losses and are retrospective, not independent qualification of sizing gains.

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

Pad optimization uses oil minus the selected water price times machine water.
E/M machines handle total water; I/S use lift water. Both allocation engines
share the objective and candidates. CFP moves are changes from the measured plant
anchor. See [pad formulation](optimization_redesign_2026-09.md) and
[CFP methodology](cfp_moves_methodology.md) for the exact constraints.

## Explore watercut sensitivity

Expand **WC uncertainty** to view sampled oil and suction-BHP ranges, initially
+/-5 percentage points. The liquid IPR anchor and GOR remain fixed. Failed
samples are disclosed; results refresh when inputs change. The range explores
the selected installed/clean scenario and does not save inputs or change the
optimizer. It is a sensitivity range, not a statistical confidence interval.
