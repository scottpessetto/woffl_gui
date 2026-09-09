# Saving a well fit for optimization

Current behavior: September 8, 2026. See [the session handoff](session_learnings_2026-09-08.md)
for verification and deployment status. The app is React/FastAPI; older Streamlit
tab/CSV instructions do not describe the current workflow.

## Prepare the well

Open the well in Solver. Check the tracker pump and installation date, wellbore
geometry, circulation, IPR anchor, reservoir pressure, WC, GOR and pressures.
The IPR rate is **formation liquid**, excluding returned power fluid. Use the
comparison test and gauge evidence to assess whether the current inputs make sense.

Use **Save well inputs** under IPR Anchor to retain supported IPR/fluid inputs.
This includes changed bubble point/formation temperature where supported. PF
pressure remains a live/run input, and tracker hardware is not rewritten by this
save. Pump loss coefficients and nozzle area have a separate save.

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
