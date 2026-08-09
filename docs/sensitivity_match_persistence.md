# Applying and saving a sensitivity permutation

Written 2026-08-08 for Scott. Everything below was measured against the live
warehouse (reads only) or driven in the running app; every claim carries a
file:line or an observed number.

**Status: steps 1, 2, 3 and 5 shipped the same day** (see "Suggested order"
at the end - the section text below still describes the pre-fix behaviour,
which is what the fixes are measured against). **Step 4 is declined, not
deferred:** Scott's call, 2026-08-08 - the optimization runs need PF surface
pressure free to vary, so pinning a match PF in prop_hist would fight them.
`ppf_surf` stays live-seeded on every open. A permutation's PF still holds
for the session (it is hand-set state) and the value that made the match
rides into the save comment, which is where a number nobody can persist
belongs.

The ask: "the applied fit from the permutation of the sensitivity doesn't
stick when save, because if you push Apply IPR it jumps away from what the
sensitivity pushed out. Come with a detailed suggestion so the user can apply
and save the best permutation values (may mean more props in prop_hist)."

There are **three independent gaps** between "the permutation is on screen"
and "the well opens on it tomorrow". Fixing any one alone does nothing
visible. They are ordered by how much they cost to fix and how much they buy.

---

## 0. What actually happens today (observed, MPC-45)

Driven in the app against live data:

| step | ppf_surf | qwf | pwf | ResP | WC | GOR | kth |
|---|---|---|---|---|---|---|---|
| open (saved values restored 2026-08-03) | 3,400 | 382 | 1,033 | 1,700 | 0.83 | 105 | 0.30 |
| Sensitivity -> Apply permutation #2 | **3,910** | **439** | 1,033 | 1,700 | 0.83 | 105 | **0.53** |
| gauge upload makes a Vogel fit possible | 3,910 | **453** | **1,051** | **1,796** | **0.85** | **213** | 0.53 |
| hand-set qwf back to 439, click "Apply IPR to inputs" | 3,910 | **453** | 1,051 | 1,796 | 0.85 | 213 | 0.53 |

Read the third row carefully. Nobody clicked anything: the fit query resolved
and the one-shot auto-apply (`web/src/pages/SolverPage.tsx:141-146`) laid the
fit over six of the permutation's fields. The fourth row is the same
overwrite on demand (`web/src/pages/solver/IprControls.tsx:157-159`). Both
call the same function, `applyIprSeeds` (`web/src/state/params.ts:150-157`),
with the same six-field payload built at `server/services/ipr.py:154-165`:
`qwf, pwf, pres, form_wc, form_gor` and `surf_pres` when the anchor test
carried a wellhead pressure.

What survived both events: `ppf_surf` and `kth`. That is not a design; it is
just the complement of the fit's seed list.

### This is not only a sensitivity problem

The same overwrite runs on **every** well open. Of the 31 wells that carry
saved IPR values in prop_hist today, a fit is available and disagrees with
the saved numbers on 12 of them:

```
MPB-32: saved qwf/pwf 434/974  -> the fit writes 215/964
MPB-35: saved qwf/pwf 668/222  -> the fit writes 322/152
MPB-37: saved qwf/pwf 533/539  -> the fit writes 302/544
MPB-39: saved qwf/pwf 442/472  -> the fit writes 894/410
MPC-14: saved qwf/pwf 212/328  -> the fit writes 224/255
(12 diverge, 5 agree, 14 have no usable fit so the saved values stand)
```

So "restores on every open" currently means "restores, then the fit
overwrites it before you look", for any unlocked field on a well whose tests
support a fit. Locked WC / GOR / ResP are the exception - `applyIprSeeds`
skips them (`params.ts:153-155`), which is why locking has felt like the only
way to make anything stick.

The auto-apply was added for a good reason and the reason is written down at
`SolverPage.tsx:122-133`: it makes the chart curve and the solve agree on
first paint, and it heals wells whose pre-2026-08-03 saved values carry the
old double-converted liquid rate. Both goals are still valid. Neither
requires overwriting a reviewed save.

---

## Gap 1 - state: the fit outranks the engineer

**Mechanism A (the one you hit).** `SolverPage.tsx:143` guards the auto-apply
with `fitAppliedFor === well`. That latch is per well in the store, so the
normal Solver -> Match Sensitivities -> Apply -> Solver loop is safe. It is
NOT safe when the fit had not yet latched: a direct link to /sensitivity, a
fit still in flight, a well whose fit only becomes possible later (the gauge
upload above). Then the fit lands after the Apply and wins.

**Mechanism B (total wipe).** Touching the Well Test History lookback or max
tests calls `setWindow` (`Sidebar.tsx:234,241`), which nulls `seededFor` and
`fitAppliedFor` (`params.ts:114`). That re-opens the Layout seeding gate
(`Layout.tsx:42`) and `applyContext` rebuilds the entire params object from
`DEFAULT_PARAMS + ctx.seeds` (`params.ts:132`). Every knob the permutation
set is gone, including the ones the fit never touches. `selectWell` does the
same (`params.ts:116-127`), so re-picking the same well from the selector
also wipes it.

**Nothing persists it either.** The params store has no localStorage by
deliberate decision (`params.ts:96-99`), and both sensitivity endpoints are
declared read-only (`server/services/sensitivity.py:8-9`). A permutation
lives until the next reset or reload. The Apply button's own tooltip says so
(`TopRunsTable.tsx:157`).

### Proposed fix

1. **Never auto-apply over a reviewed save.** The server already labels the
   seeds: `well_context` sets `ipr_source = "saved"` when the saved-values
   overlay wins (`server/services/wells.py:522`). Add that to the guard at
   `SolverPage.tsx:143`. One condition, and the 12 wells above stop being
   silently overwritten. The healing rationale for legacy double-converted
   rates is better served by a one-time data fix than by a permanent
   overwrite of every reviewed save.

2. **Make "engineer owns this field" real state.** Add a per-well
   `manualFields: Set<keyof SimParams>` to the params store. `set` and
   `setMany` (the sidebar, the sensitivity Apply, Auto-match BHP) add their
   keys; `applyIprSeeds` skips them exactly the way it skips locked fields
   today (`params.ts:150-157`); `applyContext` and `selectWell` clear the
   set. "Apply IPR to inputs" stays the explicit escape hatch: it clears
   ownership for the six fields it writes, because clicking it IS the
   instruction to take the fit.

3. **Stop `setWindow` from wiping the bench.** Changing the lookback should
   refetch tests and re-run the fit, not rebuild params from defaults. Null
   `fitAppliedFor` if you like (the fit legitimately changed), but leave
   `seededFor` alone so `applyContext` does not re-run. If a full reseed is
   genuinely wanted there, it should be a button that says so.

4. **Carry the permutation's identity.** The Apply button should also stash
   `{run index, score, knob values}` for the well so the Solver can show
   "showing permutation #2 of the 2026-08-08 study, score 0.057" and the Save
   comment box can be pre-filled with it. Zero schema cost - the comment
   table already takes free text.

Items 1 and 3 are small and I would ship them first; item 2 is the real fix
and is maybe half a day including tests.

---

## Gap 2 - persistence: Save writes 9 of the 14 knobs

`SaveIprRequest` (`server/schemas.py:710-731`) carries qwf_liq, pwf, res_pres,
form_wc, form_gor, surf_pres, ken, kth, kdi. `save_ipr_values`
(`woffl/gui/ipr_anchor.py:778-803`) pushes those as six prop rows plus up to
three `jpfric_*` rows. The live `prop_xref` whitelist has 26 ids:

```
casing_absruff, casing_inn_dia, casing_out_dia, form_gas_sg, form_gor,
form_gor_lock, form_oil_api, form_wat_sg, form_wc, form_wc_lock, ipr_pwf,
ipr_qwf_liq, ipr_wt_uid, jpfric_diffuser, jpfric_entry, jpfric_nozzle,
jpfric_throat, jpump_md, resvr_bubb, resvr_press, resvr_press_lock,
resvr_temp, surf_press, tubing_absruff, tubing_inn_dia, tubing_out_dia
```

Against the frozen 14-knob table (`server/services/sensitivity.py:114-173`):

| knob | field | prop id | saved today? |
|---|---|---|---|
| IPR anchor rate | qwf | ipr_qwf_liq | yes |
| IPR anchor BHP | pwf | ipr_pwf | yes |
| Reservoir pressure | pres | resvr_press | yes |
| Water cut | form_wc | form_wc | yes (capped 0.99) |
| Intake GOR | form_gor | form_gor | yes |
| Wellhead pressure | surf_pres | surf_press | yes |
| Entrance loss | ken | jpfric_entry | yes, when changed |
| Throat loss | kth | jpfric_throat | yes, when changed |
| Diffuser loss | kdi | jpfric_diffuser | yes, when changed |
| **Bubble point** | bubble_point | **resvr_bubb exists** | **no write path** |
| **Formation temp** | form_temp | **resvr_temp exists** | **no write path** |
| **PF surface pressure** | ppf_surf | **none** | no |
| Nozzle size | nozzle_no | none (deliberate) | no |
| Throat size | area_ratio | none (deliberate) | no |

### Proposed fix

**(a) Free, today: add bubble_point and form_temp to the save.** Both prop
ids already exist and are already in `review_persistence.FIELD_MAP:90-108`;
only the Solver's save path skips them. Two optional fields on
`SaveIprRequest`, two entries in the `save_ipr_values` payload, same
only-when-changed discipline the friction coefficients use so an untouched
default never materialises as a canon row. No ask to Kaelin.

**(b) One new prop id: the PF pressure the match needs.** The study says PF
surface pressure is one of the strongest levers on the match (on MPC-45 it
moved suction BHP by +/-150 psi, the third largest of fourteen inputs), and
it is the single most common thing a permutation changes. It was deliberately
left out of the 2026-07-30 ask (`docs/prop_hist_asks.md`, ask (c)) on the
grounds that it is "live-detected or derivable on open" - true of the PLANT
reading, not of the value a match requires. Those are different quantities
and should not share a name:

| prop_id | description | units | category |
|---|---|---|---|
| `match_ppf_surf` | PF surface pressure the reviewed match was solved at | psig | mechanical |

Deliberately NOT asked for: `jp_nozzle` / `jp_throat_ratio`. Pump truth stays
with the JP tracker (your 2026-07-30 call) and a permutation that changes the
pump is a recommendation, not a measurement. It belongs on the Batch Run /
Optimization side, where it already lives. If a match only closes with a
different pump, the honest output is "this well needs 9B", not a prop row
claiming it has one.

**(c) The save comment already carries provenance.** One click writes up to
nine rows under a single `entry_datetime` (`ipr_anchor.py:810`) and the
comment table joins on that stamp. Pre-filling the comment with "sensitivity
permutation #2, score 0.057, 2026-08-08" costs nothing and makes the Well
Database "Why" column answer the question by itself.

---

## Gap 3 - restore: the two values that are overwritten after the overlay

`server/services/wells.py` builds the well context in five stages, in this
order: (a) characteristics, (b) pump identity from the JP tracker, (c) the
Vogel fit from tests, (d) the saved-IPR overlay, (e) the live PF reading.

Stage (e) writes `seeds["ppf_surf"]` unconditionally at `wells.py:542` (pad
fallback) or `wells.py:553` (live reading) - **after** the saved overlay. So
even once `match_ppf_surf` exists, a saved value would be overwritten on
every open unless stage (e) learns to yield. Same shape for the pump: stage
(b) always takes nozzle and throat from the tracker (`wells.py:346-349`).

### Proposed fix

- Stage (e) seeds `ppf_surf` only when no `match_ppf_surf` is saved, and the
  PF card reports which one is in play ("PF 3,910 psi from the saved match"
  vs "live pad reading 3,400 psi"), with a one-click "use the live reading"
  to drop back.
- Leave the pump alone. The tracker is right about what is downhole.

---

## What shipped, 2026-08-08

**1. The fit no longer outranks the engineer.**
`SolverPage.tsx` skips the open-time auto-apply when the server reports
`ipr_source === "saved"` (new field on `WellContext`, already computed for
the optimizer's Fit column, now declared in `schemas.py` so it survives the
response model). The latch is still set so the chart's settle gate closes.
`params.setWindow` no longer nulls `seededFor`, so changing the lookback
refetches tests and re-runs the fit without rebuilding the bench from
defaults.

*Verified:* MPB-35 opened on its saved 668 BLPD @ 222 psi. Before the change
the fit wrote 322 / 152 over it on every open.

**1b. The drawn curve follows the sidebar, not the fit.** Falling out of the
above: the IPR chart and the Rate Calculator used to anchor on the server fit
ahead of the sidebar. That was invisible while the fit was auto-applied into
the sidebar on every open, but the moment the engineer's numbers outrank the
fit it drew one curve and solved another - MPB-35 opened on its saved
668 BLPD @ 222 psi while the chart header read "Qmax 330 BPD" and the
calculator quoted 322 BLPD. Both now anchor on `params.qwf / pwf / pres`, the
same numbers the solve uses, and the comparison-test branch of that
precedence is gone (a comparison test is a scatter point, not a curve). The
weak-R2 warning only shows while the curve still IS the fit.

*Verified:* MPB-35 now reads "Qmax: 697 BPD" (668 at 222 psi against ResP
1,652) and the calculator 679 BLPD; on four auto-fitted wells (MPE-24, -35,
-37, -42) the seeds still equal the fit coefficients to under 1 unit, so the
ordinary case draws exactly what it drew before.

**2. Hand-set inputs are held against the fit.**
`params.manualFields` records every field the engineer moved - sidebar typing,
an applied permutation, Auto-match BHP. `applyIprSeeds` skips those the same
way it skips locked WC/GOR/ResP. "Apply IPR to inputs" passes `release`,
which hands exactly the seeded fields back to the fit, so the button still
does what it says. The IPR Anchor card shows the count and offers "take the
fit instead".

*Verified:* on MPB-35, applied permutation 1 of 27 (ppf 2,977, kth 0.050, qwf
568), then changed the lookback 6 -> 12 months: all three survived (that
wipe used to be total). "Take the fit instead" then moved qwf 568 -> 322 and
pwf 222 -> 152 and cleared the note, leaving the two fields the fit does not
seed alone.

**3. Bubble point and formation temp are saveable.**
`SaveIprRequest.bubble_point` / `.form_temp` -> `save_ipr_values` ->
`resvr_bubb` / `resvr_temp`, inside the same one-statement batch and the same
batch stamp. The client sends one ONLY when it differs from the seed the
server assembled (`changedFromSeed` in `IprControls.tsx`), so an ordinary
save never re-writes the characterization it was handed.

*Verified:* payload captured client-side with the request intercepted (no row
written): untouched -> both `null`; after a real edit -> `bubble_point: 2999`,
`form_temp: null`. Contract tests in `test_web_save_ipr.py` and
`test_ipr_saved_values.py`.

One wrinkle worth knowing: these two restore through the characteristics
pivots, not through the saved-IPR overlay, and the chars frame is cached for
an hour (a cold chars query costs 8.2 s, so clearing it on every save would
reintroduce the hang we just removed). A fresh session inside that hour can
still show the old value. That is exactly how `resvr_temp` / `resvr_bubb`
already behave when pad review writes them.

**5. The permutation carries its identity.**
The Apply button stores a note - "Sensitivity permutation 1 of 27 (score
0.5912): PF surface pressure 2,977, Throat loss (kth) 0.050, IPR anchor rate
568" - which the IPR Anchor card displays and the save comment prefills, so
`woffl_eng_comment` records the study that produced the curve. Cleared by
"take the fit instead", a well change, or a context reseed.

**4. Declined.** PF surface pressure stays live-seeded and unpersisted, so
optimization runs keep varying it freely.
