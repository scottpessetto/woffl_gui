# WOFFL GUI prop_hist Persistence — Asks for Kaelin

**From:** Scott Pessetto, WOFFL GUI  
**To:** Kaelin Ellis (mpu.wells owner, DART author)  
**Date:** 2026-07-07  
**Subject:** Schema and access enhancements for GUI property persistence

---

## Context

The WOFFL GUI is beginning to persist engineer selections into `mpu.wells.prop_hist`. **Phase 1** (shipping now) stores the chosen IPR anchor test for each well via `ipr_wt_uid`, so that every user and session auto-defaults to the saved IPR on re-entry. Phases 2–3 extend this to IPR scalars, pump identity, test data, and pad offline flags. The schema and permission asks below unblock those next phases without impacting Phase 1.

---

## Schema Asks

### (a) Entry timestamp column + view ordering — ✅ DELIVERED 2026-07-08

**Ask:** Add `entry_ts TIMESTAMP` to `prop_hist` and update `vw_prop_mech`/`vw_prop_resvr` to `ORDER BY entry_ts DESC` (currently ordering by `entry_date DESC`).

**Why:** Same-day edits are currently unordered, making tie-breaking arbitrary when the same well/property is modified twice in one day. A timestamp eliminates the ambiguity. Further, it makes "corrections as new rows" feasible instead of relying on DELETE (see DART feedback below) — entries are now truly immutable and historyable.

**Delivered:** Kaelin shipped this 2026-07-08 by renaming/retyping the column in place rather than adding a new one: `entry_date DATE` → `entry_datetime TIMESTAMP` (existing rows survived, with dates becoming midnight timestamps). `vw_prop_mech`/`vw_prop_resvr` were updated to match. The GUI's `prop_hist_client.py` (`push_prop`/`fetch_latest_prop`) was updated the same day to write/read `entry_datetime` (a timezone-aware UTC `datetime` on write) and order by it — same-day re-pins now resolve deterministically to the later timestamp instead of being unordered. Text above is kept for the historical record.

---

### (b) Service-principal SELECT + MODIFY + use_catalog for deployed app

**Ask:** Grant `use_catalog` plus `SELECT` **and** `MODIFY` on `mpu.wells.prop_hist` to the Databricks Apps service principal running the GUI on the production cluster.

**Why:** Phase 1 currently works only locally (Scott's desktop). The deployed app container runs all code as the service principal; until it has write permission, saving stays local-only. SELECT matters independently: the GUI reads saved IPR pins by querying `prop_hist` directly (the `ipr_wt_uid` rows aren't pivoted into the views), so without SELECT the hosted app can't even *display* anchors saved locally. This is the gate for shipping to production.

---

### (c) Phase 2 prop_xref entries + categorical encoding decision

**Ask:** Add five new entries to `prop_xref` for forced/manual IPR persistence:
- `ipr_qwf_liq` (total liquid rate at the anchor, bbl/d)
- `ipr_pwf` (flowing pressure, psi)
- `form_wc` (water cut, fraction 0–1)
- `form_gor` (gas-oil ratio, scf/bbl)
- `surf_press` (surface pressure, psi)

Also decide on categorical value encoding: either (i) add a `prop_value_str` column to `prop_hist` for categorical properties like `jpump_direction` (`"reverse"` or `"forward"`), or (ii) document numeric encodings (e.g., jpump_direction: 0=reverse, 1=forward) and store integers.

**Why:** Phase 2 will let users save and auto-restore their IPR choices (e.g., user overrides WC to 0.45 → next session auto-restores 0.45). The categorical decision matters for pump identity and test flags that don't fit `double`.

---

### (d) Manual well-tests table

**Ask:** Create a new table `manual_well_tests` (or similar) to store hand-entered well tests:
```
Schema sketch:
  enthid (bigint, FK to vw_well_header)
  test_date (date)
  oil_rate (double, bbl/d)
  water_rate (double, bbl/d)
  total_rate (double, bbl/d)
  bhp (double, psi)
  gor (double, scf/bbl)
  lift_water (double, bbl/d)
  whp (double, psi)
  entered_by (string, user identity)
  entered_at (timestamp)
```

**Why:** Phase 2 lets engineers hand-enter provisional well tests (e.g., quick pressure survey before a JPCo visit) for IPR fitting without waiting for SCADA to populate `vw_well_test`. A stable table lets the workflow store and recall these tests; audit columns (`entered_by`, `entered_at`) keep provenance clear.

---

### (e) Preferred un-pin semantics for ipr_wt_uid

**Ask:** Confirm: should the GUI write `prop_value = NULL` as the marker to clear a saved IPR, or do you prefer a different convention (e.g., a separate flag column)?

**Why:** Phase 1 includes a "Clear saved IPR" button. We originally planned a negative-number sentinel (`-1`), but live data showed that's unsafe: `wt_uid` in `vw_well_test` is a signed integer spanning roughly -3.6M to +3.1M (the large majority are actually negative), so no numeric value is reserved — a real pin can collide with any sentinel we pick. The GUI now writes SQL `NULL` as the un-pin marker instead (still append-only, no DELETE); when W2 reads the pin, it treats a NULL/NaN `prop_value` as "no saved anchor" and applies no sign-based rule at all. This is simple and visible in history, but we want to align on your preference before we treat it as final.

**Update (2026-07-08):** The `entry_date` → `entry_datetime` migration in ask (a) appears to have dropped at least one existing NULL-`prop_value` row along the way — worth confirming with Kaelin whether that was incidental to the column rename/retype or something to watch for, since correctness of the un-pin convention above depends on NULL rows surviving migrations intact.

---

### (f) MODIFY on woffl_active for offline-flag round-trip (optional, lower priority)

**Ask:** Grant `MODIFY` on `woffl_active` to support Phase 2's "store offline" flag: letting engineers mark a review store as offline (e.g., for re-optimization later) and round-trip that flag to the app.

**Why:** Optional for Phase 2; listed here for completeness. Not on the critical path for Phase 1.

---

## DART Feedback (suggestions from the mppush.py reference)

DART's `push_prop` patterns are solid: parameterized SQL, prop_xref whitelist validation, and enthid resolution guards are all sound and the GUI mirrors them. Three notes:

1. **Entry user identity:** DART uses `os.getlogin()` for `entry_user`. On Databricks Apps, every user's code runs under the service principal container identity, so `os.getlogin()` returns the container user, not the engineer. The GUI resolves identity explicitly: locally via `SELECT current_user()` from the SQL session, and on the deployed app via the Streamlit user identity once the SP grant lands. We recommend `entry_user` as an explicit parameter in `push_prop` rather than reading `os.getlogin()` directly; callers supply the actual identity.

2. **`delete_prop` vs. append-only history:** DART's `delete_prop` corrects same-day edits via DELETE. With `entry_ts` in place (ask (a)), corrections become new rows instead (prop_hist remains a true history and the table's append-only feature is preserved). We recommend retiring `delete_prop` — or at minimum, permission-gating it (the SP doesn't need DELETE) to enforce append-only discipline.

3. **Unit tests on mppush:** The reference implementation would benefit from a few unit tests covering whitelist rejection, enthid guards (0 or multiple matches), and parameter correctness. Currently these are validated live only.

---

## Closing Note

**None of these asks block Phase 1.** The GUI is shipping now with Phase 1 (saved IPR anchor via `ipr_wt_uid`) against the current schema. Items (a–f) are prerequisites for Phases 2–3. Ask back if anything needs clarification or if you'd like to schedule an alignment call on the categorical encoding decision.

