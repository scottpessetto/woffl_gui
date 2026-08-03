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

**✅ DELIVERED 2026-07-30 — self-served.** The grant did NOT need Kaelin after all:
Scott's own rights on catalog `mpu` proved sufficient to grant, and
`GRANT MODIFY ON TABLE mpu.wells.prop_hist TO \`2013fc45-c30e-40ac-bef0-df0a758faa3c\``
executed under his identity and verified via `SHOW GRANTS` (MODIFY now held at TABLE
level). `ALLOW_DATABRICKS_WRITES` is enabled in `app.yaml` the same day, and hosted
attribution is handled: `app.py` registers an entry-user provider that reads
`X-Forwarded-Email` per session (falling back to `current_user()` = the SP), so hosted
saves stamp the real engineer. Remaining step: **redeploy the app**, then confirm a
Solver 📌 save round-trips. The 2026-07-29 investigation below is kept for the record.

**Status 2026-07-29 — resolved to a single outstanding grant. The original ask over-asked; see below.**

The app is **`woffl`**, deployed in Hilcorp's **dev** workspace (`https://woffl-1097886750912498.aws.databricksapps.com`, source `/Workspace/MPU/woffl/woffl_gui`, creator Scott, Medium compute). Its **App ID — which for Databricks Apps *is* the service principal client id — is `2013fc45-c30e-40ac-bef0-df0a758faa3c`** (confirmed by Scott off the app page, and independently matching a principal already present in `SHOW GRANTS` on `mpu`).

`SHOW GRANTS \`2013fc45-…\` ON CATALOG mpu / SCHEMA mpu.wells / TABLE mpu.wells.prop_hist` returns:

| Privilege | Held? | Notes |
|---|---|---|
| `USE CATALOG` | ✅ | on catalog `mpu` |
| `USE SCHEMA` | ✅ | inherited from catalog |
| `SELECT` | ✅ | inherited — **reads of `prop_hist` already work on the host** |
| `BROWSE` / `EXECUTE` / `READ VOLUME` | ✅ | |
| **`MODIFY`** | ❌ | **the only gap** |

**So two-thirds of the original ask was already satisfied.** The `use_catalog` and `SELECT` asks are moot — the app has held both all along, which means saved IPR pins should *already display* on the hosted app; only the write half is blocked. (Verifiable without any grant: the two existing pins are **S-054** → test 2026-07-14 and **C-045** → test 2026-07-04, both still resolving, so the anchor selector should default to those specific tests on the hosted app today.)

**The entire remaining ask is one statement:**

```sql
GRANT MODIFY ON TABLE mpu.wells.prop_hist TO `2013fc45-c30e-40ac-bef0-df0a758faa3c`;
```

**Two facts that shape how this gets done:**

1. **Kaelin must issue it — Scott cannot self-serve.** `mpu.wells.prop_hist` and the `mpu.wells` schema are both owned by `Kaelin.Ellis@hilcorp.com`. Scott holds `MODIFY` (which lets him *write* — proven by two real `ipr_wt_uid` rows he's already saved from his desktop, enthid 36093350 on 2026-07-07 and 36569853 on 2026-07-21) but not ownership or `MANAGE`, which is what lets you *grant to someone else*.
2. **Workspace doesn't matter.** Unity Catalog privileges are scoped to the **metastore**, not the workspace. `prop_hist` lives in metastore `60c8f8a6-2d6b-499f-a91a-7a225586319f` (`hilcorp`); the dev app SP's grants are readable from `dbc-42b811e2-2a82` precisely because both attach to it. One grant covers the app, and it can be verified from either side.

Until that grant lands, `ALLOW_DATABRICKS_WRITES` stays commented out in `app.yaml` — turning it on without `MODIFY` makes every pad-review save attempt a write, fail on permissions, and surface a warning, which is strictly worse than the current silent no-op.

**Open item — audit identity on the hosted app.** `resolve_entry_user()` falls back to `SELECT current_user()`, which on Databricks Apps resolves to the **app's service principal**, not the engineer clicking Save. Left as-is, every hosted save is stamped with one identity and the `entry_user` audit trail is lost. The fix is the one its own docstring anticipates: thread the real Streamlit user through (Apps forward it in request headers, e.g. `X-Forwarded-Email`, readable via `st.context.headers`) and set `WOFFL_ENTRY_USER`, which already takes precedence. Worth doing **before** the hosted test, so the test also validates attribution.

---

### (c) Phase 2/3 prop_xref entries — the full review-persistence set

**✅ DELIVERED 2026-07-30 — self-served.** Scott's catalog-level `MODIFY` inherits to
`prop_xref`, so the GUI-side script added the rows directly (idempotent, parameterized,
via the gated `execute_write`). What actually landed differs from the draft below after
Scott's field review:

* **Added (5):** `ipr_qwf_liq`, `ipr_pwf`, `form_wc`, `form_gor`, `surf_press` — with
  existing `resvr_press` these fully persist "the curve and rate the engineer saw fit."
* **Deliberately NOT added, per Scott:** `jp_nozzle` / `jp_throat_ratio` (pump truth
  stays with the JP tracker), `well_reviewed` / `well_offline` (workflow state, not well
  characteristics), `jpump_direction` / `field_model_code` / `ppf_surf` (live-detected
  or derivable on open).

The loop is closed in code the same day: the Solver's "📌 Save IPR as well default"
pushes the pin AND the sidebar's current values (`ipr_anchor.save_ipr_values`); pad
review saves write through `review_persistence.sync_pad`; on well-open,
`sidebar._seed_saved_ipr` restores the values with **latest-timestamp-wins** precedence
against the anchor pin. `prop_value_str` is no longer needed — every persisted field is
numeric. The historical draft ask is kept below for the record.



**Status 2026-07-30 — this ask is now THE gate on full review persistence, and it grew.**
Per Kaelin's stated preference (via Scott): the GUI's pad-review state persists into
`prop_hist` itself — timestamped property rows, latest per (well, prop) wins — rather
than any new table or file. The GUI now does this (`woffl/gui/review_persistence.py`):
every reviewed property whose prop_id already exists **writes through today** (a saved
review becomes the well's latest characterization, user-stamped), and the remaining
fields activate automatically the moment their prop_xref rows appear. Verified against
the live `prop_xref` 2026-07-30 (18 rows).

**Ask:** add the following `prop_xref` rows (all `value_type = double`; categoricals use
documented numeric encodings — option (ii) of the original ask — so NO schema change to
`prop_hist` is needed):

| prop_id | prop_name | units | category | encoding |
|---|---|---|---|---|
| `ipr_qwf_liq` | IPR total liquid rate at anchor | bbl/d | reservoir | — |
| `ipr_pwf` | IPR flowing pressure at anchor | psig | reservoir | — |
| `form_wc` | Formation water cut | fraction | reservoir | 0–1 |
| `form_gor` | Formation gas-oil ratio | scf/bbl | reservoir | — |
| `surf_press` | Wellhead surface pressure | psig | mechanical | — |
| `ppf_surf` | Power-fluid surface pressure (reviewed pin) | psig | mechanical | — |
| `jp_nozzle` | Reviewed jet pump nozzle number | unitless | mechanical | numeric nozzle no. |
| `jp_throat_ratio` | Reviewed jet pump throat ratio | unitless | mechanical | X=0 A=1 B=2 C=3 D=4 E=5 |
| `jpump_direction` | Circulation direction | unitless | mechanical | 0=reverse 1=forward |
| `field_model_code` | PVT field model | unitless | reservoir | 0=Schrader 1=Kuparuk |
| `well_offline` | Reviewed offline / bring-online-candidate flag | unitless | mechanical | 0/1 |
| `well_reviewed` | Well present in the GUI review | unitless | mechanical | 1 = reviewed; **NULL = removed** (the `ipr_wt_uid` NULL-unpin precedent) |

**Why:** these are exactly the fields a *restore* needs that don't map to existing ids —
the reviewed pump, flags, and the IPR operating point. Until they exist, persistence is
deliberately partial: canonical properties + the `ipr_wt_uid` anchor pin round-trip, but
each session still re-picks pumps and offline flags. The GUI's captions say so. Encodings
above are frozen in `review_persistence.FIELD_MAP`; nothing else changes on delivery day.

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

### (g) Free-text engineer comment saved with an IPR — needs a STRING value column

**Ask:** Two changes, both on `mpu.wells`:

1. Add a nullable string column to `prop_hist`:
   ```sql
   ALTER TABLE mpu.wells.prop_hist ADD COLUMN prop_value_str STRING;
   ```
2. Add one `prop_xref` row:
   | prop_id | prop_name | units | category | value_type |
   |---|---|---|---|---|
   | `eng_comment` | Engineer Comment on Saved IPR | unitless | reservoir | `string` |

**Why:** Scott's request — when an engineer clicks *Save IPR as well default*, they should be
able to leave a short note saying **why** they chose that anchor / those values, so the next
person opening the well sees the reasoning rather than just the numbers. Same append-only,
latest-wins semantics as every other prop; no new table, no new write path.

**Why it's blocked today (verified live 2026-08-03):** `prop_hist` has exactly one value
column and it is numeric —

```
enthid bigint | prop_id string | prop_value double | entry_datetime timestamp | entry_user string
```

There is nowhere to put text. `prop_value_str` does not exist (ask (c) retired it as
unnecessary when every persisted field was numeric — this is the case that brings it back).
`SHOW TABLES IN mpu.wells` confirms no other table can hold the comment either: `woffl_active`
is `(enthid, well_bore, is_active, entry_date, entry_user)` and `down_xref` is a code lookup.

**Why we can't self-serve it like ask (b):** the app SP
(`2013fc45-c30e-40ac-bef0-df0a758faa3c`) holds `MODIFY` on `TABLE mpu.wells.prop_hist`, which
is data-level (INSERT) only. `ALTER TABLE … ADD COLUMN` needs table ownership; `SHOW GRANTS`
shows `Kaelin.Ellis@hilcorp.com` with `ALL PRIVILEGES` on catalog `mpu`.

**Compatibility:** additive and non-breaking. Existing readers select named columns
(`prop_value`), and `vw_prop_mech` / `vw_prop_resvr` are numeric pivots that would ignore a
string column. Rows for numeric props keep `prop_value_str` NULL; the `eng_comment` row is the
mirror image — `prop_value` NULL, text in `prop_value_str`.

**GUI side, once the column lands:** `prop_hist_client.push_prop` gains a string branch (the
INSERT already uses named binds, so it's one extra parameter), the Solver's *Save IPR as well
default* control gains a note field, and the Well Database history view renders the comment
beside the values it was saved with. Roughly an afternoon — the schema is the whole blocker.

---

## DART Feedback (suggestions from the mppush.py reference)

DART's `push_prop` patterns are solid: parameterized SQL, prop_xref whitelist validation, and enthid resolution guards are all sound and the GUI mirrors them. Three notes:

1. **Entry user identity:** DART uses `os.getlogin()` for `entry_user`. On Databricks Apps, every user's code runs under the service principal container identity, so `os.getlogin()` returns the container user, not the engineer. The GUI resolves identity explicitly: locally via `SELECT current_user()` from the SQL session, and on the deployed app via the Streamlit user identity once the SP grant lands. We recommend `entry_user` as an explicit parameter in `push_prop` rather than reading `os.getlogin()` directly; callers supply the actual identity.

2. **`delete_prop` vs. append-only history:** DART's `delete_prop` corrects same-day edits via DELETE. With `entry_ts` in place (ask (a)), corrections become new rows instead (prop_hist remains a true history and the table's append-only feature is preserved). We recommend retiring `delete_prop` — or at minimum, permission-gating it (the SP doesn't need DELETE) to enforce append-only discipline.

3. **Unit tests on mppush:** The reference implementation would benefit from a few unit tests covering whitelist rejection, enthid guards (0 or multiple matches), and parameter correctness. Currently these are validated live only.

---

## Closing Note

**None of these asks block Phase 1.** The GUI is shipping now with Phase 1 (saved IPR anchor via `ipr_wt_uid`) against the current schema. Items (a–f) are prerequisites for Phases 2–3. Ask (g) is the only one that blocks a feature outright — the engineer-comment box can't be built at all until `prop_hist` has a string column. Ask back if anything needs clarification or if you'd like to schedule an alignment call on the categorical encoding decision.

