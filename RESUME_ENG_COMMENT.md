# Archived engineer-comment resume note

Superseded September 8, 2026. The August 3 crash-recovery instructions described
uncommitted files and Streamlit modules that no longer represent this repository.
Do not rerun their DDL, sample production writes or commit checklist. The original
note remains available in Git history.

The existing `mpu.wells.woffl_eng_comment` ledger is used for engineer comments
and, under context `pump_calibration_v1`, self-contained installed-pump fits.
Well-input saves and pump-calibration saves are separate; both use the existing
gated executor and request identity. Databricks INSERT rowcount may be -1;
a non-raising write is success. Structured pump records must be checked against
the 500-character limit before the comment writer can truncate them.

Continue from [the session handoff](docs/session_learnings_2026-09-08.md),
[pump calibration scope](docs/pump_calibration_scope_2026-09-08.md) and
[AGENTS.md](AGENTS.md). The latest recorded local verification is 1,861 Python
tests, 8 frontend tests, a production build and intercepted browser save checks.
No September 8 production save or deployment was performed.
