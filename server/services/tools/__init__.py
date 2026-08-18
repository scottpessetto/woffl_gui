"""Scott's Tools - the secret-menu engineering tools, server side.

Ported from ``woffl/gui/scotts_tools/`` when the Streamlit app was deleted
(2026-08-18). Each module here is the tool's ENGINE with the Streamlit
rendering stripped: pure functions plus a cached data pull, returning
JSON-ready dicts for the React pages under ``web/src/pages/tools/``.

Every tool is READ-ONLY. Nothing here writes to prop_hist or anywhere else -
the old JP Calibration tab's write-preview was always inert, and it stays
inert (it renders SQL for a human to run, it does not execute it).
"""
