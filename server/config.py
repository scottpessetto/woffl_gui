"""Server configuration and path resolution.

All paths resolve from ``Path(__file__)`` (repo convention - never cwd).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
JP_DATA_DIR = REPO_ROOT / "woffl" / "jp_data"
SURVEY_DIR = JP_DATA_DIR / "well_surveys"
WEB_DIST = REPO_ROOT / "web" / "dist"

JP_HISTORY_XLSX = DATA_DIR / "jetpump_history.xlsx"
JETPUMP_DIMENSIONS_JSON = DATA_DIR / "jetpump_dimensions.json"
JP_CHARS_CSV = JP_DATA_DIR / "jp_chars.csv"

DEFAULT_TEST_MONTHS = 6

# Cache TTLs (seconds) - mirror the Streamlit @st.cache_data sites.
TTL_WELL_TESTS = 86_400
TTL_JP_HISTORY = 86_400
TTL_EXTENDED_TESTS = 86_400
TTL_CHARS = 3_600
TTL_PF_LATEST = 3_600
TTL_PROFILES = 3_600
TTL_SAVED_IPR = 300
TTL_PROP_HISTORY = 300
TTL_WELL_SORT = 3_600
TTL_XV_STATUS = 300


def writes_enabled() -> bool:
    """Same truthy convention as databricks_client._write_gate_enabled.
    Gates the three write endpoints (save-ipr, ipr-pin delete, prop-lock)
    at the router and hides the UI save controls via /api/meta."""
    return os.environ.get("ALLOW_DATABRICKS_WRITES", "").strip().lower() in ("1", "true", "yes")


def is_deployed() -> bool:
    return bool(os.environ.get("DATABRICKS_CLIENT_ID")) and bool(
        os.environ.get("DATABRICKS_CLIENT_SECRET")
    )
