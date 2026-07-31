"""Per-well property save history — the prop_hist audit trail, readable.

Scott, 2026-07-30: *"add a view on the well database tab that allows user to
view all the history of saves on a well."* prop_hist is append-only, so the
full audit trail already exists — every 📌 save, pad-review push, friction
calibration, IPR pin and the original DART bulk load, each with value,
timestamp and user. This module fetches and shapes it; the Well Database page
renders it.

Fetch is one query (all rows for the well's enthid, prop names joined from
prop_xref). Shaping is pure and tested.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Rendered specially: the IPR anchor pin is a well-test ID, not a quantity.
PIN_PROP_ID = "ipr_wt_uid"


def fetch_prop_history(well_name: str) -> pd.DataFrame:
    """Every prop_hist row for one well, newest first, names joined from
    prop_xref. Raises on failure — the page catches and shows why."""
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.prop_hist_client import _resolve_enthid

    enthid = _resolve_enthid(well_name)
    return execute_query(
        f"""
        SELECT ph.prop_id,
               coalesce(x.prop_name, ph.prop_id) AS prop_name,
               x.units,
               coalesce(x.category, 'other') AS category,
               ph.prop_value,
               ph.entry_datetime,
               ph.entry_user
        FROM mpu.wells.prop_hist ph
        LEFT JOIN mpu.wells.prop_xref x ON ph.prop_id = x.prop_id
        WHERE ph.enthid = {int(enthid)}
        ORDER BY ph.entry_datetime DESC, ph.prop_id
        """
    )


def shape_history(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Pure: raw history rows → the view's pieces.

    Returns ``{"history", "latest", "n_edits", "n_props", "last_edit",
    "editors"}`` or None for empty input.

    * ``history`` — all rows newest-first with ``is_current`` marking each
      prop's live row (the one the well opens with) and ``display_value``
      ("(cleared)" for the NULL tombstone/un-pin rows, ints for the pin id).
    * ``latest`` — one row per prop, the well's current stored state, ordered
      by category then name.
    """
    if df is None or df.empty:
        return None

    d = df.copy()
    d["entry_datetime"] = pd.to_datetime(d["entry_datetime"], utc=True)
    d = d.sort_values(
        ["entry_datetime", "prop_id"], ascending=[False, True]
    ).reset_index(drop=True)
    d["is_current"] = ~d["prop_id"].duplicated()

    def _display(row):
        v = row["prop_value"]
        if v is None or pd.isna(v):
            return "(cleared)"
        if row["prop_id"] == PIN_PROP_ID:
            return f"test uid {int(v)}"
        return f"{float(v):g}"

    d["display_value"] = d.apply(_display, axis=1)

    latest = (
        d[d["is_current"]]
        .sort_values(["category", "prop_name"])
        .reset_index(drop=True)
    )
    return {
        "history": d,
        "latest": latest,
        "n_edits": int(len(d)),
        "n_props": int(d["prop_id"].nunique()),
        "last_edit": d["entry_datetime"].iloc[0],
        "editors": sorted(
            {str(u) for u in d["entry_user"].dropna().astype(str) if u}
        ),
    }
