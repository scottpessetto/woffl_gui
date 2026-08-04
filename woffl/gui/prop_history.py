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

from woffl.assembly.prop_hist_client import to_alaska

# Rendered specially: the IPR anchor pin is a well-test ID, not a quantity.
PIN_PROP_ID = "ipr_wt_uid"

# The saved IPR rate is the well's MEASURED TOTAL LIQUID rate (BLPD, excluding
# power fluid) — oil and water are what get derived from it, through the
# assumed / 🔒 locked water cut (see the RATE CONVENTION in
# ``woffl/gui/params.py``). Scott, 2026-08-03, on B-28's 2135.29 bbl/d: *"i dont
# understand this ipr total liquid it doesnt match any test"* — at the time it
# genuinely didn't, because the code stored `oil / (1 - WC)` and B-28's WC is
# locked at 0.83. That inversion is fixed; the rate IS the test's total fluid
# now. This column shows the phase split it implies, so nobody has to redo the
# arithmetic to see the oil behind a liquid rate.
LIQ_PROP_ID = "ipr_qwf_liq"
WC_PROP_ID = "form_wc"


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
               ph.entry_user,
               c.comment_text AS comment
        FROM mpu.wells.prop_hist ph
        LEFT JOIN mpu.wells.prop_xref x ON ph.prop_id = x.prop_id
        -- The engineer's note for the SAVE this row belongs to. prop_hist has
        -- no batch id, so the shared entry_datetime is the join key (every
        -- prop of one save is written with one stamp — see
        -- prop_hist_client.push_prop). Grouped so a retried comment write can
        -- never fan a prop row out into duplicates.
        LEFT JOIN (
            SELECT enthid, entry_datetime, MAX(comment_text) AS comment_text
            FROM mpu.wells.woffl_eng_comment
            GROUP BY enthid, entry_datetime
        ) c ON ph.enthid = c.enthid AND ph.entry_datetime = c.entry_datetime
        WHERE ph.enthid = {int(enthid)}
        ORDER BY ph.entry_datetime DESC, ph.prop_id
        """
    )


def _derivations(d: pd.DataFrame) -> pd.Series:
    """``derivation`` text per row — empty except where a stored value implies
    others that aren't stored.

    Today that is only the saved IPR liquid rate, which carries the oil and
    water splits with it. The water cut is taken from the SAME save (prop_hist's
    batch identity is the shared ``entry_datetime`` — see
    ``prop_hist_client.push_prop``), falling back to the newest WC at or before
    that stamp for rows written before WC rode along. No WC anywhere ⇒ no
    claim: the cell stays empty rather than guessing 0.5.
    """
    notes = pd.Series("", index=d.index, dtype=object)
    liq = d.index[(d["prop_id"] == LIQ_PROP_ID) & d["prop_value"].notna()]
    if len(liq) == 0:
        return notes

    wc_rows = d[(d["prop_id"] == WC_PROP_ID) & d["prop_value"].notna()]
    wc_by_stamp = dict(zip(wc_rows["entry_datetime"], wc_rows["prop_value"]))
    wc_sorted = wc_rows.sort_values("entry_datetime")

    for i in liq:
        stamp = d.at[i, "entry_datetime"]
        wc = wc_by_stamp.get(stamp)
        if wc is None:
            earlier = wc_sorted[wc_sorted["entry_datetime"] <= stamp]
            wc = earlier["prop_value"].iloc[-1] if not earlier.empty else None
        if wc is None:
            continue
        wc = float(wc)
        if not 0.0 <= wc < 1.0:
            continue
        liquid = float(d.at[i, "prop_value"])
        oil = liquid * (1.0 - wc)
        water = liquid * wc
        notes.at[i] = (
            f"→ {oil:,.0f} BOPD oil + {water:,.0f} BWPD water at WC {wc:.2f}"
        )
    return notes


def shape_history(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Pure: raw history rows → the view's pieces.

    Returns ``{"history", "latest", "n_edits", "n_props", "last_edit",
    "editors"}`` or None for empty input.

    * ``history`` — all rows newest-first with ``is_current`` marking each
      prop's live row (the one the well opens with), ``display_value``
      ("(cleared)" for the NULL tombstone/un-pin rows, ints for the pin id)
      and ``derivation`` (how a stored number was computed, where it wasn't
      measured — see :data:`LIQ_PROP_ID`).
    * ``latest`` — one row per prop, the well's current stored state, ordered
      by category then name.

    ``entry_datetime`` stays UTC — it is the ordering key and every comparison
    downstream depends on it. ``entry_datetime_ak`` is the same instant as
    Alaska wall time for DISPLAY only (Kaelin, 2026-08-03: a bare "19:22" told
    nobody anything; that was 11:22 AKDT).
    """
    if df is None or df.empty:
        return None

    d = df.copy()
    # Optional: frames from before the comment join (and unit-test fixtures)
    # won't carry it. Absent means "no note", not an error.
    if "comment" not in d.columns:
        d["comment"] = None
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
    d["derivation"] = _derivations(d)
    # Display-only, tz-naive so a widget renders the wall clock verbatim under
    # a header that names the zone. Derived AFTER sorting — converting an
    # instant can't reorder anything, but keeping the key untouched is the
    # point.
    d["entry_datetime_ak"] = d["entry_datetime"].map(to_alaska)

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
