"""SCADA tag history from the plant historian (fork-only Databricks glue).

``reporting.historian.vw_mpu_measurements`` is the exception-reported process
historian: one row per tag per *reported change*, not per fixed interval. Two
consequences every caller must respect:

* Sampling is irregular (10 s on a live flow meter, minutes on an analyzer
  that is sitting still), so any average over these rows must be TIME
  WEIGHTED. A plain ``mean()`` over-weights whatever was moving fastest.
* A value holds until the next row. Merging two tags therefore means an
  as-of/step-hold join, never an inner join on timestamp.

``MeasureTime`` is UTC. ``LocalTime`` exists on the view but is not trusted
here - callers convert explicitly to ``America/Anchorage``.

Reads only; no gate needed (see AGENTS.md section 3).
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from woffl.assembly.databricks_client import execute_query

MEASUREMENT_VIEW = "reporting.historian.vw_mpu_measurements"

# execute_query has no parameter binding, so every tag spliced into SQL is
# shape-validated first. Historian tags are plant-code identifiers only.
_TAG_SHAPE_RE = re.compile(r"^[A-Za-z0-9_]{3,64}$")


def validate_tags(tags: Iterable[str]) -> list[str]:
    """Shape-check tag names before they are spliced into read SQL.

    Args:
        tags (iterable): Historian tag names, e.g. ``MPU_FI_5365``.

    Returns:
        clean (list): The same names, de-duplicated, order preserved.

    Raises:
        ValueError: Any name that is not a bare plant-code identifier.
    """
    clean: list[str] = []
    for tag in tags:
        name = (tag or "").strip()
        if not _TAG_SHAPE_RE.match(name):
            raise ValueError(f"unsafe historian tag name: {tag!r}")
        if name not in clean:
            clean.append(name)
    return clean


def fetch_tag_history(tags: Iterable[str], days_back: int) -> pd.DataFrame:
    """Raw historian rows for a set of tags over a trailing window.

    Args:
        tags (iterable): Historian tag names.
        days_back (int): Trailing window in days, 1-400.

    Returns:
        df (DataFrame): Columns ``tag`` (str), ``t`` (tz-aware UTC), ``value``
            (float), sorted by tag then time. Empty frame when nothing matches.

    Raises:
        ValueError: Bad tag shape or a days_back outside 1-400.
    """
    clean = validate_tags(tags)
    if not clean:
        raise ValueError("no tags requested")
    days = int(days_back)
    if not 1 <= days <= 400:
        raise ValueError(f"days_back must be 1-400, got {days_back}")

    tag_list = ", ".join(f"'{t}'" for t in clean)
    query = f"""
SELECT Tag AS tag, MeasureTime AS t, Value AS value
FROM {MEASUREMENT_VIEW}
WHERE Tag IN ({tag_list})
  AND MeasureDate >= DATE_SUB(current_date(), {days})
ORDER BY Tag, MeasureTime
"""
    df = execute_query(query)
    if df is None or df.empty:
        return pd.DataFrame(columns=["tag", "t", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])
