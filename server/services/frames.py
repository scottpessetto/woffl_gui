"""DataFrame -> JSON-safe records.

One conversion path for every service so NaN/NaT/numpy scalars never leak
into responses (JSON has no NaN; the SPA treats null as "unmeasured").
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    """Coerce one cell to a JSON-safe value. Timestamps -> YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def records(df: Optional[pd.DataFrame], columns: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts.

    Args:
        df: source frame (None/empty -> []).
        columns: optional {source_col: json_key} projection + rename. When
            given, only these columns are emitted (missing ones -> None).
    """
    if df is None or df.empty:
        return []
    out: list[dict[str, Any]] = []
    if columns:
        cols = [(src, key) for src, key in columns.items()]
        for row in df.itertuples(index=False):
            raw = dict(zip(df.columns, row))
            out.append({key: json_value(raw.get(src)) for src, key in cols})
    else:
        for row in df.itertuples(index=False):
            out.append({col: json_value(val) for col, val in zip(df.columns, row)})
    return out


def opt_float(value: Any) -> Optional[float]:
    """Finite float or None (guards the `d.get(k) or default` NaN trap)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
