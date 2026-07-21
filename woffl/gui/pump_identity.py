"""Pump identity + circulation normalization for the MPU JP tracker.

The tracker (``apps.mpu_tracker.tbl_jetpump_data``) carries three columns the
app historically ignored:

* ``Circulating`` — "Reverse"/"Forward" (674/41 rows live, plus a few junk
  values), the INSTALLED configuration. Better than inferring direction from
  live daily pressures (which stays as the fallback).
* ``Manufacturer`` — "National"/"Guiberson" (682/18 live). Guiberson installs
  record a LETTER nozzle (``NozzleNumber`` = 'D', 'E', 'H'…), a blank
  ``ThroatRatio``, and the Guiberson throat number in ``ThroatNumber`` —
  which the National-only pump parsing rejected as "corrupt", so those wells
  (MPS-17/25/54…) read as having NO current pump.
* ``NozzleDiameter`` / ``ThroatDiameter`` — inches on recent records; some
  legacy rows stored AREAS (in²) in the same columns (MPS-25 2023:
  0.0241 = π/4·0.1752²), guarded here.

:func:`enrich_jp_history` runs ONCE at fetch: it normalizes the direction
into a ``Circ Direction`` column and rewrites Guiberson rows to their closest
NATIONAL equivalent (catalog-based, same ratio matching as the Pump
Equivalents tab) so every downstream consumer — pump strips, Model vs
Actual, batch auto-match, calibration — sees a valid National code. The
original label survives in ``Raw Pump`` and the row is flagged
``Pump Converted`` so nothing masquerades as a native National install.

Pure of Streamlit; the catalog is ``data/jetpump_dimensions.json``.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "jetpump_dimensions.json"
)

# National throat letters are OFFSETS from the nozzle number: a 12B runs
# throat #13, a 13C runs #15 (verified against the tracker's ThroatNumber
# column for every forward-circ National install).
_LETTER_OFFSETS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Plausible-diameter floor (inches). Tracker values below this in the
# diameter columns are legacy AREA entries (in²) — the smallest Guiberson
# nozzle (DD) is 0.0451" across, while areas run 0.0016–0.066 in².
_MIN_PLAUSIBLE_DIA = 0.04


@lru_cache(maxsize=1)
def _catalog() -> dict:
    with open(_CATALOG_PATH) as f:
        return json.load(f)


def normalize_circulating(raw) -> Optional[str]:
    """Tolerant Circulating → 'forward' / 'reverse' / None.

    The live column is mostly clean but carries a numeric, a comment
    fragment, blanks, and NULLs — anything unrecognized maps to None so the
    caller falls through to the live-pressure inference.
    """
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    s = str(raw).strip().lower()
    if s.startswith("forward") or s == "fwd":
        return "forward"
    if s.startswith("reverse") or s == "rev":
        return "reverse"
    return None


def _as_diameter(value: Optional[float]) -> Optional[float]:
    """Coerce a tracker nozzle/throat 'diameter' that may actually be an
    AREA (legacy rows) into a diameter, or None when unusable."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not (v == v) or v <= 0:
        return None
    if v < _MIN_PLAUSIBLE_DIA:
        return math.sqrt(4.0 * v / math.pi)
    return v


def _throat_key(throat_number) -> Optional[str]:
    """Guiberson catalog throat key from the tracker's numeric ThroatNumber."""
    if throat_number is None:
        return None
    try:
        f = float(throat_number)
    except (TypeError, ValueError):
        s = str(throat_number).strip()
        return s or None
    if not (f == f):
        return None
    return str(int(f)) if f == int(f) else str(f)


def guiberson_to_national(
    nozzle_letter,
    throat_number=None,
    *,
    nozzle_dia: Optional[float] = None,
    throat_dia: Optional[float] = None,
) -> Optional[dict]:
    """Closest National (nozzle, letter) for a Guiberson install.

    Diameters resolve catalog-first (the tracker's diameter columns are the
    fallback, area-guarded): nozzle by closest National diameter, throat by
    closest diameter RATIO expressed as a National letter offset — the same
    matching the Pump Equivalents tab uses in the other direction.

    Returns ``{"nozzle_no", "throat_ratio", "raw_label", "national_ratio",
    "guiberson_ratio"}`` or None when the geometry can't be resolved.
    """
    cat = _catalog()
    letter = str(nozzle_letter).strip() if nozzle_letter is not None else ""
    g_noz = cat["guiberson"]["nozzle"].get(letter)
    if g_noz is None:
        g_noz = _as_diameter(nozzle_dia)
    thr_key = _throat_key(throat_number)
    g_thr = cat["guiberson"]["throat"].get(thr_key) if thr_key else None
    if g_thr is None:
        g_thr = _as_diameter(throat_dia)
    if not g_noz or not g_thr or g_thr <= g_noz:
        return None

    nat_noz = cat["national"]["nozzle"]
    nat_thr = cat["national"]["throat"]
    noz_label, noz_dia_val = min(
        nat_noz.items(), key=lambda kv: abs(kv[1] - g_noz)
    )
    target_ratio = g_thr / g_noz

    best_letter, best_diff = None, float("inf")
    for ltr, off in _LETTER_OFFSETS.items():
        key = str(int(noz_label) + off)
        thr_dia_val = nat_thr.get(key)
        if thr_dia_val is None:
            continue
        diff = abs(thr_dia_val / noz_dia_val - target_ratio)
        if diff < best_diff:
            best_letter, best_diff = ltr, diff
    if best_letter is None:
        return None

    return {
        "nozzle_no": noz_label,
        "throat_ratio": best_letter,
        "raw_label": f"Guiberson {letter or '?'}/{thr_key or '?'}",
        "guiberson_ratio": target_ratio,
        "national_ratio": nat_thr[str(int(noz_label) + _LETTER_OFFSETS[best_letter])]
        / noz_dia_val,
    }


def _looks_guiberson(manufacturer, nozzle) -> bool:
    """Guiberson row detection: the Manufacturer column when present, else a
    letter nozzle that exists in the Guiberson catalog (covers the xlsx
    upload path, which has no Manufacturer column). National nozzles are
    strictly numeric, so a catalog letter can never be a National code."""
    m = str(manufacturer).strip().lower() if manufacturer is not None else ""
    if "guiberson" in m:
        return True
    if "national" in m:
        return False
    noz = str(nozzle).strip() if nozzle is not None else ""
    if not noz:
        return False
    try:
        int(float(noz))
        return False
    except (TypeError, ValueError):
        return noz in _catalog()["guiberson"]["nozzle"]


def enrich_jp_history(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize direction + convert Guiberson rows to National equivalents.

    Adds ``Circ Direction`` ('forward'/'reverse'/None), and for detected
    Guiberson rows rewrites ``Nozzle Number``/``Throat Ratio`` to the closest
    National labels, stashing the original in ``Raw Pump`` and flagging
    ``Pump Converted``. Rows that can't be converted are left untouched (they
    keep the existing invalid-pump handling). Tolerant of frames missing any
    of the tracker columns (the legacy xlsx upload path).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    circ_src = out["Circulating"] if "Circulating" in out.columns else None
    out["Circ Direction"] = (
        circ_src.map(normalize_circulating) if circ_src is not None else None
    )
    out["Raw Pump"] = None
    out["Pump Converted"] = False

    if "Nozzle Number" not in out.columns:
        return out

    for idx, row in out.iterrows():
        if not _looks_guiberson(row.get("Manufacturer"), row.get("Nozzle Number")):
            continue
        conv = guiberson_to_national(
            row.get("Nozzle Number"),
            row.get("Throat Number"),
            nozzle_dia=row.get("Nozzle Diameter"),
            throat_dia=row.get("Throat Diameter"),
        )
        if conv is None:
            continue
        out.at[idx, "Raw Pump"] = conv["raw_label"]
        out.at[idx, "Nozzle Number"] = conv["nozzle_no"]
        out.at[idx, "Throat Ratio"] = conv["throat_ratio"]
        out.at[idx, "Pump Converted"] = True
    return out


def tracker_direction(jp_hist, well_name: str) -> Optional[str]:
    """The CURRENT install's normalized Circulating for a well, or None.

    Reads the enriched ``Circ Direction`` column (latest Date Set row).
    Callers layer this ABOVE the live-pressure inference: the tracker states
    how the pump was actually plumbed; the pressure heuristic remains the
    fallback for wells with no (or junk) tracker direction.
    """
    if jp_hist is None or getattr(jp_hist, "empty", True):
        return None
    if "Circ Direction" not in jp_hist.columns or "Well Name" not in jp_hist.columns:
        return None
    rows = jp_hist[jp_hist["Well Name"] == well_name].dropna(subset=["Date Set"])
    if rows.empty:
        return None
    d = rows.sort_values("Date Set", ascending=False).iloc[0].get("Circ Direction")
    return d if d in ("forward", "reverse") else None
