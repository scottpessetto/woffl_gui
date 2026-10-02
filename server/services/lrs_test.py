"""Parse an LRS portable-separator "WELL TEST SUMMARY" workbook.

The sheet is a form, not a table: a label cell ("POWER FLUID RATE:") with its
value some cells to the right on the same row, then a unit. Cells are found
by LABEL, never by address, so a shifted row or column still parses.

Stateless and read-only: the Solver lays the result over the sidebar as a
manual test. Nothing is stored. The sheet carries no bottom-hole pressure.
"""

from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any, Optional

# Normalized label -> result key. Rates on the sheet are already per-day.
_NUMBER_LABELS: dict[str, str] = {
    "DURATION": "hours",
    "WELL HEAD PRESSURE": "whp",  # psi
    "POWER FLUID RATE": "pf_rate",  # BLPD
    "POWER FLUID PRESSURE": "pf_press",  # psi
    "CORRECTED FORMATION OIL RATE": "oil",  # STBOPD
    "FORMATION WATER RATE": "water",  # BWPD
    "CORRECTED WATER CUT": "wc_pct",  # %
    "CORRECTED FORMATION FLUID RATE": "total_fluid",  # STBFPD
    "CORRECTED FORMATION GAS RATE": "gas_mmscfd",  # MMSCFPD
    "CORRECTED FORMATION GOR": "gor",  # scf/stb
}
_DATE_LABELS: dict[str, str] = {
    "TEST DATE": "test_date",
    "TEST END DATE": "end_date",
    "TEST START DATE": "start_date",
}
_TEXT_LABELS: dict[str, str] = {
    "WELL": "well_raw",
    "TEST LOCATION CODE": "location",
}
_LABELS = {**_NUMBER_LABELS, **_DATE_LABELS, **_TEXT_LABELS}


def _norm(value: Any) -> str:
    """'WELL#:' -> 'WELL', 'Power  Fluid Rate: ' -> 'POWER FLUID RATE'."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[#:]", " ", value)).strip().upper()


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _date(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _read_labels(rows: list[list[Any]]) -> dict[str, Any]:
    """{result key: value} from the first cell right of each known label that
    holds a value of the label's kind. A row can carry two labels (start date
    and start time), so the search stops at the next label."""
    found: dict[str, Any] = {}
    for row in rows:
        for col, cell in enumerate(row):
            key = _LABELS.get(_norm(cell))
            if key is None or key in found:
                continue
            for other in row[col + 1 :]:
                if _norm(other) in _LABELS:
                    break
                if key in _NUMBER_LABELS.values():
                    value: Any = _number(other)
                elif key in _DATE_LABELS.values():
                    value = _date(other)
                else:
                    value = other.strip() if isinstance(other, str) and other.strip() else None
                if value is not None:
                    found[key] = value
                    break
    return found


def parse_lrs_sheet(blob: bytes, filename: str) -> dict[str, Any]:
    """One LRS well test from an .xlsx/.xlsm workbook (LrsTestResponse shape).

    Args:
        blob: workbook bytes.
        filename: upload name, for the error text and the provenance label.

    Returns:
        well (app name, or None), test_date (YYYY-MM-DD), hours, location,
        oil (BOPD), water (BWPD), total_fluid (BLPD), form_wc (fraction),
        gor (scf/stb), whp (psi), pf_rate (BWPD), pf_press (psi), missing
        (labels the sheet did not give a value for).

    Raises:
        ValueError: not a readable workbook, or no oil/liquid rate on it.
    """
    from openpyxl import load_workbook

    from woffl.assembly.well_test_client import _normalize_well_name

    if filename.lower().endswith(".xls"):
        raise ValueError("old-format .xls is not supported - save the sheet as .xlsx and load that")
    try:
        book = load_workbook(io.BytesIO(blob), data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises many types on a non-workbook
        raise ValueError("not a readable Excel workbook (.xlsx / .xlsm)") from exc

    found: dict[str, Any] = {}
    try:
        for sheet in book.worksheets:
            found = _read_labels([list(r) for r in sheet.iter_rows(values_only=True)])
            if "oil" in found or "total_fluid" in found:
                break
    finally:
        book.close()

    oil, water, total = found.get("oil"), found.get("water"), found.get("total_fluid")
    if total is None and oil is not None and water is not None:
        total = oil + water
    if oil is None and total is not None and water is not None:
        oil = total - water
    if water is None and total is not None and oil is not None:
        water = total - oil
    if oil is None or total is None or total <= 0:
        raise ValueError(
            "no formation oil / fluid rate found - expected an LRS WELL TEST SUMMARY sheet "
            "with a CORRECTED FORMATION OIL RATE row"
        )

    wc_pct = found.get("wc_pct")
    # The sheet's own corrected cut when present; else from the rates.
    form_wc = wc_pct / 100.0 if wc_pct is not None else water / total
    well_raw = found.get("well_raw")
    well = _normalize_well_name(well_raw) if well_raw else None
    labels = {v: k for k, v in _LABELS.items()}
    wanted = ("test_date", "whp", "pf_rate", "pf_press", "gor")
    return {
        "filename": filename,
        "well": well,
        "well_raw": well_raw,
        "test_date": found.get("test_date") or found.get("end_date") or found.get("start_date"),
        "hours": found.get("hours"),
        "location": found.get("location"),
        "oil": oil,
        "water": water,
        "total_fluid": total,
        "form_wc": form_wc,
        "gor": found.get("gor"),
        "whp": found.get("whp"),
        "pf_rate": found.get("pf_rate"),
        "pf_press": found.get("pf_press"),
        "missing": [labels[k].title() for k in wanted if found.get(k) is None],
    }
