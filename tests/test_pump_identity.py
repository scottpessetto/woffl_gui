"""Tests for ``woffl.gui.pump_identity`` — tracker Circulating + Guiberson
→ National conversion.

Live-data grounding (apps.mpu_tracker.tbl_jetpump_data, 2026-07-21):
``Circulating`` is 'Reverse' (674) / 'Forward' (41) plus junk ('13', a
comment fragment, blanks, NULLs); Guiberson installs store a LETTER nozzle
with the throat in ``ThroatNumber`` (MPS-17 = D/8, MPS-54 = H/13), which the
National-only pump parsing rejected as corrupt ("no current pump"); legacy
rows store AREAS in the diameter columns (0.0241 in² = π/4·0.1752²).

Catalog-derived expectations: D (0.1501") → National 9 (0.1458), ratio
1.933 → letter B; H (0.2901") → National 15 (0.3017), ratio 1.637 → A.
"""

import math

import pandas as pd
import pytest

from woffl.gui import pump_identity as pi


# ---------------------------------------------------------------------------
# normalize_circulating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Reverse", "reverse"),
        ("Forward", "forward"),
        (" forward ", "forward"),
        ("REV", "reverse"),
        ("13", None),
        ('and set JP (28 Feb)"', None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_normalize_circulating(raw, expected):
    assert pi.normalize_circulating(raw) == expected


# ---------------------------------------------------------------------------
# Guiberson → National
# ---------------------------------------------------------------------------


def test_s17_guiberson_d8_converts_to_9b():
    conv = pi.guiberson_to_national("D", 8.0)
    assert conv is not None
    assert (conv["nozzle_no"], conv["throat_ratio"]) == ("9", "B")
    assert conv["raw_label"] == "Guiberson D/8"


def test_s54_guiberson_h13_converts_to_15a():
    conv = pi.guiberson_to_national("H", 13.0)
    assert conv is not None
    assert (conv["nozzle_no"], conv["throat_ratio"]) == ("15", "A")


def test_unknown_letter_falls_back_to_tracker_diameters():
    conv = pi.guiberson_to_national("Z", None, nozzle_dia=0.1501, throat_dia=0.2901)
    assert conv is not None
    assert (conv["nozzle_no"], conv["throat_ratio"]) == ("9", "B")


def test_unresolvable_returns_none():
    assert pi.guiberson_to_national("Z", None) is None
    # Throat smaller than nozzle is geometric nonsense.
    assert (
        pi.guiberson_to_national("Z", None, nozzle_dia=0.29, throat_dia=0.15) is None
    )


def test_area_guard_on_legacy_diameter_columns():
    """MPS-25's 2023 rows store areas: 0.0241 in² is really a 0.1752" nozzle."""
    assert pi._as_diameter(0.0241) == pytest.approx(
        math.sqrt(4 * 0.0241 / math.pi)
    )
    assert pi._as_diameter(0.1752) == pytest.approx(0.1752)
    assert pi._as_diameter(None) is None
    assert pi._as_diameter(0.0) is None
    assert pi._as_diameter(float("nan")) is None


# ---------------------------------------------------------------------------
# enrich_jp_history
# ---------------------------------------------------------------------------


def _hist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Well Name": ["MPS-17", "MPB-28", "MPS-54"],
            "Date Set": pd.to_datetime(["2026-07-12", "2026-03-13", "2026-06-11"]),
            "Nozzle Number": ["D", "11", "H"],
            "Throat Ratio": ["", "A", ""],
            "Tubing Diameter": [4.5, 4.5, 4.5],
            "Circulating": ["Forward", "Reverse", "Forward"],
            "Manufacturer": ["Guiberson", "National", "Guiberson"],
            "Throat Number": [8.0, 12.0, 13.0],
            "Nozzle Diameter": [0.1501, 0.1858, 0.2901],
            "Throat Diameter": [0.2901, 0.3404, 0.4750],
        }
    )


def test_enrich_converts_guiberson_rows_and_keeps_national():
    out = pi.enrich_jp_history(_hist())

    s17 = out.iloc[0]
    assert s17["Nozzle Number"] == "9"
    assert s17["Throat Ratio"] == "B"
    assert bool(s17["Pump Converted"]) is True
    assert s17["Raw Pump"] == "Guiberson D/8"
    assert s17["Circ Direction"] == "forward"

    b28 = out.iloc[1]
    assert b28["Nozzle Number"] == "11"
    assert b28["Throat Ratio"] == "A"
    assert bool(b28["Pump Converted"]) is False
    assert b28["Circ Direction"] == "reverse"

    s54 = out.iloc[2]
    assert s54["Nozzle Number"] == "15"
    assert s54["Throat Ratio"] == "A"
    assert s54["Raw Pump"] == "Guiberson H/13"


def test_enrich_tolerates_xlsx_frames_without_tracker_columns():
    """The Excel fallback path has none of the new columns — enrich must not
    crash, and a letter nozzle with no throat info stays untouched (existing
    invalid-pump handling applies)."""
    df = pd.DataFrame(
        {
            "Well Name": ["MPS-17", "MPB-28"],
            "Date Set": pd.to_datetime(["2026-07-12", "2026-03-13"]),
            "Nozzle Number": ["D", "11"],
            "Throat Ratio": ["", "A"],
        }
    )
    out = pi.enrich_jp_history(df)
    assert out["Circ Direction"].isna().all() or (
        out["Circ Direction"].eq(None).all()
    )
    assert out.iloc[0]["Nozzle Number"] == "D"  # no throat info → untouched
    assert bool(out.iloc[0]["Pump Converted"]) is False
    assert out.iloc[1]["Nozzle Number"] == "11"


def test_letter_detection_without_manufacturer_column():
    df = _hist().drop(columns=["Manufacturer"])
    out = pi.enrich_jp_history(df)
    assert bool(out.iloc[0]["Pump Converted"]) is True
    assert out.iloc[0]["Nozzle Number"] == "9"


def test_enrich_handles_empty_and_none():
    assert pi.enrich_jp_history(None) is None
    empty = pd.DataFrame()
    assert pi.enrich_jp_history(empty) is empty


# ---------------------------------------------------------------------------
# tracker_direction + the full chain through get_current_pump
# ---------------------------------------------------------------------------


def test_tracker_direction_reads_current_install():
    out = pi.enrich_jp_history(_hist())
    assert pi.tracker_direction(out, "MPS-17") == "forward"
    assert pi.tracker_direction(out, "MPB-28") == "reverse"
    assert pi.tracker_direction(out, "MPX-99") is None
    assert pi.tracker_direction(None, "MPS-17") is None
    # Un-enriched frame (no Circ Direction column) → None, never a crash.
    assert pi.tracker_direction(_hist().drop(columns=["Circulating"]), "MPS-17") is None


def test_get_current_pump_sees_converted_national_code():
    """The money path: a Guiberson well that used to read 'no current pump'
    now returns a valid National code + direction + raw provenance."""
    from woffl.assembly.jp_history import get_current_pump

    out = pi.enrich_jp_history(_hist())
    cp = get_current_pump(out, "MPS-17")
    assert cp is not None
    assert cp["nozzle_no"] == "9"
    assert cp["throat_ratio"] == "B"
    assert cp["circ_direction"] == "forward"
    assert cp["raw_pump"] == "Guiberson D/8"
