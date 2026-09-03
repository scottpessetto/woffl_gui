"""Cross-brand jet pump equivalent lookup.

Given a National nozzle + throat, find the closest equivalent in the
Guiberson, Kobe, and Petrolift catalogs (Petrie & Smart, Jet Pumping Oil
Wells, 1983):

1. Find the brand nozzle whose diameter is closest to the National nozzle.
2. Compute the National diameter ratio (throat_dia / nozzle_dia).
3. For each brand throat, compute throat_dia / matched_nozzle_dia.
4. Pick the throat whose ratio is closest to the National ratio.
"""

from __future__ import annotations

import json
import math
from typing import Any

from server import config, schemas

_OTHER_BRANDS = ["guiberson", "kobe", "petrolift"]

# Loaded once at import; the catalog ships with the repo and never changes
# at runtime.
_CATALOG: dict[str, Any] = json.loads(
    config.JETPUMP_DIMENSIONS_JSON.read_text(encoding="utf-8")
)


def _closest_by_diameter(
    catalog_items: dict[str, float], target_dia: float
) -> tuple[str, float]:
    """(label, diameter) of the catalog entry closest to target_dia."""
    best_label, best_dia = min(
        catalog_items.items(), key=lambda kv: abs(kv[1] - target_dia)
    )
    return best_label, best_dia


def _find_equivalent(
    brand_data: dict[str, dict[str, float]],
    national_nozzle_dia: float,
    national_throat_dia: float,
) -> dict[str, Any]:
    """Closest equivalent nozzle + throat for one brand."""
    national_ratio = national_throat_dia / national_nozzle_dia

    # Step 1 - closest nozzle by diameter
    noz_label, noz_dia = _closest_by_diameter(brand_data["nozzle"], national_nozzle_dia)
    noz_area = math.pi / 4 * noz_dia**2

    # Step 2 - closest throat by diameter ratio
    best_thr_label = ""
    best_thr_dia = 0.0
    best_ratio_diff = float("inf")
    for thr_label, thr_dia in brand_data["throat"].items():
        brand_ratio = thr_dia / noz_dia
        diff = abs(brand_ratio - national_ratio)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_thr_label = thr_label
            best_thr_dia = thr_dia

    thr_area = math.pi / 4 * best_thr_dia**2
    dia_ratio = best_thr_dia / noz_dia

    return {
        "nozzle": noz_label,
        "nozzle_dia": noz_dia,
        "nozzle_area": noz_area,
        "throat": best_thr_label,
        "throat_dia": best_thr_dia,
        "throat_area": thr_area,
        "dia_ratio": dia_ratio,
    }


def equivalents(nozzle: str, throat: str) -> dict[str, Any]:
    """Cross-brand equivalents table for one National pump.

    Args:
        nozzle: National nozzle number (schemas.NOZZLE_OPTIONS).
        throat: National area ratio letter (schemas.THROAT_OPTIONS).

    Returns:
        EquivalentsResponse dict; the National reference row is first with
        is_reference=True.

    Raises:
        ValueError: nozzle/throat outside the GUI option lists (router maps
            to 422 "invalid").
    """
    if nozzle not in schemas.NOZZLE_OPTIONS:
        raise ValueError(f"Unknown nozzle {nozzle!r}; expected one of {schemas.NOZZLE_OPTIONS}")
    if throat not in schemas.THROAT_OPTIONS:
        raise ValueError(f"Unknown throat {throat!r}; expected one of {schemas.THROAT_OPTIONS}")

    from woffl.geometry.jetpump import JetPump

    # Friction coefficients are irrelevant here (no flow calculation) - any
    # defaults are fine, same as the Streamlit tab.
    local_jp = JetPump(nozzle_no=nozzle, area_ratio=throat, ken=0.03, kth=0.3, kdi=0.4)

    nat_noz_dia = float(local_jp.dnz)
    nat_thr_dia = float(local_jp.dth)

    rows: list[dict[str, Any]] = [
        {
            "brand": "National",
            "nozzle": nozzle,
            "throat": throat,
            "nozzle_dia": nat_noz_dia,
            "throat_dia": nat_thr_dia,
            "nozzle_area": math.pi / 4 * nat_noz_dia**2,
            "throat_area": math.pi / 4 * nat_thr_dia**2,
            "area_ratio_val": nat_thr_dia / nat_noz_dia,
            "is_reference": True,
        }
    ]
    for brand in _OTHER_BRANDS:
        eq = _find_equivalent(_CATALOG[brand], nat_noz_dia, nat_thr_dia)
        rows.append(
            {
                "brand": brand.capitalize(),
                "nozzle": str(eq["nozzle"]),
                "throat": str(eq["throat"]),
                "nozzle_dia": float(eq["nozzle_dia"]),
                "throat_dia": float(eq["throat_dia"]),
                "nozzle_area": float(eq["nozzle_area"]),
                "throat_area": float(eq["throat_area"]),
                "area_ratio_val": float(eq["dia_ratio"]),
                "is_reference": False,
            }
        )

    return {"nozzle_no": nozzle, "area_ratio": throat, "rows": rows}
