"""Tab: Pump Equivalent Lookup

Given a National nozzle + throat combo, finds the closest equivalent
nozzle/throat in Guiberson, Kobe, and Petrolift catalogs.

Selection is local to this tab (two selectboxes inline) so users can
explore equivalents without disturbing the sidebar pump that drives
the Solver / Batch Run views. The tab seeds its initial selection
from the sidebar so the first view matches what the user is already
analyzing.

Matching algorithm (mirrors jetpump_equivalent_rev2.xlsx):
1. Find the brand nozzle whose diameter is closest to the National nozzle.
2. Compute the National diameter ratio (throat_dia / nozzle_dia).
3. For each brand throat, compute throat_dia / matched_nozzle_dia.
4. Pick the throat whose ratio is closest to the National ratio.
"""

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from woffl.geometry.jetpump import JetPump
from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS, SimulationParams

_CATALOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "jetpump_dimensions.json"

_OTHER_BRANDS = ["guiberson", "kobe", "petrolift"]


def _load_catalog() -> dict:
    with open(_CATALOG_PATH) as f:
        return json.load(f)


def _closest_by_diameter(
    catalog_items: dict[str, float], target_dia: float
) -> tuple[str, float]:
    """Return (label, diameter) of the catalog entry closest to target_dia."""
    best_label, best_dia = min(
        catalog_items.items(), key=lambda kv: abs(kv[1] - target_dia)
    )
    return best_label, best_dia


def _find_equivalent(
    brand_data: dict,
    national_nozzle_dia: float,
    national_throat_dia: float,
) -> dict:
    """Find the closest equivalent nozzle + throat for one brand."""
    national_ratio = national_throat_dia / national_nozzle_dia

    # Step 1 — closest nozzle by diameter
    noz_label, noz_dia = _closest_by_diameter(brand_data["nozzle"], national_nozzle_dia)
    noz_area = math.pi / 4 * noz_dia**2

    # Step 2 — closest throat by diameter ratio
    best_thr_label = None
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


def render_tab(params: SimulationParams) -> None:
    """Render the Pump Equivalent tab.

    Uses two local selectboxes for nozzle + throat so the user can browse
    equivalents without changing the sidebar pump (which would re-run the
    Solver / Batch / PF Sensitivity views). First-time defaults are seeded
    from the sidebar so the tab opens on the user's current setup.
    """
    st.subheader("Cross-Brand Pump Equivalents")
    st.caption(
        "Finds the closest nozzle and throat match for a National pump "
        "across Guiberson, Kobe, and Petrolift catalogs. The selection "
        "below is local to this tab \u2014 it does **not** change the sidebar "
        "or the other views."
    )

    # Seed the local selection from the sidebar on first visit so the tab
    # opens on the user's current setup. After that, the widgets own their
    # own session_state and the sidebar can drift independently.
    if "pe_nozzle_no" not in st.session_state:
        st.session_state["pe_nozzle_no"] = params.nozzle_no
    if "pe_area_ratio" not in st.session_state:
        st.session_state["pe_area_ratio"] = params.area_ratio

    col_n, col_t = st.columns(2)
    with col_n:
        nozzle_no = st.selectbox(
            "Nozzle",
            options=NOZZLE_OPTIONS,
            key="pe_nozzle_no",
        )
    with col_t:
        area_ratio = st.selectbox(
            "Throat",
            options=THROAT_OPTIONS,
            key="pe_area_ratio",
        )

    # Build a local JetPump just to get the National diameters. The
    # friction coefficients are irrelevant here (no flow calculation
    # happens on this tab), so any defaults are fine.
    local_jp = JetPump(
        nozzle_no=nozzle_no,
        area_ratio=area_ratio,
        ken=0.03,
        kth=0.3,
        kdi=0.4,
    )

    catalog = _load_catalog()

    nat_noz_dia = local_jp.dnz
    nat_thr_dia = local_jp.dth
    nat_noz_area = math.pi / 4 * nat_noz_dia**2
    nat_thr_area = math.pi / 4 * nat_thr_dia**2
    nat_ratio = nat_thr_dia / nat_noz_dia

    # Build results table
    rows = [
        {
            "Brand": "National (selected)",
            "Nozzle": nozzle_no,
            "Throat": area_ratio,
            "Nozzle Dia (in)": nat_noz_dia,
            "Nozzle Area (in\u00b2)": nat_noz_area,
            "Throat Dia (in)": nat_thr_dia,
            "Throat Area (in\u00b2)": nat_thr_area,
            "Dia Ratio": nat_ratio,
        }
    ]

    for brand in _OTHER_BRANDS:
        eq = _find_equivalent(catalog[brand], nat_noz_dia, nat_thr_dia)
        rows.append(
            {
                "Brand": brand.capitalize(),
                "Nozzle": eq["nozzle"],
                "Throat": eq["throat"],
                "Nozzle Dia (in)": eq["nozzle_dia"],
                "Nozzle Area (in\u00b2)": eq["nozzle_area"],
                "Throat Dia (in)": eq["throat_dia"],
                "Throat Area (in\u00b2)": eq["throat_area"],
                "Dia Ratio": eq["dia_ratio"],
            }
        )

    df = pd.DataFrame(rows)

    # Format for display
    fmt = df.style.format(
        {
            "Nozzle Dia (in)": "{:.4f}",
            "Nozzle Area (in\u00b2)": "{:.6f}",
            "Throat Dia (in)": "{:.4f}",
            "Throat Area (in\u00b2)": "{:.6f}",
            "Dia Ratio": "{:.3f}",
        }
    )
    st.dataframe(fmt, use_container_width=True, hide_index=True)

    st.markdown(
        f"**Reference:** National {nozzle_no}-{area_ratio} "
        f"&mdash; nozzle {nat_noz_dia:.4f}\", throat {nat_thr_dia:.4f}\", "
        f"diameter ratio {nat_ratio:.3f}"
    )
    st.caption(
        "Nozzle matched by closest diameter. Throat matched by closest "
        "diameter ratio (throat dia / nozzle dia) to preserve pump characteristics. "
        "Source: Petrie & Smart, *Jet Pumping Oil Wells* (1983)."
    )
