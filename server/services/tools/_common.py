"""Shared helpers for Scott's Tools - the Streamlit-free port.

Port of the retired Streamlit tools' shared helpers. The original was built
on ``@st.cache_data`` fetchers and ``load_well_characteristics``; here the
data comes from the server's own cached datasources, so a tool never issues a
per-well query the fleet caches already answer (the cardinal rule).

The physics builders (``build_well_config`` / ``create_well_objects``) are
lifted verbatim - the tools' numbers must not move because they changed UI.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import pandas as pd


from server.services import datasources
from server.services import tests as tests_svc
from server.services import wells as wells_svc

from woffl.assembly.network_optimizer import WellConfig
from woffl.assembly.parallelism import worker_ceiling  # noqa: F401 - re-export for tools
from woffl.assembly.sim_factories import create_pvt_components
from woffl.flow.inflow import InFlow
from woffl.geometry import Pipe, PipeInPipe
from woffl.pvt import ResMix

log = logging.getLogger("woffl.web.tools")

pad_from_mp_name = wells_svc._pad_from_mp_name

# Pad PF defaults: still no per-well PF pressure in Databricks, so the tools
# seed from the pad table (PF-PRESSURE-DEPENDENCY) and let the page edit it.
PAD_PF_DEFAULTS = wells_svc._PAD_PF_DEFAULTS
PAD_PF_FALLBACK = wells_svc._PAD_PF_FALLBACK


def normalize_short_name(name: str) -> str:
    """B-30 -> MPB-30, I-15 -> MPI-15. Already-prefixed names pass through."""
    name = (name or "").strip()
    if name.upper().startswith("MP"):
        return name
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    return f"MP{m.group(1).upper()}-{int(m.group(2))}" if m else name


def live_pf_for_seed(well_name: str) -> Optional[dict[str, Any]]:
    """Best live PF reading to seed a well's PF pressure with, or None.

    Delegates to the server's own seed chain (test-day reading -> latest
    daily), and returns its DICT - ``{"pf_press", "pf_source", "pf_date",
    "kind"}`` - because that is what the ported tools index into. An earlier
    version of this returned a bare float and every caller died on
    ``live_pf["pf_press"]``.

    None means "no live reading"; the caller falls back to the pad default.
    """
    try:
        tests = tests_svc.tests_for_well(well_name, 6, 0)
        return wells_svc._live_pf_seed(well_name, tests)
    except Exception:  # noqa: BLE001 - the pad default is always valid
        log.warning("tools: live PF seed failed for %s", well_name, exc_info=True)
        return None


def detect_jpump_direction(well_name: str) -> str:
    """"forward" when the well's live PF reading comes from the TUBING
    (forward circulation: PF down the tubing, production up the annulus),
    else "reverse" - the same rule the sidebar seeds from (pf_pressure
    resolve_pf_pressure). Falls back to "reverse" when there is no reading."""
    live = live_pf_for_seed(well_name)
    src = str((live or {}).get("pf_source") or "").lower()
    return "forward" if src == "tubing" else "reverse"


def load_well_characteristics():
    """The fleet characteristics frame (compat name for the ported tools)."""
    df, _source = datasources.well_chars_safe()
    return df


# ── jp_chars helpers (verbatim from _common) ───────────────────────────────


def friction_coefs_from_chars(chars: Optional[dict]) -> dict:
    """Jet-pump friction coefficients from a jp_chars row.

    Reads the Databricks vw_prop_mech columns (jpfric_*); missing/NaN values
    are omitted so JetPump falls back to its class defaults (knz=0.01,
    ken=0.03, kth=0.3, kdi=0.3).
    """
    if not chars:
        return {}
    mapping = {
        "knz": "jpfric_nozzle",
        "ken": "jpfric_entry",
        "kth": "jpfric_throat",
        "kdi": "jpfric_diffuser",
    }
    out: dict = {}
    for kw, col in mapping.items():
        v = chars.get(col)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if pd.isna(fv):
            continue
        out[kw] = fv
    return out


def casing_dims_from_chars(chars: Optional[dict]) -> tuple[float, float]:
    """(casing_od, casing_thickness) from chars; fallback 6.875 / 0.5."""
    if chars:
        out_dia = chars.get("casing_out_dia")
        inn_dia = chars.get("casing_inn_dia")
        try:
            if out_dia is not None and inn_dia is not None:
                od_f = float(out_dia)
                id_f = float(inn_dia)
                if not pd.isna(od_f) and not pd.isna(id_f) and od_f > id_f > 0:
                    return od_f, (od_f - id_f) / 2.0
        except (TypeError, ValueError):
            pass
    return 6.875, 0.5


# ── fleet frames (server caches, never per-well) ───────────────────────────


def well_chars_map() -> dict[str, dict]:
    """{well: chars-row-dict} from the cached fleet characteristics frame."""
    df, _source = datasources.well_chars_safe()
    if df is None or df.empty or "Well" not in df.columns:
        return {}
    return df.set_index("Well").to_dict("index")


def _fleet_tests(months_back: int) -> pd.DataFrame:
    """The cached fleet well-test frame. Warmed for the config windows."""
    try:
        return tests_svc.fetch_all_well_tests(months_back)
    except Exception:  # noqa: BLE001 - tools degrade, they do not 500
        log.warning("tools: fleet well tests unavailable", exc_info=True)
        return pd.DataFrame()


def fetch_well_tests_raw(months_back: int) -> pd.DataFrame:
    """Fleet tests WITHOUT dropping gaugeless rows.

    The Streamlit helper of this name ran its own query that skipped the
    BHP-required filter. The server's fleet frame already keeps gaugeless
    rows (callers drop them per need), so this is the cached frame - one
    query for the fleet instead of a second parallel pull.
    """
    return _fleet_tests(months_back)


def latest_col_per_well(months_back: int, column: str) -> dict[str, float]:
    """{well: newest non-null ``column``} across the fleet test window.

    Replaces ``get_latest_whp_per_well`` / ``get_latest_bhp_per_well``, which
    each pulled their own frame; both now slice one cached fleet frame.
    """
    df = _fleet_tests(months_back)
    if df is None or df.empty or column not in df.columns:
        return {}
    valid = df.dropna(subset=[column])
    if valid.empty or "WtDate" not in valid.columns:
        return {}
    latest = valid.sort_values("WtDate").groupby("well").last()
    out: dict[str, float] = {}
    for well in latest.index:
        try:
            out[str(well)] = float(latest.loc[well, column])
        except (TypeError, ValueError):
            continue
    return out


def get_vogel_for_wells(well_names: list[str], months_back: int = 3) -> dict[str, dict]:
    """{well: vogel_row_dict} for wells with usable BHP gauge data.

    Same library chain as the tab (estimate_reservoir_pressure ->
    compute_vogel_coefficients), fed from the cached fleet frame.
    """
    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
    )

    df = _fleet_tests(months_back)
    if df is None or df.empty:
        return {}
    filtered = df[df["well"].isin(well_names)].copy()
    if filtered.empty:
        return {}
    chars_df, _ = datasources.well_chars_safe()
    try:
        merged = estimate_reservoir_pressure(filtered, jp_chars=chars_df)
        vogel = compute_vogel_coefficients(merged)
        if vogel is None or vogel.empty or "Well" not in vogel.columns:
            return {}
        return vogel.set_index("Well").to_dict("index")
    except Exception:  # noqa: BLE001 - a failed fit is "no Vogel", not an error
        log.warning("tools: Vogel fit failed", exc_info=True)
        return {}


def get_latest_whp_per_well(months_back: int) -> dict[str, float]:
    """{well: latest measured wellhead pressure}."""
    return latest_col_per_well(months_back, "whp")


def get_latest_bhp_per_well(months_back: int) -> dict[str, float]:
    """{well: latest measured BHP}. Wells absent here have no gauge."""
    return latest_col_per_well(months_back, "BHP")


# ── well config + objects (verbatim physics) ───────────────────────────────


def has_databricks_casing(chars: Optional[dict]) -> bool:
    """True when prop_hist supplies real casing dimensions for this well.

    The as-built rule: where prop_hist has the geometry the UI must not let
    it be edited (and push_prop refuses to write it). Where it has nothing,
    a Custom/hypothetical well still has to be modelable from a default.
    """
    if not chars:
        return False
    out_dia, inn_dia = chars.get("casing_out_dia"), chars.get("casing_inn_dia")
    try:
        if out_dia is None or inn_dia is None:
            return False
        od, idd = float(out_dia), float(inn_dia)
        return not pd.isna(od) and not pd.isna(idd) and od > idd > 0
    except (TypeError, ValueError):
        return False


def build_well_config(
    well_name: str,
    jp_chars_dict: dict,
    vogel_row: Optional[dict] = None,
    surf_pres: float = 210.0,
    jpump_direction: Optional[str] = None,
) -> WellConfig:
    """WellConfig from jp_chars, optionally overriding IPR with Vogel data.

    Args:
        well_name: Well identifier (e.g. "MPB-30").
        jp_chars_dict: {well: chars} from :func:`well_chars_map`.
        vogel_row: Optional Vogel coefficient row to override IPR.
        surf_pres: Wellhead pressure, psi (callers pass the latest test's).
        jpump_direction: "forward" | "reverse". None (default) live-detects
            it from the well's PF source the way the sidebar does (a tubing
            PF reading = forward circulation). Until 2026-09-01 every tool
            modeled every well REVERSE - MPS-17, MPE-17, MPL-20 and the
            F-pad forward wells had PF friction computed down the annulus
            and production up the tubing (review EVID-F22).

    Raises:
        ValueError: the well is not in the characteristics frame.
    """
    chars = jp_chars_dict.get(well_name)
    if not chars:
        raise ValueError(f"{well_name} not in jp_chars database")

    if jpump_direction is None:
        jpump_direction = detect_jpump_direction(well_name)

    is_sch = chars.get("is_sch", True)
    if isinstance(is_sch, str):
        is_sch = is_sch.lower() in ("true", "1", "yes")
    fm = "Schrader" if is_sch else "Kuparuk"

    casing_od, casing_thk = casing_dims_from_chars(chars)

    params = dict(
        well_name=well_name,
        res_pres=float(chars.get("res_pres", 1800)),
        form_temp=float(chars.get("form_temp", 75 if is_sch else 170)),
        jpump_tvd=float(chars["JP_TVD"]),
        jpump_md=float(chars.get("JP_MD", chars["JP_TVD"])),
        tubing_od=float(chars.get("out_dia", 4.5)),
        tubing_thickness=float(chars.get("thick", 0.271)),
        casing_od=casing_od,
        casing_thickness=casing_thk,
        field_model=fm,
        surf_pres=float(surf_pres),
        form_wc=0.5,
        form_gor=250.0,
        qwf=750.0,
        pwf=500.0,
        jpump_direction=jpump_direction,
    )

    if vogel_row:
        params["res_pres"] = float(vogel_row["ResP"])
        params["form_wc"] = float(vogel_row.get("form_wc", 0.5))
        params["form_gor"] = float(vogel_row.get("fgor", 250))
        params["qwf"] = float(vogel_row["qwf"])
        params["pwf"] = float(vogel_row["pwf"])

    return WellConfig(**params)


def create_well_objects(wc: WellConfig):
    """Simulation objects (mirrors NetworkOptimizer._create_well_objects)."""
    from server.services.factories import build_well_profile

    tube = Pipe(out_dia=wc.tubing_od, thick=wc.tubing_thickness)
    case = Pipe(out_dia=wc.casing_od, thick=wc.casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)

    well_profile = build_well_profile(wc.well_name, float(wc.jpump_tvd), wc.field_model)

    oil_qwf = wc.qwf * (1 - wc.form_wc)
    inflow = InFlow(qwf=oil_qwf, pwf=wc.pwf, pres=wc.res_pres)

    oil, water, gas = create_pvt_components(wc.field_model)
    res_mix = ResMix(wc=wc.form_wc, fgor=wc.form_gor, oil=oil, wat=water, gas=gas)
    prop_pf = water.condition(0, 60)

    return wellbore, well_profile, inflow, res_mix, prop_pf


def default_pad_pf(well_name: str) -> float:
    """Pad-level power-fluid pressure default for a well.

    PF-PRESSURE-DEPENDENCY: there is still no per-well PF pressure in
    Databricks, so the tools seed from the pad table and let the engineer
    edit per row - exactly as the tabs did.
    """
    return float(wells_svc._default_pad_pf(pad_from_mp_name(well_name)))


def json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records for a tool result frame."""
    from server.services import frames

    return frames.records(df)
