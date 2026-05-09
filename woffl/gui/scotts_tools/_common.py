"""Shared helpers for Scott's Tools tabs.

Name normalization, well-test fetchers, jp_chars helpers, and well-config
builders used by the PF Scenario and JP Friction Calibration tabs.
"""

import re

import pandas as pd
import streamlit as st

from woffl.assembly.network_optimizer import WellConfig, _load_well_profile
from woffl.flow.inflow import InFlow
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.gui.utils import create_pvt_components, load_well_characteristics
from woffl.pvt.resmix import ResMix


# ── name helpers ───────────────────────────────────────────────────────────


def normalize_short_name(name: str) -> str:
    """Convert shorthand well names to MP format.

    B-30 -> MPB-30, I-15 -> MPI-15.  Already-prefixed names pass through.
    """
    name = name.strip()
    if name.upper().startswith("MP"):
        return name
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    if m:
        return f"MP{m.group(1).upper()}-{int(m.group(2))}"
    return name


def pad_from_mp_name(mp_name: str) -> str:
    """MPB-30 -> B, MPI-15 -> I."""
    return mp_name.replace("MP", "").split("-")[0]


# ── IPR from well tests ────────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_well_tests(months_back: int):
    """Fetch well tests with BHP filter. Cached 24h per months_back value."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    df, _ = fetch_milne_well_tests(start_date, end_date)
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_well_tests_raw(months_back: int):
    """Fetch well tests WITHOUT dropping gaugeless rows. Cached 24h."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import (
        WELL_TEST_QUERY,
        _normalize_well_name,
        get_mpu_well_names,
    )

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    well_names = get_mpu_well_names()
    if not well_names:
        return pd.DataFrame()
    well_list = ", ".join(f"'{w}'" for w in well_names)
    df = execute_query(WELL_TEST_QUERY.format(well_list=well_list, start_date=start_date, end_date=end_date))
    if df.empty:
        return df

    rename = {"well_name": "well", "wt_date": "WtDate", "bhp": "BHP", "oil_rate": "WtOilVol", "fwat_rate": "WtWaterVol", "fgas_rate": "WtGasVol"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "WtOilVol" in df.columns and "WtWaterVol" in df.columns:
        df["WtTotalFluid"] = df["WtOilVol"] + df["WtWaterVol"]
    if "well" in df.columns:
        df["well"] = df["well"].apply(_normalize_well_name)
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], utc=True).dt.tz_localize(None)
    for col in ["BHP", "WtOilVol", "WtWaterVol", "WtTotalFluid", "WtGasVol", "lift_wat", "whp", "fgor", "form_wc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only require well + date + fluid rate (NOT BHP)
    req = [c for c in ["well", "WtDate", "WtTotalFluid"] if c in df.columns]
    df = df.dropna(subset=req)
    return df.sort_values(by=["well", "WtDate"])


def get_vogel_for_wells(well_names: list[str], months_back: int = 3) -> dict:
    """Return {well_name: vogel_row_dict} for wells with BHP gauge data."""
    from woffl.assembly.ipr_analyzer import compute_vogel_coefficients, estimate_reservoir_pressure

    try:
        wt_df = fetch_well_tests(months_back)
    except Exception:
        wt_df = st.session_state.get("all_well_tests_df")

    if wt_df is None or wt_df.empty:
        return {}

    jp_chars_df = load_well_characteristics()
    filtered = wt_df[wt_df["well"].isin(well_names)].copy()
    if filtered.empty:
        return {}

    try:
        merged = estimate_reservoir_pressure(filtered, jp_chars=jp_chars_df)
        vogel = compute_vogel_coefficients(merged)
        return vogel.set_index("Well").to_dict("index")
    except Exception:
        return {}


def get_latest_whp_per_well(months_back: int) -> dict[str, float]:
    """Return {well_name: latest_whp} from raw well tests within the lookback window.

    Pulls from the unfiltered (BHP-not-required) raw test set so wells without
    BHP gauges still get a real WHP value when one is available.
    """
    try:
        raw = fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw is None or raw.empty or "whp" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["whp"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").last()
    return {w: float(latest.loc[w, "whp"]) for w in latest.index}


def get_latest_bhp_per_well(months_back: int) -> dict[str, float]:
    """Return {well_name: latest_bhp} for wells with measured BHP in the window.

    Used as the calibration target when "Auto-calibrate per well" is selected.
    Wells absent from the returned dict have no measured BHP and cannot be
    friction-coef-calibrated.
    """
    try:
        raw = fetch_well_tests_raw(months_back)
    except Exception:
        return {}
    if raw is None or raw.empty or "BHP" not in raw.columns:
        return {}
    valid = raw.dropna(subset=["BHP"]).sort_values("WtDate")
    if valid.empty:
        return {}
    latest = valid.groupby("well").last()
    return {w: float(latest.loc[w, "BHP"]) for w in latest.index}


# ── jp_chars helpers ───────────────────────────────────────────────────────


def friction_coefs_from_chars(chars: dict | None) -> dict:
    """Extract jet-pump friction coefficients from a jp_chars row.

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


def casing_dims_from_chars(chars: dict | None) -> tuple[float, float]:
    """Return (casing_od, casing_thickness) from chars; fallback 6.875 / 0.5."""
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


# ── well config + objects ──────────────────────────────────────────────────


def build_well_config(
    well_name: str,
    jp_chars_dict: dict,
    vogel_row: dict | None = None,
    surf_pres: float = 210.0,
) -> WellConfig:
    """Build WellConfig from jp_chars, optionally overriding IPR with Vogel data.

    Args:
        well_name: Well identifier (e.g. "MPB-30")
        jp_chars_dict: Dict from load_well_characteristics().set_index("Well")
        vogel_row: Optional Vogel coefficient row to override IPR
        surf_pres: Surface (wellhead) pressure in psi (typically pulled from
            latest well test by the caller)
    """
    chars = jp_chars_dict.get(well_name)
    if not chars:
        raise ValueError(f"{well_name} not in jp_chars database")

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
    )

    # Override with well-test-derived IPR when available
    if vogel_row:
        params["res_pres"] = float(vogel_row["ResP"])
        params["form_wc"] = float(vogel_row.get("form_wc", 0.5))
        params["form_gor"] = float(vogel_row.get("fgor", 250))
        params["qwf"] = float(vogel_row["qwf"])
        params["pwf"] = float(vogel_row["pwf"])

    return WellConfig(**params)


def create_well_objects(wc: WellConfig):
    """Create simulation objects (mirrors NetworkOptimizer._create_well_objects)."""
    tube = Pipe(out_dia=wc.tubing_od, thick=wc.tubing_thickness)
    case = Pipe(out_dia=wc.casing_od, thick=wc.casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)

    jpump_md = wc.jpump_md if wc.jpump_md else wc.jpump_tvd
    well_profile = _load_well_profile(wc.well_name, jpump_md, wc.field_model)

    oil_qwf = wc.qwf * (1 - wc.form_wc)
    inflow = InFlow(qwf=oil_qwf, pwf=wc.pwf, pres=wc.res_pres)

    oil, water, gas = create_pvt_components(wc.field_model)
    res_mix = ResMix(wc=wc.form_wc, fgor=wc.form_gor, oil=oil, wat=water, gas=gas)
    prop_pf = water.condition(0, 60)

    return wellbore, well_profile, inflow, res_mix, prop_pf
