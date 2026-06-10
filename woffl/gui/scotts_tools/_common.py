"""Shared helpers for Scott's Tools tabs.

Name normalization, well-test fetchers, jp_chars helpers, and well-config
builders used by the PF Scenario and JP Friction Calibration tabs.
"""

import os
import re

import pandas as pd
import streamlit as st


# ── parallelism budget ─────────────────────────────────────────────────────

def worker_ceiling() -> int:
    """Max ProcessPool workers permitted in the current environment.

    Reads the ``WOFFL_MAX_WORKERS`` env var (default 1 — Databricks Apps
    safe), parses defensively, and clamps by ``os.cpu_count()``. Returns at
    least 1. Tabs that expose a worker slider use this as the upper bound,
    so the Databricks deployment is auto-pinned to single-threaded.
    """
    raw = os.environ.get("WOFFL_MAX_WORKERS", "1")
    try:
        env_max = max(1, int(raw))
    except (TypeError, ValueError):
        env_max = 1
    return max(1, min(env_max, os.cpu_count() or 1))

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


# Re-exported from woffl.gui.utils so existing scotts_tools call sites keep
# working without import churn. The canonical definition lives in utils.py
# alongside PAD_PF_DEFAULTS, where the sidebar's well auto-populate also
# imports from.
from woffl.gui.utils import pad_from_mp_name  # noqa: E402, F401


# ── IPR from well tests ────────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False, max_entries=4)
def fetch_well_tests(months_back: int):
    """Fetch well tests with BHP filter. Cached 24h per months_back value."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    df, _ = fetch_milne_well_tests(start_date, end_date)
    return df


@st.cache_data(ttl=86400, show_spinner=False, max_entries=4)
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

    def _num(key: str, default):
        """NaN-safe numeric lookup — Databricks chars carry missing values as
        NaN under a *present* key, so dict.get's default never fires and
        float(nan) silently poisons the solve (e.g. InFlow(pres=nan))."""
        v = chars.get(key)
        if v is None:
            return default
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return default
        return default if pd.isna(fv) else fv

    is_sch = chars.get("is_sch", True)
    if isinstance(is_sch, str):
        is_sch = is_sch.lower() in ("true", "1", "yes")
    elif pd.isna(is_sch):
        is_sch = True
    fm = "Schrader" if is_sch else "Kuparuk"

    casing_od, casing_thk = casing_dims_from_chars(chars)

    jp_tvd = _num("JP_TVD", None)
    if jp_tvd is None:
        raise ValueError(f"{well_name} has no JP_TVD in jp_chars")

    params = dict(
        well_name=well_name,
        res_pres=_num("res_pres", 1800.0),
        form_temp=_num("form_temp", 75.0 if is_sch else 170.0),
        jpump_tvd=jp_tvd,
        jpump_md=_num("JP_MD", jp_tvd),
        tubing_od=_num("out_dia", 4.5),
        tubing_thickness=_num("thick", 0.271),
        casing_od=casing_od,
        casing_thickness=casing_thk,
        field_model=fm,
        surf_pres=float(surf_pres),
        form_wc=0.5,
        form_gor=250.0,
        qwf=750.0,
        pwf=500.0,
        # Per-well PVT from vw_prop_resvr (None → field-model preset inside
        # create_pvt_components) — same treatment as NetworkOptimizer.
        oil_api=_num("oil_api", None),
        gas_sg=_num("gas_sg", None),
        wat_sg=_num("wat_sg", None),
        bubble_point=_num("bubble_point", None),
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

    # Per-well PVT overrides replace the field preset when present — keeps
    # Scott's Tools calibrations on the same fluid model the optimizer uses.
    oil, water, gas = create_pvt_components(
        field_model=wc.field_model,
        oil_api=wc.oil_api,
        gas_sg=wc.gas_sg,
        wat_sg=wc.wat_sg,
        bubble_point=wc.bubble_point,
    )
    res_mix = ResMix(wc=wc.form_wc, fgor=wc.form_gor, oil=oil, wat=water, gas=gas)
    prop_pf = water.condition(0, 60)

    return wellbore, well_profile, inflow, res_mix, prop_pf
