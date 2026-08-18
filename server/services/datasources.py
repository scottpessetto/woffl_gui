"""Databricks-backed data sources with TTL caching and soft fallbacks.

Read-only. Every cached fetcher raises on failure (failures are never
cached); the ``*_safe`` wrappers at the service edge degrade to bundled
files or empty frames so routers stay 200 where the Streamlit app did.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from server import config
from server.cache import ttl_cache

log = logging.getLogger("woffl.web.datasources")

# Columns of an empty PF-latest frame so consumers never special-case.
# mirrors woffl/gui/utils.py:_PF_LATEST_COLS
_PF_LATEST_COLS = ["well", "pf_press", "pf_source", "pf_date", "tubing_prs", "inn_ann_prs"]


# ---------------------------------------------------------------------------
# Well characteristics (vw_prop_mech + vw_prop_resvr, TVD-enriched)
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_CHARS, maxsize=2)
def well_chars() -> tuple[pd.DataFrame, list[str]]:
    """Enriched well characteristics from Databricks.

    fetch_well_props_enriched already merges local_well_overrides.csv and
    adds JP_TVD / tvd_estimated / is_sch. Raises on failure or empty result
    so a Databricks blip is never cached.
    # mirrors woffl/gui/utils.py:_load_well_characteristics_cached

    Returns:
        (chars frame, wells with estimated JP_TVD i.e. missing surveys).
    """
    from woffl.assembly.databricks_client import fetch_well_props_enriched

    df, missing = fetch_well_props_enriched()
    if df.empty:
        raise RuntimeError("vw_prop_mech returned no rows")
    return df, missing


def well_chars_safe() -> tuple[pd.DataFrame, str]:
    """Chars frame with jp_chars.csv fallback - availability beats freshness.

    The CSV fallback is rebuilt on every call (never cached) so Databricks
    is re-probed and live data returns the moment it recovers. Raises only
    when BOTH sources fail.
    # mirrors woffl/gui/utils.py:load_well_characteristics

    Returns:
        (chars frame, "databricks" | "csv_fallback").
    """
    try:
        df, _missing = well_chars()
        return df, "databricks"
    except Exception as db_err:
        log.warning("Databricks well-properties load failed: %s", db_err)
        try:
            fallback = pd.read_csv(config.JP_CHARS_CSV)
            if fallback.empty:
                raise RuntimeError("jp_chars.csv has no rows")
        except Exception as csv_err:
            raise RuntimeError(
                f"{db_err}; jp_chars.csv fallback also failed: {csv_err}"
            ) from csv_err
        return fallback, "csv_fallback"


def missing_surveys_safe() -> list[str]:
    """Wells with estimated JP_TVD (no local survey CSV); [] on any failure."""
    try:
        _df, missing = well_chars()
        return list(missing)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Live PF pressure (vw_pressure_daily)
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_PF_LATEST, maxsize=2)
def pf_latest() -> pd.DataFrame:
    """Latest valid PF surface pressure per well. Raises on failure.

    # mirrors woffl/gui/utils.py:_fetch_pf_latest_cached
    """
    from woffl.assembly.pf_pressure import fetch_pf_latest

    return fetch_pf_latest()


def pf_latest_safe() -> pd.DataFrame:
    """Soft-fail PF pull - empty frame when Databricks is unreachable.

    # mirrors woffl/gui/utils.py:load_pf_latest
    """
    try:
        return pf_latest()
    except Exception:
        return pd.DataFrame(columns=_PF_LATEST_COLS)


# ---------------------------------------------------------------------------
# Jet pump installation history (mpu_tracker, enriched)
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_JP_HISTORY, maxsize=2)
def _jp_history_databricks() -> pd.DataFrame:
    from woffl.assembly.databricks_client import fetch_jp_history
    from woffl.gui.pump_identity import enrich_jp_history

    df = fetch_jp_history()
    if df.empty:
        raise RuntimeError("mpu_tracker returned no jet pump rows")
    return enrich_jp_history(df)


@ttl_cache(config.TTL_JP_HISTORY, maxsize=2)
def _jp_history_excel() -> pd.DataFrame:
    from woffl.assembly.jp_history import parse_jp_history
    from woffl.gui.pump_identity import enrich_jp_history

    return enrich_jp_history(parse_jp_history(config.JP_HISTORY_XLSX))


def jp_history() -> tuple[pd.DataFrame, str]:
    """Enriched JP history (Circ Direction + Guiberson conversion applied).

    Databricks first; bundled xlsx fallback. Databricks is re-probed on
    every call after a failure (the failed fetch is never cached). Raises
    only when BOTH sources fail.

    Returns:
        (enriched frame, "databricks" | "excel_fallback").
    """
    try:
        return _jp_history_databricks(), "databricks"
    except Exception as db_err:
        log.warning("Databricks JP history load failed: %s", db_err)
        return _jp_history_excel(), "excel_fallback"


def jp_history_safe() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """(frame, source) or (None, None) when both sources fail."""
    try:
        return jp_history()
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Deviation surveys (local CSVs)
# ---------------------------------------------------------------------------


def has_survey(well: str) -> bool:
    """True when a local deviation survey CSV exists for the well."""
    return (config.SURVEY_DIR / f"{well} Deviation Survey.csv").is_file()


@ttl_cache(config.TTL_PROFILES, maxsize=1)
def surveyed_wells() -> frozenset[str]:
    """Every well name with a local deviation survey, from ONE listing.

    ``has_survey`` per well is a filesystem stat per well; the fleet is ~90,
    and /api/wells asked for all of them on every request. The CSVs only
    change on deploy, so one cached listing answers all of it.
    """
    suffix = " Deviation Survey.csv"
    try:
        return frozenset(
            p.name[: -len(suffix)]
            for p in config.SURVEY_DIR.iterdir()
            if p.name.endswith(suffix)
        )
    except OSError as exc:
        log.warning("could not list survey directory %s: %s", config.SURVEY_DIR, exc)
        return frozenset()


@ttl_cache(config.TTL_PROFILES, maxsize=256)
def survey(well: str) -> Optional[pd.DataFrame]:
    """Deviation survey frame (meas_depth, tvd_depth [, inclination]) or None.

    A missing/unreadable CSV caches as None - same as the Streamlit site.
    # mirrors woffl/gui/utils.py:get_well_survey_data
    """
    path = config.SURVEY_DIR / f"{well} Deviation Survey.csv"
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("Could not load survey data for %s: %s", well, exc)
        return None
