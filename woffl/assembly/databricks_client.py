"""Databricks SQL Client

Centralized Databricks connectivity module that works in two environments:
- Inside a Databricks App: Uses DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET (OAuth M2M)
- Local development: Uses databricks-sql-connector with .env credentials
  (bricks_host, bricks_http, bricks_token)
"""

import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_WAREHOUSE_ID = "698745db7da46ba3"
SCHRADER_TVD_CUTOFF = 5500.0

# OAuth token cache (deployed M2M flow). Tokens last ~1 h; minting a fresh one
# per query added ~0.5-1 s of pure overhead to every warehouse call.
_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: dict = {"token": None, "expires_at": 0.0}

# Per-thread connection cache. Streamlit reuses its script threads, so keeping
# one warehouse session per thread skips the ~1-2 s connect handshake on every
# query. ProcessPool workers each get fresh module state, so this is safe there
# too. databricks-sql connections are NOT thread-safe — hence thread-local, not
# a shared pool.
_CONN_LOCAL = threading.local()


def _is_deployed() -> bool:
    return bool(
        os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET")
    )


def _oauth_token() -> str:
    """Return a cached M2M OAuth token, refreshing ~60 s before expiry."""
    import base64
    import json
    import urllib.parse
    import urllib.request

    now = time.time()
    with _TOKEN_LOCK:
        if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"] - 60:
            return _TOKEN_CACHE["token"]

        host = os.getenv("DATABRICKS_HOST")
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")

        # Encode credentials as Basic auth
        credentials = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()

        token_url = f"https://{host}/oidc/v1/token"
        data = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "scope": "sql",
            }
        ).encode()

        req = urllib.request.Request(
            token_url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        # timeout: a stalled OIDC endpoint used to hang the whole process
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())

        _TOKEN_CACHE["token"] = payload["access_token"]
        _TOKEN_CACHE["expires_at"] = now + float(payload.get("expires_in", 3600))
        return _TOKEN_CACHE["token"]


def _new_connection():
    from databricks import sql

    if _is_deployed():
        host = os.getenv("DATABRICKS_HOST")
        http_path = f"/sql/1.0/warehouses/{DEFAULT_WAREHOUSE_ID}"
        return sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=_oauth_token(),
        )

    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("bricks_host")
    token = os.getenv("bricks_token")
    http_path = (
        os.getenv("bricks_http") or f"/sql/1.0/warehouses/{DEFAULT_WAREHOUSE_ID}"
    )

    if not all([host, token]):
        raise RuntimeError(
            "Missing local Databricks credentials. "
            "Set bricks_host, bricks_token, and optionally bricks_http in your .env file."
        )

    return sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )


def _query_via_connector(query: str) -> pd.DataFrame:
    """Execute a query on the per-thread cached connection.

    One retry with a fresh connection (and a forced token refresh) covers the
    stale-session cases: warehouse idle-stop, network blips, token expiry
    mid-session. A genuinely bad query fails twice and raises.
    """
    last_err: Exception | None = None
    for attempt in range(2):
        conn = getattr(_CONN_LOCAL, "conn", None)
        if conn is None:
            conn = _new_connection()
            _CONN_LOCAL.conn = conn
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [desc[0] for desc in (cursor.description or [])]
            finally:
                cursor.close()
            return pd.DataFrame(result, columns=columns)
        except Exception as e:
            last_err = e
            _CONN_LOCAL.conn = None
            try:
                conn.close()
            except Exception:
                pass
            with _TOKEN_LOCK:
                _TOKEN_CACHE["token"] = None  # force refresh on the retry
    raise last_err  # type: ignore[misc]


def execute_query(query: str) -> pd.DataFrame:
    return _query_via_connector(query)


JP_HISTORY_QUERY = """\
SELECT
    w.wellname AS `Well Name`,
    jp.DateSet AS `Date Set`,
    jp.DatePulled AS `Date Pulled`,
    jp.NozzleNumber AS `Nozzle Number`,
    jp.ThroatRatio AS `Throat Ratio`,
    jp.TubingDiameter AS `Tubing Diameter`
FROM apps.mpu_tracker.tbl_jetpump_data jp
JOIN apps.mpu_tracker.tbl_wells w ON jp.loc_id = w.loc_id
ORDER BY w.wellname, jp.DateSet DESC
"""


def fetch_jp_history() -> pd.DataFrame:
    """Fetch jet pump installation history from Databricks mpu_tracker."""
    df = execute_query(JP_HISTORY_QUERY)

    if "Date Set" in df.columns:
        df["Date Set"] = pd.to_datetime(df["Date Set"], errors="coerce")
    if "Date Pulled" in df.columns:
        df["Date Pulled"] = pd.to_datetime(df["Date Pulled"], errors="coerce")
    if "Well Name" in df.columns:
        df["Well Name"] = df["Well Name"].astype(str).str.strip()

    return df


WELL_PROPS_QUERY = """\
SELECT
    m.well_name,
    m.tubing_out_dia,
    m.tubing_inn_dia,
    m.tubing_absruff,
    m.casing_out_dia,
    m.casing_inn_dia,
    m.casing_absruff,
    m.jpump_md,
    m.jpfric_nozzle,
    m.jpfric_throat,
    m.jpfric_diffuser,
    m.jpfric_entry,
    r.form_oil_api,
    r.form_gas_sg,
    r.form_wat_sg,
    r.resvr_bubb,
    r.resvr_press,
    r.resvr_temp
FROM mpu.wells.vw_prop_mech m
LEFT JOIN mpu.wells.vw_prop_resvr r USING (enthid)
ORDER BY m.well_name
"""


def fetch_well_props() -> pd.DataFrame:
    """Fetch combined mechanical + reservoir well properties from Databricks.

    Joins mpu.wells.vw_prop_mech and mpu.wells.vw_prop_resvr on enthid.
    Returns a DataFrame in jp_chars-compatible schema (Well, out_dia, thick,
    JP_MD, res_pres, form_temp) plus extended PVT fields (oil_api, gas_sg,
    wat_sg, bubble_point) and casing dims.

    Well names are normalized from Databricks format ("B-028") to GUI format
    ("MPB-28") via the existing well_test_client._normalize_well_name.

    Wells defined in jp_data/local_well_overrides.csv are appended at the end
    if they aren't already present in the Databricks result — used for wells
    that exist in the field but haven't yet been added to mpu.wells.prop_hist.
    Delete the CSV (or remove the row) once Databricks catches up.
    """
    from woffl.assembly.well_test_client import _normalize_well_name

    df = execute_query(WELL_PROPS_QUERY)
    if df.empty:
        return _merge_local_overrides(df)

    # Drop orphan rows with no well_name (e.g. the nameless enthid 32795743 in
    # vw_prop_mech). Without this, the astype(str) below turns a SQL NULL into the
    # literal string "None" -- a phantom well that has no survey CSV and so trips
    # the "missing deviation surveys" banner forever (and can never be pulled).
    df = df[df["well_name"].notna() & (df["well_name"].astype(str).str.strip() != "")].copy()

    df["Well"] = df["well_name"].astype(str).str.strip().map(_normalize_well_name)
    df["out_dia"] = df["tubing_out_dia"]
    df["thick"] = (df["tubing_out_dia"] - df["tubing_inn_dia"]) / 2.0
    df["JP_MD"] = df["jpump_md"]
    df["res_pres"] = df["resvr_press"]
    df["form_temp"] = df["resvr_temp"]
    df["oil_api"] = df["form_oil_api"]
    df["gas_sg"] = df["form_gas_sg"]
    df["wat_sg"] = df["form_wat_sg"]
    df["bubble_point"] = df["resvr_bubb"]

    return _merge_local_overrides(df)


def _local_overrides_path() -> Path:
    return Path(__file__).resolve().parent.parent / "jp_data" / "local_well_overrides.csv"


def _merge_local_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Append rows from local_well_overrides.csv for wells not yet in Databricks."""
    path = _local_overrides_path()
    if not path.exists():
        return df
    try:
        overrides = pd.read_csv(path)
    except Exception:
        return df
    if overrides.empty or "Well" not in overrides.columns:
        return df

    existing = set(df["Well"].dropna().astype(str)) if "Well" in df.columns else set()
    new_rows = overrides[~overrides["Well"].astype(str).isin(existing)].copy()
    if new_rows.empty:
        return df

    # Derive the same downstream columns fetch_well_props computes from the
    # raw mech/resvr columns. NaN-safe: missing inputs leave outputs as NaN.
    if "tubing_out_dia" in new_rows.columns:
        new_rows["out_dia"] = new_rows["tubing_out_dia"]
    if {"tubing_out_dia", "tubing_inn_dia"}.issubset(new_rows.columns):
        new_rows["thick"] = (new_rows["tubing_out_dia"] - new_rows["tubing_inn_dia"]) / 2.0
    if "jpump_md" in new_rows.columns:
        new_rows["JP_MD"] = new_rows["jpump_md"]
    if "resvr_press" in new_rows.columns:
        new_rows["res_pres"] = new_rows["resvr_press"]
    if "resvr_temp" in new_rows.columns:
        new_rows["form_temp"] = new_rows["resvr_temp"]
    if "form_oil_api" in new_rows.columns:
        new_rows["oil_api"] = new_rows["form_oil_api"]
    if "form_gas_sg" in new_rows.columns:
        new_rows["gas_sg"] = new_rows["form_gas_sg"]
    if "form_wat_sg" in new_rows.columns:
        new_rows["wat_sg"] = new_rows["form_wat_sg"]
    if "resvr_bubb" in new_rows.columns:
        new_rows["bubble_point"] = new_rows["resvr_bubb"]

    return pd.concat([df, new_rows], ignore_index=True)


def _survey_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "jp_data" / "well_surveys"


def _load_survey(well_name: str) -> Optional[pd.DataFrame]:
    path = _survey_dir() / f"{well_name} Deviation Survey.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _interp_tvd(survey: Optional[pd.DataFrame], jp_md: float) -> Optional[float]:
    if survey is None or survey.empty:
        return None
    if "meas_depth" not in survey.columns or "tvd_depth" not in survey.columns:
        return None
    s = survey.dropna(subset=["meas_depth", "tvd_depth"]).sort_values("meas_depth")
    if s.empty:
        return None
    return float(np.interp(jp_md, s["meas_depth"], s["tvd_depth"]))


def _pad_prefix(well: str) -> str:
    return well.split("-", 1)[0] if "-" in well else well


def enrich_with_tvd(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add JP_TVD, tvd_estimated, and is_sch columns to a well-props DataFrame.

    JP_TVD is computed by interpolating the local deviation survey for each well
    using JP_MD. Wells without a survey CSV get an estimated JP_TVD based on the
    pad-average TVD/MD ratio (or a global ratio if the pad has no surveys at all).

    Args:
        df: DataFrame from fetch_well_props() (must have Well, JP_MD columns)

    Returns:
        (enriched DataFrame, list of well names with estimated TVD)
    """
    jp_tvd: list[float | None] = []
    estimated: list[bool] = []
    pad_ratios: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        well = row["Well"]
        jp_md = row["JP_MD"]
        if not isinstance(jp_md, (int, float)) or math.isnan(jp_md):
            jp_tvd.append(None)
            estimated.append(False)
            continue
        survey = _load_survey(well)
        tvd = _interp_tvd(survey, jp_md)
        if tvd is not None and jp_md > 0:
            pad_ratios.setdefault(_pad_prefix(well), []).append(tvd / jp_md)
        jp_tvd.append(tvd)
        estimated.append(False)

    all_ratios = [r for vals in pad_ratios.values() for r in vals]
    global_ratio = (sum(all_ratios) / len(all_ratios)) if all_ratios else 0.92

    missing: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if jp_tvd[i] is not None:
            continue
        well = row["Well"]
        jp_md = row["JP_MD"]
        if not isinstance(jp_md, (int, float)) or math.isnan(jp_md):
            continue
        ratios = pad_ratios.get(_pad_prefix(well), [])
        ratio = (sum(ratios) / len(ratios)) if ratios else global_ratio
        jp_tvd[i] = jp_md * ratio
        estimated[i] = True
        missing.append(well)

    df = df.copy()
    df["JP_TVD"] = pd.Series(jp_tvd, index=df.index)
    df["tvd_estimated"] = pd.Series(estimated, index=df.index)
    df["is_sch"] = df["JP_TVD"].fillna(0) < SCHRADER_TVD_CUTOFF
    return df, missing


def fetch_well_props_enriched() -> Tuple[pd.DataFrame, List[str]]:
    """Fetch well properties from Databricks and add JP_TVD + is_sch.

    Returns:
        (DataFrame with all jp_chars-compatible columns, list of wells with
         estimated TVD)
    """
    df = fetch_well_props()
    if df.empty:
        return df, []
    return enrich_with_tvd(df)


def load_tag_dict(custom_source=None) -> Dict[str, Tuple[str, str, str]]:
    """Load SCADA tag mapping from bhp_dict.csv."""
    if custom_source is not None:
        if isinstance(custom_source, (str, Path)):
            df = pd.read_csv(Path(custom_source))
        else:
            df = pd.read_csv(custom_source)
    else:
        current_dir = Path(__file__).parent
        dict_path = current_dir / ".." / "jp_data" / "bhp_dict.csv"
        df = pd.read_csv(dict_path)

    tag_dict = {}
    for _, row in df.iterrows():
        tag_dict[row["wellname"]] = (
            str(row["bhp_tag"]).strip(),
            str(row["headerP_tag"]).strip(),
            str(row["whp_tag"]).strip(),
        )
    return tag_dict


def get_tags_for_wells(
    wells: List[str], tag_dict: Dict[str, Tuple[str, str, str]]
) -> Tuple[Dict[str, Tuple[str, str, str]], List[str]]:
    found = {}
    missing = []
    for well in wells:
        if well in tag_dict:
            found[well] = tag_dict[well]
        else:
            missing.append(well)
    return found, missing


def query_bhp_for_well_tests(
    tag_dict: Dict[str, Tuple[str, str, str]],
    well_list: List[str],
) -> Dict[str, pd.DataFrame]:
    """Query Databricks for 6-hour time-weighted average BHP data aligned to test dates."""
    well_tags, _ = get_tags_for_wells(well_list, tag_dict)

    if not well_tags:
        return {}

    flat_tag_list = []
    for bhp_tag, headerP_tag, whp_tag in well_tags.values():
        for tag in [bhp_tag, headerP_tag, whp_tag]:
            if tag and tag not in flat_tag_list:
                flat_tag_list.append(tag)

    tag_list_str = ", ".join(f"'{tag}'" for tag in flat_tag_list)

    query = f"""
    WITH SixHourAverages AS (
        SELECT
            CAST(FLOOR(CAST(LocalTime AS BIGINT) / 21600) * 21600 AS TIMESTAMP) AS time_interval_start,
            tag,
            AVG(value) AS average_value
        FROM
            reporting.historian.vw_mpu_measurements
        WHERE
            tag IN ({tag_list_str})
        GROUP BY
            time_interval_start,
            tag
    )
    SELECT
        CAST(time_interval_start AS DATE) AS date,
        tag,
        MAX(average_value) AS max_average_value
    FROM
        SixHourAverages
    GROUP BY
        CAST(time_interval_start AS DATE),
        tag
    ORDER BY
        date, tag;
    """

    raw = execute_query(query)
    raw["date"] = pd.to_datetime(raw["date"])

    well_dfs = {}
    for well, (bhp_tag, headerP_tag, whp_tag) in well_tags.items():
        well_df = raw[raw["tag"].isin([bhp_tag, headerP_tag, whp_tag])]
        if not well_df.empty:
            well_df_pivoted = well_df.pivot(
                index="date", columns="tag", values="max_average_value"
            )
            column_mapping = {bhp_tag: "BHP", headerP_tag: "HeaderP", whp_tag: "WHP"}
            well_df_pivoted = well_df_pivoted.rename(columns=column_mapping)
            for col in ["BHP", "HeaderP", "WHP"]:
                if col in well_df_pivoted.columns:
                    well_df_pivoted[col] = pd.to_numeric(
                        well_df_pivoted[col], errors="coerce"
                    )
            well_dfs[well] = well_df_pivoted

    return well_dfs
