"""Databricks SQL Client

Centralized Databricks connectivity module that works in two environments:
- Inside a Databricks App: Uses DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET (OAuth M2M)
- Local development: Uses databricks-sql-connector with .env credentials
  (bricks_host, bricks_http, bricks_token)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_WAREHOUSE_ID = "698745db7da46ba3"


def _is_deployed() -> bool:
    return bool(os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"))


def _query_via_connector(query: str) -> pd.DataFrame:
    from databricks import sql

    if _is_deployed():
        import json
        import urllib.parse
        import urllib.request

        host = os.getenv("DATABRICKS_HOST")
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
        http_path = f"/sql/1.0/warehouses/{DEFAULT_WAREHOUSE_ID}"

        # Fetch OAuth M2M token directly
        token_url = f"https://{host}/oidc/v1/token"
        data = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "scope": "all-apis",
            }
        ).encode()
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, token_url, client_id, client_secret)
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        with opener.open(urllib.request.Request(token_url, data=data, method="POST")) as resp:
            token = json.loads(resp.read())["access_token"]

        connection = sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
        )
    else:
        from dotenv import load_dotenv

        load_dotenv()

        host = os.getenv("bricks_host")
        token = os.getenv("bricks_token")
        http_path = os.getenv("bricks_http") or f"/sql/1.0/warehouses/{DEFAULT_WAREHOUSE_ID}"

        if not all([host, token]):
            raise RuntimeError(
                "Missing local Databricks credentials. "
                "Set bricks_host, bricks_token, and optionally bricks_http in your .env file."
            )

        connection = sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
        )

    try:
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in (cursor.description or [])]
        cursor.close()
    finally:
        connection.close()

    return pd.DataFrame(result, columns=columns)


def execute_query(query: str) -> pd.DataFrame:
    return _query_via_connector(query)


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
            well_df_pivoted = well_df.pivot(index="date", columns="tag", values="max_average_value")
            column_mapping = {bhp_tag: "BHP", headerP_tag: "HeaderP", whp_tag: "WHP"}
            well_df_pivoted = well_df_pivoted.rename(columns=column_mapping)
            for col in ["BHP", "HeaderP", "WHP"]:
                if col in well_df_pivoted.columns:
                    well_df_pivoted[col] = pd.to_numeric(well_df_pivoted[col], errors="coerce")
            well_dfs[well] = well_df_pivoted

    return well_dfs
