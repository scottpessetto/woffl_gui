"""Databricks SQL Client

Centralized Databricks connectivity module that works in two environments:
- Inside a Databricks App: Uses the Databricks SDK with automatic service principal auth
- Local development: Uses databricks-sql-connector with .env credentials

Adapted from header_pressure_impact/pull_data/pull_tags.py query patterns.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# SQL warehouse ID (matches app.yaml resource)
DEFAULT_WAREHOUSE_ID = "ce196438d74e4329"


def is_databricks_app() -> bool:
    """Detect if running inside a Databricks App environment."""
    return os.getenv("DATABRICKS_APP_NAME") is not None or os.path.exists("/databricks")


def _query_via_sdk(query: str, warehouse_id: str = DEFAULT_WAREHOUSE_ID) -> pd.DataFrame:
    """Execute SQL via Databricks SDK Statement Execution API (inside Databricks App).

    Args:
        query: SQL query string
        warehouse_id: SQL warehouse ID

    Returns:
        pd.DataFrame with query results
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="120s",
    )

    if response.status and response.status.state and response.status.state.value == "FAILED":
        error_msg = response.status.error.message if response.status.error else "Unknown error"
        raise RuntimeError(f"Databricks query failed: {error_msg}")

    columns = [col.name for col in response.manifest.schema.columns]
    rows = response.result.data_array if response.result and response.result.data_array else []
    return pd.DataFrame(rows, columns=columns)


def _query_via_connector(query: str) -> pd.DataFrame:
    """Execute SQL via databricks-sql-connector (local development).

    Uses credentials from .env file: bricks_host, bricks_http, bricks_token.

    Args:
        query: SQL query string

    Returns:
        pd.DataFrame with query results
    """
    from databricks import sql
    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("bricks_host")
    http_path = os.getenv("bricks_http")
    token = os.getenv("bricks_token")

    if not all([host, http_path, token]):
        raise RuntimeError(
            "Missing Databricks credentials in .env file. " "Required: bricks_host, bricks_http, bricks_token"
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
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
    finally:
        connection.close()

    return pd.DataFrame(result, columns=columns)


def execute_query(query: str) -> pd.DataFrame:
    """Execute a SQL query against Databricks, auto-detecting the environment.

    Args:
        query: SQL query string

    Returns:
        pd.DataFrame with query results
    """
    if is_databricks_app():
        return _query_via_sdk(query)
    else:
        return _query_via_connector(query)


def load_tag_dict(custom_source=None) -> Dict[str, Tuple[str, str, str]]:
    """Load SCADA tag mapping from bhp_dict.csv.

    Maps well names to (bhp_tag, headerP_tag, whp_tag) tuples.

    Args:
        custom_source: Optional path (str/Path) or file-like object (e.g. Streamlit UploadedFile)
            for a custom tag mapping CSV. If None, uses the bundled bhp_dict.csv in jp_data/.

    Returns:
        Dictionary mapping well name to (bhp_tag, headerP_tag, whp_tag)
    """
    if custom_source is not None:
        if isinstance(custom_source, (str, Path)):
            df = pd.read_csv(Path(custom_source))
        else:
            # File-like object (e.g. Streamlit UploadedFile)
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
    """Get SCADA tags for specified wells.

    Args:
        wells: List of well names to look up
        tag_dict: Full tag dictionary from load_tag_dict()

    Returns:
        Tuple of (found_tags_dict, missing_wells_list)
    """
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
    """Query Databricks for 6-hour time-weighted average BHP data aligned to test dates.

    Returns the max 6-hour average per day per tag — best captures test conditions.

    Adapted from header_pressure_impact/pull_data/pull_tags.py query_tag_WT_average().

    Args:
        tag_dict: Dictionary mapping well names to (bhp_tag, headerP_tag, whp_tag)
        well_list: List of well names to query

    Returns:
        Dictionary mapping well name to DataFrame with BHP, HeaderP, WHP columns
        indexed by datetime
    """
    # Get tags only for requested wells
    well_tags, _ = get_tags_for_wells(well_list, tag_dict)

    if not well_tags:
        return {}

    # Build flat tag list for SQL IN clause
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

    # Pivot into per-well DataFrames
    well_dfs = {}
    for well, (bhp_tag, headerP_tag, whp_tag) in well_tags.items():
        well_df = raw[raw["tag"].isin([bhp_tag, headerP_tag, whp_tag])]
        if not well_df.empty:
            well_df_pivoted = well_df.pivot(index="date", columns="tag", values="max_average_value")
            column_mapping = {bhp_tag: "BHP", headerP_tag: "HeaderP", whp_tag: "WHP"}
            well_df_pivoted = well_df_pivoted.rename(columns=column_mapping)
            # Ensure numeric types
            for col in ["BHP", "HeaderP", "WHP"]:
                if col in well_df_pivoted.columns:
                    well_df_pivoted[col] = pd.to_numeric(well_df_pivoted[col], errors="coerce")
            well_dfs[well] = well_df_pivoted

    return well_dfs
