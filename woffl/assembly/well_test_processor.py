"""Well Test Processor

Parses FDC well test CSV files, extracts well names, cleans data,
and merges with BHP data from Databricks.

Adapted from header_pressure_impact/process_data/welltests.py and merge.py.
"""

import io
from typing import Dict, List, Optional, Union

import pandas as pd


class WellTestProcessor:
    """Parse and process FDC well test CSV files.

    Handles the FDC export format, extracting well names using the
    EntName1 column pattern and cleaning numeric columns.

    Args:
        source: File path string, file-like object, or bytes from Streamlit upload
    """

    def __init__(self, source: Union[str, io.BytesIO, io.StringIO]) -> None:
        self._raw_df = pd.read_csv(source)
        self._processed_df: Optional[pd.DataFrame] = None

    def parse(self) -> pd.DataFrame:
        """Parse the FDC CSV into a cleaned DataFrame.

        Extracts well names from EntName1, cleans numeric columns,
        and drops unnecessary columns.

        Returns:
            Cleaned DataFrame with 'well' column added
        """
        df = self._raw_df.copy()

        # Ensure WtDate is datetime
        df["WtDate"] = pd.to_datetime(df["WtDate"])

        # Sort by well and date
        df = df.sort_values(by=["EntName1", "WtDate"])

        # Extract well name from EntName1 (e.g., "B-028A" -> "MPB-28")
        df["well"] = df["EntName1"].str.extract(r"(\w+-\d+)")
        # Remove leading zeros in well number
        df["well"] = df["well"].str.replace(r"-(0)(?=\d+)", "-", regex=True)
        df["well"] = "MP" + df["well"]

        # Drop columns that conflict with gauge data or are unnecessary
        columns_to_drop = [
            "BHP",  # We use gauge BHP instead
            "RouteGroupName",
            "EntName1",
            "WtHours",
            "Choke",
            "ChangeUser",
            "Textbox29",
            "WtSeparatorTemp",
            "WtLinePressVal",
            "WtEspFrequency",
            "Textbox26",
            "WtEspAmps",
            "WtWaterCutShakeout",
            "SolidsPct",
        ]
        # Only drop columns that exist
        existing_drops = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(existing_drops, axis=1)

        # Clean numeric columns (remove commas, fill NaN, convert to float)
        numeric_columns = [
            "IA",
            "WtOilVol",
            "WtGasVol",
            "WtGasRate",
            "WtGasLiftVol",
            "WtWaterVol",
            "WtrLift",
            "WtTotalFluid",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = df[column].astype(str).str.replace(",", "").replace("nan", "0")
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        self._processed_df = df
        return df

    def get_unique_wells(self) -> List[str]:
        """Get sorted list of unique well names found in the CSV.

        Returns:
            Sorted list of well name strings
        """
        if self._processed_df is None:
            self.parse()
        return sorted(self._processed_df["well"].unique().tolist())

    def get_well_tests(self, wells: Optional[List[str]] = None) -> pd.DataFrame:
        """Get well test data, optionally filtered to specific wells.

        Args:
            wells: List of well names to include. If None, returns all wells.

        Returns:
            DataFrame with well test data
        """
        if self._processed_df is None:
            self.parse()

        if wells is None:
            return self._processed_df.copy()

        return self._processed_df[self._processed_df["well"].isin(wells)].copy()

    def get_test_date_range(self) -> tuple:
        """Get the min and max test dates in the data.

        Returns:
            Tuple of (min_date, max_date) as pd.Timestamp
        """
        if self._processed_df is None:
            self.parse()
        return (
            self._processed_df["WtDate"].min(),
            self._processed_df["WtDate"].max(),
        )

    def get_test_count_per_well(self) -> pd.Series:
        """Get number of tests per well.

        Returns:
            Series with well names as index and test counts as values
        """
        if self._processed_df is None:
            self.parse()
        return self._processed_df.groupby("well").size().sort_values(ascending=False)


def merge_tests_with_bhp(
    well_list: List[str],
    bhp_data: Dict[str, pd.DataFrame],
    well_tests: pd.DataFrame,
) -> pd.DataFrame:
    """Merge well test data with BHP data from Databricks.

    For each well, performs an inner join on date to get rows where
    both test data and BHP gauge data exist.

    Adapted from header_pressure_impact/process_data/merge.py merge_data().

    Args:
        well_list: List of well names to merge
        bhp_data: Dictionary mapping well name to DataFrame with BHP/HeaderP/WHP columns
        well_tests: DataFrame with well test data (must have 'well' and 'WtDate' columns)

    Returns:
        Merged DataFrame with test data + BHP/HeaderP/WHP columns
    """
    merged_data = pd.DataFrame()

    for well in well_list:
        if well not in bhp_data:
            continue

        filtered_tag_data = bhp_data[well].copy()
        filtered_tests = well_tests[well_tests["well"] == well].copy()

        if filtered_tests.empty or filtered_tag_data.empty:
            continue

        # Normalize timezone on WtDate
        if filtered_tests["WtDate"].dt.tz is not None:
            filtered_tests["WtDate"] = filtered_tests["WtDate"].dt.tz_convert("UTC").dt.tz_localize(None)

        # Normalize timezone on BHP data index
        if filtered_tag_data.index.tz is not None:
            filtered_tag_data.index = filtered_tag_data.index.tz_convert("UTC").tz_localize(None)

        # Convert WtDate to date-only for matching (BHP data is daily)
        filtered_tests["merge_date"] = filtered_tests["WtDate"].dt.normalize()

        # Reset index on tag data to get date column
        filtered_tag_data = filtered_tag_data.reset_index()
        if "date" in filtered_tag_data.columns:
            filtered_tag_data["merge_date"] = pd.to_datetime(filtered_tag_data["date"]).dt.normalize()
        elif "datetime" in filtered_tag_data.columns:
            filtered_tag_data["merge_date"] = pd.to_datetime(filtered_tag_data["datetime"]).dt.normalize()
        else:
            # Index was the date
            filtered_tag_data["merge_date"] = pd.to_datetime(filtered_tag_data.index).normalize()

        merged_well_data = pd.merge(
            filtered_tests,
            filtered_tag_data,
            on="merge_date",
            how="inner",
        )

        # Drop the temporary merge column
        if "merge_date" in merged_well_data.columns:
            merged_well_data = merged_well_data.drop("merge_date", axis=1)

        merged_data = pd.concat([merged_data, merged_well_data], ignore_index=True)

    return merged_data
