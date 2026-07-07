"""Well Test Processor

Parses FDC well test CSV files, extracts well names, cleans data,
and merges with BHP data from Databricks.

Adapted from header_pressure_impact/process_data/welltests.py and merge.py.
"""

import io
from typing import List, Optional, Union

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

        # Extract + normalize well name from EntName1 (e.g., "B-028A" -> "MPB-28").
        # Delegates to the canonical well_test_client._normalize_well_name so
        # this parsing doesn't drift from the other name-normalization call
        # sites (DB names are 3-digit zero-padded; it strips ONE leading zero
        # so the GUI form is 2-digit zero-padded to match jp_chars: B-028 ->
        # MPB-28, B-008 -> MPB-08, NOT MPB-8 — jp_chars keys single-digit
        # wells as MPH-08).
        from woffl.assembly.well_test_client import _normalize_well_name

        df["well"] = df["EntName1"].apply(_normalize_well_name)

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
                df[column] = (
                    df[column].astype(str).str.replace(",", "").replace("nan", "0")
                )
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
