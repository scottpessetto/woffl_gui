"""Reusable DataFrame Display Component

Consolidates the repeated pattern of rename-columns → round-numerics →
display → download-CSV that appears throughout the GUI tabs.
"""

import pandas as pd
import streamlit as st


def display_results_table(
    df: pd.DataFrame,
    columns: list[str],
    rename_map: dict[str, str],
    numeric_cols: list[str],
    round_digits: int = 1,
    download_filename: str = "results.csv",
    download_label: str = "Download CSV",
    sort_by: list[str] | None = None,
    sort_ascending: list[bool] | None = None,
    key: str | None = None,
) -> pd.DataFrame:
    """Standard pattern for displaying and downloading a results DataFrame.

    Args:
        df: Source DataFrame (will not be modified)
        columns: Column names to select from df
        rename_map: Mapping of original column names to display names
        numeric_cols: Display column names to round
        round_digits: Number of decimal places for rounding
        download_filename: Filename for the CSV download button
        download_label: Label text for the download button
        sort_by: Display column names to sort by (optional)
        sort_ascending: Sort direction for each sort column (optional)
        key: Unique Streamlit key for the download button. Defaults to
             the download_filename to avoid duplicate element ID errors.

    Returns:
        The formatted display DataFrame (for further use if needed)
    """
    display_df = df[columns].copy()
    display_df = display_df.rename(columns=rename_map)

    # Round numeric columns (only those that exist in the display df)
    existing_numeric = [c for c in numeric_cols if c in display_df.columns]
    if existing_numeric:
        display_df[existing_numeric] = display_df[existing_numeric].round(round_digits)

    # Sort if requested
    if sort_by:
        ascending = sort_ascending if sort_ascending else [True] * len(sort_by)
        display_df = display_df.sort_values(by=sort_by, ascending=ascending)

    # Display
    st.dataframe(display_df, use_container_width=True)

    # Download button — use key to avoid Streamlit duplicate element ID errors
    button_key = key if key is not None else f"dl_{download_filename}"
    csv = display_df.to_csv(index=False)
    st.download_button(
        label=download_label,
        data=csv,
        file_name=download_filename,
        mime="text/csv",
        key=button_key,
    )

    return display_df
