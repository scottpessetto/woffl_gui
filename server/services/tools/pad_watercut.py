"""Pad Water Cut - daily pad-level WC over time for pads G/H/I/J.

Port of the retired Streamlit tool. That tab was one client call wrapped in
Plotly; all of the physics/aggregation already lives in
``woffl.assembly.pad_watercut_client.fetch_pad_watercut``, so this is a cache
plus a JSON shaping step.

Method (unchanged, from the client): each well's last allocated test is
forward-filled across days; well-days with more than 6 h of shut-in are
excluded. H and I are treated as on-pad PF recycle (lift water stays), G and
J ship lift water back to the plant.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import frames

# The four pads the client models, plus the combined series it appends.
PADS = ("G", "H", "I", "J", "All")


@ttl_cache(config.TTL_CHARS, maxsize=8)
def _series(start_date: str, end_date: str) -> pd.DataFrame:
    """Daily [date, pad, oil, water, wc] for the window. One warehouse query."""
    from woffl.assembly.pad_watercut_client import fetch_pad_watercut

    return fetch_pad_watercut(start_date, end_date)


def default_window() -> tuple[str, str]:
    """The tab's default range: three years back to today."""
    today = dt.date.today()
    return (today - dt.timedelta(days=365 * 3)).isoformat(), today.isoformat()


def pad_watercut(start_date: str, end_date: str) -> dict[str, Any]:
    """One series per pad, ready to plot.

    Args:
        start_date: 'YYYY-MM-DD' inclusive.
        end_date: 'YYYY-MM-DD' inclusive, after start_date.

    Returns:
        dict: ``{"start", "end", "series": [{"pad", "points": [{date, wc,
        oil, water}...]}...]}``. Water cut is a FRACTION here; the chart
        renders the percentage, so the number the API returns matches every
        other wc in this codebase.

    Raises:
        ValueError: start_date is not before end_date (router -> 422).
    """
    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    df = _series(start_date, end_date)
    series: list[dict[str, Any]] = []
    if df is not None and not df.empty:
        for pad in PADS:
            sub = df[df["pad"] == pad]
            if sub.empty:
                continue
            sub = sub.sort_values("date")
            series.append(
                {
                    "pad": pad,
                    "points": [
                        {
                            "date": frames.json_value(r.get("date")),
                            "wc": frames.opt_float(r.get("wc")),
                            "oil": frames.opt_float(r.get("oil")),
                            "water": frames.opt_float(r.get("water")),
                        }
                        for r in sub.to_dict("records")
                    ],
                }
            )
    return {"start": start_date, "end": end_date, "series": series}
