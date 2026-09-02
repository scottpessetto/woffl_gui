"""Jet-pump MD placement on toe-up surveys (review 2026-09-01, findings 1-2).

``np.interp(tvd, vd_ray, md_ray)`` requires an increasing ``vd_ray``; 77 of
the 91 local surveys are toe-up, and on MPH-31 it returned 21,180 ft MD (the
last station) against a measured 5,144. These tests pin the two server call
sites and the optimizer seed path to the measured depth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from server.services import depth_interp, factories, optimizer_runs

SURVEYS = Path(__file__).resolve().parents[1] / "woffl" / "jp_data" / "well_surveys"
CHARS = Path(__file__).resolve().parents[1] / "woffl" / "jp_data" / "jp_chars.csv"

# (well, chars JP_TVD, chars JP_MD) - all three are toe-up laterals where
# np.interp lands at the toe.
TOE_UP = [("MPH-31", 3799.0, 5144.0), ("MPS-17", 4165.0, 4569.0), ("MPM-22", 3701.0, 4162.0)]


def _survey(well: str) -> pd.DataFrame:
    path = SURVEYS / f"{well} Deviation Survey.csv"
    if not path.exists():
        pytest.skip(f"no local survey for {well}")
    return pd.read_csv(path)


@pytest.fixture
def chars_frame(monkeypatch):
    """Serve the bundled jp_chars.csv as the chars frame (no Databricks)."""
    df = pd.read_csv(CHARS)
    monkeypatch.setattr(
        factories.datasources, "well_chars_safe", lambda: (df, "csv_fallback")
    )
    monkeypatch.setattr(
        factories.datasources,
        "survey",
        lambda well: _survey(well),
    )
    return df


@pytest.mark.parametrize("well,jp_tvd,jp_md", TOE_UP)
def test_first_crossing_matches_measured_md(well, jp_tvd, jp_md):
    df = _survey(well)
    md, vd = df["meas_depth"].to_numpy(float), df["tvd_depth"].to_numpy(float)
    assert np.any(np.diff(vd) < 0), "fixture must be toe-up for this test to mean anything"
    got = depth_interp.first_crossing_md(md, vd, jp_tvd)
    assert got is not None
    assert abs(got - jp_md) < 2.0
    # and the thing we replaced really was wrong
    assert abs(float(np.interp(jp_tvd, vd, md)) - jp_md) > 1000.0


def test_first_crossing_returns_none_when_never_reached():
    md = np.array([0.0, 1000.0, 2000.0])
    vd = np.array([0.0, 900.0, 1500.0])
    assert depth_interp.first_crossing_md(md, vd, 1600.0) is None


def test_first_crossing_hits_station_exactly():
    md = np.array([0.0, 1000.0, 2000.0])
    vd = np.array([0.0, 900.0, 1500.0])
    assert depth_interp.first_crossing_md(md, vd, 900.0) == 1000.0


@pytest.mark.parametrize("well,jp_tvd,jp_md", TOE_UP)
def test_resolver_prefers_measured_chars_md(chars_frame, well, jp_tvd, jp_md):
    df = _survey(well)
    got = factories.resolve_jetpump_md(
        well, jp_tvd, df["meas_depth"].tolist(), df["tvd_depth"].tolist()
    )
    assert got == pytest.approx(jp_md, abs=0.01)


def test_resolver_falls_back_to_crossing_on_tvd_override(chars_frame):
    """A sidebar TVD override (far from chars JP_TVD) must follow the survey,
    not the stale measured MD."""
    well, jp_tvd, jp_md = TOE_UP[0]
    df = _survey(well)
    override = jp_tvd - 300.0
    got = factories.resolve_jetpump_md(
        well, override, df["meas_depth"].tolist(), df["tvd_depth"].tolist()
    )
    assert got is not None
    assert got < jp_md  # shallower TVD -> shallower MD on the way down
    expected = depth_interp.first_crossing_md(
        df["meas_depth"].to_numpy(float), df["tvd_depth"].to_numpy(float), override
    )
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("well,jp_tvd,jp_md", TOE_UP)
def test_build_well_profile_places_pump_at_measured_md(chars_frame, well, jp_tvd, jp_md):
    from server.cache import clear_all_caches

    clear_all_caches()
    wp = factories.build_well_profile(well, jp_tvd, "Schrader")
    assert wp.jetpump_md == pytest.approx(jp_md, abs=0.01)
    assert wp.jetpump_md < float(wp.md_ray[-1]) - 1000.0  # nowhere near the toe


def test_config_from_seeds_carries_jpump_md():
    cfg = optimizer_runs._config_from_seeds(
        "MPH-31",
        "H",
        {"jpump_tvd": 3799.0, "jpump_md": 5144.0, "pres": 1500.0, "qwf": 800, "pwf": 600},
    )
    assert cfg.jpump_md == pytest.approx(5144.0)
    assert cfg.jpump_md != cfg.jpump_tvd


def test_config_from_seeds_without_md_keeps_legacy_default():
    cfg = optimizer_runs._config_from_seeds(
        "MPB-28", "B", {"jpump_tvd": 4254.0, "pres": 1500.0, "qwf": 800, "pwf": 600}
    )
    assert cfg.jpump_md == pytest.approx(4254.0)
