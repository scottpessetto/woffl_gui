"""MD <-> TVD interpolation along a deviation survey (SPE 84246).

The contract this defends:

* Between stations the path is the minimum-curvature circular arc, not the
  chord. ``test_quarter_circle_*`` pins that against the closed-form arc, so a
  regression back to ``np.interp`` fails loudly.
* A lookup at a station returns that station's *recorded* TVD - the survey
  file is the number of record, not a re-integration of its inclinations.
* TVD is not single-valued in MD once a lateral builds past 90 deg; every
  crossing MD comes back, shallowest first.
* No angles on file -> straight-line chord, reported as such.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import server.services.depth_interp as depth_interp
from server.cache import clear_all_caches
from server.main import app
from server.services.depth_interp import Trajectory, depth_lookup

# --------------------------------------------------------------------------
# Analytic reference: a quarter-circle build in one vertical plane.
# A hole that builds 0 -> 90 deg on a constant-radius arc of radius R has
# TVD(theta) = R sin(theta) and MD(theta) = R theta, so every interpolated
# depth has a closed form to check against.
# --------------------------------------------------------------------------

_RADIUS = 1000.0  # ft


def _quarter_circle(n_stations: int = 5) -> Trajectory:
    theta = np.linspace(0.0, math.pi / 2, n_stations)
    return Trajectory(
        md_ray=_RADIUS * theta,
        vd_ray=_RADIUS * np.sin(theta),
        inc_ray=np.degrees(theta),
        azi_ray=np.zeros(n_stations),
    )


def test_quarter_circle_matches_the_closed_form_arc():
    """Minimum curvature is exact on a constant-curvature arc: interpolated
    TVD must equal R sin(MD/R) everywhere, not just at the stations."""
    traj = _quarter_circle()
    assert traj.method == "minimum_curvature"
    for md in np.linspace(1.0, _RADIUS * math.pi / 2 - 1.0, 97):
        expected = _RADIUS * math.sin(md / _RADIUS)
        assert traj.vd_at(float(md)) == pytest.approx(expected, abs=1e-6)


def test_quarter_circle_beats_the_chord_by_feet():
    """The chord (plain MD/TVD interpolation) reads shallow through a build.
    If this gap ever collapses, the arc math has been interpolated away."""
    traj = _quarter_circle()
    probe = np.linspace(1.0, _RADIUS * math.pi / 2 - 1.0, 400)
    arc = np.array([traj.vd_at(float(m)) for m in probe])
    chord = np.interp(probe, traj.md, traj.vd)
    assert (arc >= chord - 1e-9).all()  # the arc bulges above its own chord
    assert (arc - chord).max() > 5.0


def test_quarter_circle_inclination_and_dogleg():
    """Hole angle at a depth is the slerped tangent; DLS is the arc's own
    curvature, 180/(pi R) deg per radian of hole -> deg/100 ft."""
    traj = _quarter_circle()
    md = _RADIUS * math.pi / 4  # 45 deg into the build
    state = traj.state_at(float(md))
    assert state["inclination"] == pytest.approx(45.0, abs=1e-6)
    assert state["azimuth"] == pytest.approx(0.0, abs=1e-6)
    assert state["dls"] == pytest.approx(math.degrees(100.0 / _RADIUS), rel=1e-9)


def test_stations_return_their_recorded_tvd():
    """Vendor TVDs are the number of record. Even when they disagree with a
    re-integration of the inclinations, a station lookup returns the file."""
    theta = np.linspace(0.0, math.pi / 2, 6)
    recorded = _RADIUS * np.sin(theta) + 0.25  # 3-inch vendor offset
    recorded[0] = 0.0
    traj = Trajectory(
        md_ray=_RADIUS * theta,
        vd_ray=recorded,
        inc_ray=np.degrees(theta),
        azi_ray=np.zeros(theta.size),
    )
    for md, vd in zip(traj.md, traj.vd):
        assert traj.vd_at(float(md)) == pytest.approx(float(vd), abs=1e-9)


def test_no_angles_falls_back_to_the_chord():
    traj = Trajectory(md_ray=[0.0, 100.0, 300.0], vd_ray=[0.0, 100.0, 250.0])
    assert traj.method == "chord"
    assert traj.vd_at(200.0) == pytest.approx(175.0)
    state = traj.state_at(200.0)
    assert state["inclination"] == pytest.approx(math.degrees(math.acos(0.75)))
    assert state["azimuth"] is None
    assert state["dls"] is None


def test_partial_angle_columns_degrade_to_chord():
    """One NaN angle must not silently drop a station out of the MD/TVD table;
    the whole well degrades to the chord and says so."""
    traj = Trajectory(
        md_ray=[0.0, 100.0, 200.0],
        vd_ray=[0.0, 100.0, 195.0],
        inc_ray=[0.0, float("nan"), 20.0],
        azi_ray=[0.0, 0.0, 0.0],
    )
    assert traj.method == "chord"
    assert traj.md.size == 3


def test_unsorted_and_duplicate_stations_are_normalized():
    traj = Trajectory(
        md_ray=[100.0, 0.0, 100.0, 300.0],
        vd_ray=[100.0, 0.0, 100.0, 250.0],
    )
    assert traj.md.tolist() == [0.0, 100.0, 300.0]


def test_too_few_stations_raises():
    with pytest.raises(ValueError, match="two usable stations"):
        Trajectory(md_ray=[0.0], vd_ray=[0.0])


# --------------------------------------------------------------------------
# TVD is not single-valued: a toe-up lateral crosses the same TVD twice.
# --------------------------------------------------------------------------


def _toe_up() -> Trajectory:
    """Build to 100 deg (toe-up), so TVD peaks mid-lateral and comes back."""
    md = np.array([0.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
    inc = np.array([0.0, 30.0, 70.0, 95.0, 100.0, 100.0])
    azi = np.zeros(md.size)
    # TVD integrated by minimum curvature so the table is self-consistent.
    tvd = [0.0]
    for i in range(md.size - 1):
        i1, i2 = math.radians(inc[i]), math.radians(inc[i + 1])
        dl = abs(i2 - i1)
        rf = 1.0 if dl < 1e-12 else math.tan(dl / 2) / (dl / 2)
        tvd.append(tvd[-1] + (md[i + 1] - md[i]) / 2 * rf * (math.cos(i1) + math.cos(i2)))
    return Trajectory(md_ray=md, vd_ray=np.array(tvd), inc_ray=inc, azi_ray=azi)


def test_toe_up_tvd_has_two_measured_depths():
    traj = _toe_up()
    peak = float(traj.vd.max())
    solutions = traj.md_at_vd(peak - 5.0)
    assert len(solutions) == 2
    assert solutions[0] < solutions[1]
    for md in solutions:
        assert traj.vd_at(md) == pytest.approx(peak - 5.0, abs=1e-4)


def test_md_tvd_round_trip():
    traj = _toe_up()
    for md in (500.0, 2500.0, 3750.0):
        vd = traj.vd_at(md)
        assert traj.md_at_vd(vd)[0] == pytest.approx(md, abs=1e-4)


# --------------------------------------------------------------------------
# depth_lookup payload + the HTTP surface, against a stubbed survey.
# --------------------------------------------------------------------------


@pytest.fixture()
def stub_survey(monkeypatch):
    """MPB-TEST gets the toe-up survey; everything else has none."""
    traj = _toe_up()
    frame = pd.DataFrame(
        {
            "meas_depth": traj.md,
            "tvd_depth": traj.vd,
            "inclination": np.degrees(np.arccos(traj.tan_ray[:, 2])),
            "azimuth": np.zeros(traj.md.size),
        }
    )
    monkeypatch.setattr(
        depth_interp.datasources,
        "survey",
        lambda well: frame if well == "MPB-TEST" else None,
    )
    clear_all_caches()
    yield traj
    clear_all_caches()


def test_depth_lookup_md_payload(stub_survey):
    hit = depth_lookup("MPB-TEST", md=2500.0)
    assert hit["given"] == "md"
    assert hit["has_survey"] is True
    assert hit["method"] == "minimum_curvature"
    assert hit["md"] == 2500.0
    assert hit["tvd"] == pytest.approx(stub_survey.vd_at(2500.0))
    assert 30.0 < hit["inclination"] < 70.0
    assert hit["station_above"]["md"] == 2000.0
    assert hit["station_below"]["md"] == 3000.0
    assert hit["at_station"] is False
    assert hit["md_solutions"] == []
    assert hit["note"] is None


def test_depth_lookup_tvd_reports_every_crossing(stub_survey):
    peak = float(stub_survey.vd.max())
    hit = depth_lookup("MPB-TEST", tvd=peak - 5.0)
    assert hit["given"] == "tvd"
    assert len(hit["md_solutions"]) == 2
    assert hit["md"] == min(hit["md_solutions"])
    assert "crossed 2 times" in hit["note"]


def test_depth_lookup_rejects_bad_input(stub_survey):
    with pytest.raises(ValueError, match="exactly one"):
        depth_lookup("MPB-TEST")
    with pytest.raises(ValueError, match="exactly one"):
        depth_lookup("MPB-TEST", md=100.0, tvd=100.0)
    with pytest.raises(ValueError, match="off the survey"):
        depth_lookup("MPB-TEST", md=99_000.0)
    with pytest.raises(ValueError, match="off the survey"):
        depth_lookup("MPB-TEST", tvd=99_000.0)


def test_depth_lookup_without_survey_uses_the_preset(stub_survey):
    hit = depth_lookup("Custom", md=4065.0, field_model="Kuparuk")
    assert hit["has_survey"] is False
    assert hit["method"] == "chord"
    assert hit["tvd"] > 0.0
    assert hit["azimuth"] is None


def test_depth_endpoint(stub_survey):
    client = TestClient(app)

    ok = client.get("/api/wells/MPB-TEST/depth", params={"md": 2500})
    assert ok.status_code == 200
    body = ok.json()
    assert body["given"] == "md"
    assert body["tvd"] == pytest.approx(stub_survey.vd_at(2500.0))

    back = client.get("/api/wells/MPB-TEST/depth", params={"tvd": body["tvd"]})
    assert back.status_code == 200
    assert back.json()["md"] == pytest.approx(2500.0, abs=1e-3)

    bad = client.get("/api/wells/MPB-TEST/depth", params={"md": 2500, "tvd": 1000})
    assert bad.status_code == 400
    assert "exactly one" in bad.json()["detail"]["message"]

    none = client.get("/api/wells/MPB-TEST/depth")
    assert none.status_code == 400


# --------------------------------------------------------------------------
# The real vendor files: their TVD column IS minimum curvature. If our arc
# disagreed with it, every in-between answer would be wrong too.
# --------------------------------------------------------------------------


def test_real_survey_reproduces_its_own_tvd_column():
    clear_all_caches()
    traj = depth_interp.trajectory("MPB-28", "Schrader")
    if not traj.has_survey:  # survey CSVs are packaged; skip if trimmed out
        pytest.skip("MPB-28 deviation survey not present")
    assert traj.method == "minimum_curvature"
    assert traj.resid is not None
    # resid is (recorded dTVD) - (arc dTVD) per segment: sub-millifoot.
    assert float(np.abs(traj.resid).max()) < 1e-3
