"""WellProfile survey validation, raw-survey traverse and depth lookups.

Guards docs/upstream_sync.md #23 (FLOW-2), #24 (FLOW-3) and #25 (FLOW-10),
review 2026-09-01. Every test here goes red if its patch is lost on an
upstream sync.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from woffl.geometry import wellprofile as wpmod
from woffl.geometry.wellprofile import WellProfile, first_crossing, validate_survey

ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = ROOT / "woffl" / "jp_data" / "well_surveys"

# Local surveys the validator is expected to reject. EMPTY since 2026-09-02:
# the five that raised (MPC-05, MPC-40, MPI-11, MPL-20, MPM-64) were not bad
# wells but a bad PULL - deviation_survey_pdb.sql read the PDB *history* view
# without its PREFERRED_FLAG filter and stacked every survey version into one
# CSV (MPC-05: 12 versions, one with TVD 0). The query now takes the preferred
# survey only and all 91 CSVs were re-pulled. The stacked MPC-05 file is kept
# under tests/fixtures/ so the rejection path stays exercised.
CORRUPT_SURVEYS: set[str] = set()
CORRUPT_FIXTURE = ROOT / "tests" / "fixtures" / "corrupt_survey_MPC-05_stacked_versions.csv"


def _load_csv(path: Path) -> tuple[list[float], list[float]]:
    df = pd.read_csv(path)
    md = df["meas_depth"].to_numpy(dtype=float)
    vd = df["tvd_depth"].to_numpy(dtype=float)
    keep = np.isfinite(md) & np.isfinite(vd)
    return md[keep].tolist(), vd[keep].tolist()


def _load(well: str) -> tuple[list[float], list[float]]:
    return _load_csv(SURVEY_DIR / f"{well} Deviation Survey.csv")


def _all_surveys() -> list[str]:
    return sorted(
        p.name.replace(" Deviation Survey.csv", "") for p in SURVEY_DIR.glob("*.csv")
    )


# ------------------------------------------------------------------
# FLOW-3: corrupt surveys are rejected, merged-run surveys still construct
# ------------------------------------------------------------------


def test_stacked_versions_survey_raises_naming_station():
    """The pre-fix MPC-05 pull (every survey version stacked) must be
    rejected with the offending station named."""
    md, vd = _load_csv(CORRUPT_FIXTURE)
    with pytest.raises(
        ValueError, match=r"conflicting vertical depths at 6090\.00 ft MD"
    ):
        WellProfile(md, vd, jetpump_md=0.85 * max(md))


def test_repulled_mpc05_constructs():
    """The preferred-survey pull of the same well is clean."""
    md, vd = _load("MPC-05")
    wp = WellProfile(md, vd, jetpump_md=0.85 * max(md))
    assert wp.jetpump_vd > 0


def test_mph31_toe_up_survey_constructs():
    """MPH-31 builds past horizontal (toe-up: TVD max above the toe TVD).
    The preferred-survey pull has no duplicated MDs; the measured pump MD
    (5,144 ft) sits within the resolver's 5 ft window of chars JP_TVD 3,799,
    which was interpolated from the pre-fix stacked file."""
    md, vd = _load("MPH-31")
    wp = WellProfile(md, vd, jetpump_md=5144)
    assert np.all(np.diff(wp.md_ray) > 0)  # strictly increasing MD
    assert wp.jetpump_vd == pytest.approx(3799.0, abs=5.0)
    assert wp.vd_ray[-1] < wp.vd_ray.max()  # genuinely toe-up


def test_merged_run_duplicates_within_tolerance_coalesce():
    """Two survey runs that agree at a shared MD to within the tolerance
    coalesce to one station (the MPH-31 case before the preferred pull)."""
    md = [0, 1000, 1000, 2000, 3000]
    vd = [0, 990.0, 993.0, 1950.0, 2800.0]  # 3 ft disagreement < 5 ft tol
    wp = WellProfile(md, vd, jetpump_md=2500)
    assert wp.md_ray.tolist() == [0, 1000, 2000, 3000]
    assert np.all(np.diff(wp.md_ray) > 0)


def test_exact_duplicate_rows_are_dropped_silently():
    md = [0, 100, 100, 200, 300]
    vd = [0, 99, 99, 195, 280]
    wp = WellProfile(md, vd, jetpump_md=250)
    assert wp.md_ray.tolist() == [0, 100, 200, 300]
    assert wp.vd_ray.tolist() == [0, 99, 195, 280]


def test_duplicate_md_within_tolerance_keeps_first_seen():
    md = [0, 100, 100, 200, 300]
    vd = [0, 99.0, 101.0, 195, 280]  # 2 ft disagreement < SURVEY_STEP_TOL_FT
    wp = WellProfile(md, vd, jetpump_md=250)
    assert wp.vd_ray.tolist() == [0, 99.0, 195, 280]


def test_duplicate_md_conflicting_tvd_raises():
    md = [0, 100, 200, 200, 300]
    vd = [0, 99, 195, 150, 280]  # 45 ft disagreement at the same MD
    with pytest.raises(
        ValueError, match=r"conflicting vertical depths at 200\.00 ft MD"
    ):
        WellProfile(md, vd, jetpump_md=250)


def test_impossible_step_raises_naming_stations():
    md = [0, 100, 200, 300]
    vd = [0, 99, 210, 280]  # +111 ft TVD over 100 ft MD
    with pytest.raises(ValueError, match=r"between 100\.00 and 200\.00 ft MD"):
        WellProfile(md, vd, jetpump_md=250)


def test_sub_tolerance_noise_still_constructs():
    excess = wpmod.SURVEY_STEP_TOL_FT - 0.1
    md = [0, 100, 200, 300]
    vd = [0, 100 + excess, 195, 280]
    wp = WellProfile(md, vd, jetpump_md=250)
    assert np.isfinite(wp.hd_ray).all()


def test_validate_survey_is_identity_on_a_clean_survey():
    md = np.array([0.0, 100.0, 200.0, 300.0])
    vd = np.array([0.0, 99.0, 195.0, 280.0])
    md2, vd2 = validate_survey(md, vd)
    assert np.array_equal(md, md2) and np.array_equal(vd, vd2)


def test_fleet_only_the_known_corrupt_surveys_raise():
    """Every other local survey must still construct; a new raise means a
    new corrupt CSV (or a tolerance that drifted)."""
    raised = set()
    for well in _all_surveys():
        md, vd = _load(well)
        try:
            WellProfile(md, vd, jetpump_md=0.85 * max(md))
        except ValueError:
            raised.add(well)
    assert raised == CORRUPT_SURVEYS


# ------------------------------------------------------------------
# FLOW-2: the traverse reads the raw survey, the fit is plot-only
# ------------------------------------------------------------------


def test_traverse_pump_tvd_equals_raw_survey_tvd_fleetwide():
    """Before the fix the fitted profile put the traverse's pump TVD off the
    raw survey (fleet median 14 ft, p90 111 ft, max 240 ft on MPE-48) while
    the power-fluid side used the raw ``jetpump_vd``."""
    worst = 0.0
    for well in _all_surveys():
        if well in CORRUPT_SURVEYS:
            continue
        md, vd = _load(well)
        jp = 0.85 * max(md)
        wp = WellProfile(md, vd, jetpump_md=jp)
        md_seg, vd_seg = wp.outflow_spacing(100)
        assert md_seg[-1] == jp
        raw = float(np.interp(jp, wp.md_ray, wp.vd_ray))
        worst = max(worst, abs(vd_seg[-1] - raw), abs(wp.jetpump_vd - raw))
    assert worst < 0.1


def test_outflow_spacing_follows_survey_kinks_not_fit():
    """A sharp build the greedy fit would round off must show up in vd_seg."""
    md = [0, 1000, 2000, 2500, 3000, 4000, 5000]
    vd = [0, 1000, 2000, 2400, 2600, 2700, 2750]
    wp = WellProfile(md, vd, jetpump_md=4500)
    md_seg, vd_seg = wp.outflow_spacing(100)
    assert np.allclose(vd_seg, np.interp(md_seg, wp.md_ray, wp.vd_ray))
    assert vd_seg[-1] == pytest.approx(2725.0)


def test_outflow_spacing_node_rule_unchanged():
    """Evenly spaced surface -> pump with the pre-existing count rule, so a
    straight-line profile (whose fit was one segment) is bit-identical."""
    wp = WellProfile(np.linspace(0, 6000, 100), np.linspace(0, 4000, 100), 6000)
    md_seg, vd_seg = wp.outflow_spacing(100)
    assert len(md_seg) == 60
    assert np.array_equal(md_seg, np.linspace(0, 6000, 60))
    assert np.allclose(vd_seg, md_seg * 4000 / 6000)


def test_segments_fit_not_run_by_construction_or_traverse():
    """Tripwire: the Nelder-Mead fit must stay off the physics path."""
    wp = WellProfile.schrader()
    wp.outflow_spacing(100)
    _ = wp.jetpump_vd
    assert wp._fit_cache is None
    # and it still works lazily for plotting
    assert len(wp.md_fit) == len(wp.vd_fit) == len(wp.hd_fit)
    assert wp.md_fit[0] == 0 and wp.vd_fit[0] == 0
    assert wp._fit_cache is not None


def test_schrader_preset_traverse_uses_raw_pump_tvd():
    wp = WellProfile.schrader()
    _, vd_seg = wp.outflow_spacing(100)
    assert vd_seg[-1] == pytest.approx(wp.jetpump_vd, abs=1e-9)
    assert vd_seg[-1] == pytest.approx(4096.77, abs=0.05)  # the fit said 4103.4


_LAZY_PROBE = r"""
import sys
from woffl.geometry.wellprofile import WellProfile
wp = WellProfile.schrader()
wp.outflow_spacing(100); wp.jetpump_vd
print("OPT=" + str("scipy.optimize" in sys.modules))
wp.md_fit
print("OPT2=" + str("scipy.optimize" in sys.modules))
"""


def test_wellprofile_imports_scipy_optimize_lazily():
    """Construction + traverse must not import scipy.optimize; the fit may."""
    out = subprocess.run(
        [sys.executable, "-c", _LAZY_PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "OPT=False" in out.stdout, out.stdout
    assert "OPT2=True" in out.stdout, out.stdout


# ------------------------------------------------------------------
# FLOW-10: range check that actually raises; shallowest-crossing md_interp
# ------------------------------------------------------------------

MD = [0, 100, 200, 300, 400, 500]
VD = [0, 95, 180, 270, 350, 375]


def test_vd_interp_out_of_range_raises():
    wp = WellProfile(MD, VD, jetpump_md=450)
    with pytest.raises(ValueError, match="not inside survey boundary"):
        wp.vd_interp(5500)  # used to return the clamped 375
    with pytest.raises(ValueError, match="not inside survey boundary"):
        wp.vd_interp(-1)
    with pytest.raises(ValueError, match="not inside survey boundary"):
        wp.hd_interp(501)
    with pytest.raises(ValueError, match="not inside survey boundary"):
        wp.md_interp(376)


def test_interp_inclusive_at_both_ends():
    wp = WellProfile(MD, VD, jetpump_md=500)
    assert wp.vd_interp(0) == 0
    assert wp.vd_interp(500) == 375
    assert wp.jetpump_vd == 375  # pump AT total depth used to raise
    assert wp.md_interp(375) == 500
    assert wp.md_interp(0) == 0


def test_md_interp_toe_up_takes_shallowest_crossing():
    """TVD 2800 is reached twice on a toe-up survey; np.interp on the
    non-monotonic vd_ray returned the toe (as it did for MPH-31)."""
    md = [0, 1000, 2000, 3000, 4000, 5000]
    vd = [0, 900, 1800, 2700, 2900, 2600]  # builds past horizontal at the toe
    wp = WellProfile(md, vd, jetpump_md=4500)
    assert wp.md_interp(2800) == pytest.approx(3500.0, abs=0.01)  # not 4333 at the toe
    assert wp.md_interp(2700) == 3000.0  # exact station hit
    assert np.interp(2800, wp.vd_ray, wp.md_ray) != pytest.approx(3500.0, abs=1)


def test_first_crossing_helper():
    x = np.array([0.0, 10.0, 20.0, 15.0])
    y = np.array([0.0, 100.0, 200.0, 300.0])
    assert first_crossing(x, y, 5.0) == 50.0
    assert first_crossing(x, y, 17.0) == 170.0  # first pass, not the fold-back
    assert first_crossing(x, y, 20.0) == 200.0
    assert first_crossing(x, y, 25.0) is None
    assert first_crossing(np.array([]), np.array([]), 1.0) is None


def test_mph31_md_interp_matches_measured_pump_depth():
    md, vd = _load("MPH-31")
    wp = WellProfile(md, vd, jetpump_md=5144)
    # the shallowest crossing of the pump's own TVD is the pump's MD, not
    # the 21,180 ft toe that np.interp on the folded vd_ray returns
    assert wp.md_interp(wp.jetpump_vd) == pytest.approx(5144.0, abs=0.01)
    assert np.interp(wp.jetpump_vd, wp.vd_ray, wp.md_ray) > 20000
