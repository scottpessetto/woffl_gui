"""MD <-> TVD lookup along a well's deviation survey.

SPE best practice for position *between* survey stations is the
**minimum-curvature method** (Sawaryn & Thorogood, SPE 84246, "A Compendium of
Directional Calculations Based on the Minimum Curvature Method"): the hole
between two stations is a circular arc lying in the plane spanned by the two
station tangent vectors, so the unit tangent rotates uniformly with measured
depth along that arc. Straight MD/TVD interpolation - what
``WellProfile._depth_interp`` does for the solver - is the *chord* of that arc
and reads shallow through a build section, by enough to matter when an
engineer is picking a landing depth off a survey.

Two implementation choices worth keeping:

* Station TVDs come from the survey file; they are not re-integrated from the
  inclinations. The recorded TVD is the number of record and re-integration
  drifts from it. The arc supplies the *shape* between stations and a
  per-segment linear residual pins both ends back onto the recorded values, so
  a lookup exactly at a station returns that station's TVD to the foot.
* TVD is not monotonic in MD once a well builds past 90 deg. A TVD lookup
  therefore returns *every* MD that reaches it, shallowest first, and the
  payload says so.

Wells with no survey CSV fall back to the field-model preset trajectory, which
carries MD/TVD only - no inclination, no azimuth - so those lookups report
``method="chord"``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
from scipy import optimize

from server import config
from server.cache import ttl_cache
from server.services import datasources

log = logging.getLogger("woffl.web.depth_interp")

_DEG = math.pi / 180.0
# Probe points per survey segment used to bracket TVD roots. A segment is a
# single circular arc, so its TVD has at most one turning point; 32 intervals
# resolve both branches of that turn far below survey precision.
_PROBES = 32
_TOL_FT = 1e-6
_MIN_DL = 1e-9  # below this the arc is a straight tangent section


class Trajectory:
    """Survey stations plus the minimum-curvature arc between them.

    Args:
        md_ray (np array): Station measured depths, feet
        vd_ray (np array): Station true vertical depths, feet
        inc_ray (np array): Station inclinations, degrees from vertical (None
            when the survey carries no angles)
        azi_ray (np array): Station azimuths, degrees (None with inc_ray)
        has_survey (bool): True when built from the well's own survey CSV
    """

    def __init__(
        self,
        md_ray: np.ndarray,
        vd_ray: np.ndarray,
        inc_ray: Optional[np.ndarray] = None,
        azi_ray: Optional[np.ndarray] = None,
        has_survey: bool = True,
    ) -> None:
        md = np.asarray(md_ray, dtype=float)
        vd = np.asarray(vd_ray, dtype=float)
        if md.size != vd.size:
            raise ValueError("survey MD and TVD columns differ in length")

        keep = np.isfinite(md) & np.isfinite(vd)
        md, vd = md[keep], vd[keep]
        order = np.argsort(md, kind="stable")
        md, vd = md[order], vd[order]
        # searchsorted and every fraction below assume strictly increasing MD.
        uniq = np.concatenate(([True], np.diff(md) > _TOL_FT))
        md, vd = md[uniq], vd[uniq]
        if md.size < 2:
            raise ValueError("survey has fewer than two usable stations")

        inc = azi = None
        if inc_ray is not None and azi_ray is not None:
            i_all = np.asarray(inc_ray, dtype=float)
            a_all = np.asarray(azi_ray, dtype=float)
            if i_all.size == keep.size and a_all.size == keep.size:
                i_all = i_all[keep][order][uniq]
                a_all = a_all[keep][order][uniq]
                # One bad angle row degrades the whole well to chord rather
                # than dropping a station out of the MD/TVD table.
                if np.isfinite(i_all).all() and np.isfinite(a_all).all():
                    inc, azi = i_all, a_all

        self.md = md
        self.vd = vd
        self.has_survey = has_survey
        self.method = "minimum_curvature" if inc is not None else "chord"

        self.dmd = np.diff(md)
        self.dvd = np.diff(vd)

        if inc is None:
            self.tan_ray: Optional[np.ndarray] = None
            self.dl: Optional[np.ndarray] = None
            self.resid: Optional[np.ndarray] = None
        else:
            i_r = inc * _DEG
            a_r = azi * _DEG  # type: ignore[operator]
            sin_i = np.sin(i_r)
            # (north, east, down) unit tangents; down is +TVD.
            self.tan_ray = np.column_stack(
                (sin_i * np.cos(a_r), sin_i * np.sin(a_r), np.cos(i_r))
            )
            dot = np.einsum("ij,ij->i", self.tan_ray[:-1], self.tan_ray[1:])
            self.dl = np.arccos(np.clip(dot, -1.0, 1.0))
            seg = np.arange(self.dmd.size)
            # Pin each arc back onto the survey's recorded station TVDs.
            self.resid = self.dvd - self._arc_dvd(seg, np.ones(seg.size))

        # Flat probe polyline over the whole hole, used to bracket TVD roots.
        probes = np.linspace(0.0, 1.0, _PROBES + 1)
        seg = np.arange(self.dmd.size)
        p_seg = np.repeat(seg, probes.size)
        p_frac = np.tile(probes, seg.size)
        self._probe_md = self.md[p_seg] + p_frac * self.dmd[p_seg]
        self._probe_vd = self.vd_at_frac(p_seg, p_frac)

    # -- arc math ----------------------------------------------------------

    @staticmethod
    def _ratio_factor(dogleg: np.ndarray) -> np.ndarray:
        """Minimum-curvature ratio factor tan(dl/2)/(dl/2); 1 in the limit."""
        half = dogleg / 2.0
        safe = np.where(half < _MIN_DL, 1.0, half)
        return np.where(half < _MIN_DL, 1.0, np.tan(safe) / safe)

    def _tangent(self, seg: np.ndarray, frac: np.ndarray) -> np.ndarray:
        """Unit tangent(s) a fraction into a segment (slerp along the arc).

        Args:
            seg (np array): Segment indices
            frac (np array): Fraction of the segment, 0-1

        Returns:
            tan_ray (np array): (n, 3) unit tangents, (north, east, down)
        """
        assert self.tan_ray is not None and self.dl is not None
        dogleg = self.dl[seg]
        straight = dogleg < _MIN_DL
        sin_dl = np.where(straight, 1.0, np.sin(dogleg))
        w1 = np.where(straight, 1.0 - frac, np.sin((1.0 - frac) * dogleg) / sin_dl)
        w2 = np.where(straight, frac, np.sin(frac * dogleg) / sin_dl)
        return w1[:, None] * self.tan_ray[seg] + w2[:, None] * self.tan_ray[seg + 1]

    def _arc_dvd(self, seg: np.ndarray, frac: np.ndarray) -> np.ndarray:
        """TVD gained from a station to a fraction into its segment, feet."""
        assert self.dl is not None and self.tan_ray is not None
        tan_z = self._tangent(seg, frac)[:, 2]
        ratio = self._ratio_factor(self.dl[seg] * frac)
        return 0.5 * frac * self.dmd[seg] * ratio * (self.tan_ray[seg][:, 2] + tan_z)

    def vd_at_frac(self, seg: np.ndarray, frac: np.ndarray) -> np.ndarray:
        """True vertical depth a fraction into a segment, feet.

        The linear ``resid`` term is what pins both ends of every segment onto
        the survey's own recorded TVDs.
        """
        if self.tan_ray is None:
            return self.vd[seg] + frac * self.dvd[seg]
        assert self.resid is not None
        return self.vd[seg] + self._arc_dvd(seg, frac) + frac * self.resid[seg]

    # -- lookups -----------------------------------------------------------

    def _locate(self, md: float) -> tuple[int, float]:
        """(segment index, fraction 0-1) holding a measured depth."""
        idx = int(np.searchsorted(self.md, md, side="right")) - 1
        idx = min(max(idx, 0), self.md.size - 2)
        frac = (md - self.md[idx]) / self.dmd[idx]
        return idx, float(min(max(frac, 0.0), 1.0))

    def vd_at(self, md: float) -> float:
        """True vertical depth at a measured depth, feet."""
        seg, frac = self._locate(md)
        return float(self.vd_at_frac(np.array([seg]), np.array([frac]))[0])

    def state_at(self, md: float) -> dict[str, Any]:
        """TVD, hole angle, and bracketing stations at a measured depth.

        Args:
            md (float): Measured depth, feet

        Returns:
            state (dict): vd (ft), inclination (deg), azimuth (deg), dls
                (deg/100 ft), at_station (bool), station_above/below (dict)
        """
        seg, frac = self._locate(md)
        seg_ray = np.array([seg])
        frac_ray = np.array([frac])
        vd = float(self.vd_at_frac(seg_ray, frac_ray)[0])

        azimuth: Optional[float] = None
        dls: Optional[float] = None
        if self.tan_ray is None:
            # No angles on file: the chord's own slope is the best available
            # hole angle, and a chord has no dogleg to report.
            cos_i = min(max(self.dvd[seg] / self.dmd[seg], -1.0), 1.0)
            inclination = math.degrees(math.acos(cos_i))
        else:
            assert self.dl is not None
            tan_v = self._tangent(seg_ray, frac_ray)[0]
            inclination = math.degrees(math.acos(min(max(float(tan_v[2]), -1.0), 1.0)))
            azimuth = math.degrees(math.atan2(float(tan_v[1]), float(tan_v[0]))) % 360.0
            dls = math.degrees(self.dl[seg]) / self.dmd[seg] * 100.0

        above, below = seg, seg + 1
        at_station = bool(frac <= _TOL_FT or frac >= 1.0 - _TOL_FT)
        return {
            "vd": vd,
            "inclination": inclination,
            "azimuth": azimuth,
            "dls": dls,
            "at_station": at_station,
            "station_above": {
                "md": float(self.md[above]),
                "tvd": float(self.vd[above]),
            },
            "station_below": {
                "md": float(self.md[below]),
                "tvd": float(self.vd[below]),
            },
        }

    def md_at_vd(self, vd_target: float) -> list[float]:
        """Every measured depth that reaches a true vertical depth, feet.

        More than one MD is the normal answer in a horizontal well: a toe-up
        lateral crosses the same TVD on the way down, along the tangent, and
        again on the way back up.

        Args:
            vd_target (float): True vertical depth, feet

        Returns:
            md_list (list): Measured depths, shallowest first, feet
        """
        gap = self._probe_vd - vd_target
        roots = [float(m) for m in self._probe_md[np.abs(gap) <= _TOL_FT]]

        low, high = gap[:-1], gap[1:]
        crossings = np.flatnonzero(
            ((low > _TOL_FT) & (high < -_TOL_FT))
            | ((low < -_TOL_FT) & (high > _TOL_FT))
        )
        for k in crossings:
            md_lo = float(self._probe_md[k])
            md_hi = float(self._probe_md[k + 1])
            if md_hi <= md_lo:  # segment seam: the two probes are one point
                continue
            roots.append(
                float(
                    optimize.brentq(
                        lambda m: self.vd_at(m) - vd_target,
                        md_lo,
                        md_hi,
                        xtol=1e-8,
                    )
                )
            )

        roots.sort()
        unique: list[float] = []
        for root in roots:
            if not unique or root - unique[-1] > 1e-3:
                unique.append(root)
        return unique


@ttl_cache(config.TTL_PROFILES, maxsize=256)
def trajectory(well: str, field_model: str) -> Trajectory:
    """The well's survey trajectory, or the field-model preset when it has none.

    Args:
        well (str): GUI well name, e.g. "MPB-28"
        field_model (str): "Schrader" | "Kuparuk" preset for the fallback

    Returns:
        traj (Trajectory): Ready-to-query trajectory
    """
    survey_df = datasources.survey(well)
    if survey_df is not None and not survey_df.empty:
        cols = set(survey_df.columns)
        if {"meas_depth", "tvd_depth"} <= cols:
            try:
                inc = azi = None
                if {"inclination", "azimuth"} <= cols:
                    inc = survey_df["inclination"].to_numpy(dtype=float)
                    azi = survey_df["azimuth"].to_numpy(dtype=float)
                return Trajectory(
                    survey_df["meas_depth"].to_numpy(dtype=float),
                    survey_df["tvd_depth"].to_numpy(dtype=float),
                    inc,
                    azi,
                    has_survey=True,
                )
            except (ValueError, TypeError) as exc:
                log.warning("Unusable survey for %s: %s", well, exc)

    from woffl.geometry.wellprofile import WellProfile

    preset = (field_model or "Schrader").lower()
    well_prof = WellProfile.kuparuk() if preset == "kuparuk" else WellProfile.schrader()
    return Trajectory(well_prof.md_ray, well_prof.vd_ray, has_survey=False)


def depth_lookup(
    well: str,
    md: Optional[float] = None,
    tvd: Optional[float] = None,
    field_model: Optional[str] = None,
) -> dict[str, Any]:
    """Convert one depth to the other along the well's survey.

    Args:
        well (str): GUI well name, e.g. "MPB-28"
        md (float): Measured depth to convert, feet (exclusive with tvd)
        tvd (float): True vertical depth to convert, feet (exclusive with md)
        field_model (str): Preset used when the well has no survey

    Returns:
        payload (dict): Dict matching schemas.DepthLookupResponse

    Raises:
        ValueError: Neither or both depths given, or the depth is off survey.
    """
    if (md is None) == (tvd is None):
        raise ValueError("give exactly one of md or tvd")

    traj = trajectory(well, field_model or "Schrader")
    md_lo, md_hi = float(traj.md[0]), float(traj.md[-1])
    vd_lo, vd_hi = float(traj.vd.min()), float(traj.vd.max())

    note: Optional[str] = None
    solutions: list[float] = []

    if md is not None:
        if not md_lo - _TOL_FT <= md <= md_hi + _TOL_FT:
            raise ValueError(
                f"{md:,.1f} ft MD is off the survey "
                f"({md_lo:,.0f} to {md_hi:,.0f} ft MD)"
            )
        md_out = min(max(float(md), md_lo), md_hi)
        given = "md"
    else:
        target = float(tvd)  # type: ignore[arg-type]
        if not vd_lo - _TOL_FT <= target <= vd_hi + _TOL_FT:
            raise ValueError(
                f"{target:,.1f} ft TVD is off the survey "
                f"({vd_lo:,.0f} to {vd_hi:,.0f} ft TVD)"
            )
        solutions = traj.md_at_vd(target)
        if not solutions:
            raise ValueError(f"{target:,.1f} ft TVD is never reached by this survey")
        md_out = solutions[0]
        if len(solutions) > 1:
            note = (
                f"{target:,.1f} ft TVD is crossed {len(solutions)} times - "
                "the shallowest MD is shown."
            )
        given = "tvd"

    state = traj.state_at(md_out)
    return {
        "well": well,
        "has_survey": traj.has_survey,
        "method": traj.method,
        "given": given,
        "md": md_out,
        "tvd": state["vd"],
        "inclination": state["inclination"],
        "azimuth": state["azimuth"],
        "dls": state["dls"],
        "md_solutions": solutions,
        "at_station": state["at_station"],
        "station_above": state["station_above"],
        "station_below": state["station_below"],
        "station_count": int(traj.md.size),
        "md_range": [md_lo, md_hi],
        "tvd_range": [vd_lo, vd_hi],
        "note": note,
    }
