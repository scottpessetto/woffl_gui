"""Smoke test for the P1-15 fix in woffl.gui.tabs.well_profile.

P1-15: the "Horizontal Deviation" plot (and the "Max Deviation" summary
figure) computed ``md_ray[i] - vd_ray[i]``, which is NOT the true horizontal
departure — it's only a valid proxy while inclination is near zero. On a
deviated well this understates (or otherwise distorts) the real horizontal
offset by a wide margin vs. the profile's own Pythagorean ``hd_ray``
(``woffl/geometry/wellprofile.py``'s ``_horz_dist``, cumulative
``sqrt(md_diff**2 - vd_diff**2)``).

This drives the private ``_render_trajectory_plots`` function directly
(Streamlit calls no-op safely outside a running app — they just warn about
a missing ScriptRunContext) and inspects the actual Plotly trace data fed
to the "Deviation" chart, rather than re-deriving the fix's arithmetic.
"""

import numpy as np

from woffl.geometry.wellprofile import WellProfile
from woffl.gui.tabs.well_profile import _render_trajectory_plots


def _deviated_profile() -> WellProfile:
    # A well that kicks off around 3000 ft MD and builds to a significant
    # horizontal departure — exactly the case where md - vd diverges hard
    # from the true (Pythagorean) horizontal distance.
    md = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    vd = [0, 1000, 2000, 2900, 3600, 4100, 4500, 4800, 5000]
    return WellProfile(md, vd, jetpump_md=4000.0)


def test_deviation_plot_uses_hd_ray_not_md_minus_vd():
    wp = _deviated_profile()
    captured = {}

    import plotly.graph_objects as go

    orig_add_trace = go.Figure.add_trace

    def spy(self, trace, *a, **k):
        if getattr(trace, "name", None) == "Deviation":
            captured["x"] = np.asarray(trace.x, dtype=float)
        return orig_add_trace(self, trace, *a, **k)

    go.Figure.add_trace = spy
    try:
        _render_trajectory_plots(wp, jpump_tvd=3600)
    finally:
        go.Figure.add_trace = orig_add_trace

    assert "x" in captured, "Deviation trace was never added to the figure"

    md_ray = np.asarray(wp.md_ray, dtype=float)
    vd_ray = np.asarray(wp.vd_ray, dtype=float)
    naive_deviation = md_ray - vd_ray

    # The fixed plot must match the profile's own hd_ray exactly...
    assert np.allclose(captured["x"], wp.hd_ray)
    # ...and must NOT match the old buggy md-vd proxy, which diverges
    # substantially (>60%) at depth on this deviated survey.
    assert not np.allclose(captured["x"], naive_deviation)
    assert abs(wp.hd_ray[-1] - naive_deviation[-1]) / naive_deviation[-1] > 0.5
