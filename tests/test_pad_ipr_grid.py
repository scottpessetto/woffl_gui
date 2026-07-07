"""Tests for the pad Results per-well IPR grid.

``build_ipr_grid_figure`` is the pure Plotly-figure builder behind
``pad_page._render_results``'s "Per-well IPR — current vs optimized"
expander — one subplot per well showing the SOLVER'S IPR (the Vogel curve
through the reviewed anchor — jetflow evaluates
``oil_flow(psu, method="vogel")``, restoring ``ee3886e`` after the woffl-2.0
sync had clobbered it back to the straight-line PI; see
``docs/upstream_sync.md`` #15), the CURRENT operating point (reviewed store
pwf/qwf), and the OPTIMIZED operating point (the optimizer's solved suction
pressure + formation liquid), connected by a dashed line, with a small
"12B -> 14C" pump-change annotation per panel.

The Vogel-not-PI choice is load-bearing: the jet-pump solver now genuinely
solves the IPR on Vogel, so drawing anything else (e.g. the straight-line
PI) would put the optimized point visibly off the curve.
``test_solver_consistent_optimized_point_sits_on_the_line`` is the
regression for that.
"""

import pytest

from woffl.assembly.network_optimizer import OptimizationResult
from woffl.gui.pad_helpers import build_ipr_grid_figure
from woffl.gui.vogel import vogel_fraction, vogel_qmax, vogel_rate


def _result(**overrides):
    defaults = dict(
        well_name="MPS-01",
        recommended_nozzle="14",
        recommended_throat="C",
        allocated_power_fluid=500,
        predicted_oil_rate=250.4,
        predicted_formation_water=100.0,
        predicted_lift_water=500.0,
        suction_pressure=1234.4,
        marginal_oil_rate=0.4,
        sonic_status=False,
        mach_te=0.6,
    )
    defaults.update(overrides)
    return OptimizationResult(**defaults)


def _entry(nozzle="12", throat="B", pwf=None, qwf=None, res_pres=None, **extra):
    e = {"review_nozzle": nozzle, "review_throat": throat}
    if pwf is not None:
        e["pwf"] = pwf
    if qwf is not None:
        e["qwf"] = qwf
    if res_pres is not None:
        e["res_pres"] = res_pres
    e.update(extra)
    return e


def _panel_titles(fig, wells):
    """Subplot-title annotations (exact well-name text) added by make_subplots."""
    return [a.text for a in fig.layout.annotations if a.text in wells]


def _pump_annotations(fig):
    return [a.text for a in fig.layout.annotations if "→" in (a.text or "")]


class TestBuildIprGridFigure:
    def test_returns_none_when_no_well_has_store_data(self):
        fig = build_ipr_grid_figure(
            [_result()],
            {"MPS-01": _entry()},  # no pwf/qwf/res_pres
            [],
        )
        assert fig is None

    def test_returns_none_for_empty_input(self):
        assert build_ipr_grid_figure([], {}, []) is None

    def test_one_panel_per_well(self):
        results = [
            _result(well_name="A", recommended_nozzle="14", recommended_throat="C"),
            _result(well_name="B", recommended_nozzle="13", recommended_throat="B"),
        ]
        active = {
            "A": _entry(pwf=1400, qwf=300, res_pres=1800),
            "B": _entry(pwf=1500, qwf=350, res_pres=2000),
            "C": _entry(pwf=1600, qwf=200, res_pres=1900),  # SI, has data
        }
        fig = build_ipr_grid_figure(results, active, ["C"])
        assert fig is not None
        assert sorted(_panel_titles(fig, ["A", "B", "C"])) == ["A", "B", "C"]

    def test_si_well_without_store_data_gets_no_panel(self):
        results = [_result(well_name="A")]
        active = {
            "A": _entry(pwf=1400, qwf=300, res_pres=1800),
            "D": _entry(),  # SI, no store data at all
        }
        fig = build_ipr_grid_figure(results, active, ["D"])
        assert fig is not None
        assert _panel_titles(fig, ["A", "D"]) == ["A"]

    def test_pwf_equals_res_pres_does_not_crash_still_gets_points(self):
        # Guarded: curve is skipped (pwf >= res_pres) but the current point
        # still renders — no ZeroDivisionError, no exception at all.
        active = {"MPS-09": _entry(pwf=1800.0, qwf=300.0, res_pres=1800.0)}
        fig = build_ipr_grid_figure([], active, ["MPS-09"])
        assert fig is not None
        marker_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(marker_traces) == 1
        assert marker_traces[0].x[0] == 300.0
        assert marker_traces[0].y[0] == 1800.0
        # No curve and no connector (no optimized point) -> zero line traces.
        line_traces = [t for t in fig.data if t.mode == "lines"]
        assert len(line_traces) == 0

    def test_si_well_has_no_optimized_trace_but_has_shut_in_annotation(self):
        active = {"MPS-09": _entry(pwf=1200, qwf=300, res_pres=1800)}
        fig = build_ipr_grid_figure([], active, ["MPS-09"])
        assert fig is not None
        assert not any(
            t.mode == "markers" and t.marker.symbol == "star" for t in fig.data
        )
        anns = _pump_annotations(fig)
        assert any(a.startswith("12B → SHUT IN") for a in anns)
        # SI well: optimized oil is 0 on the annotation's oil line.
        assert any("→ 0 bopd" in a for a in anns)

    def test_pump_change_annotation_text(self):
        results = [
            _result(well_name="MPS-01", recommended_nozzle="14", recommended_throat="C")
        ]
        active = {"MPS-01": _entry("12", "B", pwf=1400, qwf=300, res_pres=1800)}
        fig = build_ipr_grid_figure(results, active, [])
        assert fig is not None
        assert any(a.startswith("12B → 14C") for a in _pump_annotations(fig))

    def test_annotation_shows_oil_rates_current_to_optimized(self):
        # Current oil prefers the store's as-reviewed audit field; optimized
        # oil is the result's predicted rate. Both appear on line 2.
        r = _result(well_name="MPS-01", predicted_oil_rate=260.4)
        active = {
            "MPS-01": _entry(
                "12", "B", pwf=1400, qwf=300, res_pres=1800, qwf_oil_review=245.2
            )
        }
        fig = build_ipr_grid_figure([r], active, [])
        assert fig is not None
        assert any("245 → 260 bopd" in a for a in _pump_annotations(fig))

    def test_annotation_oil_falls_back_to_liquid_times_wc(self):
        # No qwf_oil_review -> oil = qwf * (1 - form_wc); missing both -> "—".
        r = _result(well_name="A", predicted_oil_rate=50.0)
        active = {
            "A": _entry(pwf=1400, qwf=300, res_pres=1800, form_wc=0.9),
            "B": _entry(pwf=1200, qwf=400, res_pres=1600),  # no oil info at all
        }
        fig = build_ipr_grid_figure([r], active, ["B"])
        assert fig is not None
        anns = _pump_annotations(fig)
        assert any("30 → 50 bopd" in a for a in anns)  # 300 * (1 - 0.9)
        assert any("— → 0 bopd" in a for a in anns)  # unknown current, SI

    def test_current_hover_includes_oil_when_known(self):
        active = {
            "A": _entry(pwf=1400, qwf=300, res_pres=1800, qwf_oil_review=245.0),
            "B": _entry(pwf=1200, qwf=400, res_pres=1600),  # oil unknown
        }
        fig = build_ipr_grid_figure([], active, ["A", "B"])
        assert fig is not None
        current = [t for t in fig.data if t.name == "Current"]
        assert len(current) == 2
        by_x = {t.x[0]: t.hovertemplate for t in current}
        assert "Oil 245 BOPD" in by_x[300]
        assert "Oil" not in by_x[400]

    def test_optimized_x_is_oil_plus_formation_water_not_total_liquid(self):
        r = _result(
            well_name="MPS-01",
            predicted_oil_rate=250.4,
            predicted_formation_water=100.0,
            predicted_lift_water=900.0,  # must NOT be included in x
            suction_pressure=1234.4,
        )
        active = {"MPS-01": _entry(pwf=1400, qwf=300, res_pres=1800)}
        fig = build_ipr_grid_figure([r], active, [])
        assert fig is not None
        (star,) = [t for t in fig.data if t.marker.symbol == "star"]
        assert star.x[0] == pytest.approx(350.4)
        assert star.y[0] == pytest.approx(1234.4)

    def test_connector_line_exists_between_current_and_optimized(self):
        r = _result(
            well_name="MPS-01",
            predicted_oil_rate=250.0,
            predicted_formation_water=100.0,
            suction_pressure=1234.0,
        )
        active = {"MPS-01": _entry(pwf=1400, qwf=300, res_pres=1800)}
        fig = build_ipr_grid_figure([r], active, [])
        assert fig is not None
        connectors = [
            t
            for t in fig.data
            if t.mode == "lines" and getattr(t.line, "dash", None) == "dash"
        ]
        assert len(connectors) == 1
        conn = connectors[0]
        assert conn.x[0] == 300 and conn.x[1] == pytest.approx(350.0)
        assert conn.y[0] == 1400 and conn.y[1] == pytest.approx(1234.0)

    def test_ipr_line_present_and_is_the_solver_vogel_curve(self):
        # The drawn IPR must be the Vogel curve the solver uses
        # (q = qmax * vogel_fraction(p, res)), hitting q=0 at res_pres and
        # passing through the anchor — NOT the straight-line PI chord.
        pwf, qwf, res_pres = 1400.0, 300.0, 1800.0
        active = {"MPS-01": _entry(pwf=pwf, qwf=qwf, res_pres=res_pres)}
        fig = build_ipr_grid_figure([], active, ["MPS-01"])
        assert fig is not None
        curve_traces = [
            t
            for t in fig.data
            if t.mode == "lines" and getattr(t.line, "dash", None) is None
        ]
        assert len(curve_traces) == 1
        curve = curve_traces[0]
        assert len(curve.x) == 40
        qmax = vogel_qmax(qwf, pwf, res_pres)
        for x, y in zip(curve.x, curve.y):
            assert x == pytest.approx(vogel_rate(y, qmax, res_pres))
        assert curve.x[0] == pytest.approx(0.0)  # zero rate at res_pres
        # Distinguish from the straight-line PI: Vogel at p = res/2 gives
        # qmax*0.7, the PI chord gives qwf*(res/2)/(res-pwf); for this anchor
        # they differ (582.5 vs 675.0).
        mid_x = vogel_rate(res_pres / 2, qmax, res_pres)
        assert mid_x == pytest.approx(582.53, abs=0.01)
        pi_mid_x = qwf * (res_pres / 2) / (res_pres - pwf)
        assert mid_x != pytest.approx(pi_mid_x)

    def test_solver_consistent_optimized_point_sits_on_the_line(self):
        # Regression: a result whose oil + formation water respect the
        # solver's Vogel relation must land exactly ON the drawn IPR curve.
        # With the straight-line PI drawn instead, this fails.
        pwf, qwf, res_pres, psu = 1400.0, 300.0, 1800.0, 1000.0
        qmax_liq = vogel_qmax(qwf, pwf, res_pres)
        liquid = qmax_liq * vogel_fraction(psu, res_pres)  # ~534.2
        r = _result(
            well_name="MPS-01",
            predicted_oil_rate=liquid * 0.1,
            predicted_formation_water=liquid * 0.9,
            suction_pressure=psu,
        )
        active = {"MPS-01": _entry(pwf=pwf, qwf=qwf, res_pres=res_pres)}
        fig = build_ipr_grid_figure([r], active, [])
        assert fig is not None
        (star,) = [t for t in fig.data if t.marker.symbol == "star"]
        # Same curve equation the IPR is drawn with, evaluated at star.y:
        expected_x = qmax_liq * vogel_fraction(star.y[0], res_pres)
        assert star.x[0] == pytest.approx(expected_x)

    def test_legend_shown_only_once_each(self):
        results = [
            _result(well_name="A"),
            _result(well_name="B", recommended_nozzle="13", recommended_throat="B"),
        ]
        active = {
            "A": _entry(pwf=1400, qwf=300, res_pres=1800),
            "B": _entry(pwf=1500, qwf=350, res_pres=2000),
        }
        fig = build_ipr_grid_figure(results, active, [])
        assert fig is not None
        current_legend = [
            t for t in fig.data if t.name == "Current" and t.showlegend is not False
        ]
        opt_legend = [
            t for t in fig.data if t.name == "Optimized" and t.showlegend is not False
        ]
        line_legend = [
            t for t in fig.data if t.name == "IPR (Vogel)" and t.showlegend is not False
        ]
        assert len(current_legend) == 1
        assert len(opt_legend) == 1
        assert len(line_legend) == 1
