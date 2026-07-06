"""Tests for the pad Results "current vs optimized" comparison table.

``build_comparison_rows`` is the pure row-builder behind
``pad_page._render_results`` — the default results view that shows each
well's CURRENT pump + measured test rates beside the optimizer's pick, so an
engineer can tell at a glance what the plan changes vs today (the confusion
the optimized-only table caused). Also covers the pad-hub spec registry.
"""

import pytest

from woffl.assembly.network_optimizer import OptimizationResult
from woffl.gui.pad_helpers import build_comparison_rows, pump_size_change


def _result(**overrides):
    defaults = dict(
        well_name="MPS-01",
        recommended_nozzle="13",
        recommended_throat="C",
        allocated_power_fluid=500,
        predicted_oil_rate=250.4,
        predicted_formation_water=100,
        predicted_lift_water=500,
        suction_pressure=1100,
        marginal_oil_rate=0.4,
        sonic_status=False,
        mach_te=0.6,
    )
    defaults.update(overrides)
    return OptimizationResult(**defaults)


def _entry(nozzle="12", throat="B"):
    return {"review_nozzle": nozzle, "review_throat": throat}


class TestBuildComparisonRows:
    def test_pump_change_row(self):
        rows = build_comparison_rows(
            [_result()],
            {"MPS-01": _entry("12", "B")},
            [],
            {"MPS-01": (200.0, 450.0)},
        )
        (row,) = rows
        assert row["Current pump"] == "12B"
        assert row["Optimized pump"] == "13C"
        assert row["Change"] == "▲ bigger"
        assert row["Current oil"] == 200
        assert row["Current PF"] == 450
        assert row["Opt oil"] == 250
        assert row["Opt PF"] == 500
        assert row["Δ oil"] == 50
        assert row["Status"] == "run"

    def test_same_pump_flagged_same(self):
        rows = build_comparison_rows(
            [_result(recommended_nozzle="12", recommended_throat="B")],
            {"MPS-01": _entry("12", "B")},
            [],
            {"MPS-01": (200.0, 450.0)},
        )
        assert rows[0]["Change"] == "same"

    def test_shut_in_well_shows_forgone_oil(self):
        # An active well the plan leaves out: its measured rates stay visible
        # and Δ oil is the production the shut-in gives up.
        rows = build_comparison_rows(
            [],
            {"MPS-02": _entry()},
            ["MPS-02"],
            {"MPS-02": (80.0, 300.0)},
        )
        (row,) = rows
        assert row["Optimized pump"] == "SHUT IN"
        assert row["Status"] == "SHUT IN"
        assert row["Change"] == "shut in"
        assert row["Opt oil"] == 0 and row["Opt PF"] == 0
        assert row["Δ oil"] == -80

    def test_hypothetical_well_has_no_current_baseline(self):
        # No installed pump, no tests: current side renders as missing and the
        # full modeled rate is the delta.
        rows = build_comparison_rows(
            [_result()],
            {"MPS-01": {"review_nozzle": "", "review_throat": ""}},
            [],
            {"MPS-01": (None, None)},
        )
        (row,) = rows
        assert row["Current pump"] == "—"
        assert row["Current oil"] is None
        assert row["Current PF"] is None
        assert row["Change"] == "—"
        assert row["Δ oil"] == 250

    def test_model_test_ratio_from_matchcheck(self):
        rows = build_comparison_rows(
            [_result()],
            {"MPS-01": _entry()},
            [],
            {"MPS-01": (200.0, 450.0)},
            matchcheck_rows={"MPS-01": {"oil_ratio": 1.3049}},
        )
        assert rows[0]["Model÷Test"] == pytest.approx(1.3)

    def test_model_test_ratio_none_without_matchcheck(self):
        rows = build_comparison_rows(
            [_result()], {"MPS-01": _entry()}, [], {"MPS-01": (200.0, 450.0)}
        )
        assert rows[0]["Model÷Test"] is None

    def test_sonic_status(self):
        rows = build_comparison_rows(
            [_result(sonic_status=True)],
            {"MPS-01": _entry()},
            [],
            {"MPS-01": (200.0, 450.0)},
        )
        assert rows[0]["Status"] == "⚠ sonic"

    def test_sorted_by_biggest_mover(self):
        results = [
            _result(well_name="SMALL", predicted_oil_rate=210.0),  # Δ +10
            _result(well_name="BIG", predicted_oil_rate=400.0),  # Δ +200
        ]
        active = {"SMALL": _entry(), "BIG": _entry(), "SI-WELL": _entry()}
        rates = {
            "SMALL": (200.0, 450.0),
            "BIG": (200.0, 450.0),
            "SI-WELL": (90.0, 300.0),  # Δ −90
        }
        rows = build_comparison_rows(results, active, ["SI-WELL"], rates)
        assert [r["Well"] for r in rows] == ["BIG", "SI-WELL", "SMALL"]

    def test_detail_columns_present(self):
        # _render_results splits these into the detail expander + the CSV.
        rows = build_comparison_rows(
            [_result()], {"MPS-01": _entry()}, [], {"MPS-01": (200.0, 450.0)}
        )
        for col in ("Form. water (BPD)", "Total WC", "Suction (psi)"):
            assert col in rows[0]


class TestPumpSizeChange:
    """Direction flag for the results table's Change column (nozzle = size)."""

    @pytest.mark.parametrize(
        "cur, opt, expected",
        [
            (("12", "B"), ("13", "C"), "▲ bigger"),
            (("13", "C"), ("12", "B"), "▼ smaller"),
            (("9", "A"), ("10", "A"), "▲ bigger"),  # numeric, not lexicographic
            (("12", "B"), ("12", "C"), "▲ throat"),
            (("12", "C"), ("12", "B"), "▼ throat"),
            (("12", "B"), ("12", "B"), "same"),
            (("", ""), ("12", "B"), "—"),  # no current pump
            (("XX", "B"), ("12", "C"), "new pump"),  # unparseable nozzle
        ],
    )
    def test_directions(self, cur, opt, expected):
        assert pump_size_change(cur[0], cur[1], opt[0], opt[1]) == expected


class TestPadHub:
    def test_spec_for_each_pad(self):
        from woffl.gui.pad_hub import _PADS, _spec_for

        pads = [_spec_for(label).pad for label in _PADS]
        assert pads == ["S", "I", "M"]
        prefixes = [_spec_for(label).prefix for label in _PADS]
        assert len(set(prefixes)) == len(prefixes)  # unique session-key spaces

    def test_spec_for_unknown_pad_raises(self):
        from woffl.gui.pad_hub import _spec_for

        with pytest.raises(ValueError):
            _spec_for("Z-Pad")
