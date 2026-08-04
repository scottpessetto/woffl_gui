"""Sidebar guarantees for as-built geometry and the liquid-rate anchor.

Two of Scott's rulings, 2026-08-03:

1. **As-built dimensions are read-only in the GUI.** Tubing/casing sizes and the
   jet-pump depth come from prop_hist (via the ``vw_prop_mech`` pivot) and are
   the data team's to change. The write side is already sealed
   (``prop_hist_client.AS_BUILT_PROP_IDS``); these pin the INPUT side, so an
   engineer can't model steel that isn't in the hole. Wells prop_hist knows
   nothing about stay editable — a Custom / hypothetical well must remain
   modelable from a default.
2. **The total liquid rate is the truth.** ``qwf`` is BLPD excluding power
   fluid; oil and water are derived through the water cut. Every seed path must
   put a LIQUID rate in that field — scaling it by ``(1 - WC)`` on the way in
   was the bug behind B-28's stored 2135.29 BLPD matching no well test.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

_st_mock = MagicMock()
_st_mock.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) else lambda fn: fn
)
sys.modules.setdefault("streamlit", _st_mock)

from woffl.gui import sidebar  # noqa: E402

# A fully-characterized well: both pipe pairs and a survey-interpolated TVD.
DB_WELL = {
    "out_dia": 4.5,
    "thick": 0.271,
    "casing_out_dia": 9.625,
    "casing_inn_dia": 8.681,
    "JP_TVD": 4254.250474,
    "JP_MD": 4753.0,
}


class _FakeSt:
    """Records every number_input call's kwargs; real dict session_state.

    ``_number_input`` renders under the WIDGET key (``<key>_input``, the
    two-tier session-state pattern), so unwrap it back to the logical key.
    """

    def __init__(self):
        self.session_state = {}
        self.inputs = {}

    def number_input(self, label, key=None, **kwargs):
        logical = str(key)[: -len("_input")] if str(key).endswith("_input") else key
        self.inputs[logical] = {"label": label, **kwargs}
        return self.session_state.get(logical, kwargs.get("min_value", 0))

    # Seed paths reach for these; none affect what the tests assert.
    def checkbox(self, *a, **k):
        return False

    def caption(self, *a, **k):
        return None

    def expander(self, *a, **k):
        raise AssertionError("tests call the widget renderers directly")


@pytest.fixture
def fake_st(monkeypatch):
    fs = _FakeSt()
    monkeypatch.setattr(sidebar, "st", fs)
    return fs


class TestAsBuiltPredicate:
    def test_all_keys_present_and_numeric(self):
        assert sidebar.as_built_from_props(DB_WELL, "out_dia", "thick") is True

    def test_no_well_data_is_not_as_built(self):
        assert sidebar.as_built_from_props(None, "out_dia") is False
        assert sidebar.as_built_from_props({}, "out_dia") is False

    def test_nan_under_a_present_key_is_not_as_built(self):
        """Databricks carries missing values as NaN under a PRESENT key, so a
        plain ``in`` / ``.get()`` check would lock the field on a hole."""
        wd = {**DB_WELL, "thick": float("nan")}
        assert sidebar.as_built_from_props(wd, "out_dia", "thick") is False

    def test_a_single_missing_key_unlocks_the_pair(self):
        wd = {k: v for k, v in DB_WELL.items() if k != "thick"}
        assert sidebar.as_built_from_props(wd, "out_dia", "thick") is False


class TestPipeWidgetsLock:
    def test_databricks_dimensions_render_disabled(self, fake_st):
        sidebar._render_pipe_params(DB_WELL)
        for key in ("tubing_od", "tubing_thickness", "casing_od", "casing_thickness"):
            assert fake_st.inputs[key]["disabled"] is True, key
            assert "As-built" in fake_st.inputs[key]["help"]

    def test_custom_well_keeps_them_editable(self, fake_st):
        """No prop_hist row ⇒ the well is modeled from defaults, and locking the
        fields would make it unmodelable."""
        sidebar._render_pipe_params(None)
        for key in ("tubing_od", "tubing_thickness", "casing_od", "casing_thickness"):
            assert fake_st.inputs[key]["disabled"] is False, key

    def test_missing_casing_unlocks_only_casing(self, fake_st):
        """Tubing known, casing absent: lock what's measured, leave the
        substituted 6.875/0.5 editable and say so."""
        wd = {k: v for k, v in DB_WELL.items() if not k.startswith("casing_")}
        sidebar._render_pipe_params(wd)
        assert fake_st.inputs["tubing_od"]["disabled"] is True
        assert fake_st.inputs["casing_od"]["disabled"] is False
        assert "6.875" in fake_st.inputs["casing_od"]["help"]

    def test_inverted_casing_pair_does_not_lock(self, fake_st):
        """ID >= OD is bad data, not a measurement — ``casing_dims_from_chars``
        substitutes the default, so the field must stay editable."""
        wd = {**DB_WELL, "casing_inn_dia": 9.625, "casing_out_dia": 8.681}
        sidebar._render_pipe_params(wd)
        assert fake_st.inputs["casing_od"]["disabled"] is False

    def test_csv_fallback_help_does_not_claim_a_prop_hist_read(self, fake_st):
        """Databricks down ⇒ the frame came from the stale jp_chars.csv. Still
        locked, but the caption must say where the number actually came from."""
        fake_st.session_state["well_chars_source"] = "csv_fallback"
        sidebar._render_pipe_params(DB_WELL)
        help_txt = fake_st.inputs["tubing_od"]["help"]
        assert "jp_chars.csv" in help_txt and "stale" in help_txt
        assert "prop_hist" not in help_txt


class TestJetPumpDepthLock:
    def test_survey_derived_tvd_is_read_only(self, fake_st):
        sidebar._render_geometry(DB_WELL)
        assert fake_st.inputs["jpump_tvd"]["disabled"] is True
        assert "jpump_md" in fake_st.inputs["jpump_tvd"]["help"]

    def test_pf_density_stays_editable(self, fake_st):
        """PF density is a fluid property the engineer sets — not as-built."""
        sidebar._render_geometry(DB_WELL)
        assert fake_st.inputs["rho_pf"].get("disabled", False) is False

    def test_hypothetical_well_can_still_set_a_depth(self, fake_st):
        sidebar._render_geometry(None)
        assert fake_st.inputs["jpump_tvd"]["disabled"] is False


class TestLiquidRateAnchor:
    def test_qwf_widget_is_labelled_and_bounded_for_liquid(self, fake_st):
        sidebar._render_formation_inflow(DB_WELL)
        qwf = fake_st.inputs["qwf"]
        assert "Total Liquid" in qwf["label"] and "BLPD" in qwf["label"]
        # A 95%-WC well's liquid rate is 20x its oil rate; the old 6000 ceiling
        # would have had clamp_seed silently truncating real anchors.
        assert qwf["max_value"] >= 20000
        assert sidebar.SEED_BOUNDS["qwf"][1] >= 20000

    def test_vogel_seed_uses_total_fluid_unscaled(self, fake_st, monkeypatch):
        """``coeff_row["qwf"]`` IS ``WtTotalFluid``. Seeding it scaled by
        (1 - WC) is what made the anchor an oil rate wearing a liquid label."""
        coeffs = pd.DataFrame(
            [
                {
                    "Well": "MPB-28",
                    "qwf": 2135.0,
                    "pwf": 1141.0,
                    "ResP": 1464.0,
                    "form_wc": 0.83,
                    "fgor": 300.0,
                    "num_tests": 4,
                    "most_recent_date": pd.Timestamp("2026-07-30"),
                }
            ]
        )
        tests = pd.DataFrame(
            [{"WtDate": pd.Timestamp("2026-07-30"), "whp": 250.0}] * 2
        )
        monkeypatch.setattr(
            "woffl.assembly.ipr_analyzer.estimate_reservoir_pressure", lambda d: d
        )
        monkeypatch.setattr(
            "woffl.assembly.ipr_analyzer.compute_vogel_coefficients", lambda d: coeffs
        )
        monkeypatch.setattr(
            "woffl.gui.utils.get_well_tests_for_well", lambda w: tests
        )

        sidebar._auto_populate_from_ipr("MPB-28")

        assert fake_st.session_state["qwf"] == 2135
        assert fake_st.session_state["form_wc"] == pytest.approx(0.83)

    def test_single_test_seed_uses_total_fluid_not_oil(self, fake_st, monkeypatch):
        tests = pd.DataFrame(
            [
                {
                    "WtDate": pd.Timestamp("2026-07-30"),
                    "WtOilVol": 363.0,
                    "WtWaterVol": 1772.0,
                    "WtTotalFluid": 2135.0,
                    "BHP": 1141.0,
                    "fgor": 300.0,
                    "whp": 250.0,
                }
            ]
        )
        monkeypatch.setattr(
            "woffl.gui.utils.get_well_tests_for_well", lambda w: tests
        )

        sidebar._auto_populate_from_ipr("MPB-28")

        assert fake_st.session_state["qwf"] == 2135  # not 363
        assert fake_st.session_state["form_wc"] == pytest.approx(0.83, abs=0.01)

    def test_single_test_without_total_reconstructs_from_oil_plus_water(
        self, fake_st, monkeypatch
    ):
        tests = pd.DataFrame(
            [
                {
                    "WtDate": pd.Timestamp("2026-07-30"),
                    "WtOilVol": 300.0,
                    "WtWaterVol": 700.0,
                    "WtTotalFluid": None,
                    "BHP": 1141.0,
                    "fgor": 300.0,
                    "whp": 250.0,
                }
            ]
        )
        monkeypatch.setattr(
            "woffl.gui.utils.get_well_tests_for_well", lambda w: tests
        )

        sidebar._auto_populate_from_ipr("MPB-28")

        assert fake_st.session_state["qwf"] == 1000
