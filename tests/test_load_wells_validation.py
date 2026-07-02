"""Regression tests for NaN-proofing in load_wells_from_dataframe (P0-4).

Databricks nulls arrive as NaN in the jp_chars dict; NaN is truthy, so the old
``cfg.get("out_dia") or 4.5`` fallbacks kept the NaN. The NaN passed WellConfig
(no bounds on the geometry fields), failed every pump combo in the sweep, and
the well silently vanished from the optimizer — the pad plan just lost a well
with no trace. These tests pin the fix: nulls fall back to defaults, and
implausible/degenerate values fail AT LOAD with the well named.
"""

import pandas as pd
import pytest

from woffl.assembly.network_optimizer import load_wells_from_dataframe

CHARS_COLUMNS = [
    "Well",
    "res_pres",
    "JP_TVD",
    "JP_MD",
    "out_dia",
    "thick",
    "casing_od",
    "casing_thick",
    "form_wc",
    "form_gor",
    "form_temp",
    "surf_pres",
    "qwf",
    "pwf",
    "is_sch",
    "oil_api",
]

GOOD_ROW = {
    "Well": "MPX-1",
    "res_pres": 1800,
    "JP_TVD": 4200,
    "JP_MD": 4300,
    "out_dia": 4.5,
    "thick": 0.271,
    "casing_od": 6.875,
    "casing_thick": 0.5,
    "form_wc": 0.6,
    "form_gor": 300,
    "form_temp": 160,
    "surf_pres": 210,
    "qwf": 900,
    "pwf": 800,
    "is_sch": True,
    "oil_api": 22.0,
}


def _load(tmp_path, row_overrides, input_df=None):
    """Write a one-well jp_chars CSV (with overrides) and load it."""
    row = {**GOOD_ROW, **row_overrides}
    chars = pd.DataFrame([row], columns=CHARS_COLUMNS)
    path = tmp_path / "jp_chars.csv"
    chars.to_csv(path, index=False)
    if input_df is None:
        input_df = pd.DataFrame({"Well": [row["Well"]]})
    return load_wells_from_dataframe(input_df, jp_chars_path=str(path))


class TestNanFallbacks:
    """Databricks nulls (NaN) must fall back to defaults, not ride through."""

    def test_happy_path(self, tmp_path):
        (cfg,) = _load(tmp_path, {})
        assert cfg.well_name == "MPX-1"
        assert cfg.tubing_od == 4.5
        assert cfg.field_model == "Schrader"

    def test_nan_geometry_falls_back(self, tmp_path):
        (cfg,) = _load(
            tmp_path,
            {"out_dia": None, "thick": None, "casing_od": None, "casing_thick": None},
        )
        assert cfg.tubing_od == 4.5
        assert cfg.tubing_thickness == 0.271
        assert cfg.casing_od == 6.875
        assert cfg.casing_thickness == 0.5
        # The point of the fix: nothing NaN survives into the config.
        for v in (
            cfg.tubing_od,
            cfg.tubing_thickness,
            cfg.casing_od,
            cfg.casing_thickness,
            cfg.qwf,
            cfg.pwf,
            cfg.form_gor,
        ):
            assert v == v  # not NaN

    def test_nan_rates_fall_back(self, tmp_path):
        (cfg,) = _load(
            tmp_path, {"qwf": None, "pwf": None, "form_gor": None, "surf_pres": None}
        )
        assert cfg.qwf == 750.0
        assert cfg.pwf == 500.0
        assert cfg.form_gor == 250.0
        assert cfg.surf_pres == 210.0

    def test_nan_is_sch_defaults_schrader_not_kuparuk(self, tmp_path):
        # Pre-fix: NaN is_sch failed the truthy-list check and silently
        # classified the well as Kuparuk (wrong PVT preset).
        (cfg,) = _load(tmp_path, {"is_sch": None})
        assert cfg.field_model == "Schrader"

    def test_nan_pvt_override_becomes_none(self, tmp_path):
        (cfg,) = _load(tmp_path, {"oil_api": None})
        assert cfg.oil_api is None  # falls to the field_model preset downstream

    def test_nan_jp_md_falls_back_to_tvd(self, tmp_path):
        (cfg,) = _load(tmp_path, {"JP_MD": None})
        assert cfg.jpump_md == cfg.jpump_tvd == 4200.0


class TestLoudLoadFailures:
    """Required/degenerate fields must fail at load with the well named."""

    def test_null_res_pres_is_named_error(self, tmp_path):
        with pytest.raises(ValueError, match=r"MPX-1.*res_pres"):
            _load(tmp_path, {"res_pres": None})

    def test_null_tvd_is_named_error(self, tmp_path):
        with pytest.raises(ValueError, match=r"MPX-1.*JP_TVD"):
            _load(tmp_path, {"JP_TVD": None})

    def test_pwf_at_or_above_res_pres_rejected(self, tmp_path):
        # Degenerate IPR: every pump combo would NaN and the well would
        # silently vanish from the optimization. Fail loud instead.
        with pytest.raises(ValueError, match=r"MPX-1.*pwf.*res_pres"):
            _load(tmp_path, {"pwf": 1900})

    def test_zero_qwf_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"MPX-1.*qwf"):
            _load(tmp_path, {"qwf": 0})

    def test_tubing_bigger_than_casing_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"MPX-1.*tubing_od"):
            _load(tmp_path, {"out_dia": 7.0, "casing_od": 6.875})

    def test_all_bad_rows_reported_together(self, tmp_path):
        chars = pd.DataFrame(
            [
                {**GOOD_ROW, "Well": "MPX-1", "pwf": 1900},
                {**GOOD_ROW, "Well": "MPX-2", "qwf": 0},
            ],
            columns=CHARS_COLUMNS,
        )
        path = tmp_path / "jp_chars.csv"
        chars.to_csv(path, index=False)
        input_df = pd.DataFrame({"Well": ["MPX-1", "MPX-2"]})
        with pytest.raises(ValueError) as exc:
            load_wells_from_dataframe(input_df, jp_chars_path=str(path))
        assert "MPX-1" in str(exc.value) and "MPX-2" in str(exc.value)


class TestDataFrameOverrides:
    def test_df_value_beats_database(self, tmp_path):
        input_df = pd.DataFrame({"Well": ["MPX-1"], "res_pres": [2100.0]})
        (cfg,) = _load(tmp_path, {}, input_df=input_df)
        assert cfg.res_pres == 2100.0

    def test_nan_df_cell_keeps_database_value(self, tmp_path):
        input_df = pd.DataFrame({"Well": ["MPX-1"], "pwf": [float("nan")]})
        (cfg,) = _load(tmp_path, {}, input_df=input_df)
        assert cfg.pwf == 800.0
