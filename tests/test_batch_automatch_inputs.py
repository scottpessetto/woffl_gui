"""Tests for ``step_review_wells._batch_automatch_inputs`` — the batch
auto-match input builder, focused on the 2026-07-21 gaugeless improvements:

* GAUGELESS wells (no BHP → ``estimate_reservoir_pressure`` drops them, no
  Vogel coeff row) used to be skipped with "IPR fit failed". Now they fall
  back to the engineer's REVIEWED store entry (res_pres / WC / GOR decided in
  the Solver), flagged via ``raw["ipr_fallback"]``; with no reviewed entry
  the skip reason tells the engineer what to do.
* Friction seeds: reviewed ``ken_well``/``kth_well``/``kdi_well`` beat the
  Databricks ``jpfric_*`` values, which beat library defaults — previously
  the batch always used generics, discarding per-well calibration.

NOTE: this file imports REAL streamlit (headless import works fine), NOT the
sys.modules MagicMock pattern other GUI tests use — this filename sorts
BEFORE test_gui_smoke in collection order, and a mock installed here would
poison the later real ``import streamlit.components.v1`` across the suite.
The function under test never touches ``st`` directly anyway.
"""

import pandas as pd
import pytest

from woffl.gui.workflow_steps import step_review_wells as srw


def _tests_df(n=3, with_bhp=False):
    return pd.DataFrame(
        {
            "WtDate": pd.to_datetime(["2026-07-01", "2026-06-01", "2026-05-01"][:n]),
            "BHP": ([600.0] * n if with_bhp else [float("nan")] * n),
            "WtTotalFluid": [400.0] * n,
            "WtOilVol": [150.0] * n,
            "WtWaterVol": [250.0] * n,
            "fgor": [320.0] * n,
            "whp": [float("nan")] * n,
        }
    )


@pytest.fixture
def mocked_sources(monkeypatch):
    """Mock every external source _batch_automatch_inputs touches. Returns a
    namespace-ish dict the tests tweak per scenario."""
    import woffl.assembly.ipr_analyzer as ipr_mod
    import woffl.assembly.jp_history as jp_mod
    import woffl.gui.utils as utils_mod

    ctx = {
        "well_data": {
            "Well": "MPS-99",
            "out_dia": 4.5,
            "thick": 0.5,
            "JP_TVD": 4200,
            "is_sch": True,
            "form_temp": 160.0,
            "jpfric_entry": 0.11,
            "jpfric_throat": 0.44,
            "jpfric_diffuser": 0.55,
            "oil_api": None,
            "gas_sg": None,
            "wat_sg": None,
            "bubble_point": None,
        },
        "tests": _tests_df(),
        "fit_raises": True,  # gaugeless: no coeff row → IndexError inside
        "pump": {"nozzle_no": "12", "throat_ratio": "B"},
    }

    monkeypatch.setattr(utils_mod, "get_well_data", lambda w: ctx["well_data"])
    monkeypatch.setattr(
        utils_mod, "get_well_tests_for_well", lambda w: ctx["tests"]
    )
    monkeypatch.setattr(
        utils_mod, "create_pipes", lambda *a, **k: (None, None, "WELLBORE")
    )
    monkeypatch.setattr(
        utils_mod, "create_well_profile_from_survey", lambda *a, **k: "WELLPROF"
    )
    monkeypatch.setattr(utils_mod, "live_pf_for_seed", lambda w: None)
    monkeypatch.setattr(
        jp_mod, "get_current_pump", lambda hist, w: ctx["pump"]
    )
    monkeypatch.setattr(srw, "_recent_test_rates", lambda w: (150.0, 2600.0))

    def fake_estimate(tests, **k):
        return tests

    def fake_coeffs(df, **k):
        if ctx["fit_raises"]:
            # Mirrors reality: no coeff row for a no-BHP well → the .iloc[0]
            # in the caller raises IndexError.
            return pd.DataFrame(columns=["Well", "ResP", "form_wc", "fgor"])
        return pd.DataFrame(
            [{"Well": "MPS-99", "ResP": 1500.0, "form_wc": 0.6, "fgor": 300.0}]
        )

    monkeypatch.setattr(ipr_mod, "estimate_reservoir_pressure", fake_estimate)
    monkeypatch.setattr(ipr_mod, "compute_vogel_coefficients", fake_coeffs)
    return ctx


def test_gaugeless_with_reviewed_entry_falls_back(mocked_sources):
    entry = {
        "res_pres": 1150.0,
        "form_wc": 0.65,
        "form_gor": 74.0,
        "surf_pres": 222.0,
        "ken_well": 0.324,
        "kth_well": 0.298,
        "kdi_well": 0.296,
    }
    kw, raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry
    )
    assert why is None
    # IPR anchor from the engineer's reviewed entry, flagged as such.
    assert kw["pres"] == pytest.approx(1150.0)
    assert kw["form_wc"] == pytest.approx(0.65)
    assert kw["form_gor"] == pytest.approx(74.0)
    assert raw["ipr_fallback"] == "reviewed entry"
    # Friction seeds from the reviewed entry (beats Databricks jpfric_*).
    assert kw["ken0"] == pytest.approx(0.324)
    assert kw["kth0"] == pytest.approx(0.298)
    assert kw["kdi0"] == pytest.approx(0.296)
    # No BHP anywhere → no bhp_target; surface pressure from the entry.
    assert kw["bhp_target"] is None
    assert kw["surf_pres"] == pytest.approx(222.0)
    assert kw["pin_ppf"] is True


def test_gaugeless_without_entry_uses_assumed_anchor(mocked_sources):
    """S-03 live lesson: the Solver models unreviewed gaugeless wells from
    its DEFAULT anchor and matches most of them — the batch must attempt the
    same (assumed ResP + test-derived WC/GOR) instead of skipping."""
    kw, raw, why = srw._batch_automatch_inputs("MPS-99", object(), 3300.0)
    assert why is None
    assert kw["pres"] == pytest.approx(srw._ASSUMED_RESP)
    # WC/GOR derived from the recent test the same way the Vogel-coeffs
    # helper derives them: water/total, and the test's fgor column.
    assert kw["form_wc"] == pytest.approx(250.0 / 400.0)
    assert kw["form_gor"] == pytest.approx(320.0)
    assert raw["ipr_fallback"] == f"assumed ResP {int(srw._ASSUMED_RESP)}"


def test_fitted_well_seeds_friction_from_databricks(mocked_sources):
    mocked_sources["fit_raises"] = False
    mocked_sources["tests"] = _tests_df(with_bhp=True)
    kw, raw, why = srw._batch_automatch_inputs("MPS-99", object(), 3300.0)
    assert why is None
    # Vogel-fit anchor (no fallback flag)...
    assert kw["pres"] == pytest.approx(1500.0)
    assert raw["ipr_fallback"] is None
    # ...with per-well Databricks friction instead of the old generics.
    assert kw["ken0"] == pytest.approx(0.11)
    assert kw["kth0"] == pytest.approx(0.44)
    assert kw["kdi0"] == pytest.approx(0.55)
    assert kw["bhp_target"] == pytest.approx(600.0)


def test_reviewed_friction_beats_databricks(mocked_sources):
    mocked_sources["fit_raises"] = False
    mocked_sources["tests"] = _tests_df(with_bhp=True)
    entry = {"ken_well": 0.2, "kth_well": None, "kdi_well": None}
    kw, _raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry
    )
    assert why is None
    assert kw["ken0"] == pytest.approx(0.2)  # reviewed wins
    assert kw["kth0"] == pytest.approx(0.44)  # falls through to Databricks
    assert kw["kdi0"] == pytest.approx(0.55)


def test_reviewed_pump_stands_in_when_tracker_has_none(mocked_sources):
    """S-Pad live run: MPS-17/25/54 skipped 'no current pump' although the
    engineer had reviewed them — the reviewed pump must stand in."""
    mocked_sources["pump"] = None
    entry = {
        "res_pres": 1150.0,
        "form_wc": 0.65,
        "form_gor": 74.0,
        "review_nozzle": "11",
        "review_throat": "A",
    }
    kw, _raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry
    )
    assert why is None
    assert kw["nozzle"] == "11"
    assert kw["throat"] == "A"


def test_no_pump_anywhere_skips(mocked_sources):
    mocked_sources["pump"] = None
    kw, _raw, why = srw._batch_automatch_inputs("MPS-99", object(), 3300.0)
    assert kw is None
    assert "pump" in why.lower()


def test_force_fit_fallback_pump_used_as_last_resort(mocked_sources):
    """Force-fit: no tracker install, no reviewed pump → the sidebar pump."""
    mocked_sources["pump"] = None
    kw, _raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, fallback_pump=("12", "B")
    )
    assert why is None
    assert kw["nozzle"] == "12"
    assert kw["throat"] == "B"


def test_reviewed_pump_beats_fallback_pump(mocked_sources):
    mocked_sources["pump"] = None
    entry = {
        "res_pres": 1150.0,
        "form_wc": 0.65,
        "form_gor": 74.0,
        "review_nozzle": "11",
        "review_throat": "A",
    }
    kw, _raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry, fallback_pump=("12", "B")
    )
    assert why is None
    assert kw["nozzle"] == "11"
    assert kw["throat"] == "A"


def test_apply_batch_row_writes_store_with_note():
    """The force-fit path saves any-status rows with a notes marker and
    honest provenance (forced/assumed for fallback-anchored wells)."""
    from types import SimpleNamespace

    raw = {
        "nozzle_no": "12",
        "area_ratio": "B",
        "jpump_direction": "forward",
        "tubing_od": 4.5,
        "tubing_thickness": 0.5,
        "casing_od": 6.875,
        "casing_thickness": 0.5,
        "form_wc": 0.65,
        "form_gor": 74,
        "form_temp": 160,
        "field_model": "Schrader",
        "oil_api": None,
        "gas_sg": None,
        "wat_sg": None,
        "bubble_point": None,
        "surf_pres": 222,
        "jpump_tvd": 4200,
        "has_bhp": False,
        "ipr_fallback": "assumed ResP 1700",
    }
    res = SimpleNamespace(
        ken=0.03, kth=0.3, kdi=0.4, ppf_surf=3300.0,
        qwf_oil=150.0, pwf=600.0, pres=1700.0,
    )
    r = SimpleNamespace(well="MPS-99", status="partial", result=res)
    store: dict = {}

    err = srw._apply_batch_row("S", store, r, raw, note="force-fit (partial)")

    assert err is None
    entry = store["MPS-99"]
    assert entry["notes"] == "force-fit (partial)"
    assert entry["ipr_source"] == "forced"
    assert entry["bhp_source"] == "assumed"
    assert entry["review_nozzle"] == "12"
    assert entry["review_throat"] == "B"
    assert entry["jpump_direction"] == "forward"
    assert entry["res_pres"] == pytest.approx(1700.0)


def test_single_test_allowed_with_reviewed_entry(mocked_sources):
    mocked_sources["tests"] = _tests_df(n=1)
    entry = {"res_pres": 1150.0, "form_wc": 0.65, "form_gor": 74.0}
    kw, _raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry
    )
    assert why is None
    assert kw["pres"] == pytest.approx(1150.0)


def test_single_test_without_entry_uses_assumed_anchor(mocked_sources):
    mocked_sources["tests"] = _tests_df(n=1)
    kw, raw, why = srw._batch_automatch_inputs("MPS-99", object(), 3300.0)
    assert why is None
    assert kw["pres"] == pytest.approx(srw._ASSUMED_RESP)
    assert raw["ipr_fallback"] == f"assumed ResP {int(srw._ASSUMED_RESP)}"


def test_no_tests_still_skips(mocked_sources):
    mocked_sources["tests"] = _tests_df(n=0)
    kw, _raw, why = srw._batch_automatch_inputs("MPS-99", object(), 3300.0)
    assert kw is None
    # Exact constant match — the force-fit path keys its default-to-offline
    # conversion on this reason string.
    assert why == srw._NO_TESTS_REASON


def test_offline_stub_entry_round_trips(mocked_sources):
    """Force-fit's no-test wells become OFFLINE store entries that must
    build a valid WellConfig (in case the engineer flips them online) and
    carry the force-fit marker."""
    entry = srw._offline_stub_entry("MPS-99")

    assert entry["offline"] is True
    assert entry["reviewed"] is True
    assert entry["notes"].startswith("force-fit")
    assert entry["ipr_source"] == "forced"
    assert entry["bhp_source"] == "assumed"
    # Databricks props flowed in (mocked well_data).
    assert entry["jpump_tvd"] == pytest.approx(4200.0)
    assert entry["form_temp"] == pytest.approx(160.0)
    wc = srw.wrs.to_well_config(entry)
    assert wc.well_name == "MPS-99"


def test_no_props_with_entry_uses_entry_geometry(mocked_sources):
    """A reviewed well with no Databricks props row must still be matchable —
    geometry, TVD, field model, and temp all come from the entry."""
    mocked_sources["well_data"] = None
    entry = {
        "res_pres": 1150.0,
        "form_wc": 0.65,
        "form_gor": 74.0,
        "jpump_tvd": 5100.0,
        "field_model": "Kuparuk",
        "tubing_od": 3.5,
        "tubing_thickness": 0.4,
        "form_temp": 170.0,
    }
    kw, raw, why = srw._batch_automatch_inputs(
        "MPS-99", object(), 3300.0, store_entry=entry
    )
    assert why is None
    assert kw["field_model"] == "kuparuk"
    assert kw["form_temp"] == pytest.approx(170.0)
    assert raw["jpump_tvd"] == 5100
    assert raw["tubing_od"] == pytest.approx(3.5)
    assert raw["field_model"] == "Kuparuk"
