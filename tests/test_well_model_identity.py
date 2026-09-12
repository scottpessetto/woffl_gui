"""Saved calibration depends on one oil curve and the stable well model."""
from copy import deepcopy

import pytest

from server import schemas
from server.services import well_model


def context():
    return {"well": "Custom", "seeds": schemas.SimParams().model_dump()}


def test_measured_composition_and_controls_preserve_model_identity():
    base = context()
    changed = deepcopy(base)
    s = changed["seeds"]
    oil = s["qwf"] * (1-s["form_wc"])
    s.update(form_wc=.81, form_gor=550.25, ppf_surf=3100.5, ken=.17)
    s["qwf"] = oil / (1-s["form_wc"])
    assert well_model.from_context(base) == well_model.from_context(changed)


def test_known_geometry_issue_cannot_be_bypassed_by_cached_profile():
    source = {**context(), "geometry_issue": "Fresh survey conflicts with saved pump depth"}
    with pytest.raises(ValueError, match="Fresh survey conflicts"):
        well_model.from_context(source)


@pytest.mark.parametrize("key,value", [("qwf", 1000.25), ("pres", 2000.5),
    ("form_temp", 99.5), ("gas_sg", .75), ("tubing_od", 3.5),
    ("jpump_direction", "forward"), ("hydraulics_model", "drift_flux"), ("surf_pres", 251.25)])
def test_curve_geometry_pvt_and_pressure_model_changes_invalidate_fit(key, value):
    base = context()
    changed = deepcopy(base)
    changed["seeds"][key] = value
    assert well_model.from_context(base)["fingerprint"] != well_model.from_context(changed)["fingerprint"]


def test_changed_survey_invalidates_model(tmp_path, monkeypatch):
    monkeypatch.setattr(well_model, "ROOT", tmp_path)
    before = well_model.from_context(context())
    survey = tmp_path / "woffl" / "jp_data" / "well_surveys" / "Custom Deviation Survey.csv"
    survey.parent.mkdir(parents=True)
    survey.write_text("MD,TVD\n0,0\n4200,4000\n", encoding="utf-8")
    after = well_model.from_context(context())
    assert before["fingerprint"] != after["fingerprint"]
    assert after["inputs"]["survey_sha256"]


def test_fresh_property_read_bypasses_warm_snapshot_and_raises_on_failure(monkeypatch):
    import pandas as pd
    from woffl.gui import ipr_anchor
    from woffl.assembly import databricks_client, prop_hist_client
    old = {"values": {"qwf_liq": 100.}}
    monkeypatch.setattr(ipr_anchor, "_saved_ipr_cache", {"Custom": old})
    monkeypatch.setattr(prop_hist_client, "_resolve_enthid", lambda well: 42)
    reads = []
    def query(sql):
        reads.append(sql)
        return pd.DataFrame()
    monkeypatch.setattr(databricks_client, "execute_query", query)
    assert ipr_anchor.load_saved_ipr("Custom") is old
    assert ipr_anchor.load_saved_ipr("Custom", fresh=True, strict=True) is None
    assert len(reads) == 1 and "WHERE enthid = 42" in reads[0]
    assert ipr_anchor._saved_ipr_cache["Custom"] is old
    def unavailable(sql):
        raise RuntimeError("warehouse unavailable")
    monkeypatch.setattr(databricks_client, "execute_query", unavailable)
    with pytest.raises(RuntimeError, match="warehouse unavailable"):
        ipr_anchor.load_saved_ipr("Custom", fresh=True, strict=True)
