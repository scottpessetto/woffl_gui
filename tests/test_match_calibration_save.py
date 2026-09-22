"""Saving a gaugeless Match test fit as the installed pump's calibration
(server.services.pump_calibration.remember_match_fit / save_match_fit).

User report 2026-09-22: after Match test + Save well inputs, a reopened well
went back to reference pump coefficients - the matched kth/kdi were never
persisted. Every write here is faked; nothing reaches Databricks.
"""

import json

import pandas as pd
import pytest

from server.services import pump_calibration as pc

WELL = "MPS-12"
PARAMS = {"nozzle_no": "11", "area_ratio": "B", "hydraulics_model": "beggs", "nozzle_area_factor": 1.0,
          "ken": 0.03, "kth": 0.3, "kdi": 0.4, "pres": 1500.0, "qwf": 900.0, "pwf": 700.0, "form_wc": 0.5,
          "surf_pres": 210.0}
MATCH = {"match_quality": "good", "pwf": 812.4, "qwf_liq": 1020.6, "form_wc": 0.4213, "ken": 0.03,
         "kth": 0.42, "kdi": 0.55, "pf_error_pct": -1.2, "oil_error_pct": 0.8, "pf_reachable": True}


@pytest.fixture()
def world(monkeypatch):
    """Fresh tracker shows 11B; the saved seeds start at the matched anchor."""
    from server.services import datasources, ipr, well_model, wells
    import woffl.assembly.jp_history as jp

    state = {"seeds": {"pres": 1500.0, "qwf": 1021, "pwf": 812, "form_wc": 0.421, "surf_pres": 210.0},
             "pump": {"nozzle_no": "11", "throat_ratio": "B", "date_set": "2026-05-01"}, "comments": []}
    monkeypatch.setattr(datasources, "jp_history_fresh", lambda: pd.DataFrame())
    monkeypatch.setattr(jp, "get_current_pump", lambda df, well: state["pump"])
    monkeypatch.setattr(wells, "well_context", lambda well, m, c, fresh=False, tracker=None: {"well": well, "seeds": dict(state["seeds"])})

    def describe(ctx, model=None):
        inputs = {k: ctx["seeds"][k] for k in sorted(ctx["seeds"])}
        return {"fingerprint": f"{abs(hash(json.dumps(inputs, sort_keys=True))):032x}"[-32:], "inputs": inputs}

    monkeypatch.setattr(well_model, "from_context", describe)
    monkeypatch.setattr(pc.history, "next_entry_datetime", lambda: "2026-09-22T20:00:00")
    monkeypatch.setattr(pc.history, "resolve_entry_user", lambda: "engineer@example.com")
    monkeypatch.setattr(pc.history, "push_eng_comment",
                        lambda well, at, who, text, context=None: state["comments"].append((well, context, json.loads(text))))
    monkeypatch.setattr(pc.snapshot, "cache_clear", lambda: None)
    monkeypatch.setattr(ipr, "_invalidate_after_write", lambda well, **kw: None)
    return state


def test_saved_match_writes_an_installation_bound_provisional_record(world):
    token = pc.remember_match_fit(WELL, PARAMS, MATCH)
    out = pc.save_match_fit(WELL, token)
    assert "11B" in out["message"] and "kth 0.420" in out["message"]
    (well, context, rec), = world["comments"]
    assert well == WELL and context == pc.CONTEXT
    assert rec["n"] == "11" and rec["t"] == "B" and rec["k"] == [0.03, 0.42, 0.55, 1.0]
    assert rec["q"]["provisional"] is True and rec["q"]["src"] == "match" and rec["q"]["n"] == 1
    pc.decode(json.dumps(rec))  # the same validator resolve() applies on reload
    with pytest.raises(ValueError, match="expired"):
        pc.save_match_fit(WELL, token)  # one save per match


def test_unsaved_matched_inputs_block_the_save_and_name_the_fields(world):
    world["seeds"]["pwf"] = 700  # the matched BHP was never saved
    token = pc.remember_match_fit(WELL, PARAMS, MATCH)
    with pytest.raises(ValueError, match="Save the matched well inputs first.*pwf"):
        pc.save_match_fit(WELL, token)
    assert world["comments"] == []


def test_pump_changed_since_the_match_is_refused(world):
    world["pump"] = {"nozzle_no": "12", "throat_ratio": "C", "date_set": "2026-09-20"}
    token = pc.remember_match_fit(WELL, PARAMS, MATCH)
    with pytest.raises(ValueError, match="ran on 11B but the tracker shows 12C"):
        pc.save_match_fit(WELL, token)
    assert world["comments"] == []


def test_failed_or_unidentified_matches_are_not_saveable(world):
    assert pc.remember_match_fit(WELL, PARAMS, {**MATCH, "match_quality": "failed"}) is None
    assert pc.remember_match_fit(WELL, PARAMS, {**MATCH, "pf_reachable": False}) is None
    token = pc.remember_match_fit(WELL, PARAMS, MATCH)
    with pytest.raises(ValueError, match="another well"):
        pc.save_match_fit("MPS-99", token)
