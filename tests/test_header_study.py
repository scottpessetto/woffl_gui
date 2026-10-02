"""Header page: relations, correlations, IPR, impact run, validation and saves.

Offline: no Databricks. The writer is mocked; ALLOW_DATABRICKS_WRITES is
never set here.
"""

import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.services import header_model as hm
from server.services import header_study as hs


# ── header_model ─────────────────────────────────────────────────────────────


def _daily(slopes, r2s):
    return pd.DataFrame({"slope": slopes, "r2": r2s})


def test_low_slope_high_rate_esp_counts_as_measured():
    # The old [0.2, 1.5] band called a well coupling at 0.05 "slugging".
    s = hm.summarize_daily(_daily([0.05, 0.06, 0.04, 0.05, 0.07, 0.3], [0.7] * 5 + [0.1]))
    assert s["status"] == "measured"
    assert s["n_fit"] == 5 and s["n_days"] == 6
    assert s["slope"] == pytest.approx(0.05)


def test_few_fit_days_is_weak_and_empty_is_no_data():
    assert hm.summarize_daily(_daily([0.5] * 3 + [0.1] * 17, [0.8] * 3 + [0.1] * 17))["status"] == "weak"
    assert hm.summarize_daily(None)["status"] == "no_data"


def test_correlation_trend_falls_with_rate_and_clamps_to_range():
    pts = [{"well": f"W{i}", "q_liq": q, "slope": s}
           for i, (q, s) in enumerate([(400, 0.95), (800, 0.72), (1200, 0.6), (2000, 0.42), (3200, 0.27)])]
    corr = hm.fit_correlation(pts)
    assert corr["kind"] == "trend" and corr["b"] < 0
    assert hm.predict_slope(corr, 500) > hm.predict_slope(corr, 2500)
    # Outside the fitted range the rate is clamped, not extrapolated.
    assert hm.predict_slope(corr, 50) == hm.predict_slope(corr, 400)
    assert hm.predict_slope(corr, 50000) == hm.predict_slope(corr, 3200)


def test_correlation_uses_median_when_trend_not_identifiable():
    pts = [{"well": "A", "q_liq": 1000, "slope": 0.4}, {"well": "B", "q_liq": 1100, "slope": 0.6},
           {"well": "C", "q_liq": 1200, "slope": 0.5}]
    corr = hm.fit_correlation(pts)
    assert corr["kind"] == "median" and corr["b"] == 0.0
    assert hm.predict_slope(corr, 99999) == pytest.approx(0.5)


def test_vogel_through_anchor_and_pi():
    ipr = {"qwf": 1000.0, "pwf": 600.0, "pres": 1800.0}
    assert hm.ipr_rate(ipr, 600.0) == pytest.approx(1000.0)
    assert hm.ipr_rate(ipr, 1800.0) == pytest.approx(0.0)
    d = 1e-3
    fd = (hm.ipr_rate(ipr, 600.0 - d) - hm.ipr_rate(ipr, 600.0 + d)) / (2 * d)
    assert hm.ipr_pi(ipr, 600.0) == pytest.approx(fd, rel=1e-6)


def test_ipr_validation_reasons():
    assert hm.ipr_valid({"qwf": 500, "pwf": 600, "pres": 1800}) is None
    assert "within" in hm.ipr_valid({"qwf": 500, "pwf": 1600, "pres": 1800})
    assert hm.ipr_valid(None) == "no IPR"
    assert hm.ipr_valid({"qwf": 0, "pwf": 600, "pres": 1800})


def test_nonjp_delta_uses_closed_loop_slope_once_and_wc_once():
    ipr = {"qwf": 1000.0, "pwf": 600.0, "pres": 1800.0}
    d = hm.nonjp_delta(10.0, 1.0, 0.5, ipr, 600.0, 0.8)
    assert d["d_whp"] == pytest.approx(10.0)
    assert d["d_bhp"] == pytest.approx(5.0)
    assert d["d_liq"] == pytest.approx(hm.ipr_rate(ipr, 605.0) - 1000.0)
    assert d["d_oil"] == pytest.approx(d["d_liq"] * 0.2)
    # Linear check: a small move is PI * dBHP.
    assert d["d_liq"] == pytest.approx(-hm.ipr_pi(ipr, 600.0) * 5.0, rel=0.01)
    assert hm.nonjp_delta(-10.0, None, 0.5, ipr, 600.0, 0.8)["d_liq"] > 0


def test_event_windows_are_hours_either_side_of_the_gap():
    w = hm.event_windows("2026-09-28T12:00", 72, 18, 6)
    assert w["pre_end"] == pd.Timestamp("2026-09-28 06:00")
    assert w["pre_start"] == pd.Timestamp("2026-09-25 06:00")
    assert w["post_start"] == pd.Timestamp("2026-09-28 18:00")
    assert w["post_end"] == pd.Timestamp("2026-09-29 12:00")


def test_window_delta_uses_medians_and_ignores_the_transition():
    idx = pd.date_range("2026-09-25", "2026-09-30", freq="h")
    s = pd.Series(400.0, index=idx)
    s[idx >= pd.Timestamp("2026-09-28 12:00")] = 410.0
    s[pd.Timestamp("2026-09-28 13:00")] = 900.0  # spike inside the gap
    s[pd.Timestamp("2026-09-26 03:00")] = 50.0   # one bad hour inside pre
    w = hm.event_windows("2026-09-28T12:00", 72, 18, 6)
    m = hm.window_delta(s, w)
    assert m["delta"] == pytest.approx(10.0)
    assert m["n_pre"] == 72 and m["n_post"] == 18


def test_run_status_levels_firm_means_own_data_or_a_saved_review():
    base = {"well": "A", "online": True, "outcome": "modeled", "relation_source": "measured",
            "ipr_source": "correlation", "pres_basis": "saved"}
    assert hm.run_status([base])["status"] == "complete"          # measured slope + its own saved ResP
    jp = {**base, "well": "J", "relation_source": "physics", "ipr_source": "jp"}
    assert hm.run_status([jp])["status"] == "complete"
    # A usable fit of the well's own gauged tests is firm.
    assert hm.run_status([{**base, "ipr_source": "fit", "ipr_fit_usable": True}])["status"] == "complete"
    # Not reviewed yet: default ResP, a borrowed correlation, a flagged fit or manual value.
    for patch in ({"pres_basis": "default"}, {"relation_source": "correlation"},
                  {"ipr_source": "fit"}, {"ipr_source": "manual"}, {"relation_source": "manual"}):
        st = hm.run_status([base, {**base, "well": "B", **patch}])
        assert st["status"] == "conditional" and st["soft"] == ["B"], patch
    # Saving any of those after review makes the well firm.
    reviewed = {**base, "well": "C", "relation_source": "correlation", "relation_saved": True,
                "ipr_source": "correlation", "pres_basis": "default", "ipr_saved": True}
    assert hm.run_status([base, reviewed])["status"] == "complete"
    assert hm.run_status([base, {**base, "well": "D", "outcome": "missing_inputs"}])["status"] == "incomplete"
    assert hm.run_status([{**base, "online": False, "outcome": "offline"}])["status"] == "complete"


# ── header_study: choices, down wells, validation ────────────────────────────


def _row(**kw):
    r = {
        "well": "MPF-01", "pad": "F", "lift": "ESP", "reservoir": "kuparuk", "pump": None,
        "age_ok": True, "looks_down": False, "online_default": True,
        "pres_well": 1800.0, "pres_basis": "default",
        "oil": 100.0, "liquid": 1000.0, "wc": 0.9, "whp_now": 410.0,
        "bhp_now": 600.0, "bhp_test": 600.0, "resvr_press": None, "has_gauge": True,
        "gauge_auto_bad": False, "gauge_note": None,
        "measured": {"slope": 0.6, "q25": 0.5, "q75": 0.7, "r2": 0.8, "n_fit": 30, "n_days": 50, "status": "measured"},
        "whp_hdr": {"slope": 0.98, "status": "measured"},
        "saved": None, "saved_ipr": None,
        "ipr_fit": {"qwf": 1000.0, "pwf": 600.0, "pres": 1500.0},
        "corr_options": {
            "ESP kuparuk": {"slope": 0.55, "lo": 0.45, "hi": 0.65, "same_lift": True},
            "ESP": {"slope": 0.5, "lo": 0.3, "hi": 0.7, "same_lift": True},
        },
        "corr_group": "ESP kuparuk",
        "ipr_options": {
            "F kuparuk": {"gauge": {"qwf": 1000.0, "pwf": 600.0, "pres": 2000.0, "pres_lo": 1500.0, "pres_hi": 2600.0},
                          "nogauge": {"qwf": 1000.0, "pwf": 700.0, "pres": 2100.0, "pres_lo": 1600.0, "pres_hi": 2700.0},
                          "same_reservoir": True},
        },
        "ipr_group": "F kuparuk",
    }
    r.update(kw)
    return r


def _choice(**kw):
    base = {"well": "MPF-01", "online": None, "gauge_bad": None, "relation": None, "corr_group": None,
            "slope": None, "ipr": None, "ipr_group": None, "qwf": None, "pwf": None, "pres": None}
    return schemas.HeaderWellChoice(**{**base, **kw})


def test_effective_defaults_follow_the_ladder():
    e = hs.effective(_row())
    assert e["rel"]["source"] == "measured" and (e["rel"]["lo"], e["rel"]["hi"]) == (0.5, 0.7)
    # IPR default: the well's own gauge data (a usable fit of its gauged tests) ...
    assert e["ipr"]["source"] == "fit" and e["ipr"]["fit_usable"] and e["gauge_ok"] and e["online"]
    # ... else its own ResP at today's rate and gauge BHP.
    assert hs.effective(_row(ipr_fit=None))["ipr"]["source"] == "correlation"
    assert hs.effective(_row(), _choice(gauge_bad=True))["ipr"]["source"] == "correlation"
    saved = _row(saved={"slope": 0.4, "whp_hdr": 1.0, "source": "correlation"},
                 saved_ipr={"qwf": 900.0, "pwf": 600.0, "pres": 2500.0, "source": "correlation"})
    s = hs.effective(saved)
    assert s["rel"]["slope"] == 0.4 and s["rel"]["source"] == "correlation" and s["rel"]["saved"]
    assert s["ipr"]["source"] == "correlation" and s["ipr"]["saved"]


def test_marking_a_gauge_bad_switches_to_correlations():
    e = hs.effective(_row(), _choice(gauge_bad=True))
    assert not e["gauge_ok"] and e["bhp_now"] is None
    assert e["rel"]["source"] == "correlation" and e["rel"]["group"] == "ESP kuparuk"
    # The gaugeless variant of the reservoir IPR: BHP from the group ratio.
    assert e["ipr"]["source"] == "correlation" and e["ipr"]["ipr"]["pwf"] == 700.0
    assert (e["ipr"]["pres_lo"], e["ipr"]["pres_hi"]) == (1600.0, 2700.0)
    # A measured relation from a bad gauge is refused, not silently used.
    m = hs.effective(_row(), _choice(gauge_bad=True, relation="measured"))
    assert m["rel"]["slope"] is None and "gauge" in m["rel"]["reason"]
    # An automatic flag can be overridden back to a working gauge.
    auto = _row(gauge_auto_bad=True, gauge_note="flat")
    assert not hs.effective(auto)["gauge_ok"]
    assert hs.effective(auto, _choice(gauge_bad=False))["gauge_ok"]


def test_assigning_a_correlation_group_and_ipr_group():
    e = hs.effective(_row(), _choice(relation="correlation", corr_group="ESP", ipr="correlation"))
    assert e["rel"]["slope"] == 0.5 and e["rel"]["group"] == "ESP"
    assert e["ipr"]["ipr"]["pres"] == 2000.0 and e["ipr"]["group"] == "F kuparuk"
    # Unknown group names fall back to the default group.
    assert hs.effective(_row(), _choice(relation="correlation", corr_group="nope"))["rel"]["group"] == "ESP kuparuk"
    # The pre-rename "assumed" choice still means the reservoir correlation.
    assert hs.effective(_row(), _choice(ipr="assumed"))["ipr"]["source"] == "correlation"


def test_resolve_ipr_rejects_anchor_near_reservoir_pressure():
    r = _row(ipr_fit={"qwf": 1000.0, "pwf": 1400.0, "pres": 1500.0})
    ip = hs.resolve_ipr(r, "fit", None)
    assert ip["ipr"] is None and "within" in ip["reason"]
    assert hs.resolve_ipr(r, "manual", {"qwf": 900, "pwf": 600, "pres": 2000})["ipr"]["pres"] == 2000


def test_looks_down_when_gauge_builds_far_above_test():
    assert hs._looks_down(1810.0, 552.0)
    assert not hs._looks_down(620.0, 552.0)
    assert not hs._looks_down(None, 552.0)
    assert "shut in" in hs._down_note(24, 1810.0, 552.0)
    assert "days old" in hs._down_note(80, 600.0, 600.0)


def test_gauge_check_catches_pf_pressure_low_readings_and_flatlines():
    assert "not a flowing BHP" in hm.gauge_problem(4629.0, 400.0, "kuparuk")      # MPL-20
    assert "not a flowing BHP" in hm.gauge_problem(4188.0, 479.0, "schrader")     # MPR-110
    # A pumped well's intake below WHP is normal (MPF-73: 385 vs 426), not a fault.
    assert hm.gauge_problem(385.0, 426.0, "kuparuk") is None
    flat = pd.Series([600.2] * 48, index=pd.date_range("2026-09-27", periods=48, freq="h"))
    assert "flat" in hm.gauge_problem(600.2, 410.0, "kuparuk", flat)
    assert hm.gauge_problem(600.0, 410.0, "kuparuk") is None


def test_validation_flags_operational_moves_and_scores_the_rest():
    rows = [
        {"well": "A", "pad": "F", "lift": "ESP", "online": True, "outcome": "modeled", "slope": 0.6,
         "d_bhp": 3.3, "d_whp": 5.5, "d_header": 5.5},
        {"well": "B", "pad": "F", "lift": "ESP", "online": True, "outcome": "modeled", "slope": 0.6,
         "d_bhp": 3.3, "d_whp": 5.5, "d_header": 5.5},
        {"well": "C", "pad": "F", "lift": "ESP", "online": True, "outcome": "modeled", "slope": 0.6,
         "d_bhp": 3.3, "d_whp": 5.5, "d_header": 5.5, "gauge_bad": True},
    ]
    event = {"wells": {
        "A": {"bhp": {"delta": 3.0}, "whp": {"delta": 5.0}},
        "B": {"bhp": {"delta": 197.0}, "whp": {"delta": 5.0}},  # ESP trip in the window
        "C": {"bhp": {"delta": 900.0}, "whp": {"delta": 5.0}},  # dead gauge: not a check at all
    }}
    v = hs._validation(rows, event)
    assert v["n"] == 2 and v["n_used"] == 1 and v["n_operational"] == 1
    assert v["bias"] == pytest.approx(0.3)
    assert v["rows"][0]["d_bhp_from_meas_whp"] == pytest.approx(3.0)


# ── correlations and IPR groups ──────────────────────────────────────────────


def _raw(name, res, q, s, status="measured", pad="F", bhp=600.0, ipr_fit=None, lift="ESP"):
    return _row(well=name, pad=pad, lift=lift, reservoir=res, liquid=q, bhp_now=bhp, bhp_test=bhp,
                measured={**_row()["measured"], "slope": s, "status": status}, ipr_fit=ipr_fit)


def test_correlations_split_by_reservoir_once_a_group_has_four_measured_wells():
    rows = ([_raw(f"K{i}", "kuparuk", q, s) for i, (q, s) in enumerate([(400, 0.95), (800, 0.8), (1200, 0.7), (2000, 0.55), (3000, 0.45)])]
            + [_raw(f"S{i}", "schrader", q, s) for i, (q, s) in enumerate([(1500, 0.1), (1800, 0.25), (2100, 0.2), (2400, 0.18)])]
            + [_raw("S-gaugeless", "schrader", 2000, None, "no_data", bhp=None), _raw("Sag-1", "sag", 900, None, "no_data")]
            + [_raw("S-dead", "schrader", 2000, 0.9, "measured")])
    rows[-1]["gauge_auto_bad"] = True  # a dead gauge's "measured" slope never feeds a correlation
    corr = hs._correlations(rows)
    assert set(corr) == {"ESP", "ESP kuparuk", "ESP schrader"}
    assert "S-dead" not in corr["ESP schrader"]["correlation"]["wells"]
    groups = hs._ipr_groups(rows)
    for r in rows:
        hs._attach_options(r, corr, groups)
    by = {r["well"]: r for r in rows}
    assert by["S-gaugeless"]["corr_group"] == "ESP schrader"
    assert by["S-gaugeless"]["corr_options"]["ESP schrader"]["slope"] < 0.3  # its Schrader peers, not the pooled line
    assert by["Sag-1"]["corr_group"] == "ESP"     # too few sag wells: lift-only fallback
    e = hs.effective(by["S-gaugeless"])
    assert e["rel"]["source"] == "correlation" and e["rel"]["lo"] <= e["rel"]["slope"] <= e["rel"]["hi"]


def test_each_well_keeps_its_own_resp_and_groups_only_supply_bhp_for_gaugeless_wells():
    saved = [_raw(f"L{i}", "kuparuk", 1000, 0.6, pad="L", bhp=600.0 + 50 * i) for i in range(3)]
    for r in saved:
        r.update(pres_well=2400.0, pres_basis="saved")
    default = _raw("L9", "kuparuk", 800, 0.6, pad="L", bhp=700.0)
    default.update(pres_well=3000.0, pres_basis="default")
    gaugeless = _raw("L-nog", "kuparuk", 700, None, "no_data", pad="L", bhp=None)
    gaugeless.update(has_gauge=False, pres_well=3000.0, pres_basis="default")
    shut = _raw("L-si", "kuparuk", 500, 0.5, pad="L", bhp=2600.0)
    shut.update(pres_well=3000.0, pres_basis="default", looks_down=True)
    rows = saved + [default, gaugeless, shut]
    groups = hs._ipr_groups(rows)
    assert set(groups) == {"kuparuk", "L kuparuk"}
    g = groups["L kuparuk"]
    assert g["ratio"]["n"] == 4 and "L-si" not in g["wells"]       # shut-in gauges never feed the ratio
    assert g["n_saved_pres"] == 3
    assert hs._shut_in_readings(rows, "kuparuk") == [{"well": "L-si", "bhp": 2600.0}]
    corr = hs._correlations(rows)
    for r in rows:
        hs._attach_options(r, corr, groups)
    # A saved ResP carries no range; a default one is +/-20%.
    s_opt = saved[0]["ipr_options"]["L kuparuk"]["gauge"]
    assert s_opt["pres"] == 2400.0 and s_opt["pres_lo"] == s_opt["pres_hi"] == 2400.0 and s_opt["pwf"] == 600.0
    d_opt = default["ipr_options"]["L kuparuk"]["gauge"]
    assert (d_opt["pres_lo"], d_opt["pres"], d_opt["pres_hi"]) == (2400.0, 3000.0, 3600.0)
    # Gaugeless: own ResP, BHP from the group's median ratio, own test rate.
    e = hs.effective(gaugeless)
    assert e["ipr"]["source"] == "correlation" and e["ipr"]["ipr"]["pres"] == 3000.0
    assert e["ipr"]["ipr"]["pwf"] == pytest.approx(round(g["ratio"]["med"] * 3000.0), abs=1)
    assert e["ipr"]["ipr"]["qwf"] == 700.0
    # A usable gauge fit is the default (the well's own data); the well-ResP option stays available.
    fitted = _raw("L7", "kuparuk", 900, 0.6, pad="L", ipr_fit={"qwf": 900.0, "pwf": 600.0, "pres": 1400.0})
    fitted.update(pres_well=1800.0, pres_basis="default")
    hs._attach_options(fitted, corr, groups)
    assert hs.effective(fitted)["ipr"]["ipr"]["pres"] == 1400.0
    assert hs.effective(fitted, _choice(ipr="correlation"))["ipr"]["ipr"]["pres"] == 1800.0


def test_range_brackets_the_base_and_orders_by_magnitude():
    ipr = {"qwf": 1000.0, "pwf": 600.0, "pres": 2000.0}
    base = hm.nonjp_delta(10.0, 1.0, 0.6, ipr, 600.0, 0.8)["d_oil"]
    lo, hi = hm.range_delta(10.0, 1.0, ipr, 600.0, 0.8, 0.5, 0.7, 1500.0, 2600.0)
    assert abs(lo) <= abs(base) <= abs(hi)
    # The low end of ResP is floored at the 300 psi drawdown rule.
    lo2, hi2 = hm.range_delta(10.0, 1.0, ipr, 600.0, 0.8, 0.6, 0.6, 100.0, 2000.0)
    assert math.isfinite(hi2) and abs(hi2) >= abs(lo2)


# ── run_impact with a fake board ─────────────────────────────────────────────


def _board(rows):
    return {"pads": ["F", "L"], "rows": rows, "built_at": "2026-09-29T00:00:00",
            "correlations": {}, "ipr_groups": {}, "header_now": {}}


def _fake_jp(detail):
    def fake(pads, jp_rows, scenarios, progress):
        oil = {k: {w: (d["d_oil"] * (list(v.values())[0] / 10.0)) for w, d in detail.items() if "d_oil" in d}
               for k, v in scenarios.items()}
        return detail, oil, []
    return fake


def test_scenario_totals_ranges_curve_and_failed_jp(monkeypatch):
    esp = _row()
    down = _row(well="MPF-14", age_ok=True, looks_down=True)
    missing = _row(well="MPF-99", corr_options={}, corr_group=None, ipr_fit=None,
                   measured={**_row()["measured"], "status": "no_data", "slope": None})
    dead = _row(well="MPF-50", gauge_auto_bad=True, gauge_note="flat")
    jp_ok = _row(well="MPF-107", lift="JP")
    jp_fail = _row(well="MPL-06", pad="L", lift="JP")
    monkeypatch.setattr(hs, "build_board", lambda pads, fit_days, progress: _board([esp, down, missing, dead, jp_ok, jp_fail]))
    monkeypatch.setattr(hs, "_jp_solve", _fake_jp({
        "MPF-107": {"d_whp": 9.8, "d_bhp": 5.0, "d_oil": -3.0, "d_liq": -10.0, "oil_base": 200.0,
                    "sonic": False, "pf": 3400, "whp_model": 420, "ipr": {"qwf": 700, "pwf": 800, "pres": 1800},
                    "ipr_source": "saved"},
        "MPL-06": {"error": "did not solve", "ipr": {"qwf": 900.0, "pwf": 1600.0, "pres": 2600.0}},
    }))
    req = schemas.HeaderRunRequest(pads=["F", "L"], delta_by_pad={"F": 10, "L": 10}, jp_method="model")
    res = hs.run_impact(req)
    by = {r["well"]: r for r in res["rows"]}
    assert by["MPF-01"]["outcome"] == "modeled" and by["MPF-01"]["d_oil"] < 0
    assert abs(by["MPF-01"]["d_oil_lo"]) <= abs(by["MPF-01"]["d_oil"]) <= abs(by["MPF-01"]["d_oil_hi"])
    assert by["MPF-14"]["outcome"] == "offline"
    assert by["MPF-99"]["outcome"] == "missing_inputs"
    # A dead gauge keeps the well online on its correlations.
    assert by["MPF-50"]["outcome"] == "modeled" and by["MPF-50"]["relation_source"] == "correlation"
    assert by["MPF-50"]["ipr_source"] == "correlation" and by["MPF-50"]["gauge_bad"]
    assert by["MPF-107"]["relation_source"] == "physics" and by["MPF-107"]["d_oil"] == -3.0
    assert by["MPL-06"]["outcome"] == "modeled" and by["MPL-06"]["ipr_source"] == "jp"
    assert "jet-pump model" in by["MPL-06"]["note"]
    total = sum(r["d_oil"] for r in res["rows"] if r["outcome"] == "modeled")
    assert res["totals"]["d_oil"] == pytest.approx(round(total, 1), abs=0.11)
    assert abs(res["totals"]["d_oil_lo"]) <= abs(res["totals"]["d_oil"]) <= abs(res["totals"]["d_oil_hi"])
    assert res["status"]["status"] == "incomplete" and res["status"]["missing"] == ["MPF-99"]
    # The curve passes through zero, reproduces the run at +10 psi and is monotone.
    c = res["curve"]
    i0, i10 = c["grid"].index(0), c["grid"].index(10)
    assert c["total"]["base"][i0] == 0.0
    assert c["total"]["base"][i10] == pytest.approx(res["totals"]["d_oil"], abs=0.3)
    assert all(a >= b for a, b in zip(c["total"]["base"], c["total"]["base"][1:]))
    assert res["board"]["rows"]  # the run carries its board for the page and for Save


def test_event_well_is_excluded_and_net_reported(monkeypatch):
    esp, l20 = _row(), _row(well="MPL-20", pad="L", lift="JP")
    monkeypatch.setattr(hs, "build_board", lambda pads, fit_days, progress: _board([esp, l20]))
    monkeypatch.setattr(hs, "_jp_solve", _fake_jp({}))
    monkeypatch.setattr(hs, "_pad_deltas", lambda req, board, progress: (
        {"F": 5.5, "L": 5.7}, {"time": req.event_time, "windows": {}, "pads": {}, "wells": {}, "pops": []}))
    req = schemas.HeaderRunRequest(pads=["F", "L"], mode="event", event_time="2026-09-28T12:00",
                                   event_well="mpl-20", event_well_oil=250)
    res = hs.run_impact(req)
    by = {r["well"]: r for r in res["rows"]}
    assert by["MPL-20"]["outcome"] == "event_well" and not by["MPL-20"]["online"]
    assert res["totals"]["net_oil"] == pytest.approx(250 + res["totals"]["d_oil"], abs=0.11)


def test_pop_candidates_from_the_downtime_log(monkeypatch):
    import woffl.assembly.databricks_client as dbc

    # The real MPL-20 record: down all of 09-28, 15.8 h down on 09-29 (the
    # PF reached the header on 09-28 ~12:00, before the log called it up).
    log = pd.DataFrame({
        "well_name": ["L-020"] * 4 + ["F-014", "F-014", "F-001"],
        "well_pad": ["L"] * 4 + ["F", "F", "F"],
        "dtdate": ["2026-09-26", "2026-09-27", "2026-09-28", "2026-09-29", "2026-09-28", "2026-09-29", "2026-09-26"],
        "hrs": [24.0, 24.0, 24.0, 15.8, 20.0, 24.0, 3.0],
    })
    monkeypatch.setattr(dbc, "execute_query", lambda sql: log)
    pops = hs._pop_candidates(["F", "L"], "2026-09-28T12:00")
    assert [(p["well"], p["kind"], p["day"]) for p in pops] == [
        ("MPL-20", "came on", "2026-09-29"), ("MPF-14", "went down", "2026-09-28")]


# ── save ─────────────────────────────────────────────────────────────────────


def _board_job(rows, kind="header_board"):
    result = {"rows": rows} if kind == "header_board" else {"board": {"rows": rows}}
    return {"status": "done", "kind": kind, "result": result}


def test_save_uses_server_values_and_one_statement_per_well(monkeypatch):
    from woffl.assembly import prop_hist_client

    rows = [_row(), _row(well="MPF-62", measured={**_row()["measured"], "status": "weak"}),
            _row(well="MPF-107", lift="JP"), _row(well="MPF-50")]
    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: _board_job(rows, "header_run") if jid == "r1" else None)
    calls = []
    monkeypatch.setattr(prop_hist_client, "resolve_entry_user", lambda: "tester@example.com")
    monkeypatch.setattr(prop_hist_client, "push_props",
                        lambda well, values, entry_user: calls.append((well, dict(values), entry_user)) or len(values))
    req = schemas.HeaderSaveRequest(board_job_id="r1", wells=[
        schemas.HeaderSaveWell(well="MPF-01", relation="measured", ipr="fit"),
        schemas.HeaderSaveWell(well="MPF-62", relation="measured"),             # weak -> refused
        schemas.HeaderSaveWell(well="MPF-107", ipr="fit"),                       # JP IPR -> refused
        schemas.HeaderSaveWell(well="MPF-62", relation="correlation", ipr="manual", qwf=400, pwf=900, pres=3000),
        # Dead gauge: assigned correlation group + reservoir IPR correlation, gaugeless variant.
        schemas.HeaderSaveWell(well="MPF-50", relation="correlation", corr_group="ESP", ipr="correlation",
                               ipr_group="F kuparuk", gauge_bad=True),
    ])
    out = hs.save(req)
    assert [c[0] for c in calls] == ["MPF-01", "MPF-62", "MPF-50"]
    v1 = calls[0][1]
    assert v1[hs.HDR_SLOPE] == 0.6 and v1[hs.HDR_REL_SOURCE] == 1.0 and v1[hs.HDR_DAYS] == 30.0
    assert v1["resvr_press"] == 1500.0 and v1[hs.HDR_IPR_SOURCE] == 1.0
    v2 = calls[1][1]
    assert v2[hs.HDR_SLOPE] == 0.55 and v2[hs.HDR_REL_SOURCE] == 2.0 and v2[hs.HDR_R2] == 0.0
    assert v2["ipr_pwf"] == 900.0 and v2[hs.HDR_IPR_SOURCE] == 3.0
    v3 = calls[2][1]
    assert v3[hs.HDR_SLOPE] == 0.5 and v3["ipr_pwf"] == 700.0 and v3[hs.HDR_IPR_SOURCE] == 2.0
    errs = {r["well"]: r["error"] for r in out["results"] if r["error"]}
    assert "weak" in errs["MPF-62"] and "Solver" in errs["MPF-107"]
    assert out["saved_wells"] == 3 and calls[0][2] == "tester@example.com"


def test_save_refuses_unknown_board_job(monkeypatch):
    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: None)
    with pytest.raises(ValueError, match="expired"):
        hs.save(schemas.HeaderSaveRequest(board_job_id="x", wells=[schemas.HeaderSaveWell(well="A", relation="measured")]))


def test_every_saved_prop_is_a_registered_id():
    assert set(hs.SAVED_PROP_IDS) >= set(hs.HDR_PROP_IDS)
    assert all(p.startswith("hdr_") for p in hs.HDR_PROP_IDS)
    assert set(hm.REL_SOURCE_CODE) == {"measured", "correlation", "manual"}
    assert set(hm.IPR_SOURCE_CODE) == {"fit", "correlation", "manual"}


# ── router ───────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    from server.main import app

    return TestClient(app)


def test_save_route_is_gated(client, monkeypatch):
    import woffl.gui.ipr_anchor as ia

    monkeypatch.setattr(ia, "writes_enabled", lambda: False)
    r = client.post("/api/header/save", json={"board_job_id": "b", "wells": [{"well": "A", "relation": "measured"}]})
    assert r.status_code == 403


def test_board_and_run_routes_start_jobs(client, monkeypatch):
    monkeypatch.setattr(hs, "start_board", lambda pads, fit_days: "job-b")
    monkeypatch.setattr(hs, "start_run", lambda req: "job-r")
    assert client.post("/api/header/board", json={"pads": ["F"]}).json() == {"job_id": "job-b"}
    assert client.post("/api/header/run", json={"pads": ["F"], "delta_by_pad": {"F": 10}}).json() == {"job_id": "job-r"}
    bad = client.post("/api/header/run", json={"pads": ["F"], "delta_by_pad": {"F": 999}})
    assert bad.status_code == 422
    assert client.post("/api/header/run", json={"pads": ["F"], "mode": "event"}).status_code == 422


def test_clean_pads_rejects_sql_shapes():
    assert hs.clean_pads(["r", "F", "f"]) == ["F", "R"]
    with pytest.raises(ValueError):
        hs.clean_pads(["F'; DROP"])
    with pytest.raises(ValueError):
        hs.clean_pads([])


# ── review cards: gauge limit, flagged fits, well detail ─────────────────────


def test_shut_in_schrader_gauges_are_plausible_but_pf_readings_are_not():
    # MPR-111 shut in reads ~2,443 psi: above the 2,200 fit cap, a real reservoir reading.
    assert hm.gauge_problem(2443.0, 758.0, "schrader") is None
    assert hm.gauge_problem(2618.0, 389.0, "schrader") is None           # MPL-46
    assert "not a flowing BHP" in hm.gauge_problem(4188.0, 479.0, "schrader")  # MPR-110 stuck
    assert "not a flowing BHP" in hm.gauge_problem(4629.0, 400.0, "kuparuk")   # MPL-20 PF pressure


def test_a_flagged_fit_can_be_chosen_and_saved_explicitly_but_is_never_the_default(monkeypatch):
    from woffl.assembly import prop_hist_client

    r = _row(ipr_fit=None, ipr_fit_any={"qwf": 1000.0, "pwf": 600.0, "pres": 1650.0})
    assert hs.effective(r)["ipr"]["source"] == "correlation"          # default skips the flagged fit
    e = hs.effective(r, _choice(ipr="fit"))
    assert e["ipr"]["source"] == "fit" and e["ipr"]["ipr"]["pres"] == 1650.0
    assert hs.effective(r, _choice(ipr="fit", gauge_bad=True))["ipr"]["ipr"] is None  # not from a bad gauge
    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: _board_job([r]))
    monkeypatch.setattr(prop_hist_client, "resolve_entry_user", lambda: "t@x")
    calls = []
    monkeypatch.setattr(prop_hist_client, "push_props", lambda w, v, entry_user: calls.append(v) or len(v))
    hs.save(schemas.HeaderSaveRequest(board_job_id="b", wells=[schemas.HeaderSaveWell(well="MPF-01", ipr="fit")]))
    assert calls[0]["resvr_press"] == 1650.0 and calls[0][hs.HDR_IPR_SOURCE] == 1.0


def test_shut_in_readings_are_reported_per_reservoir_group():
    rows = [_row(well="MPR-111", pad="R", reservoir="schrader", bhp_now=2443.0, age_ok=False),
            _row(well="MPR-102", pad="R", reservoir="schrader", bhp_now=704.0),
            _row(well="MPR-110", pad="R", reservoir="schrader", bhp_now=4188.0, age_ok=False, gauge_auto_bad=True)]
    assert hs._shut_in_readings(rows, "schrader") == [{"well": "MPR-111", "bhp": 2443.0}]
    assert hs._shut_in_readings(rows, "schrader", "L") == []


def test_well_detail_reuses_the_pad_pull_and_shapes_the_charts(monkeypatch):
    idx = pd.date_range("2026-09-01", periods=24 * 10, freq="h")
    whp = 400 + 10 * np.sin(np.arange(len(idx)) / 3.0)
    df = pd.DataFrame({"BHP": 600 + 0.6 * (whp - 400), "WHP": whp, "HeaderP": whp + 5}, index=idx)
    seen = {}

    def fake_trends(wells, start, end):
        seen["wells"] = list(wells)
        return {"MPR-102": df}

    monkeypatch.setattr(hs, "_trends", fake_trends)
    monkeypatch.setattr(hs, "_pad_wells", lambda pads: ["MPR-102", "MPR-104"])
    tests = pd.DataFrame({"well": ["MPR-102", "MPR-102", "MPF-01"], "WtDate": pd.to_datetime(["2026-08-01", "2026-09-01", "2026-09-01"]),
                          "WtTotalFluid": [1800.0, 1850.0, 900.0], "WtOilVol": [1500.0, 1450.0, 80.0],
                          "form_wc": [0.12, 0.2, 0.9], "BHP": [700.0, 690.0, 510.0], "whp": [470.0, 475.0, 410.0]})
    monkeypatch.setattr(hs.tests_svc, "fetch_all_well_tests", lambda months: tests)
    d = hs.well_detail("MPR-102", 120, ["R"])
    assert seen["wells"] == ["MPR-102", "MPR-104"]  # the board's cache key, not a per-well query
    assert d["trend"] and set(d["trend"][0]) >= {"t", "bhp", "whp", "headerp"}
    assert d["daily"] and abs(d["daily"][0]["slope"] - 0.6) < 0.01
    assert [t["date"] for t in d["tests"]] == ["2026-08-01", "2026-09-01"]


def test_well_route_validates_its_inputs(client, monkeypatch):
    monkeypatch.setattr(hs, "well_detail", lambda w, f, p: {"well": w, "fit_days": f, "pads": p})
    assert client.get("/api/header/well/MPR-102?pads=R,l").json() == {"well": "MPR-102", "fit_days": 120, "pads": ["L", "R"]}
    assert client.get("/api/header/well/DROP TABLE").status_code == 422
    assert client.get("/api/header/well/MPR-102?fit_days=5").status_code == 422


# ── saved gauge verdict (hdr_gauge_bad) ──────────────────────────────────────


def test_a_saved_gauge_verdict_overrides_the_automatic_check_until_changed():
    # Saved bad on a gauge the check thinks is fine.
    bad = _row(gauge_saved={"bad": True, "at": "2026-09-29", "by": "x"}, gauge_bad_default=True)
    e = hs.effective(bad)
    assert not e["gauge_ok"] and e["rel"]["source"] == "correlation"
    # Saved good on a gauge the check flagged (engineer overrode it).
    good = _row(gauge_auto_bad=True, gauge_saved={"bad": False, "at": "2026-09-29", "by": "x"}, gauge_bad_default=False)
    assert hs.effective(good)["gauge_ok"]
    # A session choice still wins over the saved verdict (until it is saved).
    assert hs.effective(bad, _choice(gauge_bad=False))["gauge_ok"]


def test_saving_the_gauge_verdict_writes_hdr_gauge_bad_including_gauge_only_and_jp(monkeypatch):
    from woffl.assembly import prop_hist_client

    rows = [_row(), _row(well="MPF-107", lift="JP"), _row(well="MPR-142", has_gauge=False)]
    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: _board_job(rows))
    monkeypatch.setattr(prop_hist_client, "resolve_entry_user", lambda: "t@x")
    calls = []
    monkeypatch.setattr(prop_hist_client, "push_props", lambda w, v, entry_user: calls.append((w, dict(v))) or len(v))
    out = hs.save(schemas.HeaderSaveRequest(board_job_id="b", wells=[
        schemas.HeaderSaveWell(well="MPF-01", gauge_bad=True),                       # gauge only
        schemas.HeaderSaveWell(well="MPF-107", gauge_bad=False),                     # JP: gauge only is fine
        schemas.HeaderSaveWell(well="MPR-142", gauge_bad=True),                      # no gauge to flag
        schemas.HeaderSaveWell(well="MPF-01", gauge_bad=True, relation="correlation"),
    ]))
    assert calls[0] == ("MPF-01", {hs.HDR_GAUGE_BAD: 1.0})
    assert calls[1] == ("MPF-107", {hs.HDR_GAUGE_BAD: 0.0})
    # With the gauge marked bad the relation resolves to its correlation, saved alongside.
    assert calls[2][1][hs.HDR_GAUGE_BAD] == 1.0 and calls[2][1][hs.HDR_REL_SOURCE] == 2.0
    errs = {r["well"]: r["error"] for r in out["results"] if r["error"]}
    assert "no BHP gauge" in errs["MPR-142"]
    assert hs.HDR_GAUGE_BAD in hs.SAVED_PROP_IDS


# ── jet pumps: pump model or empirical BHP~WHP relation ──────────────────────


def _jp(**kw):
    base = dict(well="MPF-107", lift="JP", pump="12B",
                measured={**_row()["measured"], "slope": 0.8, "q25": 0.7, "q75": 0.9},
                corr_options={"JP": {"slope": 0.75, "lo": 0.6, "hi": 0.9, "same_lift": True}},
                corr_group="JP", ipr_options={}, ipr_group=None, ipr_fit=None)
    return _row(**{**base, **kw})


def test_jp_wells_borrow_only_jp_correlations_and_get_no_ipr_options():
    rows = ([_raw(f"E{i}", "kuparuk", q, s) for i, (q, s) in enumerate([(800, 0.8), (1500, 0.6), (2500, 0.4)])]
            + [_raw(f"J{i}", "kuparuk", q, s, lift="JP") for i, (q, s) in enumerate([(600, 0.9), (900, 0.85)])]
            + [_raw("J-nog", "kuparuk", 700, None, "no_data", bhp=None, lift="JP")])
    corr = hs._correlations(rows)
    assert set(corr) == {"ESP", "JP"}
    assert sorted(corr["JP"]["correlation"]["wells"]) == ["J0", "J1"]    # ESP slopes never feed the JP group
    groups = hs._ipr_groups(rows)
    for r in rows:
        hs._attach_options(r, corr, groups)
    by = {r["well"]: r for r in rows}
    assert by["J-nog"]["corr_group"] == "JP" and by["J-nog"]["ipr_options"] == {}
    assert by["J-nog"]["corr_options"]["JP"]["slope"] == pytest.approx(0.875)
    assert by["E0"]["corr_group"] == "ESP"                              # ESP defaults unchanged


def test_jp_method_follows_the_run_unless_the_well_overrides_it():
    jp = _jp()
    # Default (user 2026-09-30): the relation when the well has one.
    assert hs.effective(jp)["jp_method"] == "empirical" and hs.effective(jp)["jp_note"] is None
    assert schemas.HeaderRunRequest(pads=["F"]).jp_method == "empirical"
    assert hs.effective(jp, None, "model")["jp_method"] == "model"
    assert hs.effective(jp, None, "empirical")["jp_method"] == "empirical"
    assert hs.effective(jp, _choice(well="MPF-107", jp_method="model"), "empirical")["jp_method"] == "model"
    assert hs.effective(jp, _choice(well="MPF-107", jp_method="empirical"))["jp_method"] == "empirical"
    assert hs.effective(_row(), None, "empirical")["jp_method"] is None   # not a jet pump
    # The relation ladder is the ESP one: own measured slope, else the JP group.
    e = hs.effective(jp, None, "empirical")
    assert e["rel"]["source"] == "measured" and e["rel"]["slope"] == 0.8
    assert e["ipr"]["ipr"] is None and e["ipr"]["source"] == "jp"         # filled by the run
    bad = hs.effective(jp, _choice(well="MPF-107", gauge_bad=True), "empirical")
    assert bad["rel"]["source"] == "correlation" and bad["rel"]["group"] == "JP"
    # No relation at all (no saved, no usable gauge, no JP group): the pump model.
    none = hs.effective(_jp(corr_options={}, corr_group=None), _choice(well="MPF-107", gauge_bad=True))
    assert none["jp_method"] == "model" and "pump model used" in none["jp_note"]
    # A saved relation comes first.
    saved = _jp(saved={"slope": 0.55, "whp_hdr": 1.0, "source": "measured"})
    assert hs.effective(saved)["rel"]["slope"] == 0.55 and hs.effective(saved)["rel"]["saved"]


def test_run_puts_jet_pumps_on_the_chosen_method(monkeypatch):
    esp = _row()
    jp_meas = _jp()                                                   # measured slope 0.8
    jp_model = _jp(well="MPL-06", pad="L")                            # this well stays on the model
    jp_corr = _jp(well="MPF-73", gauge_auto_bad=True)                 # bad gauge: JP correlation
    jp_none = _jp(well="MPL-20", pad="L", corr_options={}, corr_group=None,
                  measured={**_row()["measured"], "status": "no_data", "slope": None})
    monkeypatch.setattr(hs, "build_board", lambda pads, fit_days, progress: _board([esp, jp_meas, jp_model, jp_corr, jp_none]))
    seen = {}

    def fake_solve(pads, jp_rows, scenarios, progress):
        seen["model"] = sorted(r["well"] for r in jp_rows)
        solved = {"d_whp": 9.8, "d_bhp": 5.0, "d_oil": -3.0, "d_liq": -10.0, "oil_base": 200.0,
                  "sonic": False, "pf": 3400, "whp_model": 420,
                  "ipr": {"qwf": 700, "pwf": 800, "pres": 1800}, "ipr_source": "saved"}
        return _fake_jp({"MPL-06": solved, "MPL-20": {**solved, "d_oil": -1.0}})(pads, jp_rows, scenarios, progress)

    ipr = {"qwf": 900.0, "pwf": 700.0, "pres": 2600.0}

    def fake_iprs(pads, jp_rows, notes, progress):
        seen["empirical"] = sorted(r["well"] for r in jp_rows)
        return {r["well"]: {"ipr": dict(ipr), "ipr_source": "saved"} for r in jp_rows}

    monkeypatch.setattr(hs, "_jp_solve", fake_solve)
    monkeypatch.setattr(hs, "_jp_iprs", fake_iprs)
    req = schemas.HeaderRunRequest(pads=["F", "L"], delta_by_pad={"F": 10, "L": 10}, jp_method="empirical",
                                   wells=[_choice(well="MPL-06", jp_method="model")])
    res = hs.run_impact(req)
    by = {r["well"]: r for r in res["rows"]}
    assert seen == {"model": ["MPL-06", "MPL-20"], "empirical": ["MPF-107", "MPF-73"]}
    assert res["jp_method"] == "empirical"
    # Empirical: closed-loop slope x dWHP on the pump model's IPR, from the gauge BHP.
    m = by["MPF-107"]
    assert m["jp_method"] == "empirical" and m["relation_source"] == "measured" and m["ipr_source"] == "jp"
    expected = hm.nonjp_delta(10.0, 0.98, 0.8, ipr, 600.0, 0.9)
    assert m["d_bhp"] == pytest.approx(round(expected["d_bhp"], 1))
    assert m["d_oil"] == pytest.approx(round(expected["d_oil"], 1))
    assert abs(m["d_oil_lo"]) <= abs(m["d_oil"]) <= abs(m["d_oil_hi"])   # measured IQR 0.7-0.9
    assert m["ipr_saved"] and m["jp_ipr_source"] == "saved"
    # The well override stays on the pump model.
    assert by["MPL-06"]["jp_method"] == "model" and by["MPL-06"]["relation_source"] == "physics"
    assert by["MPL-06"]["d_oil"] == -3.0
    # Bad gauge: the JP correlation, on the IPR anchor's BHP (no gauge reading).
    c = by["MPF-73"]
    assert c["relation_source"] == "correlation" and c["relation_group"] == "JP" and c["bhp_basis"] == "IPR anchor"
    # No relation at all: the pump model instead, said plainly (never an ESP correlation).
    l20 = by["MPL-20"]
    assert l20["jp_method"] == "model" and l20["jp_fallback"] and l20["relation_source"] == "physics"
    assert l20["d_oil"] == -1.0 and "pump model used" in l20["note"]
    assert not by["MPL-06"]["jp_fallback"]                            # chosen, not a fallback
    # Firm: own measured slope + the pump model's IPR. Borrowed JP correlation: conditional.
    assert "MPF-107" not in res["status"]["soft"] and "MPF-73" in res["status"]["soft"]
    # The curve reproduces the run at +10 psi with both methods in it.
    cv = res["curve"]
    assert cv["total"]["base"][cv["grid"].index(10)] == pytest.approx(res["totals"]["d_oil"], abs=0.3)


def test_empirical_jp_without_a_usable_pump_ipr_has_no_estimate(monkeypatch):
    monkeypatch.setattr(hs, "build_board", lambda pads, fit_days, progress: _board([_jp()]))
    monkeypatch.setattr(hs, "_jp_solve", _fake_jp({}))
    monkeypatch.setattr(hs, "_jp_iprs", lambda pads, rows, notes, progress: {"MPF-107": {"error": "jet-pump inputs could not be loaded"}})
    res = hs.run_impact(schemas.HeaderRunRequest(pads=["F"], delta_by_pad={"F": 10}, jp_method="empirical"))
    row = res["rows"][0]
    assert row["outcome"] == "missing_inputs" and "pump-model IPR" in row["reason"]
    assert row["relation_source"] == "measured"


def test_run_route_validates_the_jp_method(client, monkeypatch):
    monkeypatch.setattr(hs, "start_run", lambda req: req.jp_method)
    ok = client.post("/api/header/run", json={"pads": ["F"], "delta_by_pad": {"F": 10}, "jp_method": "empirical",
                                              "wells": [{"well": "MPF-107", "jp_method": "model"}]})
    assert ok.json() == {"job_id": "empirical"}
    assert client.post("/api/header/run", json={"pads": ["F"], "jp_method": "vogel"}).status_code == 422
    assert client.post("/api/header/run", json={"pads": ["F"], "wells": [{"well": "A", "jp_method": "x"}]}).status_code == 422


def test_empirical_jp_accepts_the_pump_models_own_ipr_inside_300_psi(monkeypatch):
    # MPF-107 live, 2026-09-30: its pump-model IPR is 729 BLPD at 857 psi with
    # ResP 1,131 (274 psi drawdown). The page's 300 psi rule is for its own
    # assumed IPRs; the pump model runs on this one, so the relation may too.
    ipr = {"qwf": 729.0, "pwf": 857.0, "pres": 1131.0}
    assert "within 300" in hm.ipr_valid(ipr)
    assert hm.ipr_valid(ipr, min_drawdown=0.0) is None
    assert "not above" in hm.ipr_valid({**ipr, "pres": 850.0}, min_drawdown=0.0)
    jp = _jp(bhp_now=860.0, wc=0.95)
    monkeypatch.setattr(hs, "build_board", lambda pads, fit_days, progress: _board([jp]))
    monkeypatch.setattr(hs, "_jp_solve", _fake_jp({}))
    monkeypatch.setattr(hs, "_jp_iprs", lambda pads, rows, notes, progress: {"MPF-107": {"ipr": dict(ipr), "ipr_source": "vogel"}})
    res = hs.run_impact(schemas.HeaderRunRequest(pads=["F"], delta_by_pad={"F": 10}, jp_method="empirical"))
    row = res["rows"][0]
    assert row["outcome"] == "modeled" and row["d_oil"] < 0 and not row["ipr_saved"]
    # No ResP band on a pump-model IPR, and the 300 psi floor never lifts ResP above it.
    assert abs(row["d_oil_lo"]) <= abs(row["d_oil"]) <= abs(row["d_oil_hi"])
    lo, hi = hm.range_delta(10.0, 0.98, ipr, 860.0, 0.95, 0.8, 0.8, 1131.0, 1131.0)
    assert lo == pytest.approx(hi) == pytest.approx(hm.nonjp_delta(10.0, 0.98, 0.8, ipr, 860.0, 0.95)["d_oil"])


def test_jet_pump_relations_save_like_esp_ones_but_never_their_ipr(monkeypatch):
    from woffl.assembly import prop_hist_client

    rows = [_jp(), _jp(well="MPL-06", pad="L", measured={**_row()["measured"], "slope": 1.0, "status": "weak"})]
    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: _board_job(rows))
    monkeypatch.setattr(prop_hist_client, "resolve_entry_user", lambda: "t@x")
    calls = []
    monkeypatch.setattr(prop_hist_client, "push_props", lambda w, v, entry_user: calls.append((w, dict(v))) or len(v))
    out = hs.save(schemas.HeaderSaveRequest(board_job_id="b", wells=[
        schemas.HeaderSaveWell(well="MPF-107", relation="measured"),
        schemas.HeaderSaveWell(well="MPL-06", relation="measured"),                     # weak -> refused
        schemas.HeaderSaveWell(well="MPL-06", relation="correlation", corr_group="JP"),
        schemas.HeaderSaveWell(well="MPF-107", relation="manual", slope=0.5, ipr="manual", qwf=1, pwf=1, pres=900),
    ]))
    assert [c[0] for c in calls] == ["MPF-107", "MPL-06"]
    v1 = calls[0][1]
    assert v1 == {hs.HDR_SLOPE: 0.8, hs.HDR_WHP_HDR: 0.98, hs.HDR_R2: 0.8, hs.HDR_DAYS: 30.0, hs.HDR_REL_SOURCE: 1.0}
    v2 = calls[1][1]
    assert v2[hs.HDR_SLOPE] == 0.75 and v2[hs.HDR_REL_SOURCE] == 2.0 and v2[hs.HDR_R2] == 0.0
    assert not any(k in v for _, v in calls for k in hs.IPR_PROP_IDS)          # never a JP IPR
    errs = [r["error"] for r in out["results"] if r["error"]]
    assert any("weak" in e for e in errs) and any("Solver" in e for e in errs)


def test_an_implausible_measured_slope_is_never_a_default_a_correlation_point_or_a_measured_save(monkeypatch):
    # MPM-62, 2026-09-30: 15 good days whose median slope is -3.1 (a tag artefact,
    # not a flowing relation). Clipped to 0 it said "the header does not reach
    # this well" as firm data, and it tilted the M-Pad JP trend.
    assert hm.slope_plausible(0.05) and hm.slope_plausible(1.2)
    assert not hm.slope_plausible(-3.1) and not hm.slope_plausible(1.5) and not hm.slope_plausible(None)
    odd = _jp(well="MPM-62", measured={**_row()["measured"], "slope": -3.1, "q25": -4.0, "q75": -1.0})
    e = hs.effective(odd)
    assert e["rel"]["source"] == "correlation" and e["rel"]["group"] == "JP"
    assert hs.effective(odd, _choice(well="MPM-62", relation="measured"))["rel"]["source"] == "weak_measured"
    rows = [_raw(f"M{i}", "schrader", q, s, lift="JP") for i, (q, s) in enumerate([(600, 0.8), (900, 0.6), (1400, 0.4)])]
    rows.append(_raw("MPM-62", "schrader", 2000, -3.1, lift="JP"))
    assert "MPM-62" not in hs._correlations(rows)["JP"]["correlation"]["wells"]
    from woffl.assembly import prop_hist_client

    monkeypatch.setattr(hs.jobs, "get", lambda jid, kinds: _board_job([odd]))
    monkeypatch.setattr(prop_hist_client, "resolve_entry_user", lambda: "t@x")
    monkeypatch.setattr(prop_hist_client, "push_props", lambda w, v, entry_user: pytest.fail("must not write"))
    out = hs.save(schemas.HeaderSaveRequest(board_job_id="b", wells=[schemas.HeaderSaveWell(well="MPM-62", relation="measured")]))
    assert "implausible" in out["results"][0]["error"]
