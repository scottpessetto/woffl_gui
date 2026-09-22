"""Cost of PF on a fixed-speed pad + single-well pump decision
(woffl/gui/pad_marginal.py, server/services/pump_decision.py).

Fixture pad: header = 4000 - 0.05 * Q; each existing well draws
PF = 5H - 5000 and makes oil = 0.05H + 50 (10,000 BPD / 200 BOPD at 3000
psi). Two wells settle at 3000 psi; an extra 1,000 BPD drops the header to
3000 - 100/3 psi and costs each well 5/3 BOPD.
"""

from types import SimpleNamespace

import pytest

from woffl.gui import pad_marginal as pm

LEVELS = (2500.0, 2800.0, 2900.0, 3000.0, 3100.0, 3500.0)


def header_of_flow(q):
    return 4000.0 - 0.05 * q


def well(oil_slope=0.05, oil0=50.0, pf_slope=5.0, pf0=-5000.0):
    return [(h, oil_slope * h + oil0, pf_slope * h + pf0) for h in LEVELS]


def test_pfwc_lambda_round_trip():
    assert pm.pfwc_from_lambda(0.05) == pytest.approx(1 / 1.05)
    assert pm.pfwc_from_lambda(0.0) == 1.0
    assert pm.pfwc_from_lambda(None) is None
    assert pm.incremental_pfwc(50.0, 950.0) == pytest.approx(0.95)
    assert pm.incremental_pfwc(-5.0, 100.0) is None


def test_interpolation_and_cleaning():
    curve = pm.clean_curve([(3000.0, 200.0, 10000.0), (2500.0, 175.0, 7500.0), (2800.0, float("nan"), 1.0)])
    assert [p[0] for p in curve] == [2500.0, 3000.0]
    assert pm.at_header(curve, 2750.0) == pytest.approx((187.5, 8750.0))
    assert pm.at_header(curve, 9999.0) == (200.0, 10000.0)  # flat beyond the modeled range


def test_settle_on_the_curve():
    curves = {"A": well(), "B": well()}
    assert pm.settle(curves, 0.0, header_of_flow, 1000.0, 5000.0) == pytest.approx(3000.0, abs=0.5)
    assert pm.settle(curves, 1000.0, header_of_flow, 1000.0, 5000.0) == pytest.approx(3000.0 - 100 / 3, abs=0.5)


def test_pf_sensitivity_add_and_remove():
    curves = {"A": well(), "B": well()}
    s = pm.pf_sensitivity(curves, curves, header_of_flow, 3000.0, 1000.0, 5000.0, 1000.0)
    add, rem = s["add"], s["remove"]
    assert add["d_header_psi"] == pytest.approx(-100 / 3, abs=0.5)
    assert add["others_d_oil"] == pytest.approx(-10 / 3, abs=0.05)
    assert s["lambda"] == pytest.approx(10 / 3 / 1000, rel=0.02)
    assert s["pfwc"] == pytest.approx(1 / (1 + 10 / 3 / 1000), rel=1e-4)
    # the other wells give back PF as the header drops: net station flow < 1000
    assert add["others_d_pf"] == pytest.approx(-1000 / 3, rel=0.02)
    assert rem["d_header_psi"] > 0 and rem["others_d_oil"] > 0
    assert s["remove_lambda"] == pytest.approx(s["lambda"], rel=0.02)
    assert {r["well"] for r in add["wells"]} == {"A", "B"}


def test_pf_sweep_is_monotone_and_zero_at_today():
    curves = {"A": well(), "B": well()}
    sweep = pm.pf_sweep(curves, curves, header_of_flow, 3000.0, 1000.0, 5000.0, 2000.0, steps=4)
    assert [p["d_q"] for p in sweep] == [-2000.0, -1500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0, 1500.0, 2000.0]
    mid = sweep[4]
    assert mid["others_d_oil"] == 0.0 and mid["d_header_psi"] == 0.0
    oil = [p["others_d_oil"] for p in sweep]
    assert oil == sorted(oil, reverse=True)  # more draw, less oil for the others
    assert sweep[6]["others_d_oil"] == pytest.approx(-10 / 3, abs=0.05)
    assert set(sweep[0]["wells"]) == {"A", "B"} and not any(p["extrapolated"] for p in sweep)


def test_candidates_pay_the_header_drop():
    others = {"A": well(), "B": well()}
    base = well()  # target installed: same as the others
    upsize = {"pump": "13B", "pump_state": "replacement", "curve": well(oil0=150.0, pf0=-4000.0)}  # +100 oil, +1000 PF
    greedy = {"pump": "15D", "pump_state": "replacement", "curve": well(oil0=51.0, pf0=-2000.0)}  # +1 oil, +3000 PF
    same = {"pump": "12B", "pump_state": "installed", "curve": base}
    h0 = pm.settle({**others, "T": base}, 0.0, header_of_flow, 1000.0, 5000.0)
    assert h0 is not None
    rows = {r["pump"]: r for r in pm.score_candidates(others, base, [upsize, greedy, same],
                                                       header_of_flow, h0, 1000.0, 5000.0, 0.01)}
    assert rows["12B"]["net_oil"] == pytest.approx(0.0, abs=0.05)
    up = rows["13B"]
    assert up["d_header_psi"] < 0 and up["others_d_oil"] < 0
    assert up["net_oil"] == pytest.approx(up["d_oil"] + up["others_d_oil"])
    assert up["net_oil"] > 0 and up["beats_marginal"]
    assert rows["15D"]["net_oil"] < 0 and rows["15D"]["beats_marginal"] is False
    # a new well (empty slot) is charged every barrel it takes
    h_empty = pm.settle(others, 0.0, header_of_flow, 1000.0, 5000.0)
    assert h_empty is not None
    new = pm.score_candidates(others, None, [upsize], header_of_flow, h_empty, 1000.0, 5000.0, 0.01)[0]
    assert new["d_pf"] > 9000 and new["others_d_oil"] < up["others_d_oil"]


def test_pump_decision_job_with_fakes(monkeypatch):
    from server import schemas
    from server.services import optimizer_runs, pump_decision

    cfg = lambda n: SimpleNamespace(well_name=n, installed_nozzle="12", installed_throat="B")
    monkeypatch.setattr(optimizer_runs, "_build_configs", lambda *a, **k: [cfg("T"), cfg("A"), cfg("B")])
    monkeypatch.setattr(optimizer_runs, "_current_and_tests",
                        lambda names: ({n: ("12", "B") for n in names}, {n: (190.0, 9500.0) for n in names}))
    rate = lambda h: (0.05 * h + 50.0, 5.0 * h - 5000.0)
    monkeypatch.setattr(pump_decision, "_installed_at", lambda configs, levels, rho: {
        c.well_name: [(h, *rate(h)) for h in levels] for c in configs})
    monkeypatch.setattr(pump_decision, "_target_at", lambda c, levels, nz, th, rho: {
        ("12B", "installed"): [(h, *rate(h)) for h in levels],
        ("13B", "replacement"): [(h, 0.05 * h + 150.0, 5.0 * h - 4000.0) for h in levels]})
    plant = optimizer_runs._pad_plant("S")
    # three wells share header = 5000 - 0.05 Q: 5000 - 0.05 (15H - 15000) = H
    monkeypatch.setattr(plant, "header_at_flow", lambda q, n=None: 5000.0 - 0.05 * q)

    out = pump_decision._run({}, schemas.PumpDecisionRequest(target="T"))
    h0 = 5750.0 / 1.75
    assert out["target_role"] == "online" and out["header_psi"] == pytest.approx(h0, abs=1.0)
    assert out["model_pf_bpd"] == pytest.approx(3 * (5 * h0 - 5000), rel=1e-3)
    sens = out["sensitivity"]
    assert sens["add"]["d_header_psi"] < 0 and sens["lambda"] > 0
    assert {r["well"] for r in sens["add"]["wells"]} == {"A", "B"}  # the target is not an "other" well
    top = out["candidates"][0]
    assert top["pump"] == "13B" and top["net_oil"] > 0 and top["others_d_oil"] < 0
    assert out["baseline"]["pump"] == "12B"
    assert len(out["sweep"]) == 21 and out["sweep"][10]["d_q"] == 0.0


def test_pad_wide_mode_costs_every_well_and_sizes_none(monkeypatch):
    """No target: the +/- step is extra draw anywhere on the pad, charged to
    every producing well, and no pump is sized."""
    from server import schemas
    from server.services import optimizer_runs, pump_decision

    cfg = lambda n: SimpleNamespace(well_name=n, installed_nozzle="12", installed_throat="B")
    monkeypatch.setattr(optimizer_runs, "_build_configs", lambda *a, **k: [cfg("T"), cfg("A"), cfg("B")])
    monkeypatch.setattr(optimizer_runs, "_current_and_tests",
                        lambda names: ({n: ("12", "B") for n in names}, {n: (190.0, 9500.0) for n in names}))
    rate = lambda h: (0.05 * h + 50.0, 5.0 * h - 5000.0)
    monkeypatch.setattr(pump_decision, "_installed_at", lambda configs, levels, rho: {
        c.well_name: [(h, *rate(h)) for h in levels] for c in configs})
    sized = []
    monkeypatch.setattr(pump_decision, "_target_at", lambda *a, **k: sized.append(a) or {})
    plant = optimizer_runs._pad_plant("S")
    monkeypatch.setattr(plant, "header_at_flow", lambda q, n=None: 5000.0 - 0.05 * q)

    out = pump_decision._run({}, schemas.PumpDecisionRequest(target=None))
    assert out["target"] is None and out["target_role"] == "pad"
    assert sized == [] and out["candidates"] == [] and out["baseline"] is None
    assert {r["well"] for r in out["sensitivity"]["add"]["wells"]} == {"T", "A", "B"}
    assert out["sensitivity"]["add"]["others_d_oil"] < 0 and len(out["sweep"]) == 21
