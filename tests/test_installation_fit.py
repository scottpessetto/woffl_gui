"""Multi-installation well fit (server/services/installation_fit.py).

Real-solver checks use an offline WellConfig at a non-sonic operating point
(sonic tests have zero kth/kdi sensitivity by physics). Logic checks use a
fake per-test mapper so they run in milliseconds. No Databricks, no pool.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from server.main import app
from server.services import installation_fit as fm
from woffl.assembly.network_optimizer import WellConfig


def _cfg(nozzle="11", throat="C", wc=0.5, gor=250.0):
    cfg = WellConfig(well_name="MPE-42", res_pres=1500, form_temp=80, jpump_tvd=4000, qwf=1500, pwf=900)
    cfg.installed_nozzle, cfg.installed_throat, cfg.jpump_direction = nozzle, throat, "reverse"
    cfg.form_wc, cfg.form_gor = wc, gor
    return cfg


def _task(ppf=2000.0, **kw):
    return {"cfg": _cfg(**kw), "pwh": 200.0, "ppf": ppf}


# ---------------------------------------------------------------------------
# Real solver
# ---------------------------------------------------------------------------


def test_implicit_jacobian_matches_full_solve_differences():
    """d(bhp, oil, pf)/d(ipr, kth, kdi, fnz) from ONE solve plus residuals at
    fixed suction must agree with central differences of polished full solves."""
    task, phys = _task(), (1.0, 0.35, 0.45, 1.05)
    out = fm.evaluate_test(task, phys, want_jac=True)
    assert not out["sonic"], "fixture must be non-sonic to exercise implicit differentiation"
    fd = np.zeros((3, 4))
    for j, h in enumerate((0.02, 0.03, 0.03, 0.01)):
        up, dn = list(phys), list(phys)
        up[j] += h
        dn[j] -= h
        fd[:, j] = (np.array(fm.evaluate_test(task, up)["y"]) - np.array(fm.evaluate_test(task, dn)["y"])) / (2 * h)
    scale = np.abs(fd).max(axis=1, keepdims=True)
    assert np.all(np.abs(out["jac"] - fd) <= 0.12 * scale), (out["jac"], fd)


def test_continuation_lands_on_the_full_solve_root():
    """A predicted root plus Newton corrections must match a fresh solve."""
    task = _task()
    base = fm.evaluate_test(task, (1.0, 0.30, 0.40, 1.0), want_jac=True)
    new = (1.0, 0.33, 0.40, 1.02)
    dphys = np.array(new) - np.array((1.0, 0.30, 0.40, 1.0))
    guess = {"psu": base["y"][0] + float(base["jac"][0] @ dphys), "slope": base["state"]["slope"]}
    continued = fm.evaluate_test(task, new, guess=guess)
    fresh = fm.evaluate_test(task, new)
    assert continued["state"].get("stale_slope"), "continuation path was not taken"
    assert abs(continued["y"][0] - fresh["y"][0]) < 3.0
    assert abs(continued["y"][2] / fresh["y"][2] - 1) < 0.005


def test_synthetic_recovery_pools_installations():
    """Two installations generated from known losses: M2 recovers each
    installation's nozzle-area factor within two posterior sd, and beats M0."""
    rng = np.random.default_rng(7)
    true_fnz = (1.06, 0.97)
    obs = []
    for inst, (nz, th) in enumerate((("11", "C"), ("13", "C"))):
        for k in range(4):
            task = {"cfg": _cfg(nz, th, wc=0.45 + 0.03 * k, gor=250 + 20 * k),
                    "pwh": 180.0 + 5 * k, "ppf": 1880.0 + 60 * k}
            res = fm.evaluate_test(task, (1.0, 0.30, 0.40, true_fnz[inst]))
            assert "error" not in res
            bhp, oil, pf = res["y"]
            y = (bhp + rng.normal(0, 10), oil * (1 + rng.normal(0, 0.02)), pf * (1 + rng.normal(0, 0.01)))
            obs.append(fm.Observation(len(obs), inst, task, y, f"2026-0{inst + 1}-{10 + k}"))
    ladder = fm.fit_ladder(obs, ["A", "B"], models=("M0", "M2"))
    m2 = ladder["models"]["M2"]
    assert m2["cost"] < ladder["models"]["M0"]["cost"]
    got = {p["name"]: p for p in m2["params"]}
    for inst, truth in enumerate(true_fnz):
        p = got[f"fnz[{inst}]"]
        assert abs(p["value"] - truth) <= 2 * p["sd"], (p, truth)
        assert p["identified"]


def test_repeat_fits_are_deterministic():
    task = _task()
    obs = [fm.Observation(0, 0, task, (850.0, 800.0, 1850.0), "2026-01-10"),
           fm.Observation(1, 0, _task(ppf=2050.0), (840.0, 810.0, 1900.0), "2026-01-12")]
    a = fm.fit_model("M2", obs, 1)
    b = fm.fit_model("M2", obs, 1)
    assert np.array_equal(a.x, b.x) and a.cost == b.cost


# ---------------------------------------------------------------------------
# Logic with a fake per-test model
# ---------------------------------------------------------------------------


def _fake_mapper(jac_kth=(-400.0, 300.0, -200.0), jac_kdi=None, sonic=False, record=None):
    """Linear stand-in: y = base + J @ (phys - reference)."""
    def mapper(items):
        out = []
        for key, task, phys, want_jac, state, cols, guess in items:
            if record is not None:
                record.append((task["id"], phys))
            J = np.zeros((3, 4))
            J[:, fm.IPR] = (300.0, 400.0, -50.0)
            J[:, fm.KTH] = jac_kth
            J[:, fm.KDI] = jac_kdi if jac_kdi is not None else (-100.0, 50.0, 400.0)
            J[:, fm.FNZ] = (20.0, -10.0, 2000.0)
            if sonic:
                J[:, fm.KTH] = J[:, fm.KDI] = 0.0
            y = np.array(task["base"]) + J @ (np.array(phys) - np.array(fm.REFERENCE))
            res = {"y": tuple(float(v) for v in y), "liquid": float(y[1]) * 2, "sonic": sonic,
                   "state": {"sonic": sonic, "slope": [1.0, 0.0, 0.0, 0.0]}}
            if want_jac:
                res["jac"] = J
            out.append((key, res))
        return out
    return mapper


def _fake_obs(n=6, insts=(0, 1), oil_bias=0.0, start_month=1):
    obs = []
    for i in range(n):
        inst = insts[i * len(insts) // n]
        base = (800.0 + 5 * i, 400.0, 2000.0)
        cfg = _cfg()
        obs.append(fm.Observation(i, inst, {"id": i, "base": base, "cfg": cfg, "pwh": 200.0, "ppf": 2000.0},
                                  (base[0] + 8, base[1] * (1 + oil_bias), base[2] * 1.01),
                                  f"2026-{start_month + i:02d}-15"))
    return obs


def test_ipr_gate_blocks_pump_terms_unless_the_one_curve_is_refitted(monkeypatch):
    monkeypatch.setattr(fm, "ipr_consistency", lambda obs: dict(count=6, median_abs=0.4, median_signed=0.4, passes=False))
    obs = _fake_obs()
    blocked = fm.fit_ladder(obs, ["A", "B"], models=("M0", "M1", "M2"), mapper=_fake_mapper())
    assert set(blocked["skipped"]) == {"M1", "M2"} and "Refit the one IPR" in blocked["skipped"]["M1"]
    assert list(blocked["models"]) == ["M0"]
    refit = fm.fit_ladder(obs, ["A", "B"], models=("M0", "M1"), mapper=_fake_mapper(), refit_ipr=True)
    assert "M1" in refit["models"]
    assert any(p["name"] == "ipr_scale" for p in refit["models"]["M0"]["params"])


def test_collinear_losses_hold_kdi_and_say_so(monkeypatch):
    monkeypatch.setattr(fm, "ipr_consistency", lambda obs: dict(count=6, median_abs=0.0, median_signed=0.0, passes=True))
    kth = (-400.0, 300.0, -200.0)
    ladder = fm.fit_ladder(_fake_obs(), ["A", "B"], models=("M0", "M1"),
                           mapper=_fake_mapper(jac_kth=kth, jac_kdi=tuple(0.8 * v for v in kth)))
    assert ladder["losses"]["fit_kdi"] is False and "combination" in ladder["losses"]["reason"]
    assert [p["name"] for p in ladder["models"]["M1"]["params"]] == ["kth"]


def test_sonic_history_holds_both_losses_and_skips_their_rungs(monkeypatch):
    monkeypatch.setattr(fm, "ipr_consistency", lambda obs: dict(count=6, median_abs=0.0, median_signed=0.0, passes=True))
    ladder = fm.fit_ladder(_fake_obs(), ["A", "B"], models=("M0", "M1", "M2", "M3"), mapper=_fake_mapper(sonic=True))
    assert ladder["losses"]["fit_kth"] is False and "sonic" in ladder["losses"]["reason"]
    assert set(ladder["skipped"]) == {"M1", "M3"}
    assert all(p["physical"] == "fnz" for p in ladder["models"]["M2"]["params"])


def test_failed_tests_cost_more_than_a_three_sigma_miss():
    obs = _fake_obs(n=2, insts=(0,))
    problem = fm.Problem(obs, [], _fake_mapper())
    results = [{"error": "cannot lift"}, problem.run(np.zeros(0), False)[1]]
    r, failed = problem.residuals(results)
    assert failed[:3].all() and np.all(r[:3] == fm.FAILED_RESIDUAL)


def test_held_out_tests_never_enter_training(monkeypatch):
    """Every fold trains strictly before its origin minus the embargo."""
    monkeypatch.setattr(fm, "ipr_consistency", lambda obs: dict(count=12, median_abs=0.0, median_signed=0.0, passes=True))
    obs = _fake_obs(n=12, insts=(0, 1, 2))
    dates = {o.task["id"]: o.date for o in obs}
    ladder = fm.fit_ladder(obs, ["A", "B", "C"], models=("M0", "M1"), mapper=_fake_mapper())
    record: list = []
    cv = fm.cross_validate(obs, ["A", "B", "C"], ladder, mapper=_fake_mapper(record=record))
    assert cv["folds"] and cv["scores"]
    kinds = {f["kind"] for f in cv["folds"]}
    assert "changeout" in kinds
    for fold in cv["folds"]:
        origin = fm._day(fold["origin"])
        # The fold's training set is exactly the tests before the embargo.
        expected_train = sum((origin - fm._day(d)).days > fm.EMBARGO_DAYS for d in dates.values())
        assert fold["n_train"] == expected_train
    held = cv["held_predictions"]["M1"]
    assert held and all("origin" in h for h in held)
    for h in held:
        assert fm._day(obs[h["index"]].date) >= fm._day(h["origin"])


def test_one_standard_error_rule_prefers_the_simpler_model():
    ladder = {"models": {m: {"aicc": 0.0} for m in ("M0", "M1", "M2")}}
    cv = {"scores": {"M0": {"mean_loss": 3.0, "se": 0.1, "n": 40},
                     "M1": {"mean_loss": 2.5, "se": 0.4, "n": 40},
                     "M2": {"mean_loss": 2.4, "se": 0.3, "n": 40}}}
    chosen = fm.select_model(ladder, cv)
    assert chosen["model"] == "M1" and chosen["basis"] == "held_out_1se" and chosen["best"] == "M2"
    no_cv = fm.select_model({"models": {"M0": {"aicc": 5.0}, "M1": {"aicc": 1.0}}}, {"scores": {}})
    assert no_cv["model"] == "M1" and no_cv["basis"] == "aicc_unvalidated"


def test_fold_origins_split_at_changeouts_then_long_installations():
    obs = _fake_obs(n=12, insts=(0, 1))
    origins = fm.fold_origins(obs)
    assert (obs[6].date, "changeout") in origins and origins == sorted(origins)
    assert ("later_same_pump" in {k for _d, k in origins})
    assert len(origins) <= fm.MAX_FOLDS


# ---------------------------------------------------------------------------
# Assembly and API
# ---------------------------------------------------------------------------


def _history():
    tracker = pd.DataFrame([
        {"Well Name": "MPE-42", "Date Set": "2026-01-01T06:00:00Z", "Nozzle Number": "11",
         "Throat Ratio": "C", "Circ Direction": "reverse", "Tubing Diameter": 4.5},
        {"Well Name": "MPE-42", "Date Set": "2026-02-01T18:00:00Z", "Nozzle Number": "13",
         "Throat Ratio": "C", "Circ Direction": "reverse", "Tubing Diameter": 4.5},
    ])
    rows = []
    for month in (1, 2):
        for day in (5, 10, 15):
            rows.append(dict(well="MPE-42", wt_uid=f"{month}-{day}", WtDate=f"2026-{month:02d}-{day:02d}",
                             BHP=850., WtOilVol=780., WtTotalFluid=1560., form_wc=.5, fgor=250.,
                             lift_wat=1900., pf_press=2000.+day, whp=200., pf_source="annulus"))
    return tracker, pd.DataFrame(rows)


def test_prepare_uses_replay_assembly_and_honours_exclusions(monkeypatch):
    from server.services import pump_match as pm

    tracker, tests = _history()
    cfg = _cfg()
    monkeypatch.setattr(pm, "load_history", lambda well, request: (
        cfg, tracker, tests, "databricks", {}, "2026-03-01T00:00:00+00:00"))
    prep = fm.prepare("MPE-42", exclude_wt_uids=("1-5", "2-10"))
    ids = {prep["rows"][o.index]["wt_uid"] for o in prep["obs"]}
    assert ids == {"1-10", "1-15", "2-5", "2-15"}
    assert {o.inst for o in prep["obs"]} == {0, 1}
    # One saved oil IPR across both pumps; each test carries its own WC/GOR.
    oil = {round(o.task["cfg"].qwf * (1 - o.task["cfg"].form_wc), 6) for o in prep["obs"]}
    assert len(oil) == 1


def test_api_runs_a_read_only_job(monkeypatch):
    seen = {}

    def fake_run(job, well, request):
        seen["req"] = request
        return {"well": well, "selected": {"model": "M0"}, "validated_for_sizing": False}

    monkeypatch.setattr(fm, "run", fake_run)
    client = TestClient(app)
    r = client.post("/api/wells/MPE-42/installation-fit", json={"refit_ipr": True, "exclude_wt_uids": ["7"]})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get(f"/api/installation-fit/{job_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] == "done" and body["result"]["selected"]["model"] == "M0"
    assert seen["req"].refit_ipr is True and seen["req"].exclude_wt_uids == ["7"]
    assert client.post("/api/wells/bad;name/installation-fit", json={}).status_code == 404
    assert client.get("/api/installation-fit/nope").status_code == 404
    assert client.post("/api/wells/MPE-42/installation-fit", json={"months": 6}).status_code == 422
