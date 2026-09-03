"""E-Pad booster as a ``PadPlant`` - the frontier, the knobs, the wiring.

E-Pad joins S/I/M as a pad the optimizer can run. What makes it different, and
what these tests defend:

* Its capability frontier is **unimodal in flow**, which the I and M frontiers
  are not. Above ``ror_hi * hz_max/60`` no speed keeps the flow on the curve;
  BELOW ``ror_lo * hz_max/60`` the range FLOOR binds instead, the drive has to
  slow down, and deliverable pressure collapses with the square of the speed.
  Every inverse therefore scans then bisects the falling branch. A monotone
  bisection from zero flow - the shape the I/M inverses assume - reports 0.0
  here, so ``budget_at_pressure`` has its own test.
* Its configuration is not measured. No E-Pad SCADA point, no motor nameplate
  and no piping rating came with the vendor curve sheets, so the build,
  suction, speed cap and header cap are all per-run knobs. The tests pin that
  they actually move the answer AND that they never mutate the class default.

Pure static physics plus a faked NetworkOptimizer - no Databricks anywhere. If
this path grows a query these tests fail on the .env write-gate leak
(AGENTS.md section 3).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.main import app
from woffl.assembly.network_optimizer import WellConfig
from woffl.gui.e_pad_plant import INSTALLED_BUILD, PLANT, EPadPlant

# Installed build at its defaults: SM25000 26 stg, SG 1.02, 2,800 psi suction,
# 60 Hz cap, XRC 8,100-32,400 BPD at 60 Hz.
_KNEE = 8100.0  # ror_lo at 60 Hz — the frontier peak
_CEILING = 32400.0  # ror_hi at 60 Hz — the throughput limit
_SUCTION = 2800.0

_TOP_KEYS = {
    "pad",
    "coupling",
    "n_pumps",
    "sg",
    "suction_psi",
    "max_header_psi",
    "nameplate",
    "station",
    "pumps",
}


def leaves(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, node


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------


def test_plant_identity():
    assert PLANT.coupling == "free_pressure"
    assert PLANT.n_pump_options == []  # single machine
    assert PLANT.build.key == INSTALLED_BUILD
    assert PLANT.suction_psi() == _SUCTION
    assert PLANT.specific_gravity() == pytest.approx(1.02)
    assert PLANT.knee_flow() == pytest.approx(_KNEE)
    assert PLANT.flow_ceiling() == pytest.approx(_CEILING)


def test_frontier_is_unimodal_with_the_peak_at_the_range_floor_knee():
    # The whole reason this plant needs its own inverses. Below the knee the
    # recommended-range FLOOR forces a lower speed and deliverable pressure
    # collapses; above it the drive is pinned at 60 Hz and pressure falls with
    # flow the ordinary way.
    peak = PLANT.header_at_flow(_KNEE)
    assert peak == pytest.approx(4562.0, abs=2.0)
    # The rising branch starts above the model's 20 Hz speed floor: below
    # ~2,700 BPD the drive cannot slow far enough to keep the flow in range
    # at all, so the frontier is None there rather than merely low.
    assert PLANT.header_at_flow(2000.0) is None
    rising = [PLANT.header_at_flow(q) for q in (3000.0, 4000.0, 6000.0, _KNEE)]
    assert all(a < b for a, b in zip(rising, rising[1:])), rising
    falling = [
        PLANT.header_at_flow(q) for q in (_KNEE, 16000.0, 24000.0, _CEILING)
    ]
    assert all(a > b for a, b in zip(falling, falling[1:])), falling
    assert PLANT.header_at_flow(_CEILING) == pytest.approx(4005.0, abs=2.0)


def test_budget_at_pressure_survives_the_unimodal_frontier():
    # THE regression test for this plant. A plain monotone bisection from zero
    # flow tests ok(lo) first; on the collapsed low-flow branch that is False
    # for any useful pressure, so the budget would come back 0.0 and the
    # optimizer would report the pad infeasible at every header.
    assert PLANT.header_at_flow(100.0) is None  # ok(lo) really does fail
    for pressure in (3000.0, 3400.0, 3500.0, 4000.0):
        assert PLANT.budget_at_pressure(pressure) == pytest.approx(_CEILING, rel=1e-6)
    # Above the frontier's falling branch the budget starts to bite.
    assert PLANT.budget_at_pressure(4400.0) == pytest.approx(18861.0, abs=50.0)
    # And past the peak nothing is deliverable at all.
    assert PLANT.budget_at_pressure(4600.0) == 0.0


def test_budget_is_the_flow_where_the_frontier_crosses_the_pressure():
    for pressure in (4100.0, 4300.0, 4500.0):
        q = PLANT.budget_at_pressure(pressure)
        assert q > 0
        assert PLANT.header_at_flow(q) == pytest.approx(pressure, rel=1e-4)
        beyond = PLANT.header_at_flow(q * 1.01)
        assert beyond is None or beyond < pressure


def test_the_header_cap_not_the_pump_limits_the_sweep():
    # Worth stating plainly: at 3,400 psi this booster has pressure to spare
    # (its frontier sits at 4,000+ psi across the whole range), so the sweep
    # ceiling is the OPERATIONAL cap. Raising the cap raises the ceiling.
    floor, ceiling = PLANT.pressure_window()
    assert ceiling == pytest.approx(PLANT.max_header_psi)
    assert floor == pytest.approx(_SUCTION + 200.0)
    raised = EPadPlant(max_header_psi=4200.0)
    assert raised.pressure_window()[1] == pytest.approx(4200.0)


def test_flow_window_is_the_knee_to_the_ceiling():
    lo, hi = PLANT.flow_window()
    assert (lo, hi) == pytest.approx((_KNEE, _CEILING))


def test_flags_tell_recirc_from_over_capacity():
    assert PLANT.flags(20000.0) == {
        "in_range": True,
        "recirc": False,
        "over_capacity": False,
    }
    # Too little flow: the drive cannot slow far enough to keep it in range.
    assert PLANT.flags(500.0) == {
        "in_range": False,
        "recirc": True,
        "over_capacity": False,
    }
    # Too much: past the range ceiling at max speed.
    assert PLANT.flags(_CEILING * 1.2) == {
        "in_range": False,
        "recirc": False,
        "over_capacity": True,
    }


def test_warm_start_and_match_check_cap_at_the_operational_limit():
    assert PLANT.warm_start_psi() == pytest.approx(3400.0)
    # A measured PF anywhere in the operating band puts the raw frontier above
    # 4,000 psi; the match check must cap at the operational limit or every
    # well gets a spurious pass (the P0-7 family, exactly as on I-Pad).
    assert PLANT.header_at_flow(20000.0) > PLANT.max_header_psi
    assert PLANT.match_check_header(20000.0) == pytest.approx(PLANT.max_header_psi)
    # No measured PF at all falls back to the header setpoint.
    assert PLANT.match_check_header(0.0) == pytest.approx(3400.0)
    # Below the frontier knee the collapse is real and NOT capped away: a pad
    # running 3,000 BPD of PF really can only hold ~3,040 psi. This is where
    # E-Pad differs from I-Pad, whose frontier only ever falls with flow.
    assert PLANT.match_check_header(3000.0) == pytest.approx(
        PLANT.header_at_flow(3000.0)
    )


def test_envelope_reports_the_speed_and_amps_behind_each_frontier_point():
    rows = PLANT.envelope([500.0, 20000.0, _CEILING * 1.2])
    assert [r["feasible"] for r in rows] == [False, True, False]
    assert rows[0]["recirc"] is True
    assert rows[2]["recirc"] is False
    good = rows[1]
    assert good["max_discharge_psi"] == pytest.approx(PLANT.header_at_flow(20000.0))
    assert good["per_pump_bpd"] == pytest.approx(20000.0)
    pump = good["pumps"][0]
    assert pump["hz"] == pytest.approx(60.0)  # 60 Hz above the knee
    assert pump["amps"] > 0 and pump["amp_limit"] is None
    assert pump["dP"] == pytest.approx(good["max_discharge_psi"] - _SUCTION)


# ---------------------------------------------------------------------------
# Per-run configuration
# ---------------------------------------------------------------------------


def test_every_knob_moves_the_plant_and_none_mutates_the_class():
    alt = EPadPlant(
        "SN35000_18STG",
        suction_psi=2600.0,
        hz_max=55.0,
        max_header_psi=3400.0,
    )
    assert alt.build.key == "SN35000_18STG"
    assert alt.suction_psi() == 2600.0
    assert alt.hz_max == 55.0
    assert alt.max_header_psi == 3400.0
    # 950-series range 12,400-49,500 at 60 Hz, scaled to the 55 Hz cap.
    assert alt.knee_flow() == pytest.approx(12400.0 * 55.0 / 60.0)
    assert alt.flow_ceiling() == pytest.approx(49500.0 * 55.0 / 60.0)
    # The installed default is untouched by the alternative's construction.
    assert EPadPlant.max_header_psi == 3500.0
    assert PLANT.max_header_psi == 3500.0
    assert PLANT.build.key == INSTALLED_BUILD
    assert PLANT.flow_ceiling() == pytest.approx(_CEILING)


def test_the_alternative_build_moves_more_water_at_the_header():
    installed = PLANT.budget_at_pressure(3400.0)
    alt = EPadPlant("SN35000_18STG").budget_at_pressure(3400.0)
    assert alt > installed
    assert alt == pytest.approx(49500.0, rel=1e-6)


def test_a_lower_speed_cap_shrinks_the_budget():
    assert EPadPlant(hz_max=50.0).budget_at_pressure(3400.0) == pytest.approx(
        _CEILING * 50.0 / 60.0, rel=1e-6
    )


def test_an_amp_cap_shrinks_the_budget():
    free = PLANT.budget_at_pressure(3400.0)
    capped = EPadPlant(amp_limit=80.0).budget_at_pressure(3400.0)
    assert 0.0 < capped < free


def test_unknown_build_raises():
    with pytest.raises(ValueError, match="unknown E-Pad booster build"):
        EPadPlant("SM99999_9STG")


# ---------------------------------------------------------------------------
# curve_report
# ---------------------------------------------------------------------------


def test_curve_report_carries_the_contract_and_is_json_safe():
    rep = PLANT.curve_report(None)
    assert set(rep) == _TOP_KEYS
    assert rep["pad"] == "E"
    assert rep["suction_psi"] == _SUCTION
    assert rep["max_header_psi"] == PLANT.max_header_psi
    assert len(rep["pumps"]) == 1  # one machine
    for path, value in leaves(rep):
        assert isinstance(value, (bool, int, float, str, type(None))), path
    assert json.loads(json.dumps(rep)) == rep


def test_station_family_is_iso_speed_lines_with_the_cap_active():
    st = PLANT.curve_report()["station"]
    assert [c["hz"] for c in st["curves"]] == [45.0, 50.0, 55.0, 60.0]
    active = [c for c in st["curves"] if c["active"]]
    assert len(active) == 1 and active[0]["hz"] == 60.0
    # Station axis is DELIVERED header, so every line starts at suction + dP.
    assert active[0]["points"][0][1] > _SUCTION
    assert st["aor"] == [_KNEE, _CEILING]
    assert st["min_flow"] == pytest.approx(_KNEE)
    assert st["header_cap"] == PLANT.max_header_psi


def test_frontier_points_stop_where_the_range_does():
    front = PLANT.curve_report()["station"]["frontier"]
    assert "Recommended range limit" in front["label"]
    flows = [p[0] for p in front["points"]]
    assert max(flows) <= _CEILING + 1e-6
    # Every reported point is a real frontier value, and the peak is at the
    # knee, not at zero flow.
    for q, psi in front["points"]:
        assert psi == pytest.approx(PLANT.header_at_flow(q), rel=1e-9)
    peak_q = max(front["points"], key=lambda p: p[1])[0]
    assert peak_q == pytest.approx(_KNEE, rel=0.05)


def test_nameplate_says_the_model_is_not_scada_validated():
    np_ = PLANT.curve_report()["nameplate"]
    assert "NOT validated" in np_["validated"]
    assert "2,800" in np_["validated"]  # the suction assumption, stated
    assert "SM25000" in np_["model"] and "26 stg" in np_["model"]


# ---------------------------------------------------------------------------
# Optimizer wiring - the plant actually drives a free_pressure sweep
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    """Stands in for NetworkOptimizer; records the PF constraint it was built
    with so the test can read back the pressures the sweep tried."""

    seen: list = []

    def __init__(self, well_configs, pf, nozzles, throats, marginal_watercut=0.6):
        self.well_configs = well_configs
        self.power_fluid = pf
        self.marginal_watercut = marginal_watercut
        self.batch_results = {}
        type(self).seen.append(pf)

    def run_all_batch_simulations(self, max_workers=None):
        pass

    def get_pump_performance(self, well, nozzle, throat):
        return None


@pytest.fixture
def fake_core(monkeypatch):
    import woffl.assembly.network_optimizer as no_mod
    import woffl.assembly.optimization_algorithms as oa_mod
    import woffl.assembly.parallelism as common_mod

    class Optimizer(_FakeOptimizer):
        seen = []

    ns = SimpleNamespace(Optimizer=Optimizer, oil_at=lambda psi: 100.0)

    def fake_optimize(opt, method="milp", water_key=None):
        # One well, lift water = a tenth of the budget so the plant's cap is
        # exercised, oil shaped by the trial header.
        return [
            SimpleNamespace(
                well_name="E-048",
                recommended_nozzle="12",
                recommended_throat="B",
                predicted_lift_water=opt.power_fluid.total_rate * 0.1,
                predicted_oil_rate=ns.oil_at(opt.power_fluid.pressure),
            )
        ]

    monkeypatch.setattr(no_mod, "NetworkOptimizer", Optimizer)
    monkeypatch.setattr(no_mod, "reconcile_wells", lambda opt, results: {})
    monkeypatch.setattr(oa_mod, "optimize", fake_optimize)
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    return ns


def test_run_optimization_sweeps_the_e_pad_pressure_window(fake_core):
    import woffl.gui.pad_optimize as po

    # More oil at higher header - which is the truth on this pad, because the
    # booster has pressure to spare and the PF budget does not fall until well
    # above the operational cap. The sweep must therefore land ON the cap.
    fake_core.oil_at = lambda psi: psi / 10.0
    wells = [
        WellConfig(well_name="E-048", res_pres=1500, form_temp=70, jpump_tvd=4000)
    ]
    trials = []
    _results, _opt, meta = po.run_optimization(
        wells,
        PLANT,
        None,
        ["12"],
        ["B"],
        "milp",
        0.7,
        n_steps=6,
        progress=lambda *a: trials.append(a),
        refine_rounds=0,  # exact coarse grid for this pin
    )
    floor, ceiling = PLANT.pressure_window()
    assert len(trials) == 6
    assert [t[2] for t in trials] == pytest.approx(
        [floor + (ceiling - floor) * i / 5 for i in range(6)]
    )
    assert meta["header_psi"] == pytest.approx(ceiling)
    assert meta["suction_psi"] == _SUCTION
    assert meta["min_total_flow"] == pytest.approx(_KNEE)
    assert meta["in_range"] is True and meta["over_capacity"] is False
    # Every trial got the plant's budget as its PF cap.
    for pf in fake_core.Optimizer.seen:
        assert pf.total_rate == pytest.approx(
            PLANT.budget_at_pressure(pf.pressure), rel=1e-6
        )


def test_pad_plant_lookup_knows_e_and_run_lookup_honours_the_knobs():
    from server.services.optimizer_runs import (
        _PAD_DEFAULTS,
        _pad_plant,
        _pad_plant_for_run,
    )

    assert _PAD_DEFAULTS["E"] == {"n_pumps": None, "n_steps": 11}
    assert _pad_plant("E") is PLANT
    with pytest.raises(ValueError, match="expected S, I, M or E"):
        _pad_plant("Q")

    req = schemas.OptimizeRunRequest(
        kind="pad",
        pad="E",
        e_pad_build="SN35000_18STG",
        e_pad_suction_psi=2600.0,
        e_pad_hz_max=55.0,
        e_pad_max_header_psi=3400.0,
        e_pad_amp_limit_a=150.0,
    )
    configured = _pad_plant_for_run("E", req)
    assert configured is not PLANT
    assert configured.build.key == "SN35000_18STG"
    assert configured.suction_psi() == 2600.0
    assert configured.hz_max == 55.0
    assert configured.max_header_psi == 3400.0
    assert configured.amp_limit == 150.0
    # Other pads ignore the E knobs entirely.
    assert _pad_plant_for_run("I", req) is _pad_plant("I")


def test_run_request_defaults_match_the_plant_defaults():
    req = schemas.OptimizeRunRequest(kind="pad", pad="E")
    assert req.e_pad_build == INSTALLED_BUILD
    assert req.e_pad_suction_psi == _SUCTION
    assert req.e_pad_hz_max == 60.0
    assert req.e_pad_max_header_psi == EPadPlant.max_header_psi
    assert req.e_pad_amp_limit_a is None


# ---------------------------------------------------------------------------
# GET /api/optimize/pump-curve?pad=E
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_pump_curve_endpoint_serves_e_pad(client):
    r = client.get("/api/optimize/pump-curve?pad=E")
    assert r.status_code == 200
    body = schemas.PumpCurveResponse.model_validate(r.json())
    assert body.pad == "E"
    assert body.suction_psi == _SUCTION
    assert body.n_pump_options == []  # single machine
    assert len(body.pumps) == 1


def test_pump_curve_endpoint_honours_the_e_pad_knobs(client):
    r = client.get(
        "/api/optimize/pump-curve?pad=E&build=SN35000_18STG"
        "&suction_psi=2600&hz_max=55&max_header_psi=3400"
    )
    assert r.status_code == 200
    body = schemas.PumpCurveResponse.model_validate(r.json())
    assert body.suction_psi == 2600.0
    assert body.max_header_psi == 3400.0
    assert "SN35000" in body.nameplate.model
    assert body.station.aor == pytest.approx([12400.0, 49500.0])


def test_pump_curve_endpoint_rejects_e_pad_knobs_on_other_pads(client):
    r = client.get("/api/optimize/pump-curve?pad=I&suction_psi=2600")
    assert r.status_code == 422
    assert "E-Pad booster only" in r.json()["detail"]["message"]
    # And the S/I/M sheets still serve unchanged.
    assert client.get("/api/optimize/pump-curve?pad=I").status_code == 200
