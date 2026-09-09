"""WC envelopes retain liquid-rate physics, failures and the exact base solve."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server import schemas
from server.cache import clear_all_caches
from server.main import app
from server.services import solve, wc_uncertainty


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from woffl.assembly import databricks_client

    def forbidden(*args, **kwargs):
        pytest.fail("WC scenario tests must not read or write Databricks")

    clear_all_caches()
    monkeypatch.setattr(databricks_client, "execute_query", forbidden)
    monkeypatch.setattr(databricks_client, "execute_write", forbidden)
    yield
    clear_all_caches()


def prediction(oil=100., bhp=600.):
    return dict(qoil_std=oil, psu=bhp, fwat_bwpd=300., qnz_bwpd=2000.)


def test_bounds_include_interior_extrema(monkeypatch):
    def curved(_well, sp):
        return prediction(oil=100-10000*(sp.form_wc-.525)**2,
                          bhp=600+10000*(sp.form_wc-.475)**2)

    monkeypatch.setattr(solve, "solve_single", curved)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=schemas.SimParams(), uncertainty_points=10))
    assert result.sample_count == result.solved_count == 9
    assert result.complete
    assert result.oil.high == pytest.approx(100)  # interior, not either WC endpoint
    assert result.bhp.low == pytest.approx(600)
    assert result.oil.base == pytest.approx(93.75)


def test_wc_changes_oil_anchor_once_and_isolates_mutable_pvt(monkeypatch):
    from woffl.assembly import solopump

    seen = []

    def capture(**kw):
        seen.append(kw)
        return 600., False, 100., 300., 2000., .4

    monkeypatch.setattr(solopump, "jetpump_solver", capture)
    params = schemas.SimParams(form_wc=.8, qwf=1000, form_gor=420)
    original = params.model_dump_json()
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=params))
    assert result.wc_low == pytest.approx(.75)
    assert result.wc_high == pytest.approx(.85)
    assert len(seen) == 9
    for point, kw in zip(result.points, seen):
        assert kw["ipr_su"].qwf == pytest.approx(1000*(1-point.wc))
        assert kw["prop_su"].fgor == 420
    assert len({id(kw["prop_su"]) for kw in seen}) == 9
    assert len({id(kw["prop_su"].oil) for kw in seen}) == 9
    assert params.model_dump_json() == original


@pytest.mark.parametrize("wc", [0., .99])
def test_clipped_endpoints_are_unique_and_labeled(monkeypatch, wc):
    seen = []

    def capture(_well, sp):
        seen.append(sp.form_wc)
        return prediction()

    monkeypatch.setattr(solve, "solve_single", capture)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=schemas.SimParams(form_wc=wc)))
    assert result.clipped
    assert result.complete
    assert len(seen) == len(set(seen)) == 5
    assert min(seen) >= 0 and max(seen) <= .99
    assert wc in seen


def test_real_base_is_identical_to_regular_solve():
    params = schemas.SimParams()
    before = solve.solve_single("Custom", params)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=params))
    assert result.base_solved
    assert result.oil.base == before["qoil_std"]
    assert result.bhp.base == before["psu"]
    assert solve.solve_single("Custom", params) == before


def test_zero_uncertainty_solves_once(monkeypatch):
    calls = []

    def capture(_well, sp):
        calls.append(sp.form_wc)
        return prediction()

    monkeypatch.setattr(solve, "solve_single", capture)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=schemas.SimParams(), uncertainty_points=0))
    assert len(calls) == result.sample_count == 1
    assert result.oil.low == result.oil.base == result.oil.high
    assert result.bhp.low == result.bhp.base == result.bhp.high


def test_partial_failure_and_nan_never_become_finite_bounds(monkeypatch):
    def sometimes(_well, sp):
        if sp.form_wc == .5:
            raise solve.SolveFailure("convergence", "No base solution")
        if sp.form_wc < .5:
            return prediction(bhp=float("nan"))
        return prediction()

    monkeypatch.setattr(solve, "solve_single", sometimes)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=schemas.SimParams()))
    assert result.solved_count == 4
    assert not result.complete and not result.base_solved
    assert result.oil.base is result.bhp.base is None
    assert result.bhp.low == result.bhp.high == 600
    assert sum(p.error is not None for p in result.points) == 5
    assert "NaN" not in result.model_dump_json()


def test_all_failed_samples_have_no_range(monkeypatch):
    def failed(*args):
        raise solve.SolveFailure("no_solution", "No throat solution", 250)

    monkeypatch.setattr(solve, "solve_single", failed)
    result = wc_uncertainty.run(schemas.WcUncertaintyRequest(params=schemas.SimParams()))
    assert result.sample_count == 9
    assert result.solved_count == 0
    assert not result.complete
    assert result.oil is result.bhp is None


@pytest.mark.parametrize("width", [-1, 101, float("nan"), float("inf")])
def test_invalid_uncertainty_is_rejected(width):
    with pytest.raises(ValidationError):
        schemas.WcUncertaintyRequest(params=schemas.SimParams(), uncertainty_points=width)


@pytest.mark.parametrize("params", [{"model_as_water": True}, {"form_wc": .995}])
def test_unsupported_modes_return_422_without_solving(monkeypatch, params):
    monkeypatch.setattr(solve, "solve_single", lambda *a: pytest.fail("must reject before solve"))
    response = TestClient(app).post("/api/solve/wc-uncertainty", json={"params": params})
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "invalid"


def test_route_uses_cpu_limit_and_serializes_partial_failure(monkeypatch):
    from server import pool

    inside_slot = False

    @contextmanager
    def slot():
        nonlocal inside_slot
        inside_slot = True
        try:
            yield
        finally:
            inside_slot = False

    def bounded(_well, sp):
        assert inside_slot
        if sp.form_wc > .5:
            raise ValueError("cannot lift")
        return prediction()

    monkeypatch.setattr(pool, "cpu_slot", slot)
    monkeypatch.setattr(solve, "solve_single", bounded)
    response = TestClient(app).post("/api/solve/wc-uncertainty", json={"params": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["physics_model"] == schemas.MODEL_VERSION
    assert body["solved_count"] == 5
    assert not body["complete"]
    assert body["oil"] == {"low": 100, "base": 100, "high": 100}
    assert body["points"][-1]["oil"] is None
    assert body["points"][-1]["error"] == "cannot lift"
    assert not inside_slot
