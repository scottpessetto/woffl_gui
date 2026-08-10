"""_solve_combos / _solve_parallel - the combine study's fan-out layer.

A real ProcessPool child cannot see monkeypatches, so these tests swap the
pool class for ThreadPoolExecutor through the _EXECUTOR_CLS test seam; the
chunking, ordering, progress and fallback logic is identical either way.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

import server.services.sensitivity as sens
import woffl.gui.scotts_tools._common as common_mod


class _FakeParams:
    """Stands in for SimParams: model_copy just hands the update through."""

    def model_copy(self, update):
        return update


def _res(v: float) -> dict:
    """A SolveResult-shaped dict whose psu encodes the permutation id."""
    return {
        "psu": v,
        "qoil_std": v,
        "fwat_bwpd": 0.0,
        "qnz_bwpd": 1.0,
        "mach_te": 0.5,
        "sonic_status": False,
    }


@pytest.fixture()
def fake_solve(monkeypatch):
    """solve_single echoes the update's 'v'; v < 0 is a typed failure."""

    def solve_single(well, sp):
        v = sp["v"]
        if v < 0:
            raise sens.solve.SolveFailure("no_solution", "crafted failure")
        return _res(v)

    monkeypatch.setattr(sens.solve, "solve_single", solve_single)
    return solve_single


def _updates(n: int) -> list[dict]:
    return [{"v": float(i)} for i in range(n)]


def test_parallel_path_preserves_permutation_order(monkeypatch, fake_solve):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 4)
    monkeypatch.setattr(sens, "_EXECUTOR_CLS", ThreadPoolExecutor)
    ticks: list[tuple[int, int]] = []

    out = sens._solve_combos("W", _FakeParams(), _updates(120), lambda d, t: ticks.append((d, t)))

    assert [p["psu"] for p in out] == [float(i) for i in range(120)]
    # progress is monotone and finishes exactly at the total
    dones = [d for d, _ in ticks]
    assert dones == sorted(dones) and dones[-1] == 120
    assert all(t == 120 for _, t in ticks)


def test_per_point_failures_survive_the_parallel_path(monkeypatch, fake_solve):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 4)
    monkeypatch.setattr(sens, "_EXECUTOR_CLS", ThreadPoolExecutor)
    updates = _updates(60)
    updates[7] = {"v": -1.0}
    updates[41] = {"v": -1.0}

    out = sens._solve_combos("W", _FakeParams(), updates, None)

    assert out[7] == {"error": "no_solution"}
    assert out[41] == {"error": "no_solution"}
    assert out[8]["psu"] == 8.0  # neighbors untouched, order kept


class _NoPool:
    """Executor that must never be constructed - proves the serial branch."""

    def __init__(self, *a, **k):
        raise AssertionError("pool must not be used here")


def test_small_grids_stay_serial(monkeypatch, fake_solve):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 8)
    monkeypatch.setattr(sens, "_EXECUTOR_CLS", _NoPool)
    out = sens._solve_combos("W", _FakeParams(), _updates(sens._PARALLEL_MIN_RUNS - 1), None)
    assert [p["psu"] for p in out] == [float(i) for i in range(sens._PARALLEL_MIN_RUNS - 1)]


def test_ceiling_of_one_stays_serial(monkeypatch, fake_solve):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    monkeypatch.setattr(sens, "_EXECUTOR_CLS", _NoPool)
    out = sens._solve_combos("W", _FakeParams(), _updates(200), None)
    assert len(out) == 200 and out[199]["psu"] == 199.0


class _BrokenPool:
    """Executor whose construction fails - the spawn-refused case."""

    def __init__(self, *a, **k):
        raise OSError("no more processes")


def test_pool_failure_falls_back_to_serial(monkeypatch, fake_solve):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 4)
    monkeypatch.setattr(sens, "_EXECUTOR_CLS", _BrokenPool)
    ticks: list[tuple[int, int]] = []

    out = sens._solve_combos("W", _FakeParams(), _updates(80), lambda d, t: ticks.append((d, t)))

    # correct, ordered results despite the dead pool, and progress completed
    assert [p["psu"] for p in out] == [float(i) for i in range(80)]
    assert ticks[-1] == (80, 80)


def test_serial_progress_cadence_matches_the_old_loop(monkeypatch, fake_solve):
    # The deployed tier's behavior: every 25th solve plus the final one.
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    ticks: list[tuple[int, int]] = []
    sens._solve_combos("W", _FakeParams(), _updates(60), lambda d, t: ticks.append((d, t)))
    assert ticks == [(25, 60), (50, 60), (60, 60)]
