"""run_sensitivity's fan-out: every knob gets ITS OWN points back.

The knob table used to solve inside the per-knob loop. It now builds one
flat job list across all knobs, fans it out over the shared process pool,
and slices the results back by cursor. That slice is the dangerous part: a
cursor off by one knob does not crash, it silently attributes one knob's
excursions to another - a wrong answer on the page an engineer uses to
decide which knob to trust. These tests pin the mapping.

Threads, not processes, via the pool's _EXECUTOR_CLS seam - a real child
cannot see the monkeypatched solve.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import server.services.sensitivity as sens
import woffl.assembly.parallelism as common_mod
from server import pool, schemas


@pytest.fixture()
def echo_solve(monkeypatch):
    """solve_single echoes the knob field it was handed.

    Every knob moves a DIFFERENT SimParams field, so encoding the field's
    own value into the result lets a test prove each knob's points came from
    that knob's sweep and no other.
    """

    def solve_single(well, sp):
        return {
            "psu": float(sp.ken) * 1000.0,
            "qoil_std": float(sp.kth) * 1000.0,
            "fwat_bwpd": float(sp.form_gor),
            "qnz_bwpd": float(sp.jpump_tvd),
            "mach_te": 0.5,
            "sonic_status": False,
        }

    monkeypatch.setattr(sens.solve, "solve_single", solve_single)
    return solve_single


@pytest.fixture()
def thread_pool(monkeypatch):
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 2)
    monkeypatch.setattr(pool, "_EXECUTOR_CLS", ThreadPoolExecutor)
    pool.stop()
    started = pool.start()
    yield started
    pool.stop()


def _run(**kw):
    return sens.run_sensitivity("Custom", schemas.SimParams(), {}, **kw)


def test_pooled_and_serial_results_are_identical(echo_solve, thread_pool):
    """The whole point: fanning out changed no answer, only the wall clock."""
    assert thread_pool == 2
    pooled = _run()

    pool.stop()
    assert pool.workers() == 0
    serial = _run()

    assert pooled == serial


def test_every_knob_gets_its_own_sweep_back(echo_solve, thread_pool):
    """Each knob's points must reflect ITS field moving, not a neighbour's.

    ken and kth are separate knobs mapped onto separate result metrics by
    echo_solve, so a mis-sliced cursor shows up as a knob whose points do
    not vary in its own metric.
    """
    assert thread_pool == 2
    out = _run()
    by_id = {k["id"]: k for k in out["knobs"]}

    ken = by_id["ken"]
    psu = [p["psu"] for p in ken["points"] if "error" not in p]
    assert len(set(psu)) > 1, "the ken knob's own points did not vary with ken"

    kth = by_id["kth"]
    qoil = [p["qoil"] for p in kth["points"] if "error" not in p]
    assert len(set(qoil)) > 1, "the kth knob's own points did not vary with kth"


def test_point_counts_match_each_knob_sweep(echo_solve, thread_pool):
    """The cursor consumed exactly len(sweep.pairs) per knob, in order."""
    assert thread_pool == 2
    out = _run()
    sp = schemas.SimParams()
    for entry in out["knobs"]:
        knob = sens._BY_ID[entry["id"]]
        try:
            sweep = sens._knob_sweep(knob, sp, None)
        except (ValueError, KeyError):
            continue
        assert len(entry["points"]) == len(sweep.pairs), entry["id"]


def test_table_order_and_completeness_survive(echo_solve, thread_pool):
    """Every knob in the table, still in table order."""
    assert thread_pool == 2
    out = _run()
    assert [k["id"] for k in out["knobs"]] == [k.id for k in sens.KNOBS]


def test_no_pool_still_produces_the_whole_table(echo_solve):
    pool.stop()
    out = _run()
    assert [k["id"] for k in out["knobs"]] == [k.id for k in sens.KNOBS]
    assert any(k["points"] for k in out["knobs"])
