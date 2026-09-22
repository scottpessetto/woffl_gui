"""Batch sweep shows each size once (user request 2026-09-22): the installed
size as the installed pump with its fitted losses, never a second time as a
"clean" reference-loss replacement."""

from server import schemas
from server.services import solve


def _pumps(out):
    return [(r["nozzle"], r["throat"], r.get("pump_state")) for r in out["rows"]]


def test_installed_size_is_not_duplicated_as_a_clean_replacement():
    sp = schemas.SimParams(nozzle_no="12", area_ratio="B", pump_state="installed",
                           nozzle_batch_options=["12", "13"], throat_batch_options=["B"])
    pumps = _pumps(solve.run_batch("Custom", sp))
    assert pumps.count(("12", "B", "installed")) == 1
    assert ("12", "B", "replacement") not in pumps
    assert ("13", "B", "replacement") in pumps


def test_a_replacement_bench_keeps_every_catalog_size():
    sp = schemas.SimParams(nozzle_no="12", area_ratio="B", pump_state="replacement",
                           nozzle_batch_options=["12", "13"], throat_batch_options=["B"])
    pumps = _pumps(solve.run_batch("Custom", sp))
    assert ("12", "B", "replacement") in pumps and ("13", "B", "replacement") in pumps
