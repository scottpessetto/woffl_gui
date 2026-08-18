"""Tests for the pad Configure-screen additions (2026-07-21):

* ``pad_optimize.pf_what_if_rows`` / ``pf_what_if_totals`` — pure comparison
  math for the PF-pressure what-if (model all wells at their reviewed pumps
  at two forced delivered PF pressures).
* ``pad_optimize.pf_pressure_what_if`` — end-to-end against a fake
  NetworkOptimizer whose performance depends on the constraint pressure
  (the ``test_pad_optimize`` FakeOptimizer style).
* ``well_review_store.clone_entry`` — placeholder wells cloned from an
  existing reviewed well.
* ``pad_page._next_placeholder_name`` — placeholder naming.
"""

from types import SimpleNamespace

import pytest

import woffl.gui.pad_optimize as po
from woffl.assembly.network_optimizer import WellConfig


# ---------------------------------------------------------------------------
# Pure comparison math
# ---------------------------------------------------------------------------


def test_rows_projection_uses_test_anchored_ratio():
    rows = po.pf_what_if_rows(
        ["W1"],
        {"W1": ("12", "B")},
        base={"W1": (200.0, 2000.0)},
        scen={"W1": (220.0, 2200.0)},
        test_rates={"W1": (150.0, 1900.0)},
    )
    r = rows[0]
    assert r["d_oil"] == pytest.approx(20.0)
    assert r["d_pf"] == pytest.approx(200.0)
    # Projected = measured test oil × model ratio (bias cancels): 150 × 1.1
    assert r["projected_oil"] == pytest.approx(165.0)


def test_rows_handle_unsolved_and_missing_pump():
    rows = po.pf_what_if_rows(
        ["W1", "W2"],
        {"W1": ("12", "B"), "W2": None},
        base={"W1": (200.0, 2000.0), "W2": None},
        scen={"W1": None, "W2": None},
        test_rates={},
    )
    solved_one = rows[0]
    assert solved_one["oil_base"] == pytest.approx(200.0)
    assert solved_one["oil_scen"] is None
    assert solved_one["d_oil"] is None
    no_pump = rows[1]
    assert no_pump["pump"] == "—"
    assert no_pump["d_oil"] is None


def test_totals_exclude_unsolved_rows():
    rows = po.pf_what_if_rows(
        ["W1", "W2"],
        {"W1": ("12", "B"), "W2": ("11", "A")},
        base={"W1": (200.0, 2000.0), "W2": (100.0, 1000.0)},
        scen={"W1": (220.0, 2100.0), "W2": None},
        test_rates={"W1": (150.0, 1900.0)},
    )
    t = po.pf_what_if_totals(rows)
    assert t["n_solved"] == 1
    assert t["n_unsolved"] == 1
    # Only W1 (solved at BOTH pressures) participates in totals.
    assert t["oil_base"] == pytest.approx(200.0)
    assert t["oil_scen"] == pytest.approx(220.0)
    assert t["d_oil"] == pytest.approx(20.0)
    assert t["projected_d_oil"] == pytest.approx(165.0 - 150.0)


# ---------------------------------------------------------------------------
# End-to-end with a pressure-dependent fake optimizer
# ---------------------------------------------------------------------------


class FakeOptimizer:
    """Minimal NetworkOptimizer stand-in whose pump performance scales with
    the constraint pressure — so the what-if's two passes produce different
    numbers from the same table."""

    instances: list = []

    def __init__(self, well_configs, pf, nozzles, throats, marginal_watercut=0.6):
        self.well_configs = well_configs
        self.power_fluid = pf
        type(self).instances.append(self)

    def run_all_batch_simulations(self, max_workers=None):
        pass

    def get_pump_performance(self, well, nozzle, throat):
        p = self.power_fluid.pressure
        return {"oil_rate": p / 10.0, "lift_water": p}


@pytest.fixture
def fake_opt(monkeypatch):
    import woffl.assembly.network_optimizer as no_mod
    import woffl.assembly.parallelism as common_mod

    FakeOptimizer.instances = []
    monkeypatch.setattr(no_mod, "NetworkOptimizer", FakeOptimizer)
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    return FakeOptimizer


def _wells(*names):
    return [
        WellConfig(well_name=n, res_pres=1500, form_temp=70, jpump_tvd=4000)
        for n in names
    ]


def test_pf_pressure_what_if_end_to_end(fake_opt):
    configs = _wells("W1", "W2")
    current = {"W1": ("12", "B"), "W2": ("11", "A")}
    rows, totals = po.pf_pressure_what_if(
        configs, current, {"W1": (280.0, 2900.0)}, 3000.0, 3200.0
    )

    by_well = {r["well"]: r for r in rows}
    # oil = pressure/10 → 300 → 320 at each well.
    assert by_well["W1"]["oil_base"] == pytest.approx(300.0)
    assert by_well["W1"]["oil_scen"] == pytest.approx(320.0)
    assert by_well["W1"]["d_oil"] == pytest.approx(20.0)
    # Test-anchored projection: 280 × (320/300)
    assert by_well["W1"]["projected_oil"] == pytest.approx(280.0 * 320.0 / 300.0)
    # W2 has no test → model absolutes only, no projection.
    assert by_well["W2"]["projected_oil"] is None
    assert totals["d_oil"] == pytest.approx(40.0)
    # Two separate optimizer builds (one per forced pressure), and the header
    # was FORCED onto every well config in each pass.
    assert len(fake_opt.instances) == 2
    assert fake_opt.instances[0].power_fluid.pressure == pytest.approx(3000.0)
    assert fake_opt.instances[1].power_fluid.pressure == pytest.approx(3200.0)
    assert all(wc.ppf_surf_well == pytest.approx(3200.0) for wc in configs)


# ---------------------------------------------------------------------------
# Placeholder cloning
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Base vs Future comparison math
# ---------------------------------------------------------------------------


