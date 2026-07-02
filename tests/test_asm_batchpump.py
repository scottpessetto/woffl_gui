"""Batchpump Smoke Tests

Smoke tests for batch_run, process_results, and search_run.
Uses the Schrader preset well with a small jet pump grid to keep runtime short.
Asserts physically reasonable outputs — not exact values.
"""

import numpy as np
import pytest

from tests.asm_helper import jp_list, make_well
from woffl.assembly.batchpump import gradient_back
from woffl.geometry.jetpump import JetPump


def test_gradient_back_equal_water_is_nan() -> None:
    """Tripwire for the equal-water zero-denominator guard (library patch).

    Two semi-finalists with identical total water -> zero denominator in the
    backward gradient. The guard returns NaN instead of inf / ZeroDivisionError.
    """
    grad = gradient_back(np.array([10.0, 20.0]), np.array([100.0, 100.0]))
    assert np.isnan(grad[-1])

# ---- fixture ----


@pytest.fixture(scope="module")
def e41_batch():
    """Single well with batch_run and process_results completed."""
    well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
    well.batch_run(jp_list)
    well.process_results()
    return well


# ---- batch_run tests ----


class TestBatchRun:
    """Smoke tests for BatchPump.batch_run"""

    def test_returns_results(self, e41_batch):
        df = e41_batch.df
        assert len(df) == len(jp_list)

    def test_some_pumps_converge(self, e41_batch):
        df = e41_batch.df
        valid = df.dropna(subset=["qoil_std"])
        assert len(valid) > 0, "No pumps converged"

    def test_oil_rate_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["qoil_std"])
        assert (df["qoil_std"] > 0).all()

    def test_suction_pressure_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["psu_solv"])
        assert (df["psu_solv"] > 0).all()

    def test_water_rates_positive(self, e41_batch):
        df = e41_batch.df.dropna(subset=["totl_wat"])
        assert (df["totl_wat"] > 0).all()
        assert (df["lift_wat"] > 0).all()


# ---- process_results tests ----


class TestProcessResults:
    """Smoke tests for BatchPump.process_results"""

    def test_semi_finalists_exist(self, e41_batch):
        assert e41_batch.df["semi"].sum() > 0

    def test_gradients_finite(self, e41_batch):
        semi = e41_batch.df[e41_batch.df["semi"]]
        assert semi["motwr"].notna().all()
        assert semi["molwr"].notna().all()

    def test_curve_fit_coefficients(self, e41_batch):
        assert len(e41_batch.coeff_totl) == 3
        assert len(e41_batch.coeff_lift) == 3


# ---- update_press tests (guards a LIBRARY patch) ----


class TestUpdatePress:
    """Tripwire for the dotted-path update_press fix (upstream PR to kwellis/woffl).

    setattr does not traverse a dotted path, so update_press("reservoir", x)
    used to create a junk attribute "ipr_su.pres" and leave the real
    ipr_su.pres untouched (silent no-op). Goes red if that patch is lost.
    """

    def test_reservoir_actually_updates_ipr(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.update_press("reservoir", 1750)
        assert well.ipr_su.pres == 1750
        # and no junk flat attribute was created
        assert not hasattr(well, "ipr_su.pres")

    def test_wellhead_and_powerfluid_still_work(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.update_press("wellhead", 250)
        well.update_press("powerfluid", 3200)
        assert well.pwh == 250
        assert well.ppf_surf == 3200

    def test_invalid_kind_raises(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        with pytest.raises(ValueError):
            well.update_press("bogus", 100)


# ---- process_results idempotency (guards a LIBRARY patch) ----


class TestProcessResultsIdempotent:
    """Tripwire for the process_results re-merge fix (upstream PR to kwellis/woffl).

    A second process_results() used to re-merge motwr/molwr into a df that
    already had them, so pandas suffixed them _x/_y and the plain columns
    vanished. Goes red if that patch is lost.
    """

    def test_double_call_keeps_motwr_molwr(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.batch_run(jp_list)
        well.process_results()
        well.process_results()  # second call must not corrupt the columns
        cols = set(well.df.columns)
        assert {"motwr", "molwr"} <= cols
        assert not ({"motwr_x", "motwr_y", "molwr_x", "molwr_y"} & cols)
        semi = well.df[well.df["semi"]]
        assert semi["motwr"].notna().all()
        assert semi["molwr"].notna().all()


# ---- search_run tests ----


class TestSearchRun:
    """Smoke tests for BatchPump.search_run"""

    def test_returns_catalog_pump(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert len(df) == 1
        assert df["nozzle"].iloc[0] != "opt"
        assert df["throat"].iloc[0] != "opt"

    def test_oil_rate_positive(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        seed = JetPump("12", "B")
        df = well.search_run(seed, lift_cost=0.03)
        assert df["qoil_std"].iloc[0] > 0

    def test_does_not_overwrite_batch_df(self):
        well = make_well("MPE-41", qwf=246, pwf=1049, pres=1400, wc=0.894, fgor=600)
        well.batch_run(jp_list)
        batch_len = len(well.df)
        seed = JetPump("12", "B")
        well.search_run(seed, lift_cost=0.03)
        assert len(well.df) == batch_len, "search_run overwrote self.df"
