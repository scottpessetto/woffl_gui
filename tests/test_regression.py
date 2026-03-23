"""Regression tests that pin known-good outputs.

These use real woffl library objects to catch upstream library changes
that could silently break GUI behavior.
"""

import pytest

from woffl.assembly.batchpump import BatchPump
from woffl.assembly.solopump import jetpump_solver
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# ── Shared E-41 setup ──────────────────────────────────────────────────────


@pytest.fixture
def e41():
    """E-41 well objects for regression tests."""
    tube = Pipe(out_dia=4.5, thick=0.5)
    case = Pipe(out_dia=6.875, thick=0.5)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
    profile = WellProfile.schrader()
    inflow = InFlow(qwf=246, pwf=1049, pres=1400)
    oil = BlackOil.schrader()
    wat = FormWater.schrader()
    gas = FormGas.schrader()
    res_mix = ResMix(wc=0.894, fgor=600, oil=oil, wat=wat, gas=gas)
    prop_pf = wat
    return {
        "tube": tube,
        "wellbore": wellbore,
        "profile": profile,
        "inflow": inflow,
        "res_mix": res_mix,
        "prop_pf": prop_pf,
        "surf_pres": 210,
        "form_temp": 111,
        "ppf_surf": 3168,
    }


# ── Single pump solver regression ──────────────────────────────────────────


class TestSinglePumpRegression:
    def test_13a_returns_result(self, e41):
        jp = JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)
        result = jetpump_solver(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            jpump=jp,
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
        )
        assert result is not None
        psu, sonic, qoil, fwat, qnz, mach = result
        assert psu > 0
        assert qoil > 0
        assert fwat > 0
        assert qnz > 0

    def test_12b_returns_result(self, e41):
        jp = JetPump(nozzle_no="12", area_ratio="B", ken=0.03, kth=0.3, kdi=0.4)
        result = jetpump_solver(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            jpump=jp,
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
        )
        assert result is not None
        _, _, qoil, _, _, _ = result
        assert qoil > 0


# ── Batch run regression ───────────────────────────────────────────────────


class TestBatchRunRegression:
    def test_e41_batch_28_combos(self, e41):
        """7 nozzles x 4 throats = 28 rows."""
        jp_list = BatchPump.jetpump_list(
            ["9", "10", "11", "12", "13", "14", "15"],
            ["A", "B", "C", "D"],
        )
        bp = BatchPump(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
            wellname="E-41",
        )
        bp.batch_run(jp_list)
        assert len(bp.df) == 28

    def test_e41_some_sonic(self, e41):
        jp_list = BatchPump.jetpump_list(
            ["9", "10", "11", "12", "13", "14", "15"],
            ["A", "B", "C", "D"],
        )
        bp = BatchPump(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
            wellname="E-41",
        )
        bp.batch_run(jp_list)
        sonic_count = bp.df["sonic_status"].sum()
        assert sonic_count > 0, "Expected some sonic pumps"

    def test_e41_oil_always_positive(self, e41):
        jp_list = BatchPump.jetpump_list(
            ["9", "10", "11", "12", "13", "14", "15"],
            ["A", "B", "C", "D"],
        )
        bp = BatchPump(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
            wellname="E-41",
        )
        bp.batch_run(jp_list)
        successful = bp.df[bp.df["qoil_std"].notna()]
        assert (successful["qoil_std"] > 0).all()

    def test_e41_process_results(self, e41):
        """Process results should produce semi-finalists and curve fits."""
        jp_list = BatchPump.jetpump_list(
            ["9", "10", "11", "12", "13", "14", "15"],
            ["A", "B", "C", "D"],
        )
        bp = BatchPump(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
            wellname="E-41",
        )
        bp.batch_run(jp_list)
        bp.process_results()
        assert "semi" in bp.df.columns
        assert bp.df["semi"].any()

    def test_12b_has_positive_results(self, e41):
        """Pin that 12B specifically has positive output."""
        jp_list = BatchPump.jetpump_list(["12"], ["B"])
        bp = BatchPump(
            pwh=e41["surf_pres"],
            tsu=e41["form_temp"],
            ppf_surf=e41["ppf_surf"],
            wellbore=e41["wellbore"],
            wellprof=e41["profile"],
            ipr_su=e41["inflow"],
            prop_su=e41["res_mix"],
            prop_pf=e41["prop_pf"],
            wellname="E-41",
        )
        bp.batch_run(jp_list)
        row = bp.df.iloc[0]
        assert row["qoil_std"] > 0
        assert row["psu_solv"] > 0


# ── Vogel IPR regression ──────────────────────────────────────────────────


class TestVogelIPRRegression:
    def test_flow_at_pwf_equals_qwf(self):
        ipr = InFlow(qwf=750, pwf=500, pres=1700)
        q = ipr.oil_flow(500, "vogel")
        assert q == pytest.approx(750, rel=0.01)

    def test_flow_at_pres_is_zero(self):
        ipr = InFlow(qwf=750, pwf=500, pres=1700)
        q = ipr.oil_flow(1700, "vogel")
        assert q == pytest.approx(0, abs=1)

    def test_flow_at_zero_is_qmax(self):
        ipr = InFlow(qwf=750, pwf=500, pres=1700)
        qmax = ipr.oil_flow(0, "vogel")
        assert qmax > 750  # qmax should be greater than qwf

    def test_flow_monotonically_decreasing(self):
        ipr = InFlow(qwf=750, pwf=500, pres=1700)
        pressures = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
        flows = [ipr.oil_flow(p, "vogel") for p in pressures]
        for i in range(len(flows) - 1):
            assert flows[i] >= flows[i + 1]


# ── PVT Schrader vs Kuparuk regression ─────────────────────────────────────


class TestPVTRegression:
    def test_schrader_kuparuk_different_api(self):
        sch = BlackOil.schrader()
        kup = BlackOil.kuparuk()
        assert sch.oil_api != kup.oil_api

    def test_schrader_water_density(self):
        sch = FormWater.schrader()
        assert sch.density == pytest.approx(63.648, rel=0.01)

    def test_schrader_gas_sg(self):
        sch = FormGas.schrader()
        assert sch.gas_sg == pytest.approx(0.65, rel=0.01)


# ── WellProfile regression ─────────────────────────────────────────────────


class TestWellProfileRegression:
    def test_schrader_max_tvd(self):
        wp = WellProfile.schrader()
        assert wp.vd_ray[-1] == pytest.approx(4194, abs=50)

    def test_kuparuk_max_tvd(self):
        wp = WellProfile.kuparuk()
        assert wp.vd_ray[-1] == pytest.approx(9362, abs=50)

    def test_schrader_shallower_than_kuparuk(self):
        sch = WellProfile.schrader()
        kup = WellProfile.kuparuk()
        assert sch.vd_ray[-1] < kup.vd_ray[-1]
