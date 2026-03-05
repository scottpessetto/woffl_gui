import numpy as np
import pandas as pd
import pytest

from woffl.assembly.batchrun import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.batch_test is used

surf_pres = 210
jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure 3168
tsu = 80

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = Annulus(inn_pipe=tube, out_pipe=case)  # define the annulus

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

mpu_oil = BlackOil.schrader()  # class method
mpu_wat = FormWater.schrader()  # class method
mpu_gas = FormGas.schrader()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
e41_profile = WellProfile.schrader()

nozs = ["9", "10", "11", "12", "13", "14", "15", "16"]
thrs = ["X", "A", "B", "C", "D", "E"]

jp_list = BatchPump.jetpump_list(nozs, thrs)
e41_batch = BatchPump(surf_pres, tsu, rho_pf, ppf_surf, tube, e41_profile, e41_ipr, e41_res, wellname="MPE-41")

df = e41_batch.batch_run(jp_list, parallel=False)


def test_batch_row_count() -> None:
    assert len(df) == 48


def test_no_errors() -> None:
    assert (df["error"] == "na").all(), "All pumps should solve without error"


def test_sonic_count() -> None:
    assert df["sonic_status"].sum() == 13


def test_9X_reference() -> None:
    """Nozzle 9, Throat X — known sonic case."""
    row = df[(df["nozzle"] == "9") & (df["throat"] == "X")].iloc[0]
    assert row["qoil_std"] == pytest.approx(58.83, rel=0.01)
    assert row["totl_wat"] == pytest.approx(1947.73, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.935, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1316.07, rel=0.01)


def test_9D_reference() -> None:
    """Nozzle 9, Throat D — subsonic case."""
    row = df[(df["nozzle"] == "9") & (df["throat"] == "D")].iloc[0]
    assert row["qoil_std"] == pytest.approx(141.72, rel=0.01)
    assert row["totl_wat"] == pytest.approx(2557.22, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.227, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1197.78, rel=0.01)


def test_12B_reference() -> None:
    """Nozzle 12, Throat B — mid-range pump."""
    row = df[(df["nozzle"] == "12") & (df["throat"] == "B")].iloc[0]
    assert row["qoil_std"] == pytest.approx(202.04, rel=0.01)
    assert row["totl_wat"] == pytest.approx(4622.79, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.428, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1111.72, rel=0.01)


def test_16E_reference() -> None:
    """Nozzle 16, Throat E — largest pump."""
    row = df[(df["nozzle"] == "16") & (df["throat"] == "E")].iloc[0]
    assert row["qoil_std"] == pytest.approx(101.02, rel=0.01)
    assert row["totl_wat"] == pytest.approx(8014.69, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.0167, rel=0.02)
    assert row["psu_solv"] == pytest.approx(1255.86, rel=0.01)


def test_oil_always_positive() -> None:
    assert (df["qoil_std"] > 0).all()


def test_process_results() -> None:
    df_proc = e41_batch.process_results()
    assert "semi" in df_proc.columns
    assert "motwr" in df_proc.columns
    assert "molwr" in df_proc.columns
    assert df_proc["semi"].sum() > 0, "Should have at least one semi-finalist"


if __name__ == "__main__":
    print(df)
    df = e41_batch.process_results()
    print(df)
    e41_batch.plot_data(water="lift", curve=True)
    e41_batch.plot_derv(water="lift")
