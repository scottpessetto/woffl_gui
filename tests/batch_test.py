import numpy as np
import pandas as pd
import pytest

from woffl.assembly.batchpump import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m tests.batch_test is used

surf_pres = 210
jpump_tvd = 4065  # feet, interpolated off well profile
ppf_surf = 3168  # psi, power fluid surf pressure 3168
tsu = 80

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)  # define the wellbore

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
e41_batch = BatchPump(
    surf_pres,
    tsu,
    ppf_surf,
    wellbore,
    e41_profile,
    e41_ipr,
    e41_res,
    mpu_wat,
    jpump_direction="reverse",
    wellname="MPE-41",
)

df = e41_batch.batch_run(jp_list)


def test_batch_row_count() -> None:
    assert len(df) == 48


def test_no_errors() -> None:
    assert (df["error"] == "na").all(), "All pumps should solve without error"


def test_sonic_count() -> None:
    # 7 -> 6 with the 9b20c65 Beggs-Brill corrections (see note below): the
    # higher discharge back-pressure dropped one marginal combo below Mach 1
    # at the throat entry. Remaining sonic: 9X, 9A, 10X, 11X, 12X, 13X.
    # 6 -> 3 on 2026-09-01 (upstream_sync.md #16, Payne re-floor): the
    # discharge back-pressure rose again and 9A, 12X, 13X came off the
    # throat-entry choke. Remaining sonic: 9X, 10X, 11X.
    assert df["sonic_status"].sum() == 3
    sonic = df[df["sonic_status"]]
    assert sorted(sonic["nozzle"] + sonic["throat"]) == ["10X", "11X", "9X"]


# Reference values re-baselined 2026-06 after fixing the solopump psu secant
# (it previously used the bracket endpoints instead of the last two iterates,
# so every "solved" psu was a single linear interpolation with the discharge
# residual left 50-100 psid out of balance). The values below were verified by
# re-evaluating discharge_residual at each solved psu: all within +/-0.5 psid.
# mach_te also shifted from the FormGas.compress fix (absolute pressure + no
# state mutation), which corrected the mixture speed of sound.
#
# Re-baselined again 2026-06 after the BlackOil below-bubble compressibility fix
# (McCain Eq.5 now takes Rsb — solution GOR at the bubble point — instead of
# Rs at the current pressure; library patch). That raises sub-bubble oil
# compressibility, nudging the mixture sound speed and hence mach_te by ~1%.
# qoil_std / totl_wat / psu_solv are unchanged (the operating point is the same);
# only the derived mach_te moved. 9X and 12B crossed the 1% tolerance and were
# updated; 9D's shift stayed within tolerance.
#
# Re-baselined 2026-07-06 for commit 9b20c65, which wired the canonical
# Beggs-Brill corrections into the outflow traverse (Ek acceleration term,
# HL >= no-slip holdup floor, C >= 0 clamp — all previously UNDERSTATING the
# tubing gradient). The discharge back-pressure the pump works against rose,
# so psu_solv moves UP and qoil/totl_wat/mach_te move DOWN, scaling with total
# rate: 16E (~7,300 bwpd, highest velocity) moves most, 9X/12B stay in
# tolerance. Attribution verified by bisection: at 9b20c65 with ONLY
# outflow.py/twophase.py reverted, every pre-9b20c65 pin passes — the
# solopump/jetflow solver patches in the same commit were bit-identical here.
#
# Re-baselined 2026-07-06 (restored ee3886e Vogel IPR, clobbered by the
# woffl-2.0 sync): jetflow.throat_entry_zero_tde/throat_entry_mach_one now
# evaluate ipr_su.oil_flow(psu, method="vogel") instead of "pidx" (see
# docs/upstream_sync.md #15). At all four reference psu_solv values (1123 to
# 1323 psig — strictly between the E-41 anchor pwf=1049 and pres=1400), the
# Vogel curve sits above the straight-line PI chord, so qoil_std/totl_wat move
# UP. 9X's mach_te and 9D/16E's mach_te/psu_solv also shifted (operating point
# re-balances); 12B stayed within the existing 1% tolerance and its pin is
# unchanged.
#
# Re-baselined 2026-09-01 for docs/upstream_sync.md #16 (review FLOW-1): the
# slip holdup is re-floored at no-slip AFTER the Payne (1979) multiplier in
# outflow.beggs_diff_press. Payne had been pulling HL below lambda_L on the
# gassier / higher-velocity segments of the traverse, understating the static
# gradient, so the discharge back-pressure rose and psu_solv moves UP while
# qoil/totl_wat/mach_te move DOWN. Scales with how far off-design the pump
# is: 9X (sonic, one residual eval) is bit-identical; 12B -4% oil; 9D -9%;
# 16E (Mach 0.006, hydrostatics dominate) -51%. Verified the pre-fix pins
# pass with ONLY the one-line re-floor reverted.
#
# Re-baselined 2026-09-02 for docs/upstream_sync.md #17 (review PVT-F1):
# ResMix.cmix now takes the ACOUSTIC oil compressibility (Vasquez-Beggs at
# Rs(p), BlackOil.compress_acoustic) instead of the McCain material-balance
# co, which below the bubble point includes liberated-gas volume and gave
# pure-oil "sound speeds" of 100-900 ft/s. The mixture sound speed rises, so
# mach_te DROPS on every row: 9X 0.9572 -> 0.8935 (-6.7%), 9D 0.2133 ->
# 0.1930 (-9.5%), 12B 0.3784 -> 0.3486 (-7.9%), 16E 0.0061 -> 0.0054 (-11%).
# qoil_std / totl_wat / psu_solv are bit-identical (the operating point does
# not depend on cmix) and the sonic set is still 9X/10X/11X: the sonic flag
# comes from the throat-entry sweep hitting Mach 1 BEFORE the energy zero,
# and the E-41 gas fraction keeps those three choked.
#
# Re-baselined 2026-09-02 for docs/upstream_sync.md #19 (review PVT-F3): the
# gas z-factor is Dranchuk-Abu-Kassem instead of the "grad school" cubic
# (which drifted -10 % at ppr 4.5 and had the wrong sign of dz/dp beyond
# ppr ~3, overstating cg). Everything stays inside the existing tolerances;
# the pins are refreshed to the current values so they are not stale:
# mach_te 9X 0.8935 -> 0.8866 (-0.8%), 9D 0.1930 -> 0.1924, 12B unchanged,
# 16E 0.0054 -> 0.0053; qoil_std -0.04%, totl_wat -0.09%, psu_solv +0.01%
# (gas density at the suction moved ~1%). Sonic set unchanged.
#
# Re-baselined 2026-09-02 for docs/upstream_sync.md #23 (review FLOW-2): the
# production traverse now interpolates its node TVDs from the RAW survey
# instead of the greedy AIC/BIC segments fit, so both sides of the pump see
# the same TVD. On the Schrader preset (MPE-42) the fit sat 6.65 ft DEEP at
# the 6693 ft pump (4103.4 vs 4096.8 ft), a ~2.9 psi overstated production
# hydrostatic; removing it lowers the discharge back-pressure, so psu_solv
# moves DOWN and qoil/totl_wat/mach_te UP. 9X (sonic, one residual eval) is
# unchanged; 9D +0.3% oil, 12B +0.2%, 16E (Mach 0.006, hydrostatics
# dominate) 36.45 -> 38.37 (+5.3%). Verified by toggling: with the old fit
# spacing monkeypatched back in, every row reproduces the previous pins; the
# same-pass FLOW-12 friction blend (#27) moves nothing here (every segment is
# turbulent).
#
# Re-baselined 2026-09-02 for docs/upstream_sync.md #32 (review FLOW-9):
# psu_minimize now converges on the choke residual itself (|tee| <= 1 % of the
# entry kinetic energy) with tde interpolated AT Mach 1 instead of the nearest
# subsonic sweep point, so every choke floor moves by a fraction of a psi
# (worst 1.15 psi over the 48 E-41 pumps) and, because the floor is the
# secant's lower seed, every psu_solv shifts by <= 0.016 % (<= 0.21 psi) and
# qoil by <= 0.16 %. The three sonic rows return the floor itself: 9X
# 1323.31 -> 1323.40 psu, 59.01 -> 58.95 oil. Its mach_te 0.8869 -> 0.8488
# (-4.3 %) is the one pin outside tolerance: at a choke, mach_te is the last
# DISCRETE subsonic point of the 25-psi throat-entry sweep, so a floor shift of
# 0.09 psi that moves the clamp one step is worth ~4 % of Mach - the value is
# quantized, not a physics change. 9D / 12B / 16E round to the same pins.
# Bit-identity of the rest of this pass (FLOW-7 JetBook lists, SOLV-F9 seed
# reuse, SOLV-P3 secant reuse, SOLV-F2/F3, FLOW-5/6) was verified on all 48
# rows with FLOW-9 alone toggled off.


def test_9X_reference() -> None:
    """Nozzle 9, Throat X — known sonic case."""
    row = df[(df["nozzle"] == "9") & (df["throat"] == "X")].iloc[0]
    assert row["qoil_std"] == pytest.approx(58.95, rel=0.01)
    assert row["totl_wat"] == pytest.approx(1932.59, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.8488, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1323.40, rel=0.01)


def test_9D_reference() -> None:
    """Nozzle 9, Throat D — subsonic case."""
    row = df[(df["nozzle"] == "9") & (df["throat"] == "D")].iloc[0]
    assert row["qoil_std"] == pytest.approx(131.78, rel=0.01)
    assert row["totl_wat"] == pytest.approx(2454.76, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.1941, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1222.98, rel=0.01)


def test_12B_reference() -> None:
    """Nozzle 12, Throat B — mid-range pump."""
    row = df[(df["nozzle"] == "12") & (df["throat"] == "B")].iloc[0]
    assert row["qoil_std"] == pytest.approx(190.45, rel=0.01)
    assert row["totl_wat"] == pytest.approx(4456.81, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.3513, rel=0.01)
    assert row["psu_solv"] == pytest.approx(1136.49, rel=0.01)


def test_16E_reference() -> None:
    """Nozzle 16, Throat E — largest pump."""
    row = df[(df["nozzle"] == "16") & (df["throat"] == "E")].iloc[0]
    assert row["qoil_std"] == pytest.approx(38.37, rel=0.01)
    assert row["totl_wat"] == pytest.approx(7010.63, rel=0.01)
    assert row["mach_te"] == pytest.approx(0.0056, rel=0.02)
    assert row["psu_solv"] == pytest.approx(1350.57, rel=0.01)


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
