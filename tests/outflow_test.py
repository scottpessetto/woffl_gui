import matplotlib.pyplot as plt
import numpy as np
import pytest

import woffl.flow.outflow as of
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# only works if the command python -m tests.outflow_test is used
# mirror the hysys stuff
md_list = np.linspace(0, 6000, 100)
vd_list = np.linspace(0, 4000, 100)

# well with 600 fgor, 90% wc, 100 bopd
mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()
form_gor = 600  # scf/stb
form_wc = 0.9
qoil_std = 100  # stbopd

test_prop = ResMix(form_wc, form_gor, mpu_oil, mpu_wat, mpu_gas)
wellprof = WellProfile(md_list, vd_list, 6000)
tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wellbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)

ptop = 350  # psig
ttop = 100  # deg f

md_seg, prs_ray, slh_ray = of.production_top_down_press(
    ptop, ttop, qoil_std, test_prop, wellbore, wellprof
)


def test_bottom_pressure() -> None:
    # Re-baselined 1754.88 -> 1856.89 for commit 9b20c65, which wired the
    # canonical Beggs-Brill corrections into the traverse: the Ek acceleration
    # term (beggs_ek existed but had no call site), the HL >= no-slip holdup
    # floor, and the C >= 0 incline-factor clamp. All three fixed a gradient
    # UNDERSTATEMENT, so the top-down traverse now ends ~102 psi higher.
    # Verified by bisection: reverting only outflow.py/twophase.py to the
    # pre-9b20c65 versions reproduces the old value exactly.
    # Re-baselined 1856.89 -> 1911.31 on 2026-09-01 (upstream_sync.md #16):
    # the slip holdup is now re-floored at no-slip AFTER the Payne (1979)
    # multiplier. On this 600 scf/stb, 90% WC fixture Payne had been pulling
    # HL below lambda_L on most segments, understating the static gradient.
    assert prs_ray[-1] == pytest.approx(1911.31, rel=0.001)


def test_top_pressure() -> None:
    assert prs_ray[0] == pytest.approx(350.0, rel=0.001)


def test_pressure_monotonic_increasing() -> None:
    assert np.all(np.diff(prs_ray) > 0), "Pressure should increase going down"


def test_array_lengths() -> None:
    assert len(prs_ray) == len(md_seg)
    assert len(slh_ray) == len(md_seg) - 1


def test_holdup_range() -> None:
    assert np.all(slh_ray >= 0) and np.all(
        slh_ray <= 1
    ), "Liquid holdup must be between 0 and 1"


def test_gas_free_segment_holds_full_liquid_holdup() -> None:
    """Guards upstream_sync.md patch 16 (review 2026-09-01, FLOW-1).

    Above the bubble point with fgor below Rsb there is NO free gas, so the
    no-slip holdup is exactly 1 and the slip holdup must be too. Before the
    re-floor, the Payne multiplier ran after the HL >= lambda_L floor and a
    gas-free segment came back at HL = 0.924 - 7.6% of the liquid column
    replaced by gas density that is not there, biasing low-GOR wells 7-20%
    optimistic on oil.
    """
    gas_free = ResMix(0.5, 100, mpu_oil, mpu_wat, mpu_gas)  # fgor 100 < Rsb 247
    gas_free.condition(2500, 100)  # above Pb 1750: all gas dissolved
    assert gas_free.nslh() == pytest.approx(1.0)
    _dp_stat, _dp_fric, slh = of.beggs_diff_press(
        2500, 100, tubing.inn_dia, tubing.inn_area, 0.0018, -100.0, -100.0, 300, gas_free
    )
    assert slh == pytest.approx(1.0)


def test_slip_holdup_never_below_no_slip_after_payne() -> None:
    """The canonical Beggs-Brill restriction HL >= lambda_L must survive the
    Payne (1979) inclination multiplier on an uphill gassy segment."""
    gassy = ResMix(0.9, 600, mpu_oil, mpu_wat, mpu_gas)
    for press in (400, 800, 1200):
        gassy.condition(press, 100)
        nslh = gassy.nslh()
        _dp_stat, _dp_fric, slh = of.beggs_diff_press(
            press, 100, tubing.inn_dia, tubing.inn_area, 0.0018, -100.0, -100.0, 100, gassy
        )
        assert nslh - 1e-9 <= slh <= 1.0


def test_zero_flow_traverse_does_not_raise() -> None:
    """Guards upstream_sync.md #26 (review 2026-09-01, FLOW-11) end to end.

    qoil_std = 0 gives vmix = 0 and froude = 0 in every segment; the holdup
    path used to raise a bare ZeroDivisionError there (only the friction
    factor had a zero-flow guard). The traverse must now run as a static
    column at the no-slip holdup.
    """
    md_seg0, prs0, slh0 = of.production_top_down_press(
        ptop, ttop, 0.0, test_prop, wellbore, wellprof
    )
    assert len(prs0) == len(md_seg0)
    assert np.isfinite(prs0).all()
    assert np.all(np.diff(prs0) > 0)
    assert np.all((slh0 > 0) & (slh0 <= 1))


def test_pipe_wall_must_leave_an_inner_diameter() -> None:
    """Guards upstream_sync.md #28 (review 2026-09-01, section 5): the wall
    is on both sides, so 2 * thick must be < out_dia. The old check was
    ``thick > out_dia`` and accepted a negative inner diameter."""
    with pytest.raises(ValueError, match="leaves no inner diameter"):
        Pipe(out_dia=1.0, thick=0.6)  # inn_dia would be -0.2
    with pytest.raises(ValueError, match="leaves no inner diameter"):
        Pipe(out_dia=1.0, thick=0.5)  # inn_dia would be exactly 0
    assert Pipe(out_dia=1.0, thick=0.49).inn_dia == pytest.approx(0.02)
    assert Pipe(out_dia=4.5, thick=0.271).inn_dia == pytest.approx(3.958)


if __name__ == "__main__":
    slh_ray_plot = np.append(
        slh_ray, np.nan
    )  # add a nan to make same length for graphing
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(md_seg, prs_ray, linestyle="--", color="b", label="Pressure")
    ax1.set_ylabel("Pressure, PSIG")
    ax1.set_xlabel("Measured Depth, Feet")
    ax2.plot(md_seg, slh_ray_plot, linestyle="-", color="r", label="Holdup")
    ax2.set_ylabel("Slip Liquid Holdup")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
    plt.show()

    print(f"Bottom Pressure: {round(prs_ray[-1], 2)} psi")
