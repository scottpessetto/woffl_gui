import pytest

from woffl.pvt import FormWater

# "test" water properties...
water = FormWater(wat_sg=1)


def test_water_density() -> None:
    assert water.density == pytest.approx(62.4)


def test_water_viscosity() -> None:
    assert water.viscosity == pytest.approx(1.121034293, rel=1e-8)


def test_water_compressibility() -> None:
    assert water.compress == pytest.approx(3.211846719e-6, rel=1e-8)  # psi^-1


def test_water_tension() -> None:
    assert water.tension == pytest.approx(0.005, rel=0.01)  # lbf/ft


def test_wat_sg_bounds_inclusive() -> None:
    # Tripwire for the inclusive-bounds library patch (docstring range 0.5-1.5).
    FormWater(wat_sg=0.5)
    FormWater(wat_sg=1.5)
    with pytest.raises(ValueError):
        FormWater(wat_sg=1.6)
