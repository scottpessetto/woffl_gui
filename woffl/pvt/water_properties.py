"""Liquid-water properties from IAPWS, without a new runtime dependency.

Equations and numerical coefficients: IAPWS R7-97(2012), region 1,
Tables 2-4; IAPWS R12-08, Eqs. 10-12, Tables 1-2 (industrial form).
https://iapws.org/documents/release/IF97-Rev.download
https://iapws.org/technical-guidance/release/viscosity.download

Attribution: International Association for the Properties of Water and Steam.
These are pure-water properties, not a salinity correlation.
"""
from functools import lru_cache
import math

# [LIBRARY change -> upstream PR to kwellis/woffl]
PSI_MPA = 0.006894757293168
_R = 0.461526  # kJ/(kg K), IF97 specification
_TERMS = (
    (0, -2, .14632971213167), (0, -1, -.84548187169114),
    (0, 0, -3.756360367204), (0, 1, 3.3855169168385),
    (0, 2, -.95791963387872), (0, 3, .15772038513228),
    (0, 4, -.016616417199501), (0, 5, .00081214629983568),
    (1, -9, .00028319080123804), (1, -7, -.00060706301565874),
    (1, -1, -.018990068218419), (1, 0, -.032529748770505),
    (1, 1, -.021841717175414), (1, 3, -.00005283835796993),
    (2, -3, -.00047184321073267), (2, 0, -.00030001780793026),
    (2, 1, .000047661393906987), (2, 3, -.0000044141845330846),
    (2, 17, -7.2694996297594e-16), (3, -4, -.000031679644845054),
    (3, 0, -.0000028270797985312), (3, 6, -8.5205128120103e-10),
    (4, -5, -.0000022425281908), (4, -2, -6.5171222895601e-7),
    (4, 10, -1.4341729937924e-13), (5, -8, -4.0516996860117e-7),
    (8, -11, -1.2734301741641e-9), (8, -6, -1.7424871230634e-10),
    (21, -29, -6.8762131295531e-19), (23, -31, 1.4478307828521e-20),
    (29, -38, 2.6335781662795e-23), (30, -39, -1.1947622640071e-23),
    (31, -40, 1.8228094581404e-24), (32, -41, -9.3537087292458e-26),
)
_MU_TERMS = (
    (0, 0, .520094), (1, 0, .0850895), (2, 0, -1.08374), (3, 0, -.289555),
    (0, 1, .222531), (1, 1, .999115), (2, 1, 1.88797), (3, 1, 1.26613),
    (5, 1, .120573), (0, 2, -.281378), (1, 2, -.906851), (2, 2, -.772479),
    (3, 2, -.489837), (4, 2, -.257040), (0, 3, .161913), (1, 3, .257399),
    (0, 4, -.0325372), (3, 4, .0698452), (4, 5, .00872102),
    (3, 6, -.00435673), (5, 6, -.000593264),
)


def saturation_pressure(temp_k):
    """IF97 region 4, Eq. 30, saturation pressure (MPa) at temperature (K)."""
    n = (1167.0521452767, -724213.16703206, -17.073846940092,
         12020.82470247, -3232555.0322333, 14.91510861353,
         -4823.2657361591, 405113.40542057, -.23855557567849, 650.17534844798)
    theta = temp_k + n[8] / (temp_k - n[9])
    a = theta*theta + n[0]*theta + n[1]
    b = n[2]*theta*theta + n[3]*theta + n[4]
    c = n[5]*theta*theta + n[6]*theta + n[7]
    return (2*c / (-b + math.sqrt(b*b - 4*a*c)))**4


def viscosity_cp(temp_k, density_kgm3):
    """IAPWS industrial dynamic viscosity (cP), pure water."""
    t, d = temp_k / 647.096, density_kgm3 / 322.
    mu0 = 100*math.sqrt(t) / sum(h/t**i for i, h in enumerate((1.67752, 2.20462, .6366564, -.241605)))
    mu1 = math.exp(d*sum(h*(1/t-1)**i*(d-1)**j for i, j, h in _MU_TERMS))
    return mu0*mu1 / 1000.


@lru_cache(maxsize=8192)
def liquid_properties(pressure_mpa, temp_k):
    """Return immutable (density kg/m3, isothermal compressibility 1/MPa, cP).

    Only stable liquid in IF97 region 1 is supported. Exact float keys and
    a bounded cache amortize repeated PVT paths; no fluid objects are cached.
    """
    if not (273.15 <= temp_k <= 623.15 and 0 < pressure_mpa <= 100):
        raise ValueError("water PVT requires 273.15-623.15 K and 0-100 MPa absolute")
    if pressure_mpa < saturation_pressure(temp_k):
        raise ValueError("water PVT requires liquid pressure at or above saturation")
    x, y = 7.1 - pressure_mpa/16.53, 1386./temp_k - 1.222
    gp = gpp = 0.
    for i, j, n in _TERMS:
        if i:
            term = n*i*x**(i-1)*y**j
            gp -= term
            gpp += term*(i-1)/x
    volume = _R*temp_k*gp / 16530.
    density = 1/volume
    compressibility = -gpp/gp/16.53
    if density <= 0 or compressibility <= 0:
        raise ValueError("invalid liquid-water state")
    return density, compressibility, viscosity_cp(temp_k, density)
