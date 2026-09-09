"""Shared, unscaled throat-entry energy balance (field units).

The material path remains isothermal with the configured equilibrium PVT.
Wood Mach is diagnostic; choking follows the reachable energy turning point.
This is not a slip or finite-rate gas-release model.
"""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import math
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from woffl.flow.errors import ThroatEntryNoSolution
from woffl.flow.jetplot import JetBook, ThroatEntryChoked

# [LIBRARY change -> upstream PR to kwellis/woffl]
MODEL_VERSION = "entry-energy-v2"
PRESSURE_MIN = 50.0  # psig; a numerical bound, never a sonic criterion
PVT_STEP = 10.0  # psi; refine independently of the display book's grid
GC_PSI = 32.174 * 144
ENERGY_RTOL = 1e-6
_PATHS = ContextVar("entry_energy_paths", default=None)


def scoped_paths(fn):
    """Reuse immutable PVT paths within one solve/batch, isolated per caller."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if _PATHS.get() is not None:
            return fn(*args, **kwargs)
        token = _PATHS.set({})
        try:
            return fn(*args, **kwargs)
        finally:
            _PATHS.reset(token)
    return wrapped


def retired_mach(value):
    """Accept old call signatures while explicitly retiring their multiplier."""
    if not math.isfinite(value) or value < 1:
        raise ValueError("mach_crit must be finite and >= 1")
    if value != 1:
        warnings.warn(
            "mach_crit is retired by entry-energy-v1 and has no effect; "
            "refit calibrations made with the former Mach multiplier",
            DeprecationWarning, stacklevel=3,
        )


class MaterialPath:
    """Positive specific volume and its exact interpolant pressure integral.

    The same interpolated density is used for velocity, pressure work and
    the limiting derivative. Conserved stream mass is taken from ResMix at
    one reference condition. No shared mutable fluid is retained.
    """
    def __init__(self, temp, pressure_max, fluid, step=None):
        if pressure_max <= PRESSURE_MIN:
            raise ThroatEntryNoSolution("reservoir pressure is below the throat-entry pressure range")
        step = PVT_STEP if step is None else step
        self.pressure_max = float(pressure_max)
        grid = np.linspace(PRESSURE_MIN, pressure_max,
                           max(2, int(math.ceil((pressure_max-PRESSURE_MIN)/step))+1))
        prop = deepcopy(fluid)
        volume, sound = [], []
        for pressure in grid:
            prop.condition(float(pressure), temp)
            rho = prop.rho_mix()
            c = prop.cmix()
            if not (math.isfinite(rho) and rho > 0 and math.isfinite(c) and c > 0):
                raise ThroatEntryNoSolution("non-positive or non-finite throat-entry fluid properties")
            volume.append(1/rho)
            sound.append(c)
        self.mass_per_rate = prop.rho_mix() * sum(prop.insitu_volm_flow(1.0))
        self.volume = PchipInterpolator(grid, volume, extrapolate=False)
        self.dv = self.volume.derivative()
        self.work = self.volume.antiderivative()
        self.sound = PchipInterpolator(grid, sound, extrapolate=False)
        self.grid = grid

    def balance(self, psu, rate, ken, area):
        return Balance(self, psu, rate, ken, area)


def material_path(tsu, ipr, prop, psu=None):
    pressure_max = max(ipr.pres-10., psu or PRESSURE_MIN)
    paths = _PATHS.get()
    # Fluid composition does not change within a solve/batch; condition()
    # changes pressure/temperature caches only. Cache never outlives caller.
    key = (id(prop), tsu, pressure_max, PVT_STEP)
    if paths is None:
        return MaterialPath(tsu, pressure_max, prop)
    if key not in paths:
        paths[key] = MaterialPath(tsu, pressure_max, prop)
    return paths[key]


class Balance:
    def __init__(self, path, psu, rate, ken, area):
        if not (PRESSURE_MIN <= psu <= path.pressure_max):
            raise ThroatEntryNoSolution("suction lies outside the throat-entry pressure range")
        if not all(math.isfinite(v) for v in (area, ken, rate)) or area <= 0 or ken < 0 or rate < 0:
            raise ValueError("entry area must be positive; loss and flow must be non-negative")
        self.path, self.psu, self.rate = path, float(psu), rate
        self.flux = rate * path.mass_per_rate / area
        self.coeff = (1+ken)*self.flux**2 / 2
        self.work_su = float(path.work(psu))
        self.entry_ke = float(self.coeff*path.volume(psu)**2)

    def energy(self, pressure):
        return self.coeff*self.path.volume(pressure)**2 + GC_PSI*(self.path.work(pressure)-self.work_su)

    def derivative(self, pressure):
        return self.path.volume(pressure)*(GC_PSI+2*self.coeff*self.path.dv(pressure))

    def limit(self):
        """First minimum reached from suction; never jump to a later branch."""
        if self.derivative(self.psu) <= 0:
            return self.psu, "inlet_infeasible"
        if self.coeff > 0:
            roots = self.path.dv.solve(-GC_PSI/(2*self.coeff), extrapolate=False)
            for root in sorted((float(r) for r in roots if PRESSURE_MIN < r < self.psu), reverse=True):
                delta = min(.001, (self.psu-root)/2, (root-PRESSURE_MIN)/2)
                if self.derivative(root-delta) < 0 and self.derivative(root+delta) > 0:
                    return root, "energy_minimum"
        return PRESSURE_MIN, "pressure_bound"

    def operating_state(self, limit=None):
        pmin, reason = self.limit() if limit is None else limit
        emin = float(self.energy(pmin))
        tolerance = ENERGY_RTOL*max(self.entry_ke, 1.)
        if emin > tolerance or reason == "inlet_infeasible":
            raise ThroatEntryChoked("throat-entry energy balance cannot close on the reachable branch")
        if abs(emin) <= tolerance:
            pressure = pmin
        else:
            pressure = brentq(self.energy, pmin, self.psu, xtol=1e-7)
        v = float(self.flux*self.path.volume(pressure))
        return pressure, v, float(1/self.path.volume(pressure)), float(v/self.path.sound(pressure))

    def book(self):
        limit = self.limit()
        pressure, reason = limit
        grid = self.path.grid[(self.path.grid > pressure) & (self.path.grid < self.psu)][::-1]
        grid = np.unique(np.r_[self.psu, grid, pressure])[::-1]
        volume = self.path.volume(grid)
        vel = self.flux*volume
        snd = self.path.sound(grid)
        ke = self.coeff*volume**2
        ee = GC_PSI*(self.path.work(grid)-self.work_su)
        book = EntryBook.__new__(EntryBook)
        for name, values in dict(prs=grid, vel=vel, rho=1/volume, snd=snd, kde=ke,
                                 ede=ee, tde=ke+ee, mach=vel/snd, grad=self.derivative(grid)).items():
            setattr(book, name, list(values))
        book._arrays = {}
        book.limit_reason = reason
        book.limit_pressure = pressure
        book.minimum_energy = float(self.energy(pressure))
        book.model_version = MODEL_VERSION
        try:
            book.entry_state = self.operating_state(limit)
        except ThroatEntryChoked:
            book.entry_state = None
        return book


class EntryBook(JetBook):
    """JetBook interface with an exact root on its shared energy interpolant."""
    def dete_zero(self):
        if self.entry_state is None:
            raise ThroatEntryChoked("throat-entry energy balance cannot close on the reachable branch")
        return self.entry_state

    def copy(self):
        return deepcopy(self)


def entry_book(psu, tsu, ken, ate, ipr_su, prop_su):
    path = material_path(tsu, ipr_su, prop_su, psu)
    rate = ipr_su.oil_flow(psu, method="vogel")
    return rate, path.balance(psu, rate, ken, ate).book()


def suction_limit(tsu, ken, ate, ipr_su, prop_su):
    path = material_path(tsu, ipr_su, prop_su)
    def residual(psu):
        rate = ipr_su.oil_flow(psu, method="vogel")
        bal = path.balance(psu, rate, ken, ate)
        return float(bal.energy(bal.limit()[0]))
    lo, hi = PRESSURE_MIN, path.pressure_max
    if residual(hi) > 0:
        raise ThroatEntryNoSolution("throat-entry energy balance cannot close below reservoir pressure")
    psu = lo if residual(lo) <= 0 else brentq(residual, lo, hi, xtol=1e-7)
    rate = ipr_su.oil_flow(psu, method="vogel")
    return psu, rate, path.balance(psu, rate, ken, ate).book()
