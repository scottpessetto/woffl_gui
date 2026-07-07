"""Canonical Vogel IPR curve math for the GUI.

Single source of truth for the Vogel forward/inverse formulas used OUTSIDE
the RP-fit search in :mod:`woffl.assembly.ipr_analyzer`. Two GUI call sites
used to re-derive the same polynomial from scratch instead of going through
the library's :class:`woffl.flow.inflow.InFlow` (the canonical single test
point implementation):

- ``woffl.gui.workflow_steps.step2_review_ipr._template_to_vogel_coeffs``
  re-derived the forward qmax formula (``x*x`` instead of ``x**2``, but the
  same math) to re-visualize a hand-edited, re-uploaded template.
- ``woffl.gui.optimization_viz.create_ipr_comparison_pdf`` re-derived both
  the forward curve AND an inverse rate -> pwf solve for the PDF's
  "Current" operating point marker.

Both now import the helpers below instead of hand-rolling the polynomial.

NOT used by (and must never be swapped into) ``ipr_analyzer``'s RP-search
objective (``_normalized_curve_sse`` / ``_calculate_global_sse`` /
``estimate_reservoir_pressure``) or its R-squared helper
(``_calculate_r_squared``). Those implement a deliberately axis-normalized
anchor search (see that module's docstrings and the CLAUDE.md gotcha "The
Vogel RP fit objective is axis-normalized -- do NOT simplify it back to
rate-only SSE"); routing them through generic helpers is exactly the kind of
"simplification" that gotcha warns against, so they intentionally keep their
own inline arithmetic, untouched by this consolidation. The regression guard
is ``tests/test_ipr_analyzer.py`` ::
``TestEstimateReservoirPressure.test_flat_cloud_does_not_rail_to_cap``.
"""

from __future__ import annotations

import math
from typing import Optional, TypeVar

import numpy as np

from woffl.flow.inflow import InFlow

#: A plain float or anything that supports elementwise +, -, *, ** and /
#: (numpy arrays, pandas Series). All helpers here are pure arithmetic, so
#: they work unchanged for either.
Numeric = TypeVar("Numeric", float, np.ndarray)


def vogel_fraction(pwf: Numeric, pres: Numeric) -> Numeric:
    """Vogel q/qmax fraction at ``pwf`` for reservoir pressure ``pres``.

    ``1 - 0.2*(pwf/pres) - 0.8*(pwf/pres)**2``. Works elementwise on numpy
    arrays / pandas Series as well as plain floats.
    """
    ratio = pwf / pres
    return 1 - 0.2 * ratio - 0.8 * ratio**2


def vogel_qmax(qwf: Numeric, pwf: Numeric, pres: Numeric) -> Numeric:
    """Vogel max theoretical flowrate from one test point (qwf, pwf, pres).

    Delegates to :meth:`woffl.flow.inflow.InFlow.vogel_qmax` -- the
    library's canonical formula -- so there is exactly one implementation of
    the forward derivation. Also works elementwise on arrays/Series.
    """
    return InFlow.vogel_qmax(qwf, pwf, pres)


def vogel_rate(pwf: Numeric, qmax: Numeric, pres: Numeric) -> Numeric:
    """Forward Vogel rate at ``pwf`` given an already-known ``qmax``.

    Same math as ``InFlow.oil_flow(pwf, "vogel")``'s vogel branch, but takes
    ``qmax`` directly and works elementwise -- for callers building a whole
    curve (e.g. an array of pressures) without constructing one ``InFlow``
    per point.
    """
    return qmax * vogel_fraction(pwf, pres)


def vogel_pwf_from_rate(
    rate: Optional[float], qmax: Optional[float], pres: float
) -> Optional[float]:
    """Inverse Vogel: solve for pwf given a rate strictly below qmax.

    Returns ``None`` when there is no usable solution on the curve: ``rate``
    or ``qmax`` missing, ``rate`` non-positive, or ``rate`` at/above ``qmax``
    (mirrors the ``0 < rate < qmax`` guard the original call site used
    before ever computing the discriminant). Also returns ``None`` on a
    negative discriminant, which shouldn't occur once that guard holds but
    was checked at the original call site too.

    Note the operation order in the return expression --
    ``(-0.2 + sqrt(disc)) / 1.6 * pres`` -- is deliberate: floating-point
    division then multiplication is NOT bit-identical to multiplication
    then division, and this matches the pre-consolidation call site.
    """
    if rate is None or qmax is None or not (0 < rate < qmax):
        return None
    disc = 0.04 + 3.2 * (1 - rate / qmax)
    if disc < 0:
        return None
    return (-0.2 + math.sqrt(disc)) / 1.6 * pres
