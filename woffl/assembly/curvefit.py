"""Curve Fit Functions

Functions that fit the jet pump data of oil rate vs water for increasing pump sizes.
Consider changing the name from curvefit.py to curvemodel.py (not quite as elegent)
"""

import numpy as np
from scipy import optimize as opt


def exp_model(x: float, a: float, b: float, c: float) -> float:
    """Exponential Curve Fit

    Args:
        x (float): Water Rate, bwpd
        a (float): Asymptote of the Curve
        b (float): Constant
        c (float): Constant

    Returns
        y (float): Oil Rate, bopd
    """
    return a - b * np.exp(-c * x)


def exp_deriv(x: float, b: float, c: float) -> float:
    """Derivative of Exponential Curve Fit

    Args:
        x (float): Water Rate, bwpd
        b (float): Constant
        c (float): Constant

    Returns
        s (float): Marginal Oil - Water Ratio, bbl/bbl
    """
    return c * b * np.exp(-c * x)


def rev_exp_deriv(s: float, b: float, c: float) -> float:
    """Reverse Derivative of Exponential Curve Fit

    Args:
        s (float): Marginal Oil - Water Ratio, bbl/bbl
        b (float): Constant
        c (float): Constant

    Returns
        x (float): Water Rate, bwpd
    """
    if s == 0:
        s = 0.00001
    x = -1 / c * np.log(s / (c * b))
    x = max(x, 0)  # make sure s doesn't drop below zero
    return x


def batch_curve_fit(qoil_filt: np.ndarray, qwat_filt: np.ndarray, origin: bool = True) -> tuple[float, float, float]:
    """Batch Curve Fit

    Curve fit the filtered datapoints from the Batch Results

    Args:
        qoil_filt (list): Filtered Oil Array, bopd
        qwat_filt (list): Filtered Water Array, bwpd
        origin (bool): Add point to encourage intercepting at (0,0)

    Returns:
        coeff (float): a, b and c coefficients for curve fit
    """
    # add a point at 0,0 to force intercepting origin
    if origin:
        qoil_filt = np.append(qoil_filt, 0.0)
        qwat_filt = np.append(qwat_filt, 0.0)

    initial_guesses = [max(qoil_filt), max(qoil_filt), 0.001]
    coeff, _ = opt.curve_fit(exp_model, qwat_filt, qoil_filt, p0=initial_guesses)
    return coeff
