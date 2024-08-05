"""Curve Fit Functions

Functions that fit the jet pump data of oil rate vs water for increasing pump sizes
"""

import numpy as np


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
    x = -1 / c * np.log(s / (c * b))
    x = max(x, 0)  # make sure s doesn't drop below zero
    return x
