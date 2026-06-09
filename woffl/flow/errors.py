"""Typed Exceptions for the Jet Pump Solvers

Gives callers a stable exception family to catch instead of bare ValueError,
IndexError, ZeroDivisionError, or math domain errors bubbling out of the
numerical internals.

All errors subclass ValueError, so existing ``except ValueError`` handlers
(batch runs, the GUI solver wrapper) keep working unchanged.
ThroatEntryNoSolution additionally subclasses IndexError because GUI callers
historically caught IndexError from ``JetBook._dete_zero``; the dual
inheritance keeps those handlers working while they migrate to typed names.
"""


class JetPumpError(ValueError):
    """Base class for jet pump solver failures."""


class ConvergenceError(JetPumpError):
    """An iterative solve (secant / Newton loop) failed to converge."""


class ThroatEntryNoSolution(JetPumpError, IndexError):
    """The throat-entry sweep produced no valid zero crossing of the total
    differential energy. Typical causes: GOR or suction pressure too low."""
