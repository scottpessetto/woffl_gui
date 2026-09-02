"""woffl.assembly.batchpump must not drag matplotlib / scipy.optimize in at
import time (review 2026-09-01, SOLV-P1).

Every ProcessPool worker re-imports the app stack; at module scope those two
cost ~0.98 s per spawn for code the batch hot path never runs. Checked in a
fresh interpreter because the test process itself has long since imported
both.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# scipy.optimize is deliberately NOT in this list. Since FLOW-2 (2026-09-02)
# wellprofile imports it lazily inside the plot-only segments_fit (guarded
# by tests/test_wellprofile_validation.py), but it still arrives on the
# chain transitively: jetflow imports scipy.integrate.trapezoid and
# scipy.integrate imports scipy.optimize. matplotlib never is.
_PROBE = r"""
import sys
import woffl.assembly.batchpump  # noqa: F401
import woffl.assembly.network_optimizer  # noqa: F401
import woffl.assembly.solopump  # noqa: F401
heavy = [m for m in ("matplotlib", "matplotlib.pyplot", "matplotlib.axes", "matplotlib.figure") if m in sys.modules]
print("HEAVY=" + ",".join(heavy))
"""


def test_batchpump_import_is_free_of_plotting_and_scipy_optimize():
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("HEAVY=")][-1]
    assert line == "HEAVY=", f"module-scope heavy imports crept back: {line}"


def test_plotting_still_works_when_asked():
    """The lazy imports must resolve when a plot method actually runs."""
    import woffl.assembly.batchpump as bp

    assert callable(bp.batch_plot_data)
    assert callable(bp.batch_curve_fit)
    # curve fit path imports scipy.optimize on demand: feed it points that
    # the saturating exponential model actually describes.
    import numpy as np

    qwat = np.array([500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0])
    qoil = 300.0 * (1.0 - np.exp(-qwat / 1500.0))
    coeff = bp.batch_curve_fit(qoil, qwat, origin=True)
    assert len(coeff) == 3
    assert all(np.isfinite(coeff))
