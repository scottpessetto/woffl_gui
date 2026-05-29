"""Shared explainer widgets for the GUI.

Reusable Streamlit help blocks so the same authoritative text renders in more
than one place without drifting.
"""

import streamlit as st

_KCOEF_TEXT = """
The friction coefficients are **dimensionless energy-loss factors** in
the four pressure-drop stages of the jet pump. Each captures the fraction
of dynamic head lost to friction/turbulence in its section — higher value
means a less efficient (more lossy) component.

**The four coefficients:**

- **`knz` (nozzle)** — held fixed at 0.01. Loss as power fluid accelerates
  through the nozzle. Primarily affects PF flow rate and nozzle exit
  velocity. Default 0.01 is good when measured PF rates match the model.
- **`ken` (entrance / suction)** — calibrated, range [0.005, 0.20]. Loss
  as formation fluid enters the throat from the suction side. Higher
  `ken` means it's harder for produced fluid to flow into the throat →
  pump can't pull suction pressure as far down → **higher modeled BHP**.
  Affects drawdown directly.
- **`kth` (throat / mixing)** — calibrated, range [0.05, 1.0]. Loss
  during mixing of high-velocity power fluid with low-velocity formation
  fluid in the throat (constant-area mixing chamber). The biggest
  dissipative section in a jet pump. Higher `kth` means worse momentum
  transfer → less pressure built up downstream → pump needs higher
  suction (BHP). Affects both BHP and PF rate.
- **`kdi` (diffuser)** — calibrated, range [0.05, 1.0]. Loss as the
  mixed stream decelerates in the diverging diffuser, converting kinetic
  energy back into static pressure. Higher `kdi` means less pressure
  recovery → lower discharge pressure → pump needs more suction (BHP) to
  lift fluid out. Primarily affects BHP.

**Why these values change in practice:**

The defaults come from idealized Cunningham-style jet pump theory. Real
pumps deviate because of:

- **Wear / erosion** — sand or solids enlarging or roughening the
  throat/diffuser surfaces
- **Scale / deposits** — restricting flow areas, increasing turbulence
- **Manufacturing tolerances** — actual nozzle/throat geometry differs
  slightly from catalog
- **Fluid-property assumptions** — viscosity, density, or two-phase
  effects not captured by single-phase loss correlations
- **Geometry simplifications** — the model uses a 1D approximation; real
  flow has 3D structure

Calibrating per-well fits a one-number-per-component "wear / efficiency
factor" so the model matches actual measured BHP. The coefficients absorb
whatever the pump physics + simplified model couldn't predict from
spec-sheet geometry alone.
"""


def render_kcoef_explainer(expanded: bool = False) -> None:
    """Render the ken/kth/kdi/knz friction-coefficient explainer expander."""
    with st.expander("What do these coefficients represent?", expanded=expanded):
        st.markdown(_KCOEF_TEXT)
