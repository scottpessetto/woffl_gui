"""One "Pad Optimization" app mode: pad selector → the shared pad page.

Completes R-1 §3 (docs/code_review_2026-07-01.md): S/I/M were three separate
app modes even after their pages unified onto :func:`woffl.gui.pad_page.
run_pad_page`. A single mode with a pad selector keeps the mode strip short as
pads are added — a new pad is one ``PadSpec`` + a plant subclass (or
``FixedHeaderPlant``) plus one entry in ``_PADS`` here.

Specs are imported lazily so picking S doesn't pay for I/M module import.
"""

from __future__ import annotations

import streamlit as st

_PADS = ["S-Pad", "I-Pad", "M-Pad"]
_SHADOW_KEY = "pad_hub_pad_shadow"


def _spec_for(pad: str):
    if pad == "S-Pad":
        from woffl.gui.s_pad_page import SPEC
    elif pad == "I-Pad":
        from woffl.gui.i_pad_page import SPEC
    elif pad == "M-Pad":
        from woffl.gui.m_pad_page import SPEC
    else:  # unreachable via the radio; explicit beats a NameError
        raise ValueError(f"Unknown pad: {pad}")
    return SPEC


def run_pad_hub() -> None:
    from woffl.gui.pad_page import run_pad_page

    # Restore the selection from a non-widget shadow key: leaving this mode
    # garbage-collects the radio's widget state, and without the shadow a
    # detour (Pad Optimization → Single Well → back) would snap the selector
    # to S-Pad. Streamlit ignores ``index`` when widget state survived, so
    # this is a no-op then (the CLAUDE.md widget-GC pattern).
    shadow = st.session_state.get(_SHADOW_KEY, _PADS[0])
    idx = _PADS.index(shadow) if shadow in _PADS else 0
    pad = st.radio("Pad", _PADS, index=idx, horizontal=True, key="pad_hub_pad")
    st.session_state[_SHADOW_KEY] = pad

    # Per-pad state (stage, results, match check) is keyed on each spec's
    # unique prefix, so switching pads here parks one pad's run and resumes
    # the other — same behavior as the old three separate modes.
    run_pad_page(_spec_for(pad))
