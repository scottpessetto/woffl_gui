"""Dedicated S-Pad optimization page — thin :class:`~woffl.gui.pad_page.PadSpec`.

S-Pad: 3 parallel fixed-speed boosters, so the delivered header pressure is a
CURVE of total PF flow (``coupling="fixed_curve"``) — the optimizer is solved
to a fixed point against it (see :mod:`woffl.gui.s_pad_plant`). The 3-stage
flow (Review Wells → Configure & Run → Results) is the shared
:func:`woffl.gui.pad_page.run_pad_page` (R-1 Phase C); this module keeps only
the S-specific pieces: the booster-curve plot with the thrust window shading,
the per-pump metric/warnings, and the page copy.
"""

import streamlit as st

from woffl.gui import s_pad_plant
from woffl.gui.pad_page import PadSpec, run_pad_page

# ---------------------------------------------------------------------------
# Pad-specific render hooks
# ---------------------------------------------------------------------------


def _flow_caption(n_pumps) -> str:
    lo, hi = s_pad_plant.recommended_flow_per_pump()
    return (
        f"Per-pump thrust window: {lo:,.0f}–{hi:,.0f} BPD  ·  "
        f"station capacity ({n_pumps} pumps): {s_pad_plant.station_capacity(n_pumps):,.0f} BPD"
    )


def _station_metric3(meta: dict) -> tuple[str, str]:
    return f"Per pump (×{meta['n_pumps']})", f"{meta['per_pump_bpd']:,.0f} BPD"


def _station_extras(meta: dict) -> None:
    if not meta["in_range"]:
        lo, hi = s_pad_plant.recommended_flow_per_pump()
        st.warning(
            f"Per-pump flow {meta['per_pump_bpd']:,.0f} BPD is OUTSIDE the "
            f"{lo:,.0f}–{hi:,.0f} thrust window — pump damage / off-curve risk."
        )
    if not meta["converged"]:
        st.warning(
            "Pump-curve coupling did NOT fully converge in the iteration cap "
            "— review the trend below."
        )


def _scenario_warnings(scn: dict, n_pumps) -> None:
    if not scn["in_range"]:
        st.warning("Scenario per-pump flow is outside the thrust window.")


def _render_curve_plot(n_pumps: int, op_points=None) -> None:
    """Combined ``n_pumps``-pump booster curve (header pressure vs total flow),
    thrust window shaded, with optional labeled operating-point markers.

    ``op_points``: list of dicts {label, total_pf_bpd, header_psi, color?} — e.g.
    the optimized point and a manual-scenario point on the same curve.
    """
    import plotly.graph_objects as go

    lo, hi = s_pad_plant.recommended_flow_per_pump()
    down_x, up_x = lo * n_pumps, hi * n_pumps
    # Plot ~12% past the up-thrust limit so the up-thrust region is visible
    # (the per-pump polynomial is valid out to ~21,000 BPD/pump).
    x_max = up_x * 1.12
    n_pts = 72
    xs = [x_max * i / n_pts for i in range(1, n_pts + 1)]
    ys = [s_pad_plant.discharge_pressure(x, n_pumps) for x in xs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"{n_pumps}-pump curve",
            line=dict(color="#1f77b4"),
            hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
        )
    )
    # Recommended (safe) window in green, between the down-thrust and up-thrust
    # limits; the up-thrust region beyond the high limit shaded red.
    fig.add_vrect(
        x0=down_x,
        x1=up_x,
        fillcolor="green",
        opacity=0.08,
        line_width=0,
        annotation_text="recommended window",
        annotation_position="top left",
    )
    fig.add_vrect(x0=up_x, x1=x_max, fillcolor="red", opacity=0.07, line_width=0)
    fig.add_vline(
        x=down_x,
        line=dict(color="#888", width=1, dash="dot"),
        annotation_text=f"down-thrust  {down_x:,.0f} BPD",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=up_x,
        line=dict(color="#d62728", width=1.5, dash="dash"),
        annotation_text=f"up-thrust limit  {up_x:,.0f} BPD",
        annotation_position="top right",
    )
    for p in op_points or []:
        fig.add_trace(
            go.Scatter(
                x=[p["total_pf_bpd"]],
                y=[p["header_psi"]],
                mode="markers+text",
                marker=dict(size=14, color=p.get("color", "#d62728"), symbol="x"),
                name=p["label"],
                text=[
                    f"  {p['label']}: {p['header_psi']:.0f} psi @ {p['total_pf_bpd']:,.0f} BPD"
                ],
                textposition="middle right",
                hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Total station flow (BPD)",
        yaxis_title="Header pressure (psi)",
        height=380,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Spec + entry point
# ---------------------------------------------------------------------------

SPEC = PadSpec(
    pad="S",
    prefix="sp",
    plant=s_pad_plant.PLANT,
    title="S-Pad Optimization 🛢",
    subtitle=(
        "Review each S-Pad well, then optimize the pad against the 3-pump booster curve."
    ),
    configure_caption=(
        "Delivered power-fluid pressure is set by the 3-pump booster curve — "
        "more total PF flow means lower header pressure. The optimizer is solved "
        "to a fixed point against that curve."
    ),
    n_pump_options=(3, 2),
    n_pumps_label="Booster pumps online",
    n_pumps_help="All 3 normally; drop to 2 to model one offline.",
    marginal_wc_help="S-Pad default 1.0 (no marginal-WC cutoff).",
    flow_caption=_flow_caption,
    plot_heading="##### Combined pump curve",
    render_plot=_render_curve_plot,
    matchcheck_caption=(
        "Before optimizing, model each well at its current pump + chosen IPR and "
        "compare to its recent tests (median). Fix the ✗ wells first — a model that can't "
        "reproduce a well's current oil/PF won't optimize it reliably."
    ),
    matchcheck_note="⚠ = off, ✓ = match. Oil×/PF× = model ÷ test. Worst matches first.",
    run_bar_text="Solving pump-curve coupling…",
    run_spinner_text="Running batch sweeps + optimization (this can take a minute)…",
    station_metric3=_station_metric3,
    render_station_extras=_station_extras,
    results_plot_heading="#### Booster curve",
    comparator_caption=(
        "Choose a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the booster curve — side by side with a baseline."
    ),
    cmp_base_help=(
        "Existing = each well's CURRENT measured oil/PF from its recent tests (median) "
        "(the natural baseline for 'I only change a couple of wells'). "
        "Optimized = the optimizer's picks. Pair Existing with Custom → "
        "Fill all from latest installed → change a few wells."
    ),
    scenario_spinner_text="Running scenario (coupled to the booster curve)…",
    render_scenario_warnings=_scenario_warnings,
    show_per_pump=True,
    where_markdown="**Where each option operates on the 3-pump booster curve**",
    bias_note=(
        " **Model÷Test** = model's predicted current rate ÷ the measured test "
        "(>1 = model over-predicts; that well's IPR match is loose — tighten it first)."
    ),
    delta_caption=(
        "**Δ oil** = net per-well change ({base} → scenario), including the "
        "ripple on unchanged wells (header pressure {h0:,.0f} → "
        "{h1:,.0f} psi). ★ = model couldn't solve; estimated from latest "
        "test × the average ripple of the solving wells. Sorted by biggest mover."
    ),
)


def run_s_pad_page() -> None:
    run_pad_page(SPEC)
