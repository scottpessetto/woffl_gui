"""Dedicated I-Pad optimization page — thin :class:`~woffl.gui.pad_page.PadSpec`.

I-Pad: 2 pumps in SERIES, VFD-driven, amp-limited (see
:mod:`woffl.gui.i_pad_plant`). There's no fixed-speed curve — the train's
capability is a falling frontier ``max_discharge_pressure(flow)`` and the
delivered header pressure is an optimizer DECISION VARIABLE
(``coupling="free_pressure"``): a lower pressure lets the pumps push more PF
within their amp limits (Scott, 2026-06-16 — "no hard cap; lowering discharge
pressure to get more flow is a valid option"). The shared page sweeps candidate
pressures and keeps the most-oil point; the operational discharge cap (3,500
psi) lives on :class:`~woffl.gui.pad_plant_base.IPadPlant` as
``max_header_psi``.

The 3-stage flow is the shared :func:`woffl.gui.pad_page.run_pad_page` (R-1
Phase C); this module keeps only the I-specific pieces: the amp-limited
frontier plot, the per-pump amps table (the binding constraint), and the page
copy. No pump-count radio — the LP+HP series train is fixed.
"""

import streamlit as st

from woffl.gui import i_pad_plant as plant
from woffl.gui.pad_page import PadSpec, run_pad_page

# ---------------------------------------------------------------------------
# Pad-specific render hooks
# ---------------------------------------------------------------------------


def _station_metric3(meta: dict) -> tuple[str, str]:
    return "Frontier budget", f"{meta['frontier_cap_bpd']:,.0f} BPD"


def _station_extras(meta: dict) -> None:
    """Per-pump amp headroom — the real I-Pad constraint."""
    import pandas as pd

    st.markdown("##### Booster amps (the binding constraint)")
    pump_rows = []
    for p in meta["pumps"]:
        if p["hz"] is None:
            pump_rows.append(
                {
                    "Pump": p["name"],
                    "Speed": "—",
                    "Amps": "—",
                    "Limit (A)": round(p["amp_limit"]),
                    "Headroom": "INFEASIBLE",
                }
            )
            continue
        head = p["amp_limit"] - p["amps"]
        pump_rows.append(
            {
                "Pump": p["name"],
                "Speed": f"{p['hz']:.1f} Hz",
                "Amps": round(p["amps"], 1),
                "Limit (A)": round(p["amp_limit"]),
                "Headroom": f"{head:.1f} A ({head / p['amp_limit'] * 100:.0f}%)",
            }
        )
    st.dataframe(pd.DataFrame(pump_rows), use_container_width=True, hide_index=True)
    if meta["amp_limited"]:
        st.caption(
            "A pump is at its amp limit — the train is pushing as much PF as "
            "it can at this pressure. Lower pressure trades for more flow."
        )


def _scenario_warnings(scn: dict, n_pumps) -> None:
    if scn.get("over_capacity"):
        st.warning(
            "Scenario draws more PF than the train can push at any pressure — "
            "it would force the header below what the wells need. Size some pumps down."
        )


def _render_frontier_plot(op_points=None) -> None:
    """I-Pad amp-limited frontier: max deliverable header (HP-discharge) pressure
    vs total PF flow, with optional operating-point markers. Falls with flow;
    ends where a pump can no longer pass the flow within its amp limit."""
    import plotly.graph_objects as go

    # Sweep flow until the frontier returns None (a pump can't pass it).
    xs, ys = [], []
    q = 4000.0
    while q <= 4.0 * plant._max_valid_flow():
        p = plant.max_discharge_pressure(q)
        if p is None:
            break
        xs.append(q)
        ys.append(p)
        q += 2000.0

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="amp-limited frontier",
                line=dict(color="#1f77b4"),
                hovertemplate="%{x:.0f} BPD<br>max %{y:.0f} psi<extra></extra>",
            )
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
        xaxis_title="Total power-fluid flow (BPD)",
        yaxis_title="Max deliverable header pressure (psi)",
        height=380,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot(n_pumps, op_points=None) -> None:
    # shared-page plot hook signature is (n_pumps, op_points); the fixed
    # LP+HP series train has no pump-count concept
    _render_frontier_plot(op_points)


# ---------------------------------------------------------------------------
# Spec + entry point
# ---------------------------------------------------------------------------

SPEC = PadSpec(
    pad="I",
    prefix="ip",
    plant=plant.PLANT,
    title="I-Pad Optimization 🛢",
    subtitle=(
        "Review each I-Pad well, then optimize the pad against the "
        "two-pump series booster train (amp-limited, pressure-free)."
    ),
    configure_caption=(
        "I-Pad's two boosters run in **series on VFDs**, so delivered pressure isn't "
        "a fixed curve — the real limit is motor amps. The optimizer treats the "
        "**header pressure as a free variable**: it can lower pressure to push more "
        "power fluid within the amp limits, and keeps whatever yields the most oil."
    ),
    n_pump_options=None,  # fixed LP+HP series train — no pump-count choice
    marginal_wc_help="Default 1.0 (no marginal-WC cutoff).",
    plot_heading="##### Booster train capability (amp-limited frontier)",
    render_plot=_plot,
    matchcheck_caption=(
        "Before optimizing, model each well at its current pump + chosen IPR and "
        "compare to its recent tests (median). Fix the ✗ wells first."
    ),
    matchcheck_note="⚠ = off, ✓ = match. Worst matches first.",
    run_bar_text="Sweeping header pressure on the amp frontier…",
    run_spinner_text=(
        "Running batch sweeps across the pressure frontier " "(this can take a minute)…"
    ),
    n_steps=11,
    station_metric3=_station_metric3,
    render_station_extras=_station_extras,
    results_plot_heading="#### Amp-limited frontier",
    sweep_expander_label="Pressure sweep (oil vs header pressure explored)",
    comparator_caption=(
        "Pick a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the amp-limited frontier — next to a baseline. "
        "Size pumps up and watch the PF rise and the pressure fall."
    ),
    cmp_base_help=(
        "Existing = each well's CURRENT measured oil/PF from its recent tests "
        "(the natural 'I only change a couple of wells' baseline). "
        "Optimized = the optimizer's picks."
    ),
    scenario_spinner_text="Running scenario (coupled to the frontier)…",
    render_scenario_warnings=_scenario_warnings,
    where_markdown="**Where each option sits on the amp-limited frontier**",
    bias_note=(
        " **Model÷Test** = model's predicted current rate ÷ the measured "
        "test (>1 = the well's IPR match is loose — tighten it first)."
    ),
    delta_caption=(
        "**Δ oil** = per-well change ({base} → scenario), including the "
        "ripple on unchanged wells as the header moves "
        "({h0:,.0f} → {h1:,.0f} psi). "
        "★ = model couldn't solve; estimated from recent-test rate × the average "
        "ripple of the solvers. Sorted by biggest mover."
    ),
)


def run_i_pad_page() -> None:
    run_pad_page(SPEC)
