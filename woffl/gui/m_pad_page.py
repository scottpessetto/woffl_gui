"""Dedicated M-Pad (Moose Pad) optimization page — thin
:class:`~woffl.gui.pad_page.PadSpec`.

M-Pad: HYBRID station (see :mod:`woffl.gui.m_pad_plant`) — power fluid is
delivered by the HP bank, up to 3 REDA M675 pumps in PARALLEL, fed at the
LP-held ~1,400 psig, holding the ~3,500 psig PF header. Amps have headroom,
so unlike I-Pad the binding limit is **min-flow** (the recirculation /
high-diff shutdown), not amps; the real lever is the pumps-online count
(fewer pumps → lower recirc floor). Pressure still floats (capped 3,500,
``coupling="free_pressure"``) and the frontier bounds the high-flow end.
Head is field-derated ~0.91 (wear).

The 3-stage flow is the shared :func:`woffl.gui.pad_page.run_pad_page` (R-1
Phase C); this module keeps only the M-specific pieces: the HP-bank frontier
plot with recirc shading, the recirc warnings, the pump speed/amps table, and
the page copy. The HP-pumps-online radio offers [3, 2, 1].
"""

import streamlit as st

from woffl.gui import m_pad_plant as plant
from woffl.gui.pad_page import PadSpec, run_pad_page

_MAX_HEADER_PSI = 3500.0  # PF-header discharge cap (PIC-4231 setpoint)

# ---------------------------------------------------------------------------
# Pad-specific render hooks
# ---------------------------------------------------------------------------


def _flow_caption(n_pumps) -> str:
    mn, mx = plant.min_total_flow(n_pumps), plant.max_total_flow(n_pumps)
    return (
        f"With **{n_pumps} HP pump(s)**: min-flow (recirc) floor "
        f"**{mn:,.0f} BPD**, off-curve ceiling {mx:,.0f} BPD total PF."
    )


def _station_metric3(meta: dict) -> tuple[str, str]:
    return "HP pumps online", f"{meta['n_pumps']}"


def _station_extras(meta: dict) -> None:
    import pandas as pd

    n_pumps = meta["n_pumps"]
    if meta.get("recirc"):
        st.warning(
            f"Total PF {meta['total_pf_bpd']:,.0f} BPD is BELOW the {n_pumps}-pump "
            f"min-flow floor ({meta['min_total_flow']:,.0f} BPD) — the HP pumps would "
            "be in recirculation / low-flow shutdown. **Drop to fewer HP pumps** to "
            "lower the floor, or this demand can't run as configured."
        )

    st.markdown("##### HP pump speed & amps")
    pump_rows = []
    for p in meta["pumps"]:
        if p.get("hz") is None:
            pump_rows.append(
                {
                    "Pump": p["name"],
                    "Online": p.get("n", n_pumps),
                    "Speed": "—",
                    "Amps/pump": "—",
                    "Limit (A)": round(p["amp_limit"]),
                }
            )
            continue
        head = p["amp_limit"] - p["amps"]
        pump_rows.append(
            {
                "Pump": p["name"],
                "Online": p.get("n", n_pumps),
                "Speed": f"{p['hz']:.1f} Hz",
                "Amps/pump": round(p["amps"], 1),
                "Limit (A)": round(p["amp_limit"]),
                "Headroom": f"{head:.0f} A ({head / p['amp_limit'] * 100:.0f}%)",
            }
        )
    st.dataframe(pd.DataFrame(pump_rows), use_container_width=True, hide_index=True)


def _scenario_warnings(scn: dict, n_pumps) -> None:
    if scn.get("recirc"):
        st.warning(
            f"Scenario total PF is below the {n_pumps}-pump min-flow floor — "
            "recirc risk. Drop an HP pump or raise demand."
        )
    if scn.get("over_capacity"):
        st.warning(
            "Scenario draws more PF than the HP bank can push at 3,500 — size some pumps down."
        )


def _render_frontier_plot(n_pumps, op_points=None) -> None:
    """HP-bank frontier: max deliverable header vs total PF (capped at 3,500),
    with the min-flow recirc region shaded and optional operating-point markers."""
    import plotly.graph_objects as go

    mn = plant.min_total_flow(n_pumps)
    mx = plant.max_total_flow(n_pumps)
    xs, ys = [], []
    q = max(mn * 0.5, 2000.0)
    step = max((mx - q) / 60.0, 1000.0)
    while q <= mx:
        p = plant.max_discharge_pressure(q, n_pumps)
        if p is not None:
            xs.append(q)
            ys.append(min(_MAX_HEADER_PSI, p))
        q += step

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{n_pumps}-pump frontier (capped 3,500)",
                line=dict(color="#1f77b4"),
                hovertemplate="%{x:.0f} BPD<br>%{y:.0f} psi<extra></extra>",
            )
        )
    fig.add_vrect(
        x0=(xs[0] if xs else 0),
        x1=mn,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
        annotation_text="recirc / min-flow",
        annotation_position="top left",
    )
    fig.add_vline(
        x=mn,
        line=dict(color="#d62728", width=1.5, dash="dash"),
        annotation_text=f"min-flow  {mn:,.0f} BPD",
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
        xaxis_title="Total power-fluid flow (BPD)",
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
    pad="M",
    prefix="mp",
    plant=plant.PLANT,
    title="M-Pad Optimization 🛢",
    subtitle=(
        "Review each Moose Pad well, then optimize the pad against the HP "
        "booster bank (parallel, VFD, min-flow-limited)."
    ),
    configure_caption=(
        "M-Pad's HP bank (parallel REDA M675, VFD) holds the ~3,500 psi PF header "
        "with amp headroom, so the real limit is **min-flow** (recirc) — the lever "
        "is how many HP pumps are online. Pressure floats (capped 3,500); the "
        "optimizer keeps the most-oil point above the min-flow floor."
    ),
    no_active_warning="No active reviewed wells. Go back to **Review Wells**.",
    n_pump_options=(3, 2, 1),
    n_pumps_label="HP pumps online",
    n_pumps_help="3 normally; fewer drops the min-flow floor for low PF demand.",
    flow_caption=_flow_caption,
    plot_heading="##### HP-bank capability frontier",
    render_plot=_render_frontier_plot,
    matchcheck_caption=(
        "Model each well at its current pump + chosen IPR vs recent tests. Fix ✗ first."
    ),
    matchcheck_note="Worst first.",
    run_bar_text="Optimizing the HP bank…",
    run_spinner_text="Running batch sweeps across the pressure frontier…",
    n_steps=9,
    station_metric3=_station_metric3,
    render_station_extras=_station_extras,
    results_plot_heading="#### HP-bank frontier",
    sweep_expander_label="Pressure sweep explored",
    comparator_caption=(
        "Pick a pump (or shut-in) per well and see the resulting oil + header "
        "pressure — coupled to the HP-bank frontier — next to a baseline."
    ),
    cmp_base_help=(
        "Existing = each well's CURRENT measured oil/PF from recent tests. "
        "Optimized = the optimizer's picks."
    ),
    scn_preset_help=None,
    custom_expander_label="Per-well pump selection",
    filled_from_current_label="latest installed",
    scenario_spinner_text="Running scenario (coupled to the frontier)…",
    render_scenario_warnings=_scenario_warnings,
    subs_caption="⚠ Chosen pump infeasible — substituted: ",
    star_caption="★ Model couldn't solve — estimated from recent-test rate × ripple: ",
    infeas_caption="✗ No feasible pump (counted as 0): ",
    where_markdown="**Where each option sits on the frontier**",
    bias_note=" **Model÷Test** = model ÷ measured (>1 = loose IPR match).",
    delta_caption=(
        "**Δ oil** = per-well change ({base} → scenario). "
        "★ = estimated. Sorted by biggest mover."
    ),
)


def run_m_pad_page() -> None:
    run_pad_page(SPEC)
