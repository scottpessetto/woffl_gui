"""PW Pressure Optimization — the CFP plant and its J/G/C/B wells.

Composed from the pad-page pieces rather than forcing ``PadSpec`` /
``run_pad_page`` to go multi-pad, so S/I/M stay untouched. Four stages:

0. **Plant Dashboard** — are we optimized on discharge pressure today?
   (``cfp_tradeoff`` verdict + history + the model-accuracy gate).
1. **Review Wells** — pad picker over J/G/C/B, the existing per-pad review
   flow; entries flagged *offline* become the bring-online candidates.
2. **Configure & Analyze** — anchor on the MEASURED discharge and today's
   online set, then build WOFFL response surfaces. Deliberately NO exogenous
   water and NO summing of well tests (Scott, 2026-07-30) — the moves engine
   models deltas from today, so injection/disposal/carryover cancel.
3. **Today's Moves** — the decision: resize JPs, shut in, bring online, or
   BOL + offset pairs, each priced in fleet BOPD; the oil-vs-pressure
   frontier; the shadow price of a psi. Engine: ``woffl/gui/cfp_moves.py``;
   formulation + literature: ``docs/cfp_moves_methodology.md``.
"""

import pandas as pd
import streamlit as st

from woffl.assembly import cfp_plant as _cfp
from woffl.gui import cfp_optimize as co
from woffl.gui.cfp_pad_plant import PLANT
from woffl.gui.workflow_steps import well_review_store as wrs
from woffl.gui.workflow_steps.step_review_wells import render_review_stage, store_for

PADS = ["B", "G", "C", "J"]
_PREFIX = "cfp"
_STAGES = [
    "0 · Plant Dashboard",
    "1 · Review Wells",
    "2 · Configure & Analyze",
    "3 · Today's Moves",
]

# Pads whose power fluid rides the plant discharge — the oil exposed to a
# pressure sag. C-Pad is boosted on-pad and is deliberately excluded.
_EXPOSED_PADS = ("B", "G", "J")

# Historian tags for the dashboard. Machine flows are THE confirmed per-machine
# tags (cfp_plant.MACHINE_FLOW_TAGS, verified against Scott's SCADA screen
# 2026-07-30) — NOT MPU_FIC_5488/5489, which are a different stream and briefly
# drove a wrong "two-machine plant" conclusion.
_HIST_TAGS = {
    "MPU_PIC_5418": "disch",
    "MPU_FIC_5419S": "m_a",
    "MPU_FIC_5420S": "m_b",
    "MPU_FIC_5421S": "m_c",
    "MPU_MOD 54_ProdWaterAvgFlowRate_Calc": "prod_w",
    "MPU_MOD 54_PlantWaterAvgFlowRate_Calc": "plant_w",
}



# ── live measured inputs ────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _measured_pad_pf() -> dict:
    """Live per-pad PF from the header CLUSTER (not the median — B/G/J/C are
    ESP-mixed, so a median averages header wells against unrelated ones)."""
    from woffl.assembly.pf_pressure import fetch_pf_latest, pad_pf_cluster

    try:
        return pad_pf_cluster(fetch_pf_latest())
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _plant_history(days: int = 150) -> "pd.DataFrame":
    """Daily plant discharge + flows from the historian.

    Soft-fails to empty: the hosted app's service principal cannot read the
    `reporting` catalog (see docs/prop_hist_asks.md), so the dashboard must
    degrade rather than crash there.
    """
    from woffl.assembly.databricks_client import execute_query

    try:
        raw = execute_query(
            f"""
            SELECT MeasureDate AS day, Tag, avg(Value) AS v
            FROM reporting.historian.vw_mpu_measurements
            WHERE MeasureDate >= DATE_SUB(current_date(), {int(days)})
              AND Tag IN ({",".join(f"'{t}'" for t in _HIST_TAGS)})
            GROUP BY MeasureDate, Tag
            """
        )
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    raw["k"] = raw["Tag"].map(_HIST_TAGS)
    p = raw.pivot(index="day", columns="k", values="v")
    if "disch" in p.columns:
        p = p[p["disch"] > 2000]  # drop plant-down days
    machine_cols = [c for c in ("m_a", "m_b", "m_c") if c in p.columns]
    if machine_cols:
        p["m_total"] = p[machine_cols].sum(axis=1)
    return p.sort_index()


def current_state(hist: "pd.DataFrame", days: int = 14) -> dict:
    """Where the plant sits now — the trailing mean of the history.

    Pure given the frame, so the dashboard's headline is testable.
    """
    from woffl.assembly import cfp_plant as cfp

    if hist is None or hist.empty or "disch" not in hist.columns:
        return {
            "discharge_psi": cfp.MEASURED_DISCHARGE_PSI,
            "prod_w": cfp.MEASURED_PRODUCED_WATER_BWPD,
            "source": "fallback (no historian access)",
            "days": 0,
        }
    tail = hist.tail(int(days))
    out = {"source": f"historian, {len(tail)}-day mean", "days": len(tail)}
    for col, key in (("disch", "discharge_psi"), ("prod_w", "prod_w"),
                     ("plant_w", "plant_w"), ("m_total", "m_total")):
        if col in tail.columns and tail[col].notna().any():
            out[key] = float(tail[col].mean())
    out.setdefault("discharge_psi", cfp.MEASURED_DISCHARGE_PSI)
    out.setdefault("prod_w", cfp.MEASURED_PRODUCED_WATER_BWPD)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _current_pad_oil() -> dict:
    """Latest allocated-test oil per pad, BOPD — the oil exposed to a PF sag."""
    from woffl.assembly.databricks_client import execute_query

    try:
        df = execute_query(
            """
            WITH ranked AS (
                SELECT well_name, form_oil, substring(well_name, 1, 1) AS pad,
                       ROW_NUMBER() OVER (
                           PARTITION BY well_name ORDER BY wt_date DESC
                       ) AS rn
                FROM mpu.wells.vw_well_test
                WHERE allocated = True AND wt_date >= DATE_SUB(current_date(), 120)
            )
            SELECT pad, sum(coalesce(form_oil, 0)) AS oil
            FROM ranked WHERE rn = 1 GROUP BY pad
            """
        )
        return {str(r["pad"]): float(r["oil"]) for _, r in df.iterrows()}
    except Exception:
        return {}


# NOTE deliberately absent: any "exogenous water" input or bottom-up summing of
# well tests into a plant load. Scott rejected that framing 2026-07-30 — the
# moves engine (cfp_moves) anchors on the MEASURED discharge and models only
# deltas, so injection/disposal/carryover all cancel. See
# docs/cfp_moves_methodology.md.


# ── stage 0: the dashboard — are we optimized on discharge pressure? ────────


def _render_dashboard() -> None:
    from woffl.assembly import cfp_plant as cfp
    from woffl.gui import cfp_tradeoff as ct

    st.markdown(
        "**Are we running at the right discharge pressure?** Higher pressure "
        "means better lift on B/G/J; more water means more oil from the wells "
        "making it. This is where those two meet."
    )

    hist = _plant_history()
    now = current_state(hist)
    pad_oil = _current_pad_oil()
    exposed_oil = sum(pad_oil.get(p, 0.0) for p in _EXPOSED_PADS)

    # ── where we are ────────────────────────────────────────────────────────
    st.markdown("##### Where the plant is now")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Discharge",
        f"{now['discharge_psi']:,.0f} psi",
        delta=f"{now['discharge_psi'] - cfp.MAX_DISCHARGE_PSI:,.0f} vs trip",
        delta_color="off",
    )
    c2.metric("Produced water", f"{now.get('prod_w', 0):,.0f} BWPD")
    if "m_total" in now:
        c3.metric("Machine flow (A+B+C)", f"{now['m_total']:,.0f} BWPD")
    else:
        c3.metric("Machine flow", "—")
    c4.metric("Oil exposed to PF", f"{exposed_oil:,.0f} BOPD", help="B + G + J")
    st.caption(
        f"Source: {now['source']}. Trip at {cfp.MAX_DISCHARGE_PSI:,.0f} psi — the "
        "piping rating; above it the pumps shut down."
    )

    # ── the knobs behind the verdict ────────────────────────────────────────
    with st.expander("Assumptions behind the verdict", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            slope = st.slider(
                "Pressure cost (psi per 1,000 BWPD)",
                min_value=ct.PSI_PER_KBWPD_LOW,
                max_value=ct.PSI_PER_KBWPD_HIGH,
                value=ct.PSI_PER_KBWPD_MID,
                step=0.5,
                key=f"{_PREFIX}_slope",
                help=(
                    "Well constrained, 9-26. Three independent estimates: the "
                    "plant's own Mar->Jul trend (12.2), April's within-month fit "
                    "(9.0, r²=0.80), and the SCADA-validated pump curve (26)."
                ),
            )
        with a2:
            responsive = st.slider(
                "Fraction of wells that respond to PF",
                0.0, 1.0, 0.5, 0.05,
                key=f"{_PREFIX}_responsive",
                help=(
                    "THE shaky input, and it drives the answer. Measured +2.4-2.9% "
                    "oil per 108 psi on lift-limited wells and EXACTLY 0% on "
                    "inflow-limited ones — those are IPR-bound, so more power fluid "
                    "buys nothing."
                ),
            )
        with a3:
            marg_wc = st.slider(
                "Water cut of the marginal barrel",
                0.80, 0.999, 0.90, 0.005, format="%.3f",
                key=f"{_PREFIX}_marg_wc",
                help=(
                    "The WC of the water actually on the table — the next well or "
                    "pump size up, or the worst well you'd cut."
                ),
            )

    inp = ct.TradeoffInputs(
        exposed_oil_bopd=exposed_oil or 7793.0,
        current_water_bwpd=now.get("prod_w", cfp.MEASURED_PRODUCED_WATER_BWPD),
        current_discharge_psi=now["discharge_psi"],
        max_discharge_psi=cfp.MAX_DISCHARGE_PSI,
        psi_per_kbwpd=slope,
        responsive_frac=responsive,
    )
    v = ct.verdict(inp, marginal_wc=marg_wc)

    # ── the verdict ─────────────────────────────────────────────────────────
    st.markdown("##### Verdict")
    headline = {
        "more_water": "🟢 **Bring more water on** — you are below the line.",
        "cut_water": "🔴 **Cut water / downsize** — you are past the line.",
        "hold": "🟡 **You're at the line** — hold.",
        "unknown": "⚪ Not enough information.",
    }[v["action"]]
    (st.success if v["action"] == "more_water" else
     st.error if v["action"] == "cut_water" else st.info)(
        f"{headline}\n\n{v['reason']}"
    )

    mv = st.session_state.get(f"{_PREFIX}_moves")
    if mv:
        psi_worth, psi_src = mv["lambda_bopd_per_psi"], "from the per-well WOFFL surfaces (Today's Moves run)"
    else:
        psi_worth, psi_src = ct.bopd_per_psi(inp), "from the assumptions above — run Today's Moves for the per-well figure"

    m0, m1, m2, m3, m4 = st.columns(5)
    m0.metric(
        "1 psi is worth",
        f"{psi_worth:+,.1f} BOPD",
        help=f"d(fleet oil)/d(discharge) — the price of pressure ({psi_src}).",
    )
    m1.metric("Break-even WC", f"{v['breakeven_wc']:.1%}")
    m2.metric("Marginal barrel", f"{marg_wc:.1%}")
    m3.metric(
        "Net per 1,000 BWPD",
        f"{v.get('net_bopd_per_kbwpd', 0):+,.0f} BOPD",
        help="Oil the water brings, minus oil the pressure sag costs.",
    )
    m4.metric(
        "Cut to reach the trip",
        f"{v['water_to_cut_for_trip_bwpd']:,.0f} BWPD",
        help=(
            f"Shedding this much would take discharge to {cfp.MAX_DISCHARGE_PSI:,.0f} "
            "psi. Cutting beyond it buys NO further pressure — pure oil loss."
        ),
    )

    # ── how much should you believe any of this? ────────────────────────────
    _render_model_confidence(now)

    # ── the tradeoff curve ──────────────────────────────────────────────────
    st.markdown("##### The tradeoff")
    rows = ct.tradeoff_curve(inp, marginal_wc=marg_wc, span_bwpd=20000.0, steps=41)
    _plot_tradeoff(rows, inp, v)

    st.caption(
        "Left of centre = cutting water. The line flattens where discharge hits "
        f"the {cfp.MAX_DISCHARGE_PSI:,.0f} psi trip: past there, cutting costs oil "
        "and buys nothing."
    )

    # ── does the answer survive the uncertainty? ────────────────────────────
    st.markdown("##### Does the answer hold across the uncertainty?")
    sens = ct.sensitivity_table(inp, marginal_wc=marg_wc)
    sdf = pd.DataFrame(
        [
            {
                "Pressure cost": r["slope"],
                "Responsive": f"{r['responsive_frac']:.0%}",
                "Cost (BOPD/1k)": round(r["cost_bopd_per_kbwpd"], 1),
                "Break-even WC": f"{r['breakeven_wc']:.1%}",
                "Net (BOPD/1k)": round(r["net_bopd_per_kbwpd"], 1),
                "Verdict": r["action"].replace("_", " "),
            }
            for r in sens
        ]
    )
    st.dataframe(sdf, use_container_width=True, hide_index=True)
    actions = {r["action"] for r in sens}
    if len(actions) == 1:
        st.success(
            f"Every corner of the uncertainty box agrees: **{actions.pop().replace('_', ' ')}**."
        )
    else:
        st.warning(
            "The corners disagree — the answer depends on how many wells actually "
            "respond to PF pressure. Worth measuring per-well before acting."
        )

    # ── history ─────────────────────────────────────────────────────────────
    st.markdown("##### How we got here")
    if hist is None or hist.empty:
        st.info(
            "Plant history unavailable — the `reporting` catalog isn't readable "
            "from here. It works on Scott's desktop; the hosted app needs a grant "
            "(see docs/prop_hist_asks.md)."
        )
    else:
        _plot_history(hist)
        monthly = hist.copy()
        monthly["month"] = pd.to_datetime(monthly.index).to_period("M").astype(str)
        agg = monthly.groupby("month").mean(numeric_only=True).round(0)
        keep = [c for c in ("disch", "m_total", "prod_w", "plant_w") if c in agg.columns]
        agg = agg[keep].rename(
            columns={
                "disch": "Discharge (psi)",
                "m_total": "Machine flow (BWPD)",
                "prod_w": "Produced water (BWPD)",
                "plant_w": "Plant water (BWPD)",
            }
        )
        st.dataframe(agg, use_container_width=True)
        st.caption(
            "Discharge rises as machine flow falls — the pump curve, and the "
            "trade you are already making by hand."
        )


_TRUST_UI = {
    "good": ("✅", st.success, "Models are trustworthy"),
    "fair": ("⚠️", st.warning, "Models are partly trustworthy"),
    "poor": ("🚫", st.error, "Models are NOT trustworthy"),
    "none": ("⚪", st.info, "Models unchecked"),
}


def _render_model_confidence(now: dict) -> None:
    """Do the per-well JP models actually reproduce their own well tests?

    Everything above is computed FROM those models, so this gates how much of it
    to believe. Runs on demand — it costs a real batch solve per well.
    """
    from woffl.gui import cfp_optimize as co

    st.markdown("##### Do the well models match reality?")
    st.caption(
        "Every number above comes from the per-well jet-pump models. This runs "
        "each well at its **current pump** and its pad's **current delivered PF**, "
        "then compares against that well's own last test. ✓ = within 0.80-1.25x."
    )

    key = f"{_PREFIX}_matchcheck"
    cached = st.session_state.get(key)
    if st.button("▶ Check model accuracy", key=f"{_PREFIX}_run_matchcheck"):
        pad_configs = {
            p: wrs.store_to_well_configs(wrs.active_entries(store_for(p))) for p in PADS
        }
        pad_configs = {p: w for p, w in pad_configs.items() if w}
        if not pad_configs:
            st.warning(
                "No reviewed wells yet — review some on stage 1 first, then the "
                "check can run against their models."
            )
            return
        from woffl.gui.pad_helpers import parse_pump, recent_test_rates

        choices, rates = {}, {}
        for pad in pad_configs:
            store = wrs.active_entries(store_for(pad))
            for well, entry in store.items():
                cc = parse_pump(
                    f"{entry.get('review_nozzle', '')}{entry.get('review_throat', '')}"
                )
                if cc:
                    choices[well] = cc
                rates[well] = recent_test_rates(well)
        try:
            with st.spinner("Modelling every well at its current pump…"):
                rows, per_pad = co.cfp_match_check(
                    pad_configs,
                    PLANT,
                    now["discharge_psi"],
                    choices,
                    rates,
                    c_pad_pf_psi=float(
                        st.session_state.get(f"{_PREFIX}_c_pad_pf", 3400.0)
                    ),
                    measured_pad_pf={
                        p: v["psi"] for p, v in _measured_pad_pf().items()
                    },
                )
            cached = {"rows": rows, "per_pad": per_pad}
            st.session_state[key] = cached
        except Exception as e:
            st.error(f"Model check failed: {e}")
            return

    if not cached:
        st.info(
            "Not checked yet. Until you run this, treat the verdict's DIRECTION "
            "as informative and its BOPD figures as unverified."
        )
        return

    rows = cached["rows"]
    summary = co.match_summary(rows)
    icon, box, title = _TRUST_UI.get(summary["trust"], _TRUST_UI["none"])
    box(f"{icon} **{title}** — {summary['reason']}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wells checked", summary["n"])
    c2.metric("Match on both", f"{summary['both_ok']} ({summary['frac_ok']:.0%})")
    c3.metric("Oil busts", summary["oil_bust"])
    c4.metric("PF busts", summary["pf_bust"])

    df = pd.DataFrame(
        [
            {
                "Well": r["well"],
                "Pad": r["pad"],
                "Pump": r["pump"],
                "PF (psi)": round(r["delivered_pf"]) if r["delivered_pf"] else None,
                "Test oil": round(r["test_oil"]) if r["test_oil"] else None,
                "Model oil": round(r["model_oil"]) if r["model_oil"] is not None else None,
                "Oil ×": round(r["oil_ratio"], 2) if r["oil_ratio"] else None,
                "Oil": r["oil_flag"],
                "Test PF": round(r["test_pf"]) if r["test_pf"] else None,
                "Model PF": round(r["model_pf"]) if r["model_pf"] is not None else None,
                "PF ×": round(r["pf_ratio"], 2) if r["pf_ratio"] else None,
                "PF": r["pf_flag"],
            }
            for r in sorted(
                rows,
                key=lambda r: max(
                    abs((r.get("oil_ratio") or 1.0) - 1.0),
                    abs((r.get("pf_ratio") or 1.0) - 1.0),
                ),
                reverse=True,
            )
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "Worst mismatches first. **Oil off → the IPR is loose** (re-anchor it on a "
        "better test in the Solver). **PF off → the pump model is wrong** (nozzle "
        "wear, wrong pump on record, or friction coefficients) — the Solver's "
        "auto-match and nozzle-wear tools fix those."
    )


def _plot_tradeoff(rows: list, inp, v: dict) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["delta_water_bwpd"] for r in rows],
            y=[r["delta_oil_bopd"] for r in rows],
            mode="lines",
            name="Δ oil",
            line=dict(color="#2E7D32", width=3),
            hovertemplate="%{x:,.0f} BWPD<br>%{y:+,.0f} BOPD<extra></extra>",
        )
    )
    tripped = [r for r in rows if r["at_trip"]]
    if tripped:
        fig.add_trace(
            go.Scatter(
                x=[r["delta_water_bwpd"] for r in tripped],
                y=[r["delta_oil_bopd"] for r in tripped],
                mode="lines",
                name="at the trip — no more pressure to gain",
                line=dict(color="#D32F2F", width=3),
                hovertemplate="%{x:,.0f} BWPD<br>%{y:+,.0f} BOPD<extra></extra>",
            )
        )
    fig.add_vline(x=0, line=dict(color="#555", dash="dot"))
    fig.add_hline(y=0, line=dict(color="#555", dash="dot"))
    fig.add_annotation(
        x=0, y=0, text="you are here", showarrow=True, arrowhead=2, ay=-40
    )
    fig.update_layout(
        xaxis_title="Change in water handled (BWPD)   ← cut · add →",
        yaxis_title="Change in total oil (BOPD)",
        height=380,
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_history(hist: "pd.DataFrame") -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "disch" in hist.columns:
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist["disch"], name="Discharge (psi)",
                line=dict(color="#E65100", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} psi<extra></extra>",
            ),
            secondary_y=True,
        )
    for col, name, colour in (
        ("m_total", "Machine flow A+B+C", "#1565C0"),
        ("prod_w", "Produced water", "#6A1B9A"),
    ):
        if col in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist[col], name=name,
                    line=dict(color=colour, width=1.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} BWPD<extra></extra>",
                ),
                secondary_y=False,
            )
    from woffl.assembly import cfp_plant as cfp

    fig.add_hline(
        y=cfp.MAX_DISCHARGE_PSI, line=dict(color="#D32F2F", dash="dash"),
        annotation_text="2,900 psi trip", secondary_y=True,
    )
    fig.update_yaxes(title_text="BWPD", secondary_y=False)
    fig.update_yaxes(title_text="psi", showgrid=False, secondary_y=True)
    fig.update_layout(
        height=380, hovermode="x unified", margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── stage 1: review ─────────────────────────────────────────────────────────


def _render_review() -> None:
    st.markdown(
        "Review each well on each CFP pad, exactly as on the S/I/M pages. The "
        "four pads keep separate stores and are optimized **together** in "
        "stage 2, because they share one plant."
    )
    shadow = st.session_state.get(f"{_PREFIX}_review_pad_shadow", PADS[0])
    idx = PADS.index(shadow) if shadow in PADS else 0
    pad = st.radio(
        "Pad to review", PADS, index=idx, horizontal=True, key=f"{_PREFIX}_review_pad"
    )
    st.session_state[f"{_PREFIX}_review_pad_shadow"] = pad

    counts = " · ".join(
        f"**{p}**: {len(wrs.active_entries(store_for(p)))}" for p in PADS
    )
    st.caption(f"Active wells reviewed — {counts}")
    st.divider()
    render_review_stage(pad)


# ── stage 2: configure & analyze ────────────────────────────────────────────


_SURF_CACHE_KEY = f"{_PREFIX}_surfaces_cache"


def _collect_fleet():
    """Online wells + bring-online candidates from the per-pad review stores.

    Online = reviewed and not flagged offline. BOL candidates = reviewed
    entries flagged *offline* — the store's existing parked/future flag,
    reused as the bring-online list. An online well with no reviewed pump
    can't be anchored ("what's in the hole today?") and is reported back
    rather than silently dropped.
    """
    pad_configs, online, current, skipped = {}, {}, {}, []
    for pad in PADS:
        store = store_for(pad)
        cfgs = []
        for well, entry in store.items():
            if not entry.get("reviewed"):
                continue
            is_online = not entry.get("offline")
            n, t = entry.get("review_nozzle"), entry.get("review_throat")
            if is_online and not (n and t):
                skipped.append(well)
                continue
            cfgs.append(wrs.to_well_config(entry))
            online[well] = is_online
            current[well] = (str(n), str(t)) if (n and t) else None
        if cfgs:
            pad_configs[pad] = cfgs
    return pad_configs, online, current, skipped


def _render_configure() -> None:
    pad_configs, online, current, skipped = _collect_fleet()
    if not pad_configs:
        st.warning("No reviewed wells on any CFP pad. Review some wells first.")
        return

    n_on = sum(1 for v in online.values() if v)
    n_bol = sum(1 for v in online.values() if not v)
    st.markdown(
        f"**{n_on} online well(s)** and **{n_bol} bring-online candidate(s)** "
        "across "
        + ", ".join(f"{p} ({len(w)})" for p, w in sorted(pad_configs.items()))
        + ". Entries flagged *offline* on the Review stage are the BOL list."
    )
    if skipped:
        st.warning(
            "No reviewed pump for: " + ", ".join(sorted(skipped)) + " — set a "
            "current nozzle/throat for each on the Review stage (the model "
            "can't anchor a well without knowing what's in the hole today). "
            "Excluded from this run."
        )

    hist = _plant_history()
    now = current_state(hist)

    st.markdown("##### Anchor — today, measured")
    st.caption(
        "Every result is a **delta from here**. No exogenous water, no summing "
        "of well tests — injection, disposal and carryover all cancel in the "
        "deltas (docs/cfp_moves_methodology.md)."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        p0 = st.number_input(
            "PW discharge today (psi)",
            2300.0,
            2900.0,
            value=float(round(now["discharge_psi"])),
            step=5.0,
            key=f"{_PREFIX}_p0",
            help=f"Defaults to the live reading ({now['source']}).",
        )
    with c2:
        slope = st.slider(
            "Machine slope (psi per 1,000 BPD)",
            9.0,
            17.5,
            value=float(_cfp.MEASURED_PSI_PER_KBPD),
            step=0.25,
            key=f"{_PREFIX}_slope_cfg",
            help=(
                "Measured 13.7 (r²=0.54) on the real machine tags "
                "(FIC_5419S/20S/21S); pump-curve fit 17.5; operating trend "
                "12.2. Re-run at both ends if a conclusion looks close."
            ),
        )
    with c3:
        c_pad_pf = st.number_input(
            "C-Pad booster PF (psi)",
            1000.0,
            5000.0,
            value=float(round(_measured_pad_pf().get("C", {}).get("psi", 3400.0))),
            step=25.0,
            key=f"{_PREFIX}_c_pad_pf",
            help=(
                "C-Pad is boosted on-pad — its PF doesn't ride the discharge, "
                "but its water still moves the machines."
            ),
        )

    from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

    c4, c5 = st.columns(2)
    with c4:
        nozzles = st.multiselect(
            "Nozzle sizes to consider",
            NOZZLE_OPTIONS,
            default=["9", "10", "11", "12", "13", "14"],
            key=f"{_PREFIX}_nozzles",
        )
    with c5:
        throats = st.multiselect(
            "Throat ratios to consider",
            THROAT_OPTIONS,
            default=["A", "B", "C"],
            key=f"{_PREFIX}_throats",
        )
    st.caption(
        "Each well's CURRENT pump is always included automatically; every size "
        "here becomes a resize option for every well."
    )

    if not nozzles or not throats:
        st.warning("Pick at least one nozzle and one throat.")
        return

    st.divider()
    if st.button("▶ Analyze today's moves", type="primary", key=f"{_PREFIX}_run"):
        _run_moves(
            pad_configs, online, current, nozzles, throats, p0, slope, c_pad_pf
        )


def _fleet_signature(pad_configs, online, current, nozzles, throats, p0) -> tuple:
    """Cache key for the response surfaces — any input that changes the WOFFL
    physics must appear here (the CLAUDE.md sweep-signature rule)."""
    store_sigs = tuple(
        wrs.store_signature(
            {wc.well_name: store_for(p).get(wc.well_name, {}) for wc in wells}
        )
        for p, wells in sorted(pad_configs.items())
    )
    return (
        store_sigs,
        tuple(sorted(online.items())),
        tuple(sorted(current.items())),
        tuple(sorted(nozzles)),
        tuple(sorted(throats)),
        round(float(p0), 1),
    )


def _run_moves(
    pad_configs, online, current, nozzles, throats, p0, slope, c_pad_pf
) -> None:
    from woffl.gui import cfp_moves as cmv

    # Grid spans well below today (BOL sag) up to just under the trip.
    lo = p0 - 300.0
    grid = [round(lo + i * (2880.0 - lo) / 6.0, 1) for i in range(7)]
    sig = _fleet_signature(pad_configs, online, current, nozzles, throats, p0)
    cached = st.session_state.get(_SURF_CACHE_KEY)

    if cached and cached.get("sig") == sig:
        surfaces = cached["surfaces"]
    else:
        bar = st.progress(0.0, text="Building well response surfaces…")

        def _prog(i, n, pressure):
            bar.progress(
                min(i / n, 1.0),
                text=f"WOFFL batch {i}/{n} at discharge {pressure:,.0f} psi…",
            )

        try:
            with st.spinner(
                "Modelling every well at every size across the discharge grid "
                "(cached until an input changes)…"
            ):
                surfaces = cmv.build_response_surfaces(
                    pad_configs,
                    online,
                    current,
                    PLANT,
                    p_grid=grid,
                    nozzles=nozzles,
                    throats=throats,
                    p0=p0,
                    c_pad_pf_psi=c_pad_pf,
                    measured_pad_pf={
                        p: v["psi"] for p, v in _measured_pad_pf().items()
                    },
                    progress=_prog,
                )
        except Exception as e:
            bar.empty()
            st.error(f"Surface build failed: {e}")
            return
        bar.empty()
        st.session_state[_SURF_CACHE_KEY] = {"sig": sig, "surfaces": surfaces}

    plant = cmv.anchor(
        surfaces, psi_per_kbpd=slope, trip_psi=_cfp.MAX_DISCHARGE_PSI
    )
    try:
        summary = cmv.moves_summary(surfaces, plant)
    except Exception as e:
        st.error(f"Move analysis failed: {e}")
        return
    st.session_state[f"{_PREFIX}_moves"] = summary
    st.session_state[f"{_PREFIX}_page_stage"] = 3
    st.rerun()


# ── stage 3: today's moves ──────────────────────────────────────────────────


_MOVE_TYPE_LABEL = {
    "resize": "Resize",
    "shut_in": "Shut in",
    "bring_online": "Bring online",
}


def _fmt_label(lab) -> str:
    return {"SI": "shut in", "OFF": "offline", None: "—"}.get(lab, str(lab))


def _render_results() -> None:
    out = st.session_state.get(f"{_PREFIX}_moves")
    if not out:
        st.info("No analysis yet — run **Analyze today's moves** in stage 2.")
        return

    today = out["today"]
    lam = out["lambda_bopd_per_psi"]
    plan = out["plan"]

    st.markdown("##### Today")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Discharge (anchor)", f"{today['pressure']:,.0f} psi")
    m2.metric("Modeled fleet oil", f"{today['oil']:,.0f} BOPD")
    m3.metric(
        "Online / BOL candidates",
        f"{today['n_online']} / {today['n_bol_candidates']}",
    )
    m4.metric(
        "1 psi is worth",
        f"{lam:+,.1f} BOPD",
        help=(
            "d(fleet oil)/d(discharge) at today's configuration — the price "
            "of pressure. A knob's ΔP × this ≈ what the rest of the fleet "
            "gains or loses from the pressure move alone."
        ),
    )

    st.markdown("##### The answer")
    if plan and out["plan_gain"] > 1.0:
        st.success(
            f"**Best plan: {plan['n_changes']} change(s) → "
            f"{out['plan_gain']:+,.0f} BOPD**, landing at "
            f"{plan['pressure']:,.0f} psi discharge."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Action": _MOVE_TYPE_LABEL.get(a["type"], a["type"]),
                        "Well": a["well"],
                        "Pad": a["pad"],
                        "From": _fmt_label(a["from"]),
                        "To": _fmt_label(a["to"]),
                        "Own Δoil (BOPD)": round(a["own_oil_delta"]),
                        "Δwater (BPD)": round(a["own_water_delta"]),
                    }
                    for a in plan["actions"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif plan:
        st.info(
            "**Today's configuration is already at (or within noise of) the "
            "optimum** — no combination of resizes, shut-ins or bring-ons "
            "beats it by more than 1 BOPD under the current assumptions."
        )
    if plan and plan.get("at_trip"):
        st.warning(
            "The plan sits at the trip cap — disposal would have to re-trim, "
            "and any further water shedding is pure oil loss."
        )

    st.markdown("##### Every knob, priced")
    st.caption(
        "One change from today per row, exactly re-settled: the well's own oil "
        "change plus what the pressure move does to everyone else."
    )
    singles = out["singles"]
    tabs = st.tabs(["Resize", "Shut in", "Bring online"])
    for tab, mtype in zip(tabs, ("resize", "shut_in", "bring_online")):
        with tab:
            rows = [m for m in singles if m["type"] == mtype][:15]
            if not rows:
                st.caption("No candidates.")
                continue
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Well": m["well"],
                            "Pad": m["pad"],
                            "From": _fmt_label(m["from"]),
                            "To": _fmt_label(m["to"]),
                            "Fleet Δoil (BOPD)": round(m["fleet_oil_delta"], 1),
                            "Own Δoil (BOPD)": round(m["own_oil_delta"], 1),
                            "ΔP (psi)": round(m["pressure_delta"], 1),
                            "P after": round(m["pressure_after"]),
                            "Note": "⚠ at trip" if m["at_trip"] else "",
                        }
                        for m in rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    if out["pairs"]:
        with st.expander(
            f"Bring-on + offset pairs ({len(out['pairs'])}) — BOL a well AND "
            "hold pressure with a downsize/SI elsewhere",
            expanded=True,
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Bring online": (
                                f"{p['bring_on']['well']} @ {p['bring_on']['to']}"
                            ),
                            "Offset": (
                                f"{p['offset']['well']} → "
                                f"{_fmt_label(p['offset']['to'])}"
                            ),
                            "Fleet Δoil (BOPD)": round(p["fleet_oil_delta"], 1),
                            "ΔP (psi)": round(p["pressure_delta"], 1),
                            "P after": round(p["pressure_after"]),
                        }
                        for p in out["pairs"][:10]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("##### Oil vs pressure — the whole frontier")
    _plot_frontier(out)
    st.caption(
        "Each point is the best fleet configuration at one water price (the "
        "equal-slope sweep). Today and the plan are marked; past the trip line "
        "disposal re-trims and shedding stops paying. Method + literature: "
        "docs/cfp_moves_methodology.md."
    )


def _plot_frontier(out: dict) -> None:
    import plotly.graph_objects as go

    fr = out["frontier"]
    today = out["today"]
    plan = out["plan"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[s["pressure"] for s in fr],
            y=[s["oil"] for s in fr],
            mode="lines+markers",
            name="frontier",
            line=dict(color="#1565C0", width=2),
            hovertemplate="%{x:,.0f} psi<br>%{y:,.0f} BOPD<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[today["pressure"]],
            y=[today["oil"]],
            mode="markers+text",
            name="today",
            text=["today"],
            textposition="bottom center",
            marker=dict(color="#E65100", size=14, symbol="diamond"),
        )
    )
    if plan:
        fig.add_trace(
            go.Scatter(
                x=[plan["pressure"]],
                y=[plan["oil"]],
                mode="markers+text",
                name="best plan",
                text=["plan"],
                textposition="top center",
                marker=dict(color="#2E7D32", size=14, symbol="star"),
            )
        )
    fig.add_vline(
        x=_cfp.MAX_DISCHARGE_PSI,
        line=dict(color="#D32F2F", dash="dash"),
        annotation_text="2,900 trip",
    )
    fig.update_layout(
        xaxis_title="PW discharge (psi)",
        yaxis_title="Fleet oil (BOPD)",
        height=380,
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── page entry ──────────────────────────────────────────────────────────────


def run_cfp_page() -> None:
    st.title("PW Pressure Optimization — CFP · J / G / C / B")
    st.caption(
        "The four CFP-side pads share one produced-water plant, so they are "
        "optimized together: plant discharge is a decision (set by disposal "
        "throttling), and each pad receives its own delivered power fluid."
    )

    # Autosave/restore the four pads' review stores to Databricks, and the
    # page's own knobs — BEFORE any widget renders (seeding rule).
    from woffl.gui import review_persistence as rp

    sync_results = {p: rp.sync_pad(p) for p in PADS}
    rp.render_caption(sync_results)

    stage_key = f"{_PREFIX}_page_stage"
    stage = st.session_state.setdefault(stage_key, 0)
    have_results = bool(st.session_state.get(f"{_PREFIX}_moves"))
    have_wells = any(wrs.active_entries(store_for(p)) for p in PADS)

    cols = st.columns(len(_STAGES))
    for i, label in enumerate(_STAGES):
        # Stage 0 (dashboard) and 1 (review) always open; 2 needs wells, 3 a run.
        unlocked = i <= 1 or (i == 2 and have_wells) or (i == 3 and have_results)
        with cols[i]:
            if i == stage:
                st.markdown(f"**:blue[{label}]**")
            elif unlocked:
                if st.button(label, key=f"{_PREFIX}_nav_{i}", use_container_width=True):
                    st.session_state[stage_key] = i
                    st.rerun()
            else:
                st.markdown(f":gray[{label}]")
    st.divider()

    if stage == 0:
        _render_dashboard()
    elif stage == 1:
        _render_review()
    elif stage == 2:
        _render_configure()
    else:
        _render_results()
