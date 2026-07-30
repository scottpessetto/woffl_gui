"""Water vs pressure — the case for how to run the CFP plant.

Scott, 2026-07-30: *"the whole point of this tool is to make the case on what is
the right way to run? higher water pressure or more water"*, and the mechanism:
*"I don't hold 2900 psi because I am trying to produce as much oil as possible.
So I bring on wells or jet pumps and push the water handled higher causing
discharge pressure to sag."*

So the operating point is already an optimization made by hand. This module says
whether it's on the right side of the line.

THE TRADE, IN ONE LINE
----------------------
A new barrel of water brings its own oil, and costs a little oil on every
pressure-responsive well already online. Break-even is the water cut at which
those two cancel::

    cost   = exposed_oil x responsive_frac x oil_sens_per_psi x psi_per_kbwpd
    gain   = 1000 x (1 - wc) / wc                       [BOPD per 1,000 BWPD]
    wc*    = 1000 / (1000 + cost)

Below ``wc*`` bring the water on; above it, the barrel costs more than it pays.

THE TWO INPUTS, AND WHICH ONE IS SHAKY
--------------------------------------
* ``psi_per_kbwpd`` — well constrained, **9-26**, centre ~12-17. Three
  independent estimates agree: Scott's own Mar->Jul trend (+131 psi for −10,750
  BWPD of machine flow = 12.2), April's within-month fit (9.0 at r²=0.80), and
  the SCADA-validated pump curve (~26).
* ``oil_sens_frac_per_psi`` x ``responsive_frac`` — **shaky, and it dominates the
  answer.** Measured +2.4-2.9% oil per 108 psi on lift-limited wells
  (~0.00025/psi) and **exactly 0% on inflow-limited ones** — those are IPR-bound,
  so more power fluid buys nothing. What fraction of the real fleet responds is
  unknown; ``responsive_frac`` makes that assumption explicit instead of hiding
  it. Sweep it rather than trusting one value.

THE KINK
--------
Cutting water raises discharge only until the piping trip (2,900 psi). Past that
the pressure is capped, so further cuts are pure oil loss with no offsetting
gain. :func:`tradeoff_curve` models that kink — it is what makes "cut water to
raise PF" stop paying well before you run out of wells to cut.
"""

from dataclasses import dataclass
from typing import Optional

# Bracket from three independent estimates; see the module docstring.
PSI_PER_KBWPD_LOW = 9.0
PSI_PER_KBWPD_MID = 12.2
PSI_PER_KBWPD_HIGH = 26.0

# Measured on lift-limited wells over the 108 psi lever. Inflow-limited wells
# measured exactly 0.0, hence `responsive_frac`.
OIL_SENS_FRAC_PER_PSI = 0.00025


@dataclass
class TradeoffInputs:
    """Everything the verdict depends on, all explicit."""

    exposed_oil_bopd: float          # oil on pads whose PF rides the discharge
    current_water_bwpd: float        # what the plant handles today
    current_discharge_psi: float
    max_discharge_psi: float = 2900.0
    psi_per_kbwpd: float = PSI_PER_KBWPD_MID
    oil_sens_frac_per_psi: float = OIL_SENS_FRAC_PER_PSI
    responsive_frac: float = 0.5
    # WC of the water you'd actually bring on next (or cut). None = unknown.
    marginal_wc: Optional[float] = None


def oil_from_water(water_bwpd: float, wc: float) -> float:
    """Oil that accompanies ``water_bwpd`` at water cut ``wc``."""
    wc = min(max(float(wc), 1e-9), 1.0 - 1e-9)
    return float(water_bwpd) * (1.0 - wc) / wc


def bopd_per_psi(inp: TradeoffInputs) -> float:
    """The communication number: one psi of discharge is worth this many BOPD.

    ``exposed_oil × responsive_frac × oil_sens_frac_per_psi`` — the
    assumption-based estimate the dashboard shows before a Today's Moves run
    supplies the per-well figure (``cfp_moves.shadow_price_today``).
    """
    return (
        float(inp.exposed_oil_bopd)
        * float(inp.responsive_frac)
        * float(inp.oil_sens_frac_per_psi)
    )


def marginal_cost_bopd_per_kbwpd(inp: TradeoffInputs) -> float:
    """Oil given up across the fleet per +1,000 BWPD, via the pressure sag."""
    return (
        float(inp.exposed_oil_bopd)
        * float(inp.responsive_frac)
        * float(inp.oil_sens_frac_per_psi)
        * float(inp.psi_per_kbwpd)
    )


def breakeven_wc(inp: TradeoffInputs) -> float:
    """Water cut at which a new barrel exactly pays for the pressure it costs.

    Above it the barrel is not worth handling; below it, bring it on.
    """
    cost = marginal_cost_bopd_per_kbwpd(inp)
    return 1000.0 / (1000.0 + cost)


def discharge_at_water(inp: TradeoffInputs, water_bwpd: float) -> float:
    """Discharge pressure at a given handled-water rate, capped at the trip."""
    delta_k = (float(water_bwpd) - float(inp.current_water_bwpd)) / 1000.0
    psi = float(inp.current_discharge_psi) - delta_k * float(inp.psi_per_kbwpd)
    return min(psi, float(inp.max_discharge_psi))


def tradeoff_curve(
    inp: TradeoffInputs,
    *,
    span_bwpd: float = 20000.0,
    steps: int = 41,
    marginal_wc: Optional[float] = None,
) -> list:
    """Total oil against water handled, across ±``span_bwpd`` of today.

    Every point is relative to today, so the current operating point sits at
    ``delta_water = 0`` with ``delta_oil = 0``. Includes the 2,900 psi kink:
    once the trip caps the pressure, cutting further stops buying anything.
    """
    wc = inp.marginal_wc if marginal_wc is None else marginal_wc
    if wc is None:
        raise ValueError("marginal_wc is required — the trade has no answer without it")

    base_psi = discharge_at_water(inp, inp.current_water_bwpd)
    rows = []
    for i in range(steps):
        frac = -1.0 + 2.0 * i / (steps - 1)
        dw = frac * float(span_bwpd)
        water = float(inp.current_water_bwpd) + dw
        if water < 0:
            continue
        psi = discharge_at_water(inp, water)
        d_psi = psi - base_psi
        # Oil that rides in with (or leaves with) the water…
        d_oil_water = oil_from_water(dw, wc)
        # …and the fleet-wide response to the pressure change.
        d_oil_press = (
            float(inp.exposed_oil_bopd)
            * float(inp.responsive_frac)
            * float(inp.oil_sens_frac_per_psi)
            * d_psi
        )
        rows.append(
            {
                "water_bwpd": water,
                "delta_water_bwpd": dw,
                "discharge_psi": psi,
                "delta_psi": d_psi,
                "delta_oil_from_water": d_oil_water,
                "delta_oil_from_pressure": d_oil_press,
                "delta_oil_bopd": d_oil_water + d_oil_press,
                "at_trip": psi >= float(inp.max_discharge_psi) - 1e-9,
            }
        )
    return rows


def water_at_trip(inp: TradeoffInputs) -> float:
    """Handled-water rate at which discharge reaches the trip.

    Cutting below this buys no more pressure — the point where "cut water to
    raise PF" stops paying entirely.
    """
    head = float(inp.max_discharge_psi) - float(inp.current_discharge_psi)
    if inp.psi_per_kbwpd <= 0:
        return float("nan")
    return float(inp.current_water_bwpd) - (head / float(inp.psi_per_kbwpd)) * 1000.0


def verdict(inp: TradeoffInputs, marginal_wc: Optional[float] = None) -> dict:
    """The headline: bring water on, cut it, or hold.

    ``marginal_wc`` is the water cut of the barrel actually on the table — the
    next well or pump size up, or the worst well you'd cut. The verdict compares
    it against break-even.
    """
    wc = inp.marginal_wc if marginal_wc is None else marginal_wc
    cost = marginal_cost_bopd_per_kbwpd(inp)
    be = breakeven_wc(inp)
    trip_water = water_at_trip(inp)
    out = {
        "cost_bopd_per_kbwpd": cost,
        "breakeven_wc": be,
        "psi_per_kbwpd": float(inp.psi_per_kbwpd),
        "headroom_psi": float(inp.max_discharge_psi) - float(inp.current_discharge_psi),
        "water_at_trip_bwpd": trip_water,
        "water_to_cut_for_trip_bwpd": max(
            0.0, float(inp.current_water_bwpd) - trip_water
        ),
        "marginal_wc": wc,
    }
    if wc is None:
        out.update({"action": "unknown", "reason": "no marginal water cut supplied"})
        return out

    gain = oil_from_water(1000.0, wc)
    out["gain_bopd_per_kbwpd"] = gain
    out["net_bopd_per_kbwpd"] = gain - cost
    if wc < be - 1e-6:
        out["action"] = "more_water"
        out["reason"] = (
            f"water at {wc:.1%} WC brings {gain:,.0f} BOPD per 1,000 BWPD and costs "
            f"{cost:,.0f} in lost pressure — net {gain - cost:+,.0f}. Break-even is "
            f"{be:.1%} WC."
        )
    elif wc > be + 1e-6:
        out["action"] = "cut_water"
        out["reason"] = (
            f"water at {wc:.1%} WC brings only {gain:,.0f} BOPD per 1,000 BWPD against "
            f"{cost:,.0f} of lost pressure — net {gain - cost:+,.0f}. Cutting it (or "
            f"downsizing the pump) pays. Break-even is {be:.1%} WC."
        )
    else:
        out["action"] = "hold"
        out["reason"] = f"the marginal barrel is right at break-even ({be:.1%} WC)."
    return out


def sensitivity_table(inp: TradeoffInputs, marginal_wc: float) -> list:
    """Break-even across the plausible slope x responsiveness grid.

    The honest way to present this: the pressure cost is well constrained, the
    oil response is not, so show the answer over the whole uncertainty box rather
    than a single number that hides it.
    """
    rows = []
    for label, slope in (
        ("low (9)", PSI_PER_KBWPD_LOW),
        ("mid (12.2)", PSI_PER_KBWPD_MID),
        ("high (26)", PSI_PER_KBWPD_HIGH),
    ):
        for resp in (0.25, 0.5, 1.0):
            probe = TradeoffInputs(
                exposed_oil_bopd=inp.exposed_oil_bopd,
                current_water_bwpd=inp.current_water_bwpd,
                current_discharge_psi=inp.current_discharge_psi,
                max_discharge_psi=inp.max_discharge_psi,
                psi_per_kbwpd=slope,
                oil_sens_frac_per_psi=inp.oil_sens_frac_per_psi,
                responsive_frac=resp,
            )
            v = verdict(probe, marginal_wc)
            rows.append(
                {
                    "slope": label,
                    "responsive_frac": resp,
                    "cost_bopd_per_kbwpd": v["cost_bopd_per_kbwpd"],
                    "breakeven_wc": v["breakeven_wc"],
                    "net_bopd_per_kbwpd": v.get("net_bopd_per_kbwpd"),
                    "action": v["action"],
                }
            )
    return rows
