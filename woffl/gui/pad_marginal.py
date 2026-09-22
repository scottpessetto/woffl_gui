"""Cost of power fluid on a fixed-speed pad (S-Pad): marginal PF water cut
and single-well pump decisions.

The engineer's question: "if one well takes another 1,000 BPD of PF - or
gives 1,000 BPD back - what does that cost, in barrels of oil, across all the
OTHER wells?" S-Pad's boosters always run at 60 Hz, so the header is not a
setpoint: it sits where the pump curve meets the wells' total PF demand.
More draw walks the curve down, every well sees a lower PF pressure, and each
loses some oil (and PF, which partly offsets the drop).

Everything here is priced on that coupled curve:

* **Operating point** - the header ``H0`` where
  ``header_at_flow(sum PF_i(H0)) == H0``, every well on its installed pump.
* **PF sensitivity** - an extra draw ``+/-dQ`` (a stand-in for one well
  taking more or less PF) resettles the header; the other wells' oil change
  is the cost (or gain). ``lambda = oil lost / dQ`` and the marginal PF water
  cut is ``1 / (1 + lambda)``: extra oil bought with extra PF must come at an
  incremental PF water cut below it.
* **Candidates** - each pump size for the target well, resettled on the curve
  with the target's own header-dependent PF. Net pad oil = the target's gain
  plus every other well's change, all at the new header. No linearization:
  a large upsize is charged the real, deeper header drop.

PURE compute over per-well response curves ``[(header_psi, oil, pf), ...]``
(installed pumps modeled at a few headers, linearly interpolated), so tests
need no solver. PF water cut = PF / (PF + oil), the stream a PF-only pad's
boosters handle (Well Sort's S-Pad basis).
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Callable, Mapping, Optional, Sequence

# Response points: (wellhead PF psi, oil BOPD, PF BPD).
Point = tuple[float, float, float]
Curve = Sequence[Point]
HeaderOfFlow = Callable[[float], Optional[float]]

# Offsets around the operating point where each well's response is modeled.
FINE_OFFSETS_PSI = (-400.0, -200.0, -100.0, 0.0, 100.0, 200.0)
SENSITIVITY_BPD = 1000.0


def pfwc_from_lambda(lam: Optional[float]) -> Optional[float]:
    """Marginal PF water cut equivalent to an oil-per-PF price."""
    if lam is None or not isfinite(lam) or lam < 0:
        return None
    return 1.0 / (1.0 + lam)


def incremental_pfwc(d_oil: float, d_pf: float) -> Optional[float]:
    """PF water cut of an increment: ``dPF / (dPF + dOil)``. None unless the
    increment spends PF and adds oil (the only case a cut is meaningful)."""
    if d_pf <= 0 or d_oil <= 0:
        return None
    return d_pf / (d_pf + d_oil)


def clean_curve(points: Curve) -> list[Point]:
    """Finite, positive-PF points sorted by header, one per header."""
    out: dict[float, Point] = {}
    for h, oil, pf in points:
        h, oil, pf = float(h), float(oil), float(pf)
        if isfinite(h) and isfinite(oil) and isfinite(pf) and pf > 0 and oil >= 0:
            out[h] = (h, oil, pf)
    return [out[h] for h in sorted(out)]


def at_header(curve: Sequence[Point], header: float) -> tuple[float, float]:
    """(oil, PF) at ``header`` by linear interpolation; flat beyond the
    modeled range (candidates report when a settle leaves it)."""
    if not curve:
        return 0.0, 0.0
    if header <= curve[0][0]:
        return curve[0][1], curve[0][2]
    if header >= curve[-1][0]:
        return curve[-1][1], curve[-1][2]
    for (h0, o0, p0), (h1, o1, p1) in zip(curve, curve[1:]):
        if h0 <= header <= h1:
            t = (header - h0) / (h1 - h0)
            return o0 + t * (o1 - o0), p0 + t * (p1 - p0)
    return curve[-1][1], curve[-1][2]


def settle(curves: Mapping[str, Sequence[Point]], extra_pf: float, header_of_flow: HeaderOfFlow,
           lo: float, hi: float, tol: float = 0.5) -> Optional[float]:
    """Header where the pump curve meets the wells' demand plus ``extra_pf``.

    ``g(H) = header_of_flow(sum PF(H) + extra) - H`` falls with H (more
    pressure draws more PF, which the curve answers with less head), so the
    root is unique; bisection within ``[lo, hi]``, clamped to the band's
    ends. None when the curve has no head for the flow at the floor.
    """
    def g(h: float) -> Optional[float]:
        q = sum(at_header(c, h)[1] for c in curves.values()) + extra_pf
        delivered = header_of_flow(max(q, 0.0))
        return None if delivered is None else delivered - h

    g_lo, g_hi = g(lo), g(hi)
    if g_lo is None:
        return None
    if g_lo <= 0:
        return lo
    if g_hi is not None and g_hi >= 0:
        return hi
    a, b = lo, hi
    for _ in range(80):
        if b - a <= tol:
            break
        m = 0.5 * (a + b)
        gm = g(m)
        if gm is not None and gm > 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def well_rates(curves: Mapping[str, Sequence[Point]], header: float) -> dict[str, tuple[float, float]]:
    return {w: at_header(c, header) for w, c in curves.items()}


def pf_sensitivity(others: Mapping[str, Sequence[Point]], base_curves: Mapping[str, Sequence[Point]],
                   header_of_flow: HeaderOfFlow, h0: float, lo: float, hi: float,
                   d_q: float = SENSITIVITY_BPD) -> dict[str, Any]:
    """What an extra ``+d_q`` (or ``-d_q``) BPD of PF draw costs (or gives)
    the OTHER wells, resettled on the curve. ``base_curves`` is every well
    at today's operating point (the target included when it produces)."""
    base = well_rates(others, h0)
    out: dict[str, Any] = {"d_q": d_q}
    for sign, key in ((1.0, "add"), (-1.0, "remove")):
        h = settle(base_curves, sign * d_q, header_of_flow, lo, hi)
        if h is None:
            out[key] = None
            continue
        now = well_rates(others, h)
        per = [{"well": w, "d_oil": now[w][0] - base[w][0], "d_pf": now[w][1] - base[w][1]} for w in others]
        d_oil = sum(r["d_oil"] for r in per)
        out[key] = {
            "header_psi": h,
            "d_header_psi": h - h0,
            "others_d_oil": d_oil,
            "others_d_pf": sum(r["d_pf"] for r in per),
            # oil per BPD of the extra draw: lost when adding, gained when removing
            "lambda": max(0.0, -sign * d_oil / d_q),
            "wells": sorted(per, key=lambda r: r["d_oil"] * sign),
        }
    add, rem = out.get("add"), out.get("remove")
    out["lambda"] = add["lambda"] if add else None
    out["pfwc"] = pfwc_from_lambda(out["lambda"])
    out["remove_lambda"] = rem["lambda"] if rem else None
    out["remove_pfwc"] = pfwc_from_lambda(out["remove_lambda"])
    return out


def pf_sweep(others: Mapping[str, Sequence[Point]], base_curves: Mapping[str, Sequence[Point]],
             header_of_flow: HeaderOfFlow, h0: float, lo: float, hi: float,
             span_bpd: float, steps: int = 10) -> list[dict]:
    """The chart of the sensitivity: extra draw from ``-span_bpd`` to
    ``+span_bpd`` in ``2 * steps`` increments, each resettled on the curve,
    with the other wells' total and per-well oil change. ``extrapolated``
    marks headers outside the other wells' modeled range."""
    base = well_rates(others, h0)
    lo_mod = min((c[0][0] for c in others.values() if c), default=h0)
    hi_mod = max((c[-1][0] for c in others.values() if c), default=h0)
    out = []
    for i in range(-steps, steps + 1):
        d_q = span_bpd * i / steps
        h = h0 if i == 0 else settle(base_curves, d_q, header_of_flow, lo, hi)
        if h is None:
            continue
        now = well_rates(others, h)
        wells = {w: now[w][0] - base[w][0] for w in others}
        out.append({
            "d_q": d_q,
            "header_psi": h,
            "d_header_psi": h - h0,
            "others_d_oil": sum(wells.values()),
            "others_d_pf": sum(now[w][1] - base[w][1] for w in others),
            "wells": wells,
            "extrapolated": not (lo_mod - 1e-6 <= h <= hi_mod + 1e-6),
        })
    return out


def score_candidates(others: Mapping[str, Sequence[Point]], target_base: Optional[Sequence[Point]],
                     candidates: Sequence[dict], header_of_flow: HeaderOfFlow, h0: float,
                     lo: float, hi: float, lam: Optional[float]) -> list[dict]:
    """Resettle the pad for every candidate pump in the target well.

    ``target_base`` is the installed pump's curve (None for a new or
    offline well: an empty slot). Each candidate carries ``curve``. Net pad
    oil = target oil change + every other well's change at the candidate's
    header. ``beats_marginal`` is the linear screen: the target's increment
    has a PF water cut below the pad marginal ``1 / (1 + lam)``.
    """
    base_t = at_header(target_base, h0) if target_base else (0.0, 0.0)
    base_o = well_rates(others, h0)
    marginal = pfwc_from_lambda(lam)
    rows = []
    for cand in candidates:
        curve = cand["curve"]
        row = {k: v for k, v in cand.items() if k != "curve"}
        h = settle({**others, "__target__": curve}, 0.0, header_of_flow, lo, hi)
        if h is None:
            rows.append({**row, "header_psi": None, "net_oil": None, "beats_marginal": None})
            continue
        oil_t, pf_t = at_header(curve, h)
        now = well_rates(others, h)
        d_oil_t, d_pf_t = oil_t - base_t[0], pf_t - base_t[1]
        others_d_oil = sum(now[w][0] - base_o[w][0] for w in others)
        inc = incremental_pfwc(d_oil_t, d_pf_t)
        if d_pf_t <= 0 and d_oil_t >= 0 and (d_pf_t < 0 or d_oil_t > 0):
            beats = True  # more oil for no more PF
        elif d_pf_t > 0 and d_oil_t <= 0:
            beats = False  # more PF for no more oil
        elif inc is None or marginal is None:
            beats = None
        else:
            beats = inc < marginal
        rows.append({
            **row,
            "oil": oil_t, "pf": pf_t,
            "header_psi": h, "d_header_psi": h - h0,
            "d_oil": d_oil_t, "d_pf": d_pf_t,
            "inc_pfwc": inc,
            "oil_per_mbpd": d_oil_t / d_pf_t * 1000.0 if d_pf_t > 0 else None,
            "others_d_oil": others_d_oil,
            "net_oil": d_oil_t + others_d_oil,
            "beats_marginal": beats,
            "extrapolated": bool(curve) and not (curve[0][0] - 1e-6 <= h <= curve[-1][0] + 1e-6),
        })
    rows.sort(key=lambda r: (r["net_oil"] is None, -(r["net_oil"] or 0.0)))
    return rows
