"""Shut-in ladder — is it worth shutting wells in to buy CFP power-fluid pressure?

Answers the operating question directly: for each well, does the pressure its
water frees up buy more oil across the rest of the field than the well itself
makes? Walk the online list worst-first, and the peak of the resulting curve is
the answer. The water cut at the peak is the "above X% WC, shut in" threshold.

Pure — no Streamlit, no Databricks. The page feeds it rows and renders the curve.

WHICH WATER COUNTS (Scott, 2026-07-29)
--------------------------------------
* **POPS pads** (E/F/H/I/M/S) handle their own lift water on-pad, so only their
  **formation water** reaches the plant.
* **Non-POPS pads** (B/G/C/J, L, K, R, D…) send **all** their water.
* Any pad can be filtered out.

Caveat worth knowing: M/F/E are *full* POPS — nominally they handle formation
water on-pad too, so counting all of it as plant load overstates them. Their real
contribution is imperfect-disposal carryover. Filtering them out models that; the
default follows Scott's stated rule.

WHAT THIS CAN AND CANNOT TELL YOU
---------------------------------
The **ranking and the shape** are trustworthy now. The **magnitude is not**, for
two reasons, and both cut toward "keep the wells on":

1. The lever is capped. The plant runs at ~2,792 psi against a 2,900 psi piping
   trip, so delivered PF can rise **at most ~108 psi** no matter how much water
   you shed.
2. How much water buys that 108 psi is uncertain by 10x — 6,171 BWPD on the
   provisional curve, 17,763 on the measured PlantWater fit, or 60,335 on the
   measured ProdWater fit. That last exceeds ALL 40,562 BWPD of controllable
   CFP-pad water, i.e. the lever wouldn't exist at all.

``oil_sens_frac_per_psi`` is therefore an EXPLICIT input rather than something
inferred and hidden. At 0.0 the ladder shows the pure arithmetic — oil doesn't
respond to pressure, so shutting in is always a straight loss and the curve falls
monotonically. That null result is correct and is the right default to argue
against.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from woffl.assembly.network_optimizer import derive_pad

# Pads with on-pad water handling. The canonical list lives in
# ``scotts_tools.well_sort._DEFAULT_POPS_PADS`` (Well Sort lets the engineer
# edit it live); this is the static default for the pure engine.
POPS_PADS: tuple[str, ...] = ("E", "F", "H", "I", "M", "S")

RANKINGS = ("marginal", "wc", "volume")


@dataclass
class LadderWell:
    """One online producer as the ladder sees it."""

    well: str
    oil_bopd: float
    form_wat_bwpd: float
    lift_wat_bwpd: float = 0.0
    pad: str = ""
    # Fractional oil change per psi of delivered PF. 0.0 = no response.
    oil_sens_frac_per_psi: float = 0.0
    # Delivered PF this well sees today, i.e. the baseline the sensitivity is
    # measured from. None → the ladder uses the pad's baseline.
    pf_baseline_psi: Optional[float] = None

    def __post_init__(self) -> None:
        self.pad = derive_pad(self.pad or self.well)

    @property
    def total_wat_bwpd(self) -> float:
        return float(self.form_wat_bwpd) + float(self.lift_wat_bwpd)

    @property
    def total_wc(self) -> float:
        """Total water cut on the produced stream (oil + all water)."""
        denom = float(self.oil_bopd) + self.total_wat_bwpd
        return self.total_wat_bwpd / denom if denom > 0 else 1.0

    def plant_water_bwpd(self, pops_pads: Iterable[str] = POPS_PADS) -> float:
        """Water this well contributes to the CENTRAL plant load.

        POPS pads handle lift on-pad, so only formation water travels; every
        other pad sends the lot.
        """
        if self.pad in set(pops_pads):
            return float(self.form_wat_bwpd)
        return self.total_wat_bwpd

    def oil_per_plant_barrel(self, pops_pads: Iterable[str] = POPS_PADS) -> float:
        """Oil made per barrel of water this well puts through the plant.

        The metric that actually decides whether shutting in pays. A 99%-WC
        stripper making 40 BWPD scores WELL here (it frees almost no capacity, so
        it costs little to keep), while raw WC would condemn it — which is why
        ``"wc"`` ranking and ``"marginal"`` ranking disagree.
        """
        w = self.plant_water_bwpd(pops_pads)
        if w <= 0:
            return float("inf")  # contributes no plant load — never shut it in
        return float(self.oil_bopd) / w


@dataclass
class LadderRung:
    """The field with the worst ``k`` wells shut in."""

    k: int
    shut_in: list
    plant_load_bwpd: float
    discharge_psi: Optional[float]
    per_pad_pf: dict
    total_oil_bopd: float
    oil_delta_bopd: float
    feasible: bool
    trusted_band: bool
    # The well shut in AT this rung (None at k=0) and its WC — this is what makes
    # the "above X% WC" threshold readable off the curve.
    marginal_well: Optional[str] = None
    marginal_well_wc: Optional[float] = None
    note: str = ""


def rank_wells(
    wells: list, ranking: str = "marginal", pops_pads: Iterable[str] = POPS_PADS
) -> list:
    """Order wells worst-first for the ladder.

    * ``"marginal"`` — least oil per barrel of plant water first. The metric that
      decides whether SI pays.
    * ``"wc"`` — highest total water cut first. How the field talks about it, but
      it condemns high-WC strippers that free almost no capacity.
    * ``"volume"`` — most plant water first. Maximizes pressure freed per well
      shut in while ignoring the oil it costs.
    """
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}, got {ranking!r}")
    if ranking == "marginal":
        key = lambda w: (w.oil_per_plant_barrel(pops_pads), -w.plant_water_bwpd(pops_pads))
    elif ranking == "wc":
        key = lambda w: (-w.total_wc, -w.plant_water_bwpd(pops_pads))
    else:
        key = lambda w: (-w.plant_water_bwpd(pops_pads), w.oil_per_plant_barrel(pops_pads))
    return sorted(wells, key=key)


def _oil_at(well: LadderWell, pf_psi: Optional[float], baseline_psi: float) -> float:
    """A well's oil at a delivered PF, from its own linear sensitivity."""
    if pf_psi is None:
        return float(well.oil_bopd)
    base = well.pf_baseline_psi if well.pf_baseline_psi is not None else baseline_psi
    delta = float(pf_psi) - float(base)
    scaled = float(well.oil_bopd) * (1.0 + well.oil_sens_frac_per_psi * delta)
    return max(scaled, 0.0)


def build_ladder(
    wells: list,
    plant,
    *,
    exogenous_bwpd: float,
    baseline_discharge_psi: float,
    c_pad_pf_psi: float = 3400.0,
    measured_pad_pf: Optional[dict] = None,
    ranking: str = "marginal",
    pads_included: Optional[Iterable[str]] = None,
    pops_pads: Iterable[str] = POPS_PADS,
    max_rungs: Optional[int] = None,
) -> list:
    """Walk the shut-in ladder and return one :class:`LadderRung` per step.

    ``pads_included`` filters which pads participate AT ALL — excluded wells
    contribute neither water nor oil, which is how you model "M/F/E really only
    send carryover" or "don't touch J".

    ``baseline_discharge_psi`` is where the plant sits today (the measured value,
    not a curve inversion) — every well's sensitivity is referenced to the PF it
    sees at that discharge, so rung 0 reproduces today's oil exactly.
    """
    if pads_included is not None:
        keep = {str(p).strip().upper()[:1] for p in pads_included}
        wells = [w for w in wells if w.pad in keep]
    if not wells:
        return []

    ordered = rank_wells(wells, ranking, pops_pads)
    n = len(ordered) if max_rungs is None else min(len(ordered), int(max_rungs))

    def _pf_map(disch: Optional[float]) -> dict:
        pads = sorted({w.pad for w in ordered})
        if disch is None:
            return {}
        from woffl.gui.cfp_optimize import delivered_by_pad

        per_pad, _clamped = delivered_by_pad(
            plant, disch, pads, c_pad_pf_psi=c_pad_pf_psi,
            measured_pad_pf=measured_pad_pf,
        )
        return per_pad

    baseline_pf = _pf_map(baseline_discharge_psi)

    rungs: list = []
    baseline_oil: Optional[float] = None
    for k in range(n + 1):
        shut = [w.well for w in ordered[:k]]
        live = ordered[k:]
        load = sum(w.plant_water_bwpd(pops_pads) for w in live) + float(exogenous_bwpd)
        disch = plant.header_at_flow(load)
        feasible = disch is not None
        per_pad = _pf_map(disch)
        oil = sum(
            _oil_at(
                w,
                per_pad.get(w.pad),
                baseline_pf.get(w.pad, baseline_discharge_psi),
            )
            for w in live
        )
        if baseline_oil is None:
            baseline_oil = oil
        marg = ordered[k - 1] if k > 0 else None
        rungs.append(
            LadderRung(
                k=k,
                shut_in=shut,
                plant_load_bwpd=load,
                discharge_psi=disch,
                per_pad_pf=per_pad,
                total_oil_bopd=oil,
                oil_delta_bopd=oil - baseline_oil,
                feasible=feasible,
                trusted_band=(
                    bool(plant.flags(load)["trusted_band"]) if feasible else False
                ),
                marginal_well=marg.well if marg else None,
                marginal_well_wc=marg.total_wc if marg else None,
                note=(
                    ""
                    if feasible
                    else "plant cannot pass this much water at any modelable pressure"
                ),
            )
        )
    return rungs


def best_rung(rungs: list) -> Optional[LadderRung]:
    """The feasible rung with the most oil. Ties go to the FEWEST shut-ins —
    never recommend shutting a well in for nothing."""
    feasible = [r for r in rungs if r.feasible]
    if not feasible:
        return None
    best = max(feasible, key=lambda r: (r.total_oil_bopd, -r.k))
    return best


def wc_threshold(rungs: list) -> Optional[float]:
    """The "shut in above this WC" number: the water cut of the last well the
    peak rung shuts in. ``None`` when the peak is "shut nothing in" — which is
    the honest answer whenever the pressure lever doesn't pay.
    """
    best = best_rung(rungs)
    if best is None or best.k == 0:
        return None
    return best.marginal_well_wc


def ladder_summary(rungs: list) -> dict:
    """Headline read of a ladder: peak, threshold, and what it's worth."""
    best = best_rung(rungs)
    base = next((r for r in rungs if r.k == 0), None)
    if best is None or base is None:
        return {"feasible": False}
    return {
        "feasible": True,
        "best_k": best.k,
        "shut_in": list(best.shut_in),
        "oil_gain_bopd": best.total_oil_bopd - base.total_oil_bopd,
        "baseline_oil_bopd": base.total_oil_bopd,
        "best_oil_bopd": best.total_oil_bopd,
        "wc_threshold": wc_threshold(rungs),
        "baseline_discharge_psi": base.discharge_psi,
        "best_discharge_psi": best.discharge_psi,
        "pressure_gain_psi": (
            (best.discharge_psi - base.discharge_psi)
            if (best.discharge_psi and base.discharge_psi)
            else None
        ),
        "water_shed_bwpd": base.plant_load_bwpd - best.plant_load_bwpd,
        "recommend_si": best.k > 0,
    }
