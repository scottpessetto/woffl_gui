"""Pydantic models - the API contract shared with web/src/api/types.ts.

Field names follow the domain vocabulary verbatim (psu, qoil_std, form_wc,
ppf_surf ...) so numbers can be traced from SQL through the solver to the
screen without a translation table. Units are the woffl API-boundary units:
psig / psid, degF, BOPD / BWPD / BLPD, scf/bbl, ft, inches, lbm/ft3,
watercut as a 0-1 fraction.

Table-shaped payloads (well tests, chars rows, prop history) are
list[dict] on purpose: the client renders them via per-page column configs
and the exact column sets are owned by the services that build them.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Simulation parameters (mirror of woffl.gui.params.SimulationParams)
# ---------------------------------------------------------------------------

NOZZLE_OPTIONS = ["8", "9", "10", "11", "12", "13", "14", "15"]
THROAT_OPTIONS = ["X", "A", "B", "C", "D", "E"]


class SimParams(BaseModel):
    """All parameters for a single-well jetpump simulation.

    ``qwf`` is TOTAL LIQUID (BLPD) at ``pwf`` - the rate convention shared
    with vw_well_test.WtTotalFluid and prop_hist.ipr_qwf_liq. Oil / water /
    inflow rates are derived, never stored (see SimulationParams properties).
    """

    # Jetpump
    nozzle_no: str = "12"
    area_ratio: str = "B"
    ken: float = Field(0.03, ge=0.001, le=0.40)
    kth: float = Field(0.3, ge=0.05, le=1.0)
    kdi: float = Field(0.4, ge=0.05, le=1.0)
    # Multi-point event calibration knobs (defaults reproduce historic solves
    # byte-identically): critical Mach choking threshold and nozzle area
    # factor (washout; applied as dnz_eff = dnz_catalog * sqrt(factor)).
    mach_crit: float = Field(1.0, ge=1.0, le=2.5)
    nozzle_area_factor: float = Field(1.0, ge=0.8, le=1.3)
    jpump_direction: Literal["forward", "reverse"] = "reverse"

    # Pipe (inches)
    tubing_od: float = Field(4.5, ge=2.0, le=9.0)
    tubing_thickness: float = Field(0.5, ge=0.1, le=2.0)
    casing_od: float = Field(6.875, ge=4.0, le=17.0)
    casing_thickness: float = Field(0.5, ge=0.1, le=2.0)

    # Formation
    form_wc: float = Field(0.50, ge=0.0, le=1.0)
    form_gor: float = Field(250, ge=20, le=10000)  # scf/bbl
    form_temp: float = Field(70, ge=32, le=350)  # degF
    field_model: Literal["Schrader", "Kuparuk"] = "Schrader"
    model_as_water: bool = False

    # PVT overrides (None = field_model preset)
    oil_api: Optional[float] = Field(None, ge=11.0, le=39.0)
    gas_sg: Optional[float] = Field(None, ge=0.51, le=1.19)
    wat_sg: Optional[float] = Field(None, ge=0.51, le=1.49)
    bubble_point: Optional[float] = Field(None, ge=1001.0, le=2999.0)  # psig

    # Well
    surf_pres: float = Field(210, ge=10, le=600)  # psi wellhead
    jpump_tvd: float = Field(4065, ge=2500, le=8000)  # ft
    rho_pf: float = Field(62.4, ge=50.0, le=70.0)  # lbm/ft3
    ppf_surf: float = Field(3168, ge=800, le=5500)  # psi PF surface

    # Inflow (qwf = TOTAL LIQUID BLPD)
    qwf: float = Field(750, ge=10, le=20000)
    pwf: float = Field(500, ge=100, le=2500)
    pres: float = Field(1700, ge=400, le=5000)

    # Batch sweep
    nozzle_batch_options: list[str] = Field(
        default_factory=lambda: ["9", "10", "11", "12", "13", "14", "15"]
    )
    throat_batch_options: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D"])
    water_type: Literal["total", "formation"] = "total"
    marginal_watercut: float = Field(0.94, ge=0.0, le=1.0)

    # Power-fluid range sweep (psi)
    power_fluid_min: float = Field(1800, ge=1000, le=5000)
    power_fluid_max: float = Field(3600, ge=1000, le=5000)
    power_fluid_step: float = Field(200, ge=50, le=500)

    def to_simulation_params(self, selected_well: str, well_data: Optional[dict] = None):
        """Build the GUI dataclass the factories and solver wrappers consume."""
        from woffl.gui.params import SimulationParams

        return SimulationParams(
            nozzle_no=self.nozzle_no,
            area_ratio=self.area_ratio,
            ken=self.ken,
            kth=self.kth,
            kdi=self.kdi,
            jpump_direction=self.jpump_direction,
            tubing_od=self.tubing_od,
            tubing_thickness=self.tubing_thickness,
            casing_od=self.casing_od,
            casing_thickness=self.casing_thickness,
            form_wc=1.0 if self.model_as_water else self.form_wc,
            form_gor=int(self.form_gor),
            form_temp=int(self.form_temp),
            field_model=self.field_model,
            model_as_water=self.model_as_water,
            oil_api=self.oil_api,
            gas_sg=self.gas_sg,
            wat_sg=self.wat_sg,
            bubble_point=self.bubble_point,
            surf_pres=int(self.surf_pres),
            jpump_tvd=int(self.jpump_tvd),
            rho_pf=self.rho_pf,
            ppf_surf=int(self.ppf_surf),
            qwf=int(self.qwf),
            pwf=int(self.pwf),
            pres=int(self.pres),
            nozzle_batch_options=list(self.nozzle_batch_options),
            throat_batch_options=list(self.throat_batch_options),
            water_type=self.water_type,
            marginal_watercut=self.marginal_watercut,
            power_fluid_min=int(self.power_fluid_min),
            power_fluid_max=int(self.power_fluid_max),
            power_fluid_step=int(self.power_fluid_step),
            selected_well=selected_well,
            well_data=well_data,
        )


# ---------------------------------------------------------------------------
# Meta / wells
# ---------------------------------------------------------------------------


class MetaResponse(BaseModel):
    app: str = "WOFFL"
    version: str
    user: Optional[str] = None  # X-Forwarded-Email when hosted
    writes_enabled: bool  # reported for display only; v1 has no write endpoints
    warehouse_id: str
    deployed: bool


class WarmupStatus(BaseModel):
    """server.warmup progress - the ops answer to "is the fleet warm yet?"."""

    running: bool
    passes: int  # completed warm passes since process start
    interval_sec: Optional[int] = None  # 0 = single pass, no loop
    retention_sec: Optional[float] = None  # how long a warmed entry stays servable
    workers: Optional[int] = None  # concurrent warehouse connections
    wells_enabled: Optional[bool] = None
    started_at: Optional[str] = None  # UTC ISO-8601
    last_pass_at: Optional[str] = None
    last_pass_sec: Optional[float] = None
    fleet_total: int = 0
    fleet_ok: int = 0
    fleet_failed: list[str] = []  # target labels that raised
    wells_total: int = 0
    wells_ok: int = 0
    wells_failed: int = 0
    wells_failed_sample: list[str] = []  # first few failures, not the fleet


class WellListItem(BaseModel):
    name: str  # canonical GUI name, e.g. "MPB-28"
    pad: str  # letter prefix, e.g. "B"
    is_sch: Optional[bool] = None  # Schrader (JP_TVD < 5500 ft)
    jp_tvd: Optional[float] = None
    tvd_estimated: bool = False
    has_survey: bool = False


class WellsResponse(BaseModel):
    wells: list[WellListItem]
    source: Literal["databricks", "csv_fallback"]


class PumpInfo(BaseModel):
    nozzle_no: Optional[str] = None
    throat_ratio: Optional[str] = None
    tubing_od: Optional[float] = None
    date_set: Optional[str] = None  # YYYY-MM-DD


class PfSeed(BaseModel):
    ppf_surf: float
    direction: Optional[Literal["forward", "reverse"]] = None
    kind: Literal["test day", "latest daily", "fallback"]
    pf_press: Optional[float] = None
    pf_source: Optional[str] = None  # "annulus" | "tubing"
    pf_date: Optional[str] = None


class PropLock(BaseModel):
    locked: bool = False
    value: Optional[float] = None


class WellContext(BaseModel):
    """Everything the client needs when a well is selected - the server-side
    replay of the sidebar seeding pipeline, returned as data.

    ``seeds`` is a partial SimParams: apply it wholesale over the defaults.
    Precedence (chars -> pump history -> IPR fit -> saved IPR overlay ->
    live PF) is already resolved server-side.
    """

    well: str
    chars: dict[str, Any]  # raw characteristics row (NaN -> null)
    chars_source: Literal["databricks", "csv_fallback"]
    seeds: dict[str, Any]  # partial SimParams field -> value
    as_built_locks: dict[str, bool]  # tubing / casing / jpump_tvd
    prop_locks: dict[str, PropLock]  # form_wc / form_gor / res_pres
    pump: Optional[PumpInfo] = None
    pf: Optional[PfSeed] = None
    ipr_info: Optional[str] = None  # human caption, e.g. "IPR values loaded from N tests"
    saved_ipr_info: Optional[str] = None  # "Restored saved IPR values (date - user)"
    # Where the inflow seeds came from. Already computed for the optimizer's
    # Fit column; the Solver needs it too, because a "saved" curve outranks
    # the open-time Vogel fit and must not be overwritten by it.
    # "manual" = saved values with NO pinned well test behind them: a point the
    # engineer chose (a joint match, a backmatched BHP, a permutation), not a
    # measured rate/pressure pair. Same precedence as "saved", different truth
    # claim, so it is labelled differently everywhere it feeds a decision.
    ipr_source: Optional[Literal["saved", "manual", "vogel", "single_test"]] = None
    ipr_r2: Optional[float] = None  # only meaningful when ipr_source is "vogel"
    test_count: int = 0


class WellTestsResponse(BaseModel):
    well: str
    tests: list[dict[str, Any]]
    # rows: wt_uid, date (YYYY-MM-DD), oil (BOPD), water (BWPD), gas,
    # total_fluid (BLPD), form_wc (fraction, unclamped), bhp (psi|null),
    # fgor (scf/bbl), lift_wat (BWPD), whp (psi), pf_press, pf_source


class ResponseHistoryDay(BaseModel):
    """One flowing day of the suction-response scatter (advanced panel)."""

    date: str  # YYYY-MM-DD
    ppf: float  # resolved PF surface pressure, psi
    bhp: float  # daily bottomhole pressure, psi
    era: Literal["current", "prior"]  # vs the current pump's Date Set
    buildup: bool  # BHP >= res_pres: shut-in buildup, not a flowing point


class ResponseHistoryEvidence(BaseModel):
    """Mined field-evidence summary shown beside the scatter (evidence.py)."""

    floor: Optional[float] = None  # P5 flowing BHP, psi
    psu_ref: Optional[float] = None  # recent flowing BHP median, psi
    beta: Optional[float] = None  # -dBHP/dPpf response slope
    beta_source: str = "default"  # "well" | "pad" | "default"
    n_pairs: int = 0


class ResponseHistoryResponse(BaseModel):
    """Daily (PF, BHP) scatter vs the current pump era, for the response
    panel. The model overlay comes from POST /api/compute/pf-range
    client-side - this payload is measured data + mined evidence only.
    """

    days: list[ResponseHistoryDay]
    era_start: Optional[str] = None  # current pump Date Set, YYYY-MM-DD
    pump: Optional[str] = None  # "14B"-style current pump label
    evidence: Optional[ResponseHistoryEvidence] = None
    res_pres: Optional[float] = None  # buildup threshold used, psi


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    well: str = "Custom"
    params: SimParams


class SolveResult(BaseModel):
    psu: float  # suction pressure, psig
    sonic_status: bool
    qoil_std: float  # BOPD
    fwat_bwpd: float  # formation water, BWPD
    qnz_bwpd: float  # power fluid, BWPD
    mach_te: float
    dewatering: bool = False  # model_as_water run: qoil_std is 0 by construction
    total_water: float  # fwat + qnz, BWPD


class SolveErrorDetail(BaseModel):
    error: Literal["no_solution", "convergence", "invalid", "all_water"]
    message: str
    suggested_gor: Optional[float] = None  # GOR auto-recovery hint (250)


# ---------------------------------------------------------------------------
# IPR fit
# ---------------------------------------------------------------------------


class PadFitWell(BaseModel):
    """One well's saved-fit readiness for the Optimization pad board."""

    well: str
    pad: str
    has_curve: bool  # saved ipr_qwf_liq + ipr_pwf pair exists
    saved_at: Optional[str] = None  # values-save timestamp (None = never)
    saved_by: Optional[str] = None
    has_friction: bool  # any calibrated ken/kth/kdi stored
    friction_keys: list[str] = []
    locks: dict[str, bool] = {}
    pin_at: Optional[str] = None
    pin_user: Optional[str] = None


class PadFitStatusResponse(BaseModel):
    """GET /optimize/pad-status - fit readiness for a pad's wells, plus any
    `extra` wells requested (donor wells that future wells match; donors may
    live on any pad)."""

    pad: str
    wells: list[PadFitWell]
    extras: list[PadFitWell]


class FutureWellSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=24)
    match: str  # donor well whose saved fit models it


class OptimizeRunRequest(BaseModel):
    """Start an optimization run. kind=pad runs one of the S/I/M PadPlants
    through pad_optimize.run_optimization; kind=cfp runs the anchored-delta
    moves engine over the B/G/C/J CFP pads. Well models hydrate from saved
    fits; `offline` wells are excluded; `future` wells clone their donor."""

    kind: Literal["pad", "cfp"]
    pad: Optional[Literal["S", "I", "M"]] = None  # required when kind=pad
    offline: list[str] = []
    future: list[FutureWellSpec] = []
    nozzles: list[str] = ["9", "10", "11", "12", "13", "14", "15"]
    throats: list[str] = ["A", "B", "C", "D"]
    # pad-run knobs (mirror pad_page Configure stage)
    # strategy: "jpco" resizes pumps across nozzles x throats
    # (run_optimization); "choke" HOLDS every installed pump and only chokes
    # back / shuts in wells (run_choke_optimization) - the short-term plan
    # when a PF booster pump is down and a changeout costs a day per pump.
    strategy: Literal["jpco", "choke"] = "jpco"
    method: Literal["milp", "mckp"] = "milp"
    marginal_wc: Optional[float] = Field(None, ge=0.0, le=1.0)  # None = auto-derive
    parsimony_bopd: float = Field(20.0, ge=0.0, le=500.0)
    n_pumps: Optional[int] = Field(None, ge=1, le=3)
    n_steps: Optional[int] = Field(None, ge=3, le=21)
    # cfp-run knobs (mirror cfp_pad_page Configure stage)
    p0_psi: float = Field(2792.0, ge=2300.0, le=2900.0)
    psi_per_kbpd: float = Field(13.69, ge=9.0, le=17.5)
    c_pad_pf_psi: float = Field(3400.0, ge=1000.0, le=5000.0)
    # Pads in the run (default: the four CFP pads). Any non-POPs pad may
    # join - its water rides the CFP machines; PF for pads beyond B/G/J is
    # modeled as boosted on-pad at c_pad_pf_psi (the C-Pad treatment). POPs
    # pads separate water on-pad, so they never load the machines: rejected.
    cfp_pads: list[str] = ["B", "G", "C", "J"]

    @field_validator("cfp_pads")
    @classmethod
    def _cfp_pads_not_pops(cls, v: list[str]) -> list[str]:
        from woffl.assembly.well_sort_engine import DEFAULT_POPS_PADS

        pads = [p.strip().upper() for p in v if p.strip()]
        pops = [p for p in pads if p in DEFAULT_POPS_PADS]
        if pops:
            raise ValueError(
                f"POPs pads ({', '.join(sorted(set(pops)))}) separate water on-pad - "
                "their water never rides the CFP machines, so they cannot join a CFP run"
            )
        return pads


class OptimizeRunStarted(BaseModel):
    job_id: str


class MatchHealthRequest(BaseModel):
    """Start a match-health scorecard job for one pad's active wells."""

    pad: Literal["S", "I", "M"]


class EventCalibrationRequest(BaseModel):
    """Start a multi-point event-calibration job for one well."""

    well: str = Field(..., min_length=1, max_length=16)


class OptimizeJobStatus(BaseModel):
    """Poll envelope. `result` is the run-type-specific payload (pad: rows +
    meta from pad_optimize; cfp: the moves_summary), JSON-flattened."""

    job_id: str
    kind: Literal["pad", "cfp", "match_health", "event_cal"]
    status: Literal["running", "done", "error"]
    progress: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: str
    seconds: float


class PumpCurveNameplate(BaseModel):
    """Vendor/equipment identity for a pad's booster plant - the block an
    engineer reads off a curve sheet before trusting the curve."""

    equipment: str
    model: str
    arrangement: str
    speed: str
    source: str
    validated: str


class PumpCurveLine(BaseModel):
    """One station line: delivered header pressure vs TOTAL station flow.
    `points` are [flow_bpd, discharge_psi]. Exactly one line in a family
    carries active=True (the configured pump count / speed)."""

    label: str
    n_pumps: Optional[int] = None
    hz: Optional[float] = None
    active: bool = False
    points: list[list[float]]


class PumpStationCurve(BaseModel):
    """Station-level view: the curve family, the capability frontier, and the
    flow markers (BEP, preferred and allowable operating regions, minimum
    continuous flow) expressed as TOTAL station flow in BPD."""

    curves: list[PumpCurveLine]
    frontier: Optional[PumpCurveLine] = None
    bep: Optional[float] = None
    por: Optional[list[float]] = None
    aor: Optional[list[float]] = None
    min_flow: Optional[float] = None
    header_cap: Optional[float] = None


class PumpMachineCurve(BaseModel):
    """One machine's vendor curve sheet at PER-PUMP flow. `points` are
    [flow_bpd, head_ft, bhp, eff_pct]; `head_derated` is the same flow grid
    scaled by the field wear factor when the pad models one."""

    label: str
    hz: float
    points: list[list[float]]
    head_derated: Optional[list[list[float]]] = None
    derate_note: Optional[str] = None
    bep: Optional[float] = None
    por: Optional[list[float]] = None
    aor: Optional[list[float]] = None
    min_flow: Optional[float] = None


class PumpCurveResponse(BaseModel):
    """GET /optimize/pump-curve - the pad plant's industry-format curve set:
    head / BHP / efficiency vs flow per machine, the station family of
    delivered header pressure vs total flow, BEP, the preferred and allowable
    operating regions, and the capability frontier.

    Pure static physics read off the plant model and its data files - no run
    state, no Databricks, nothing well-specific - so it renders before a run
    and is cached hard. The duty point the optimizer landed on is overlaid
    client-side from the pad run meta; it is deliberately not in this payload.
    """

    pad: str
    coupling: str
    n_pumps: Optional[int]
    # Selectable online-pump counts (plant.n_pump_options, e.g. M = [3, 2, 1])
    # so the client can offer a "pumps online" control without hardcoding
    # pads. [] = fixed train (I-Pad) - nothing to choose.
    n_pump_options: list[int] = []
    sg: float
    suction_psi: float
    max_header_psi: Optional[float]
    nameplate: PumpCurveNameplate
    station: PumpStationCurve
    pumps: list[PumpMachineCurve]


class CalibrateRequest(BaseModel):
    """Run BHP friction calibration: fit (ken, kth, kdi) so the modeled
    suction pressure matches the selected test's measured BHP.

    ONLY the three friction coefficients are searched (fric_calibration's
    KEN/KTH/KDI bounds) - wellbore geometry (pump depth, casing/tubing
    dimensions) enters as fixed as-built inputs and is never varied."""

    well: str
    params: SimParams
    target_bhp: float = Field(..., gt=0.0, le=10_000.0)
    # Test-day wellhead pressure; None/invalid falls back to params.surf_pres
    # (mirror of build_calibration_inputs' model_surf_pres rule).
    test_whp: Optional[float] = None


class CalibrateResponse(BaseModel):
    converged: bool
    # "pinned": the final solve was sonic (target BHP on the cavitation
    # floor) - a single BHP point cannot identify friction there, so the
    # coefficients come back at their SEEDS and `message` explains why.
    match_quality: Literal["good", "fair", "poor", "failed", "pinned"]
    bounded: bool  # a coef sits on its search bound
    sonic: bool  # throat-choked: friction cannot lower BHP further
    ken: float
    kth: float
    kdi: float
    target_bhp: float
    modeled_bhp: Optional[float] = None
    bhp_error: Optional[float] = None  # modeled - target (psi)
    iterations: int
    starts_tried: int
    # Explanation for special outcomes (set on "pinned" runs); None otherwise.
    message: Optional[str] = None


class SensitivityPoint(BaseModel):
    """One solve at one swept value. Nulls mean the solver failed there."""

    value: float  # the swept value; catalog index for discrete knobs
    label: str  # display value, e.g. "0.30" or "14C"
    psu: Optional[float] = None  # suction BHP, psig
    qoil: Optional[float] = None  # STBOPD
    qliq: Optional[float] = None  # oil + formation water, BLPD
    qpf: Optional[float] = None  # power fluid, BWPD
    mach: Optional[float] = None
    sonic: Optional[bool] = None
    error: Optional[str] = None  # short reason when the solve failed


class KnobBounds(BaseModel):
    """Engineer override for one knob's swept range, in the knob's OWN units.

    Continuous knobs: absolute field values (GOR in scf/bbl, ken unitless,
    pressures in psi). Catalog knobs (nozzle_no, area_ratio): 0-based indices
    into NOZZLE_OPTIONS / THROAT_OPTIONS.

    ``low`` > ``high`` swaps rather than errors, ``low`` == ``high`` sweeps a
    single point, and both ends are clamped into the range the sidebar itself
    enforces so the sweep can never propose a value the engineer could not
    type.
    """

    low: float
    high: float
    # Points across the range, clamped to 2-15 by the service. None = the
    # knob's own default.
    steps: Optional[int] = None


class SensitivityKnob(BaseModel):
    """One calibration knob swept across its defensible range.

    ``low`` / ``high`` are the signed extreme excursions the sweep produced
    per match quantity, which is what the tornado draws. ``basis`` is the one
    line explaining WHY the range is what it is.
    """

    id: str
    label: str
    unit: str  # "psi", "scf/bbl", "" for unitless
    baseline_label: str
    basis: str  # one line: WHY this range (goes in the tooltip)
    points: list[SensitivityPoint]
    # Signed excursions from baseline over the whole sweep, per metric.
    # None when every solve on that side failed.
    low: dict[str, Optional[float]]  # keys: psu, qoil, qliq, qpf
    high: dict[str, Optional[float]]
    # True when the knob moves NOTHING measurably (all four metrics within tol
    # across the entire sweep). This is the headline finding on a choked well.
    inert: bool
    # Everything below describes the RANGE, so the client can render a bounds
    # editor without duplicating the knob table.
    field: str  # the SimParams field this knob drives
    kind: str  # "mult" | "abs" | "delta" | "catalog"
    default_low: float  # resolved ABSOLUTE low with no override, pre-clamp
    default_high: float  # resolved ABSOLUTE high with no override, pre-clamp
    swept_low: float  # what was actually swept (post-clamp)
    swept_high: float
    clamp_low: Optional[float]  # hard limit the sidebar/model enforces
    clamp_high: Optional[float]  # None when the field has no upper bound
    options: Optional[list[str]]  # catalog knobs only: the full option list
    overridden: bool  # an override was supplied AND applied


class SensitivityResponse(BaseModel):
    """Per-knob sensitivity of the match quantities. Read-only diagnostic:
    nothing here changes the model or is persisted."""

    baseline: SensitivityPoint
    knobs: list[SensitivityKnob]
    # Measured test values to compare against, echoed back when supplied.
    target_psu: Optional[float] = None
    target_qoil: Optional[float] = None
    target_qliq: Optional[float] = None
    target_qpf: Optional[float] = None
    notes: list[str] = []


class SensitivityRequest(BaseModel):
    """Sweep every calibration knob around the current operating point.

    Read-only: the params are the sidebar's current values and nothing is
    written back. The targets are the measured test values the tornado draws
    its reach lines from.
    """

    well: str
    params: SimParams
    # Measured test values for the reference lines; all optional.
    target_psu: Optional[float] = None
    target_qoil: Optional[float] = None
    target_qliq: Optional[float] = None
    target_qpf: Optional[float] = None
    # Per-knob range overrides, keyed by knob id. An entry for an unknown
    # knob is ignored with a note; missing knobs keep their table range.
    bounds: dict[str, KnobBounds] = {}


class CombineKnob(BaseModel):
    """One knob to vary inside a combined-permutations study.

    ``low`` / ``high`` are in the knob's own units, same convention as
    KnobBounds. ``levels`` is how many values to take across that range
    (2 = corners only).
    """

    id: str
    low: float
    high: float
    levels: int = Field(3, ge=2, le=7)


class CombineRun(BaseModel):
    """One permutation. Nulls where the solver failed."""

    values: dict[str, float]  # knob id -> swept value
    labels: dict[str, str]  # knob id -> display value
    psu: Optional[float] = None  # suction BHP, psig
    qoil: Optional[float] = None  # STBOPD
    qliq: Optional[float] = None  # oil + formation water, BLPD
    qpf: Optional[float] = None  # power fluid, BWPD
    sonic: Optional[bool] = None
    error: Optional[str] = None  # short reason when the solve failed
    # Root-mean-square fractional error across the SUPPLIED targets only.
    # None when no target was supplied or the run failed.
    score: Optional[float] = None


class CombineRequest(BaseModel):
    """Vary several knobs TOGETHER and report what the combination reaches.

    The question single-knob sensitivity cannot answer: when no one knob
    closes the gap to the measured test, does any combination inside the
    engineer's believable ranges? Read-only - nothing is persisted.
    """

    well: str
    params: SimParams
    # Measured test values; the score and the reachability verdict are
    # computed against whichever of these are supplied.
    target_psu: Optional[float] = None
    target_qoil: Optional[float] = None
    target_qliq: Optional[float] = None
    target_qpf: Optional[float] = None
    # Knobs to vary together. Empty is an error, not an empty study.
    knobs: list[CombineKnob]


class CombineResponse(BaseModel):
    """Full factorial over the selected knobs. Read-only diagnostic:
    nothing here changes the model or is persisted."""

    baseline: SensitivityPoint
    runs: list[CombineRun]
    # Reachable [min, max] per metric across every solved run. A metric with
    # no solved run is absent.
    envelope: dict[str, list[float]]
    # Per metric: is the supplied target inside the envelope? Absent key when
    # no target was supplied for that metric.
    reachable: dict[str, bool]
    best_index: Optional[int] = None  # index into runs, lowest score
    n_runs: int
    n_failed: int
    notes: list[str] = []


class CombineStarted(BaseModel):
    """POST /sensitivity/combine - the study runs as a background job."""

    job_id: str


class CombineJobStatus(BaseModel):
    """Poll envelope for one combined-permutations study. `result` populates
    when status becomes done."""

    job_id: str
    kind: Literal["sensitivity"]
    status: Literal["running", "done", "error"]
    progress: Optional[str] = None
    result: Optional[CombineResponse] = None
    error: Optional[str] = None
    started_at: str
    seconds: float


class GaugeDay(BaseModel):
    """One daily-median BHP from an uploaded memory gauge."""

    date: str  # YYYY-MM-DD
    bhp: float


class IprFitRequest(BaseModel):
    well: str
    anchor_mode: Literal["recent", "median", "median_liq", "specific"] = "recent"
    anchor_date: Optional[str] = None  # YYYY-MM-DD, required for "specific"
    field_model: Literal["Schrader", "Kuparuk"] = "Schrader"
    # le=60: a memory-gauge window can reach years back, past the sidebar's
    # 24-month cap (the client widens months to cover the gauge coverage).
    months: int = Field(6, ge=1, le=60)
    cap: int = Field(0, ge=0, le=50)
    # Memory-gauge daily medians: test rows whose date has an entry get
    # their BHP OVERRIDDEN before the fit (mirror of
    # memory_gauge.apply_to_well_tests) - gauge wins wherever it has
    # coverage, including tests with no Databricks BHP at all.
    bhp_overrides: Optional[list[GaugeDay]] = None


class GaugeFileMeta(BaseModel):
    """Per-file parse summary (upload preview: raw extremes stay visible)."""

    filename: str
    start_date: str
    end_date: str
    sample_count: int  # RAW pre-downsample points
    pressure_min: float
    pressure_max: float


class GaugeParseResponse(BaseModel):
    """Combined memory-gauge data for one well (all uploaded files).

    The client re-sends EVERY file when one is added/removed, so the
    combination (timestamp dedupe across files -> daily medians) always
    runs server-side in memory_gauge.MemoryGaugeData - byte-identical to
    the Streamlit path, no client-side math.
    """

    files: list[GaugeFileMeta]
    daily: list[GaugeDay]
    start_date: str
    end_date: str
    sample_count: int


class IprCoeffs(BaseModel):
    res_p: float  # psi
    qmax: Optional[float] = None  # BLPD at bhp=0
    qwf: float  # BLPD total fluid at anchor
    pwf: float  # psi at anchor
    form_wc: float  # fraction
    fgor: float  # scf/bbl
    r2: Optional[float] = None
    num_tests: int
    most_recent_date: Optional[str] = None
    anchor_label: Optional[str] = None
    anchor_date: Optional[str] = None


class IprFitResponse(BaseModel):
    well: str
    coeffs: IprCoeffs
    seeds: dict[str, Any]  # sidebar fields to apply: qwf, pwf, pres, form_wc, form_gor, surf_pres?
    weak: bool = False  # R2 below the weak-fit threshold


class IprPinResponse(BaseModel):
    status: Literal["none", "applied", "stale"]
    wt_uid: Optional[float] = None
    date_token: Optional[str] = None
    entry_user: Optional[str] = None
    entry_datetime: Optional[str] = None


class SaveIprRequest(BaseModel):
    """The Solver's "Save as well default" payload - mirror of the Streamlit
    button (_render_ipr_pin_controls): pin the resolved anchor test AND push
    the sidebar's current curve/rate values in one click. Bounds mirror the
    SimParams widget bounds; ipr_anchor.save_ipr_values re-caps WC at 0.99."""

    qwf_liq: float = Field(..., gt=0.0, le=50_000.0)  # TOTAL LIQUID (BLPD), stored verbatim
    pwf: float = Field(..., ge=50.0, le=5_000.0)
    res_pres: float = Field(..., ge=100.0, le=10_000.0)
    form_wc: float = Field(..., ge=0.0, le=1.0)
    form_gor: float = Field(..., ge=0.0, le=20_000.0)
    surf_pres: Optional[float] = Field(None, ge=0.0, le=5_000.0)
    # BHP-calibrated friction rides along; save_ipr_values skips unchanged /
    # never-calibrated-default values so no noise rows materialize.
    ken: Optional[float] = None
    kth: Optional[float] = None
    kdi: Optional[float] = None
    # Event-calibration knobs ride the same discipline (skip-at-1.0 unless a
    # saved override exists). Bounds mirror SimParams.
    nozzle_area_factor: Optional[float] = Field(None, ge=0.8, le=1.3)
    mach_crit: Optional[float] = Field(None, ge=1.0, le=2.5)
    # Characterization values the sensitivity study can move. Both prop ids
    # (resvr_bubb / resvr_temp) already exist; the client sends one only when
    # the engineer changed it off the seeded value, so a save never re-writes
    # the characterization it was handed. Bounds mirror PARAM_BOUNDS.
    bubble_point: Optional[float] = Field(None, ge=1_001.0, le=2_999.0)
    form_temp: Optional[float] = Field(None, ge=32.0, le=350.0)
    comment: Optional[str] = Field(None, max_length=500)
    # Anchor pin: the CLIENT-resolved anchor test. None = values-only save
    # (forced/manual IPR - nothing pinnable).
    pin_wt_uid: Optional[float] = None
    pin_date: Optional[str] = None  # YYYY-MM-DD, for the confirmation label
    # A manual-point save: drop any pinned anchor test first, so the well
    # reopens reading "manual" instead of claiming a test it was not derived
    # from. Appends the cleared marker (prop_hist is append-only), never a
    # DELETE.
    unpin: bool = False


class SaveIprResponse(BaseModel):
    pinned: bool
    pin_skipped: bool  # expected no-pin (manual/provisional anchor), not an error
    pin_message: Optional[str] = None
    n_values: int  # prop rows written by the values save
    values_message: str


class ClearIprPinResponse(BaseModel):
    cleared: bool
    message: str


class PropLockRequest(BaseModel):
    """Toggle a per-well field lock (ipr_anchor.LOCKABLE_FIELDS) - "the
    automated seed for this field is systematically wrong on this well; my
    saved value stands until I unlock". Locking also pushes `value` (the
    sidebar's current number) so the locked value is pinned in the same
    click; unlocking writes the 0.0 unlocked marker."""

    field: Literal["form_wc", "form_gor", "res_pres"]
    locked: bool
    value: Optional[float] = None  # pushed only when locking; WC re-capped at 0.99


class PropLockResponse(BaseModel):
    ok: bool
    message: str
    field: str
    locked: bool  # the field's lock state AFTER this call
    value: Optional[float] = None  # the value pinned with the lock, if any


# ---------------------------------------------------------------------------
# Batch sweep
# ---------------------------------------------------------------------------


class BatchRequest(BaseModel):
    well: str = "Custom"
    params: SimParams


class BatchStats(BaseModel):
    total: int
    successful: int
    success_pct: float


class BatchRecommendation(BaseModel):
    nozzle: str
    throat: str
    qoil_std: float
    water_rate: float
    marginal_ratio: Optional[float] = None
    recommendation_type: Literal["optimal", "best_available"]
    theoretical_water_rate: Optional[float] = None
    theoretical_oil_rate: Optional[float] = None


class BatchResponse(BaseModel):
    rows: list[dict[str, Any]]
    # row: nozzle, throat, qoil_std, form_wat, lift_wat, totl_wat, psu_solv,
    #      mach_te, sonic_status, semi (+ mofwr when water_type=formation)
    stats: BatchStats
    recommended: Optional[BatchRecommendation] = None
    fit_curve: Optional[dict[str, list[float]]] = None  # {x: [...], y: [...]} exp fit, water axis per x_mode
    x_mode: Literal["total", "formation"] = "total"


# ---------------------------------------------------------------------------
# Power-fluid range sweep
# ---------------------------------------------------------------------------


class PfRangeRequest(BaseModel):
    well: str = "Custom"
    params: SimParams


class PfRangeResponse(BaseModel):
    rows: list[dict[str, Any]]
    # row: pump ("12B"), nozzle, throat, power_fluid_pressure (psi),
    #      qoil_std, form_wat, lift_wat, totl_wat, psu_solv, mach_te, sonic_status
    pressures: list[float]


# ---------------------------------------------------------------------------
# Pressure profile (traverse)
# ---------------------------------------------------------------------------


class PressureProfileRequest(BaseModel):
    well: str = "Custom"
    params: SimParams


class PressureProfileResponse(BaseModel):
    prod: dict[str, list[float]]  # {md: [...], press: [...]} production string
    pf: dict[str, list[float]]  # power-fluid string
    diff: dict[str, list[float]]  # {md: [...], dp: [...]} PF - prod differential
    jpump_md: float
    metrics: dict[str, float]  # psu, prod_at_jp, pf_at_jp, dp_at_jp (psi)


# ---------------------------------------------------------------------------
# Well profile (survey)
# ---------------------------------------------------------------------------


class WellProfileResponse(BaseModel):
    well: str
    has_survey: bool
    md: list[float]
    vd: list[float]
    hd: list[float]
    md_filtered: list[float]
    vd_filtered: list[float]
    jetpump_md: Optional[float] = None
    jetpump_vd: Optional[float] = None
    inclination: Optional[dict[str, list[float]]] = None  # {md: [...], deg: [...]}


class DepthStation(BaseModel):
    md: float
    tvd: float


class DepthLookupResponse(BaseModel):
    """One MD <-> TVD conversion along the well's deviation survey."""

    well: str
    has_survey: bool
    method: Literal["minimum_curvature", "chord"]
    given: Literal["md", "tvd"]
    md: float
    tvd: float
    inclination: Optional[float] = None  # deg from vertical
    azimuth: Optional[float] = None  # deg
    dls: Optional[float] = None  # dogleg severity, deg/100 ft
    md_solutions: list[float] = Field(default_factory=list)  # every MD at that TVD
    at_station: bool = False
    station_above: Optional[DepthStation] = None
    station_below: Optional[DepthStation] = None
    station_count: int
    md_range: list[float]  # [shallowest, deepest] survey MD
    tvd_range: list[float]
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Pump equivalents
# ---------------------------------------------------------------------------


class EquivalentsResponse(BaseModel):
    nozzle_no: str
    area_ratio: str
    rows: list[dict[str, Any]]
    # row: brand, nozzle, throat, nozzle_dia (in), throat_dia (in),
    #      nozzle_area (in2), throat_area (in2), area_ratio_val, is_reference


# ---------------------------------------------------------------------------
# JP history
# ---------------------------------------------------------------------------


class JpHistoryResponse(BaseModel):
    well: str
    installs: list[dict[str, Any]]
    # row: date_set, date_pulled, nozzle, throat, tubing_od, circulating,
    #      manufacturer, raw_pump, pump_converted
    tests: list[dict[str, Any]]
    # extended window rows: date, oil_rate, fwat_rate, lift_wat, bhp, pf_press
    bhp_daily: list[dict[str, Any]]  # {date, bhp}
    current_pump: Optional[str] = None
    source: Literal["databricks", "excel_fallback"]


# ---------------------------------------------------------------------------
# Well database
# ---------------------------------------------------------------------------


class WellDatabaseResponse(BaseModel):
    rows: list[dict[str, Any]]  # chars table rows (see services/wells.py)
    source: Literal["databricks", "csv_fallback"]
    missing_surveys: list[str]


class AgingPumpsResponse(BaseModel):
    rows: list[dict[str, Any]]  # well, pump, date_set, days_in_hole, ...


class PropHistoryResponse(BaseModel):
    well: str
    current: list[dict[str, Any]]  # latest value per prop
    history: list[dict[str, Any]]  # full audit rows


# ---------------------------------------------------------------------------
# Well Sort
# ---------------------------------------------------------------------------


class WellSortTablesResponse(BaseModel):
    online: list[dict[str, Any]]  # see services/well_sort._ONLINE_COLUMNS
    offline: list[dict[str, Any]]  # see services/well_sort._SHUT_COLUMNS
    ltsi: list[dict[str, Any]]  # same shape as offline
    all_pads: list[str]
    producers: list[str]
    xv_available: bool
    tests_window_days: int
    outliers_flagged: int
    just_restarted: int
    default_pops_pads: list[str]
    pump_limit_presets: dict[str, int]
    pops_pump_handles: dict[str, str]  # pad -> "total" | "lift"


class WellSortEventsResponse(BaseModel):
    rows: list[dict[str, Any]]  # see services/well_sort._EVENT_COLUMNS


class MarginalWcResponse(BaseModel):
    marginal_wc: float
    well: str
    pad: str
    total_field_water: float
    well_count: int
    threshold_pct: float
    marg_idx: int
    cum_water_at_marginal: Optional[float] = None
    rows: list[dict[str, Any]]  # ranked walk, worst WC first


class PadMarginalWcResponse(BaseModel):
    marginal_wc: float
    well: str
    pad: str
    pad_water: float
    pump_limit: float
    headroom: Optional[float] = None
    well_count: int
    water_basis: Literal["total", "lift"]
    rows: list[dict[str, Any]]  # ranked by pad-stream WC, worst first


class TriageResponse(BaseModel):
    marginal_wc: float
    well: str
    pad: str
    threshold_pct: float
    raw_worst_wc: Optional[float] = None
    raw_worst_well: Optional[str] = None
    raw_worst_water: Optional[float] = None
    xv_available: bool
    online: list[dict[str, Any]]  # online cols + decision_code/why/rank
    shut: list[dict[str, Any]]  # shut cols + decision_code/why/rank


class WellSortRefreshResponse(BaseModel):
    cleared: int


# ---------------------------------------------------------------------------
# Scott's Tools (the secret menu)
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """One entry in the secret menu."""

    id: str
    label: str
    caption: str
    path: str


# The menu, in the order the Streamlit page listed its tabs. Well Sort is
# absent on purpose: it was never on the secret menu (it was a top-level mode)
# and it is already a first-class page in this app.
TOOL_CATALOG: list[dict[str, str]] = [
    {
        "id": "pf-scenario",
        "label": "PF Scenario",
        "caption": "Oil and BHP at two power-fluid pressures, well by well.",
        "path": "/tools/pf-scenario",
    },
    {
        "id": "header-impact",
        "label": "Header Pressure Impact",
        "caption": "What moving a pad header does to BHP and oil, all lift types.",
        "path": "/tools/header-impact",
    },
    {
        "id": "jp-calibration",
        "label": "JP Friction Calibration",
        "caption": "Fit ken/kth/kdi per well against measured BHP.",
        "path": "/tools/jp-calibration",
    },
    {
        "id": "fric-trend",
        "label": "JP Fric Trend",
        "caption": "Fitted friction coefficients across a well's test history.",
        "path": "/tools/fric-trend",
    },
    {
        "id": "jp-washout",
        "label": "JP Wash-Out",
        "caption": "Pumps needing more PF pressure than the surface can deliver.",
        "path": "/tools/jp-washout",
    },
    {
        "id": "pad-watercut",
        "label": "Pad Water Cut",
        "caption": "Daily pad-level water cut for G, H, I and J.",
        "path": "/tools/pad-watercut",
    },
    {
        "id": "sep-oil-loss",
        "label": "Separator Oil Loss",
        "caption": "Oil leaving with the first-stage water leg (FI-5365 / AI-5317).",
        "path": "/tools/sep-oil-loss",
    },
    {
        "id": "test-harness",
        "label": "Test Harness",
        "caption": "Curated sanity cases run against today's live data.",
        "path": "/tools/test-harness",
    },
]


class ToolJobStarted(BaseModel):
    job_id: str


class ToolJobStatus(BaseModel):
    job_id: str
    kind: str
    status: str
    progress: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    seconds: Optional[float] = None


class ToolRowsResponse(BaseModel):
    """Generic table payload - the tools all return rows plus their inputs."""

    model_config = {"extra": "allow"}

    rows: list[dict[str, Any]] = []


class HarnessCase(BaseModel):
    name: str
    description: str


class HarnessCasesResponse(BaseModel):
    cases: list[HarnessCase]


class WashoutRequest(BaseModel):
    months_back: int = Field(6, ge=1, le=60)
    # Surface PF infrastructure cap: above it the pump, not the pressure,
    # is the problem.
    ppf_limit: float = Field(3400.0, ge=1000.0, le=6000.0)


class FricTrendRequest(BaseModel):
    wells: list[str]
    months_back: int = Field(12, ge=1, le=60)


class CalibrationRequest(BaseModel):
    wells: Optional[list[str]] = None
    months_back: int = Field(6, ge=1, le=60)


class HeaderImpactRequest(BaseModel):
    pads: list[str]
    # Negative = drawing the header down, which is the usual direction.
    delta_p: float = Field(-50.0, ge=-300.0, le=300.0)
    months_back: int = Field(6, ge=1, le=60)
    # PF-PRESSURE-DEPENDENCY: per-pad PF overrides, since there is still no
    # per-well PF pressure in Databricks.
    pad_pf: Optional[dict[str, int]] = None


class PfScenarioRequest(BaseModel):
    wells: list[str]
    pf_a: float = Field(..., ge=1000.0, le=5000.0)
    pf_b: float = Field(..., ge=1000.0, le=5000.0)
    months_back: int = Field(6, ge=1, le=60)


class ToolCatalogResponse(BaseModel):
    tools: list[ToolInfo]


class DateWindow(BaseModel):
    start: str
    end: str


class PadWatercutPoint(BaseModel):
    date: Optional[str] = None
    wc: Optional[float] = None
    oil: Optional[float] = None
    water: Optional[float] = None


class PadWatercutSeries(BaseModel):
    pad: str
    points: list[PadWatercutPoint]


class PadWatercutResponse(BaseModel):
    start: str
    end: str
    series: list[PadWatercutSeries]


# ── Separator Oil Loss ─────────────────────────────────────────────────────


class SepLossPeriod(BaseModel):
    """One look-back roll-up. Barrels are a band, never a single number."""

    label: str
    days: int
    hours: float  # valid (separator running) hours in the look-back
    downtime_hours: float
    flow_avg: float  # BPD, time-weighted
    wc_avg: float  # %, time-weighted
    base_avg: float  # %, the analyzer's own film-corrected plateau
    bbl_upper: float  # meter as read, film-corrected, capped at field oil
    bbl_lower: float  # oil fraction of the leg capped at max_oil_frac
    bopd_upper: float
    bopd_lower: float
    pct_field_upper: Optional[float] = None
    pct_field_lower: Optional[float] = None
    upset_hours: float
    events: int


class SepLossDay(BaseModel):
    """One FIELD calendar day. Alaska local, so a night-shift upset lands in
    the day the crew would name."""

    date: str  # YYYY-MM-DD
    hours: float  # separator running hours inside the day
    covered_hours: float  # how much of the day the window spans
    bbl_upper: float
    bbl_lower: float
    pct_field_upper: Optional[float] = None
    pct_field_lower: Optional[float] = None
    upset_hours: float
    events: int
    partial: bool  # clipped by the window or cut by downtime; not comparable


class SepLossEvent(BaseModel):
    start: str
    end: str
    hours: float
    wc_min: float
    wc_avg: float
    flow_avg: float
    bbl_upper: float
    bbl_lower: float
    level_min: Optional[float] = None  # controlled level, %
    level_avg: Optional[float] = None
    level_sp_avg: Optional[float] = None  # that loop's setpoint, %
    level_dev_avg: Optional[float] = None  # level - setpoint, points
    # "at setpoint" is the interesting one: level held where it was asked and
    # the water leg ran oil anyway, so separation failed, not level control.
    kind: Literal["level loss", "off setpoint", "at setpoint"]


class SepOilLossResponse(BaseModel):
    flow_tag: str
    wc_tag: str
    level_tag: Optional[str] = None  # MPU_LIC_5365CV1, the CONTROLLED level
    level_sp_tag: Optional[str] = None
    days: int
    start: Optional[str] = None
    end: Optional[str] = None
    field_oil_bopd: float
    max_oil_frac: float
    flow_min_bpd: float
    upset_drop_pts: float
    valid_hours: float
    excluded_hours: float
    periods: list[SepLossPeriod] = Field(default_factory=list)
    daily: list[SepLossDay] = Field(default_factory=list)
    events: list[SepLossEvent] = Field(default_factory=list)
    # Parallel arrays keyed by `t`; nulls where a tag had no reading yet.
    series: dict[str, list[Any]] = Field(default_factory=dict)


class SepOilLossDayResponse(BaseModel):
    """One field day at full resolution - the drill-down behind a daily bar."""

    date: str
    days: int
    summary: Optional[SepLossDay] = None
    events: list[SepLossEvent] = Field(default_factory=list)
    series: dict[str, list[Any]] = Field(default_factory=dict)
