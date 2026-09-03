/**
 * API contract - the TypeScript mirror of server/schemas.py.
 * Field names follow the domain vocabulary verbatim (psu, qoil_std, form_wc,
 * ppf_surf ...). Units: psig/psid, degF, BOPD/BWPD/BLPD, scf/bbl, ft, inches,
 * lbm/ft3, watercut as a 0-1 fraction.
 */

export type FieldModel = "Schrader" | "Kuparuk";
export type JpumpDirection = "forward" | "reverse";
export type WaterType = "total" | "formation";

export const NOZZLE_OPTIONS = ["8", "9", "10", "11", "12", "13", "14", "15"] as const;
export const THROAT_OPTIONS = ["X", "A", "B", "C", "D", "E"] as const;

export interface SimParams {
  // Jetpump
  nozzle_no: string;
  area_ratio: string;
  ken: number;
  kth: number;
  kdi: number;
  // Multi-point event-calibration knobs (1.0/1.0 = classic behavior):
  // critical Mach choking threshold and nozzle flow-area multiplier
  // (washout wear; server applies dnz_eff = dnz_catalog * sqrt(factor)).
  mach_crit: number;
  nozzle_area_factor: number;
  jpump_direction: JpumpDirection;
  // Pipe (inches)
  tubing_od: number;
  tubing_thickness: number;
  casing_od: number;
  casing_thickness: number;
  // Formation
  form_wc: number; // fraction 0-1
  form_gor: number; // scf/bbl
  form_temp: number; // degF
  field_model: FieldModel;
  model_as_water: boolean;
  // PVT overrides (null = field preset)
  oil_api: number | null;
  gas_sg: number | null;
  wat_sg: number | null;
  bubble_point: number | null; // psig
  // Well
  surf_pres: number; // psi wellhead
  jpump_tvd: number; // ft
  rho_pf: number; // lbm/ft3
  ppf_surf: number; // psi PF surface
  // Inflow - qwf is TOTAL LIQUID (BLPD)
  qwf: number;
  pwf: number; // psi
  pres: number; // psi reservoir
  // Batch sweep
  nozzle_batch_options: string[];
  throat_batch_options: string[];
  water_type: WaterType;
  marginal_watercut: number;
  // PF range sweep (psi)
  power_fluid_min: number;
  power_fluid_max: number;
  power_fluid_step: number;
}

/** Widget bounds - mirror of sidebar.SEED_BOUNDS + option lists. */
export const PARAM_BOUNDS: Partial<Record<keyof SimParams, [number, number]>> = {
  qwf: [10, 20000],
  pwf: [100, 2500],
  pres: [400, 5000],
  form_wc: [0.0, 1.0],
  form_gor: [20, 10000],
  form_temp: [32, 350],
  surf_pres: [10, 600],
  ppf_surf: [800, 5500],
  ken: [0.001, 0.4],
  kth: [0.05, 1.0],
  kdi: [0.05, 1.0],
  mach_crit: [1.0, 2.5],
  nozzle_area_factor: [0.8, 1.3],
  jpump_tvd: [2500, 8000],
  rho_pf: [50.0, 70.0],
  oil_api: [11.0, 39.0],
  bubble_point: [1001.0, 2999.0],
  gas_sg: [0.51, 1.19],
  wat_sg: [0.51, 1.49],
  tubing_od: [2.0, 9.0],
  tubing_thickness: [0.1, 2.0],
  casing_od: [4.0, 17.0],
  casing_thickness: [0.1, 2.0],
  marginal_watercut: [0.0, 1.0],
  power_fluid_min: [1000, 5000],
  power_fluid_max: [1000, 5000],
  power_fluid_step: [50, 500],
};

export const DEFAULT_PARAMS: SimParams = {
  nozzle_no: "12",
  area_ratio: "B",
  ken: 0.03,
  kth: 0.3,
  kdi: 0.4,
  mach_crit: 1.0,
  nozzle_area_factor: 1.0,
  jpump_direction: "reverse",
  tubing_od: 4.5,
  tubing_thickness: 0.5,
  casing_od: 6.875,
  casing_thickness: 0.5,
  form_wc: 0.5,
  form_gor: 250,
  form_temp: 70,
  field_model: "Schrader",
  model_as_water: false,
  oil_api: 22.0,
  gas_sg: 0.65,
  wat_sg: 1.02,
  bubble_point: 1750.0,
  surf_pres: 210,
  jpump_tvd: 4065,
  rho_pf: 62.4,
  ppf_surf: 3168,
  qwf: 750,
  pwf: 500,
  pres: 1700,
  nozzle_batch_options: ["9", "10", "11", "12", "13", "14", "15"],
  throat_batch_options: ["A", "B", "C", "D"],
  water_type: "total",
  marginal_watercut: 0.94,
  power_fluid_min: 1800,
  power_fluid_max: 3600,
  power_fluid_step: 200,
};

/** Derived rates (mirror of SimulationParams properties). */
export const qwfOil = (p: SimParams): number => p.qwf * (1 - p.form_wc);
export const qwfWater = (p: SimParams): number => p.qwf * p.form_wc;

// ---------------------------------------------------------------------------
// Meta / wells
// ---------------------------------------------------------------------------

export interface MetaResponse {
  app: string;
  version: string;
  user: string | null;
  writes_enabled: boolean;
  warehouse_id: string;
  deployed: boolean;
}

export interface WellListItem {
  name: string; // "MPB-28"
  pad: string; // "B"
  is_sch: boolean | null;
  jp_tvd: number | null;
  tvd_estimated: boolean;
  has_survey: boolean;
}

export interface WellsResponse {
  wells: WellListItem[];
  source: "databricks" | "csv_fallback";
}

export interface PumpInfo {
  nozzle_no: string | null;
  throat_ratio: string | null;
  tubing_od: number | null;
  date_set: string | null;
  /** "databricks" = live tracker; "excel_fallback" = bundled snapshot (may be months stale). */
  source?: "databricks" | "excel_fallback" | null;
}

export interface PfSeed {
  ppf_surf: number;
  direction: JpumpDirection | null;
  kind: "test day" | "latest daily" | "fallback";
  pf_press: number | null;
  pf_source: string | null;
  pf_date: string | null;
}

export interface PropLock {
  locked: boolean;
  value: number | null;
}

export interface WellContext {
  well: string;
  chars: Record<string, unknown>;
  chars_source: "databricks" | "csv_fallback";
  seeds: Partial<SimParams>;
  /** Measured jet-pump MD (ft) for the optimizer's WellConfig; not a SimParams field. */
  jpump_md: number | null;
  /** Seeds the widget bounds altered on the way in, e.g. "pres: 5200 -> 5000 (...)". */
  clamped?: string[];
  as_built_locks: { tubing: boolean; casing: boolean; jpump_tvd: boolean };
  prop_locks: { form_wc: PropLock; form_gor: PropLock; res_pres: PropLock };
  pump: PumpInfo | null;
  pf: PfSeed | null;
  ipr_info: string | null;
  saved_ipr_info: string | null;
  /** "saved" = engineer-reviewed values with a pinned well test behind them.
   *  "manual" = the same, with NO test behind them: a point the engineer
   *  chose. Both outrank the fit; only one is a measurement. */
  ipr_source: "saved" | "manual" | "vogel" | "single_test" | null;
  ipr_r2: number | null;
  test_count: number;
}

/** One well-test row (units in comments; nullable when unmeasured). */
export interface WellTestRow {
  wt_uid: number | null;
  date: string; // YYYY-MM-DD
  oil: number | null; // BOPD
  water: number | null; // BWPD formation
  gas: number | null; // MCF/D
  total_fluid: number | null; // BLPD
  form_wc: number | null; // fraction, unclamped (out-of-range is a signal)
  bhp: number | null; // psi
  fgor: number | null; // scf/bbl
  lift_wat: number | null; // BWPD power fluid
  whp: number | null; // psi
  pf_press: number | null; // psi
  pf_source: string | null;
  [key: string]: unknown;
}

export interface WellTestsResponse {
  well: string;
  tests: WellTestRow[];
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

export interface SolveRequest {
  well: string;
  params: SimParams;
}

export interface SolveResult {
  psu: number; // psig
  sonic_status: boolean;
  qoil_std: number; // BOPD
  fwat_bwpd: number; // BWPD
  qnz_bwpd: number; // BWPD power fluid
  mach_te: number;
  dewatering: boolean;
  total_water: number; // BWPD
}

export interface ApiErrorDetail {
  error: "no_solution" | "convergence" | "invalid" | "all_water" | "internal" | "http";
  message: string;
  suggested_gor?: number | null;
}

// ---------------------------------------------------------------------------
// IPR
// ---------------------------------------------------------------------------

/** IPR anchor selection. "manual" is not a test at all - the sidebar's own
 *  qwf/pwf IS the anchor, which is what a joint match or a backmatched BHP
 *  produces. It disables the test-derived fit rather than competing with it. */
export type AnchorMode = "recent" | "median" | "median_liq" | "specific" | "manual";

/** One well's saved-fit readiness on the Optimization pad board. */
export interface PadFitWell {
  well: string;
  pad: string;
  has_curve: boolean;
  saved_at: string | null;
  saved_by: string | null;
  has_friction: boolean;
  friction_keys: string[];
  locks: Record<string, boolean>;
  pin_at: string | null;
  pin_user: string | null;
}

export interface PadFitStatusResponse {
  pad: string;
  wells: PadFitWell[];
  extras: PadFitWell[];
}

/** A pad with a booster-plant model, i.e. one the pad optimizer can run.
 *  The ONE definition - the run request, the match-health request, the run
 *  panel and the page tabs all narrow to this. */
export type RunPad = "S" | "I" | "M" | "E";

/** Which E-Pad booster build the run assumes is in the ground. */
export type EPadBuild = "SM25000_26STG" | "SN35000_18STG";

/** POST /optimize/run - mirror of server.schemas.OptimizeRunRequest. */
export interface OptimizeRunRequest {
  kind: "pad" | "cfp";
  pad: RunPad | null;
  offline: string[];
  future: { name: string; match: string }[];
  nozzles: string[];
  throats: string[];
  /** "jpco" resizes pumps; "choke" holds every installed pump and only
   *  chokes back / shuts in wells (short-term plan for a PF pump outage). */
  strategy: "jpco" | "choke";
  method: "milp" | "mckp";
  /** Water price λ, BOPD given up per BPD of lift water, in the knapsack
   *  objective oil − λ·water. null = auto (the plant budget's own shadow
   *  price). Wins over marginal_wc when both are set. */
  lambda_bopd_per_bpd: number | null;
  marginal_wc: number | null; // legacy gate, mapped to λ = (1 − wc) / wc
  parsimony_bopd: number; // DEPRECATED: accepted and ignored by the server
  n_pumps: number | null; // null = pad default
  n_steps: number | null;
  /** Pins the header for free-pressure pads (I/M/E) to one trial instead of
   *  sweeping it. null = sweep. Ignored for the fixed-curve S-Pad, whose
   *  header is a function of flow. */
  setpoint_psi: number | null;
  p0_psi: number;
  psi_per_kbpd: number;
  c_pad_pf_psi: number;
  cfp_pads: string[]; // which of B/G/C/J participate (cfp runs)
  /** E-Pad booster configuration. None of these four is a measured E-Pad
   *  tag - no SCADA point, no motor nameplate and no piping rating came with
   *  the vendor curve sheets - so a run states them. Ignored on other pads. */
  e_pad_build: EPadBuild;
  e_pad_suction_psi: number;
  e_pad_hz_max: number;
  e_pad_max_header_psi: number;
  e_pad_amp_limit_a: number | null;
}

export interface PadRunRow {
  well: string;
  current_pump: string | null;
  test_oil: number | null;
  test_pf: number | null;
  pump: string | null; // null = not in plan / shut-in
  oil: number | null;
  pf: number | null;
  form_water: number | null;
  suction: number | null;
  marginal_oil: number | null;
  sonic: boolean | null;
  /** Where this well's inflow curve came from: "saved" = engineer-reviewed
   *  with a pinned test, "manual" = engineer-chosen point with no test behind
   *  it, "vogel" = an automatic fit over recent tests, "single_test" = one
   *  test, null = neither, so the run used generic defaults. */
  ipr_source: "saved" | "manual" | "vogel" | "single_test" | null;
  /** Vogel fit quality when ipr_source is "vogel". */
  ipr_r2: number | null;
  /** ken/kth/kdi came from a BHP calibration rather than library defaults. */
  has_friction: boolean;
}

export interface PadRunResult {
  pad: string;
  rows: PadRunRow[];
  meta: Record<string, unknown>; // pad_optimize meta contract, JSON-flattened
  notes: string[];
  n_wells: number;
}

/** One well in a choke/shut-in plan (strategy="choke"): the installed pump
 *  is never changed - the action is the PF setting. */
export interface ChokePlanRow {
  well: string;
  pump: string | null; // installed pump, unchanged
  /** "model" = saved-fit solve; "test" = held at measured rates (model
   *  would not solve, so only shut-in was offered); "none" = excluded. */
  basis: "model" | "test" | "none";
  action: "full" | "choke" | "shut" | "hold" | "excluded";
  delivered_psi: number | null; // PF pressure at the wellhead after the choke
  choke_dp_psi: number | null; // pinched across the wellhead PF throttle
  /** Full-open reference (raw, highest solvable ladder level <= header). */
  delivered_full_psi: number | null;
  oil_full: number | null;
  pf_full: number | null;
  /** IPR landing: suction pressure (pump flowing BHP) at the chosen and
   *  full-open settings, plus reservoir pressure for drawdown. */
  psu: number | null;
  psu_full: number | null;
  /** Cavitation floor (sonic throat entry) at the chosen / full-open point:
   *  psu and oil are pinned there, only PF responds to delivered pressure. */
  sonic: boolean | null;
  sonic_full: boolean | null;
  /** Field-data suction response for wells whose modeled cavitation floor is
   *  contradicted by measured BHP history; absent on old payloads. floor and
   *  violation populate whenever evidence exists for a model-basis well;
   *  beta/beta_source only when the response was corrected. */
  evidence_floor_psi: number | null;
  floor_violation_psi: number | null;
  response_beta: number | null;
  beta_source: string | null;
  suction_basis: "model" | "evidence" | null;
  res_pres: number | null;
  /** Vogel inflow curve samples [oil_bopd, pwf_psi], res_pres down to 0; null off model basis. */
  ipr_curve: [number, number][] | null;
  pf: number | null;
  oil: number | null;
  d_oil_vs_full: number | null;
  d_pf_vs_full: number | null;
  test_oil: number | null;
  test_pf: number | null;
  projected_oil: number | null; // test oil x model ratio (bias cancels)
  next_trim_bopd_per_bpd: number | null; // cost of one more trim step
  ipr_source: "saved" | "manual" | "vogel" | "single_test" | null;
  ipr_r2: number | null;
  has_friction: boolean;
}

/** One action in a ladder rung's best response (full-open rows omitted). */
export interface ChokeLadderAction {
  well: string;
  action: "choke" | "shut" | "hold";
  set_psi: number | null; // delivered PF psi to pinch to; null for shut/hold
}

/** One rung of the header-drop decision ladder: if the PF bank degrades
 *  until the all-run header settles drop_psi below the plan's winning
 *  header, the best response and what it gains over doing nothing. */
export interface ChokeLadderRung {
  drop_psi: number; // below the plan's winning header
  settles_psi: number; // where the all-run header would sag to
  run_all_oil_bopd: number; // pad oil if no action is taken at that sag
  best_header_psi: number; // header the best response holds instead
  plan_oil_bopd: number; // pad oil under the best response
  gain_bopd: number; // plan_oil - run_all_oil
  /** Non-full rows only; empty = no change needed. */
  actions: ChokeLadderAction[];
}

export interface ChokePlanResult {
  pad: string;
  plan: ChokePlanRow[];
  meta: Record<string, unknown> & {
    /** Header-drop decision ladder, sorted by drop_psi ascending.
     *  Absent on runs made before this feature. */
    ladder?: ChokeLadderRung[];
  }; // run_choke_optimization meta contract
  notes: string[];
  n_wells: number;
}

export interface CfpMoveRow {
  well: string;
  pad: string;
  type: "resize" | "shut_in" | "bring_online";
  from: string | null;
  to: string | null;
  fleet_oil_delta: number;
  own_oil_delta: number;
  pressure_delta: number;
  pressure_after: number;
  at_trip: boolean;
  own_water_delta: number | null; // BWPD at the move's settled discharge
}

export interface CfpFrontierPoint {
  lam: number;
  pressure: number;
  oil: number;
  water: number;
  at_trip: boolean;
}

export interface CfpPlanAction {
  well: string;
  pad: string;
  type: "resize" | "shut_in" | "bring_online";
  from: string | null;
  to: string | null;
  own_oil_delta: number;
  own_water_delta: number;
}

export interface CfpPlan {
  lam: number;
  pressure: number;
  oil: number;
  water: number;
  at_trip: boolean;
  actions: CfpPlanAction[];
  n_changes: number;
  choices: Record<string, string>;
}

/** Today-vs-plan per well, read off the same response surfaces. */
export interface CfpWellRow {
  well: string;
  pad: string;
  online: boolean;
  baseline_label: string;
  plan_label: string;
  baseline_oil: number;
  plan_oil: number;
  baseline_water: number;
  plan_water: number;
  changed: boolean;
}

export interface CfpRunResult {
  pads: string[];
  notes: string[];
  n_wells: number;
  p0_psi: number;
  summary: {
    today: { pressure: number; oil: number; water: number; n_online: number; n_bol_candidates: number };
    lambda_bopd_per_psi: number | null;
    singles: CfpMoveRow[];
    n_positive_singles: number;
    pairs: Record<string, unknown>[];
    frontier: CfpFrontierPoint[];
    plan: CfpPlan | null;
    plan_gain: number | null;
    baseline: Record<string, string>;
  };
  wells: CfpWellRow[];
}

export interface OptimizeJobStatus {
  job_id: string;
  kind: "pad" | "cfp" | "match_health" | "event_cal";
  status: "running" | "done" | "error";
  progress: string | null;
  result:
    | PadRunResult
    | ChokePlanResult
    | CfpRunResult
    | MatchHealthResult
    | EventCalibrationResult
    | null;
  error: string | null;
  started_at: string;
  seconds: number;
}

export interface OptimizeRunStarted {
  job_id: string;
}

/** POST /optimize/match-health - start a scorecard job for one pad. */
export interface MatchHealthRequest {
  pad: RunPad;
}

export type MatchHealthVerdict = "contradicted" | "railed-cal" | "weak-fit" | "ok";

/** One well on the match-health scorecard. Every column is null-safe:
 * evidence columns are null when the warehouse was unreachable, model
 * columns when the well had no current pump or did not solve. */
export interface MatchHealthRow {
  well: string;
  pump: string | null;
  ipr_source: string | null;
  ipr_r2: number | null;
  test_oil: number | null;
  model_oil: number | null;
  model_test_oil_ratio: number | null;
  oil_flag: string | null;
  test_pf: number | null;
  model_pf: number | null;
  model_test_pf_ratio: number | null;
  pf_flag: string | null;
  model_psu: number | null;
  sonic: boolean | null;
  evidence_floor: number | null;
  floor_violation: number | null;
  beta: number | null;
  beta_source: string | null;
  n_pairs: number | null;
  ken: number | null;
  kth: number | null;
  kdi: number | null;
  ken_railed: boolean;
  kth_railed: boolean;
  kdi_railed: boolean;
  last_test_date: string | null;
  verdict: MatchHealthVerdict;
}

export interface MatchHealthResult {
  pad: string;
  rows: MatchHealthRow[];
  header_psi: number | null;
  notes: string[];
  n_wells: number;
}

/** POST /match-test - gaugeless test match: infer the anchor BHP from the
 * test's power-fluid rate and fit kth/kdi so the installed pump reproduces
 * the test's oil and PF. Mirror of server.schemas.MatchTestRequest. */
export interface MatchTestRequest {
  well: string;
  params: SimParams;
  test_oil: number; // STBOPD
  test_water: number; // formation BWPD
  test_pf: number; // power fluid BWPD
  test_whp: number | null;
  test_pf_press: number | null;
  test_date: string | null;
}

export interface MatchTestScanPoint {
  pwf: number;
  psu: number | null;
  oil: number | null;
  pf: number | null;
  sonic: boolean | null;
}

/** Mirror of server.schemas.MatchTestResponse. */
export interface MatchTestResponse {
  match_quality: "good" | "fair" | "poor" | "failed";
  converged: boolean;
  bounded: boolean;
  sonic: boolean;
  pwf: number | null; // inferred anchor BHP, psi
  qwf_liq: number; // anchor TOTAL liquid, BLPD
  form_wc: number; // the test's water cut
  kth: number;
  kdi: number;
  ken: number; // held
  modeled_bhp: number | null;
  modeled_oil: number | null;
  modeled_water: number | null;
  modeled_pf: number | null;
  score: number | null;
  oil_error_pct: number | null;
  pf_error_pct: number | null;
  pwh_used: number;
  ppf_surf_used: number;
  seed_pwf: number | null;
  scan: MatchTestScanPoint[];
  iterations: number;
  starts_tried: number;
  message: string | null;
  caveat: string;
  /** false = the test's PF is outside what the catalog nozzle passes at any
   *  BHP at this PF pressure: the BHP is NOT identified. */
  pf_reachable: boolean;
  pf_model_min: number | null;
  pf_model_max: number | null;
  /** BHP change worth a 2 % PF error on this well, psi - the resolution of
   *  the inferred BHP. */
  bhp_resolution_psi: number | null;
  pf_per_100psi: number | null;
  /** Only when pf_reachable is false: the nozzle AREA factor that would let
   *  the catalog nozzle pass the test's PF at the fitted point,
   *  (pf_test / modeled_pf)^2 - the sidebar's nozzle_area_factor knob
   *  (bounded 0.8 - 1.3) in the same units. */
  area_factor_needed: number | null;
}

/** POST /optimize/event-calibration - start a multi-point era-history
 * calibration job for one well; poll via /optimize/run/{job_id}. */
export interface EventCalibrationRequest {
  well: string;
}

/** The fitted knobs and fit quality. Null when the job refused to fit
 * (see EventCalibrationResult.refusal). */
export interface EventCalFit {
  ken: number;
  kth: number;
  kdi: number;
  /** nozzle_area_factor: nozzle flow-area multiplier (>1 = washout). */
  fnz: number;
  mach_crit: number;
  rms_bhp_psi: number;
  rms_pf_pct: number;
  rms_dbhp_psi: number | null;
  n_used: number;
  n_dropped: number;
  /** Params fitted onto a search bound - treat as low confidence. */
  railed: string[];
  /** Model's suction response -dBHP/dPpf at the fitted point. */
  implied_beta: number | null;
  message: string;
}

/** The single-point fallback leg's result - the old Auto-match BHP
 * mechanics run server-side when the era fit is impossible (young era).
 * Only ken/kth/kdi are fitted; fnz/mach_crit are untouched. */
export interface SinglePointMatch {
  ken: number;
  kth: number;
  kdi: number;
  modeled_bhp: number | null;
  target_bhp: number | null;
  // "pinned": sonic well - target BHP sits on the cavitation floor, so a
  // single BHP point cannot identify friction; coefs come back at their
  // seeds. "failed": no valid operating point at any friction setting.
  match_quality: "good" | "fair" | "poor" | "failed" | "pinned";
  /** Explanation for special outcomes (set on "pinned" runs). */
  message: string | null;
}

export interface EventCalibrationResult {
  well: string;
  pump: string | null;
  era_start: string | null;
  n_daily: number;
  n_test: number;
  /** PF-pressure spread across the era's points, psi. */
  ppf_spread: number;
  /** Non-null when the job declined to fit (thin/degenerate history). */
  refusal: string | null;
  /** "event" = multi-point era fit; "single_point" = young-era fallback
   * that matched the latest test's BHP instead. */
  method: "event" | "single_point";
  /** The builder's refusal that triggered the fallback (single_point only). */
  fallback_reason: string | null;
  /** The fallback match (single_point only); null on the event method. */
  single: SinglePointMatch | null;
  fit: EventCalFit | null;
  /** Measured suction response mined from event pairs, for cross-check. */
  mined_beta: number | null;
  mined_beta_source: string | null;
  /** The coefficients currently saved on the well, for comparison. */
  current: { ken: number | null; kth: number | null; kdi: number | null };
}

/** GET /optimize/pump-curve - mirror of server.schemas.PumpCurveResponse.
 * Industry-format booster-pump curves for one pad's plant: station head
 * capability, the vendor machine curves, and their operating regions. */
export interface PumpCurveNameplate {
  equipment: string;
  model: string;
  arrangement: string;
  speed: string;
  source: string;
  validated: string;
}

export interface PumpCurveLine {
  label: string;
  n_pumps: number | null;
  hz: number | null;
  active: boolean;
  points: number[][]; // [flow_bpd, discharge_psi]
}

export interface PumpStationCurve {
  curves: PumpCurveLine[];
  frontier: PumpCurveLine | null;
  bep: number | null;
  por: number[] | null;
  aor: number[] | null;
  min_flow: number | null;
  header_cap: number | null;
}

export interface PumpMachineCurve {
  label: string;
  hz: number;
  points: number[][]; // [flow_bpd, head_ft, bhp, eff_pct]
  head_derated: number[][] | null;
  derate_note: string | null;
  bep: number | null;
  por: number[] | null;
  aor: number[] | null;
  min_flow: number | null;
}

export interface PumpCurveResponse {
  pad: string;
  coupling: string;
  n_pumps: number | null;
  /** Selectable online-pump counts (e.g. M = [3, 2, 1]); [] = fixed train. */
  n_pump_options: number[];
  sg: number;
  suction_psi: number;
  max_header_psi: number | null;
  nameplate: PumpCurveNameplate;
  station: PumpStationCurve;
  pumps: PumpMachineCurve[];
}

// ---------------------------------------------------------------------------
// E-Pad booster candidate comparison
// ---------------------------------------------------------------------------

/** POST /optimize/e-pad-booster - mirror of server.schemas.EPadBoosterRequest.
 * The duty an engineer sweeps: how much dP the booster has to make, and the
 * fluid / wear / speed / motor limits it has to make it under. Omitted fields
 * take the server defaults (600 psid from 2,800 psi suction = the 3,400 psig
 * E-Pad power-fluid header). */
export interface EPadBoosterRequest {
  dp_psid: number;
  suction_psi: number;
  sg: number;
  /** head-only wear derate; 1.00 = as-new */
  condition: number;
  hz_max: number;
  amps_per_bhp: number;
  /** null = report amps, enforce no cap */
  amp_limit_a: number | null;
}

export interface EPadTarget {
  dp_psid: number;
  suction_psi: number;
  discharge_psi: number;
  sg: number;
  condition: number;
  hz_max: number;
  amps_per_bhp: number;
  amp_limit_a: number | null;
  /** E-Pad's PF header setpoint (psig) - what the one-click duty button aims at */
  header_default_psi: number;
}

export interface EPadNotes {
  amps: string;
  condition: string;
  housing_pressure: string;
  not_enforced: string[];
  stage_table: string;
}

export interface EPadPumpNameplate {
  key: string;
  label: string;
  installed: boolean;
  model: string;
  stage_type: string;
  series_housing: string;
  arrangement: string;
  n_stages: number;
  motor: string;
  amp_limit_a: number | null;
  amps_per_bhp: number;
  /** catalog reference only - NOT enforced; see EPadNotes.housing_pressure */
  shaft_limit_hp: number;
  housing_pressure_limit_psi: number;
  source: string;
}

/** One flow on the constant-dP locus. `hz` and everything after it are null
 * when no speed holds the required dP at this flow. `ror_lo`/`ror_hi` are the
 * recommended range AT THAT SPEED, which is why they move point to point. */
export interface EPadPoint {
  q_bpd: number;
  hz: number | null;
  dp_psid: number | null;
  discharge_psi: number | null;
  head_ft: number | null;
  bhp: number | null;
  amps: number | null;
  amp_headroom_a: number | null;
  eff_pct: number | null;
  pct_of_bep: number | null;
  ror_lo: number | null;
  ror_hi: number | null;
  in_ror: boolean;
  amp_ok: boolean;
  ok: boolean;
  blocked_by: string | null;
}

export interface EPadSpeedCurve {
  hz: number;
  label: string;
  points: number[][]; // [flow_bpd, dp_psid, bhp, amps, eff_pct]
}

/** One rung of the fixed-speed ladder: pin the drive here and this is the
 * flow that comes out at the required dP, checked against the recommended
 * range AT THAT SPEED. `q_bpd` and everything downstream are null when the
 * speed's whole curve sits below the required dP. */
export interface EPadSpeedRow {
  hz: number;
  /** true on the solved duty speed - the row that explains the headline */
  is_duty: boolean;
  q_bpd: number | null;
  discharge_psi: number | null;
  ror_lo: number;
  ror_hi: number;
  pct_of_ror_hi: number | null;
  in_ror: boolean;
  bhp: number | null;
  amps: number | null;
  amp_ok: boolean;
  eff_pct: number | null;
  blocked_by: string | null;
}

/** The other operating policy: run flat out at the speed cap and choke the
 * surplus pressure off. More water, more shaft power, a throttling loss.
 * Null when the speed cap cannot make the required dP in range at all. */
export interface EPadThrottled {
  q_bpd: number;
  hz: number;
  dp_made_psid: number;
  discharge_psi: number;
  throttle_psid: number;
  /** hydraulic HP burned across the choke */
  throttle_hhp: number;
  bhp: number;
  amps: number;
  amp_headroom_a: number | null;
  eff_pct: number;
  in_ror: boolean;
}

export interface EPadCandidate {
  nameplate: EPadPumpNameplate;
  bep_60hz: number;
  ror_60hz: number[];
  max_valid_flow_60hz: number;
  /** the flow the constant-dP locus ends at: past it the build cannot hold
   *  the dP (out of head at hz_max, or it would over-deliver) */
  q_ceiling: number;
  /** the MAX-flow feasible point - the deliverable water rate at this duty,
   *  reached by SLOWING the drive until it makes exactly that dP */
  duty: EPadPoint | null;
  /** the bottom of the same window - the turndown */
  min_duty: EPadPoint | null;
  /** the run-flat-out-and-choke alternative to `duty` */
  throttled: EPadThrottled | null;
  window: number[] | null;
  limited_by: string;
  infeasible_reason: string | null;
  locus: EPadPoint[];
  curves: EPadSpeedCurve[];
  speed_table: EPadSpeedRow[];
  machine: PumpMachineCurve;
}

export interface EPadBoosterResponse {
  pad: string;
  target: EPadTarget;
  notes: EPadNotes;
  /** installed build first */
  candidates: EPadCandidate[];
}


// ---------------------------------------------------------------------------
// Sensitivity
// ---------------------------------------------------------------------------

/** One solve at one swept value - mirror of server.schemas.SensitivityPoint.
 * Null metrics mean the solver failed at that value; `error` says why. */
export interface SensitivityPoint {
  value: number; // the swept value; catalog index for discrete knobs
  label: string; // display value, e.g. "0.30" or "14C"
  psu: number | null; // suction BHP, psig
  qoil: number | null; // STBOPD
  qliq: number | null; // oil + formation water, BLPD
  qpf: number | null; // power fluid, BWPD
  mach: number | null;
  sonic: boolean | null;
  error: string | null; // short reason when the solve failed
}

/** Engineer override for one knob's swept range, in the knob's OWN units -
 * mirror of server.schemas.KnobBounds.
 *
 * Continuous knobs take ABSOLUTE field values (form_gor in scf/bbl, ken
 * unitless, pressures in psi), NOT multipliers or deltas. Catalog knobs
 * (nozzle_no, area_ratio) take 0-based indices into the knob's `options`
 * list. Which one a knob is is `SensitivityKnob.kind`. */
export interface KnobBounds {
  low: number;
  high: number;
  steps: number | null; // 2-15; null = the knob's default
}

/** One calibration knob swept over its range - mirror of
 * server.schemas.SensitivityKnob. */
export interface SensitivityKnob {
  id: string;
  label: string;
  unit: string; // "psi", "scf/bbl", "" for unitless
  baseline_label: string;
  basis: string; // one line: WHY this range (goes in the tooltip)
  points: SensitivityPoint[];
  /** Signed excursions from baseline over the whole sweep, keyed
   * psu | qoil | qliq | qpf - same units as SensitivityPoint (psig, STBOPD,
   * BLPD, BWPD). A metric is null when every solve on that side failed. */
  low: Record<string, number | null>;
  high: Record<string, number | null>;
  /** True when the knob moves NOTHING measurably across its whole sweep -
   * the headline finding on a choked well. */
  inert: boolean;
  field: string; // the SimParams field this knob drives
  kind: string; // "mult" | "abs" | "delta" | "catalog"
  default_low: number; // resolved ABSOLUTE low with no override, knob units
  default_high: number; // resolved ABSOLUTE high with no override, knob units
  swept_low: number; // what was actually swept (post-clamp), knob units
  swept_high: number;
  clamp_low: number | null; // hard limit the sidebar/model enforces
  clamp_high: number | null;
  options: string[] | null; // catalog knobs only: the full option list
  overridden: boolean; // an override was supplied AND applied
}

/** POST /sensitivity - mirror of server.schemas.SensitivityResponse.
 * Read-only diagnostic: nothing here changes the model or is persisted. */
export interface SensitivityResponse {
  baseline: SensitivityPoint;
  knobs: SensitivityKnob[];
  // Measured test values to compare against, echoed back when supplied.
  target_psu: number | null; // psig
  target_qoil: number | null; // STBOPD
  target_qliq: number | null; // BLPD
  target_qpf: number | null; // BWPD
  notes: string[];
}

/** Mirror of server.schemas.SensitivityRequest. */
export interface SensitivityRequest {
  well: string;
  params: SimParams;
  // Measured test values for the reference lines; all optional.
  target_psu: number | null; // psig
  target_qoil: number | null; // STBOPD
  target_qliq: number | null; // BLPD
  target_qpf: number | null; // BWPD
  // Knob id -> engineer override of that knob's swept range. Absent ids keep
  // their default range.
  bounds: Record<string, KnobBounds>;
}

/** One knob to vary in the combined study - mirror of
 * server.schemas.CombineKnob. Same units as KnobBounds. */
export interface CombineKnob {
  id: string;
  low: number;
  high: number;
  levels: number; // 2-7 values across [low, high]; 2 = corners only
}

/** One permutation of the combined study - mirror of
 * server.schemas.CombineRun. Nulls where the solver failed. */
export interface CombineRun {
  values: Record<string, number>; // knob id -> swept value, knob units
  labels: Record<string, string>; // knob id -> display value
  psu: number | null; // suction BHP, psig
  qoil: number | null; // STBOPD
  qliq: number | null; // oil + formation water, BLPD
  qpf: number | null; // power fluid, BWPD
  sonic: boolean | null;
  error: string | null; // short reason when the solve failed
  /** Root-mean-square fractional error across the SUPPLIED targets only.
   * Unitless; lower is better. Null when no target was supplied or the run
   * failed. */
  score: number | null;
}

/** POST /sensitivity/combine - mirror of server.schemas.CombineRequest. */
export interface CombineRequest {
  well: string;
  params: SimParams;
  // Measured test values scored against; all optional.
  target_psu: number | null; // psig
  target_qoil: number | null; // STBOPD
  target_qliq: number | null; // BLPD
  target_qpf: number | null; // BWPD
  // Knobs varied together, each with its own range and level count. The
  // factorial is prod(levels) runs; the server rejects more than 1200.
  knobs: CombineKnob[];
}

/** Mirror of server.schemas.CombineResponse. Read-only diagnostic. */
export interface CombineResponse {
  baseline: SensitivityPoint;
  runs: CombineRun[];
  /** Reachable [min, max] per metric across every solved run, keyed
   * psu | qoil | qliq | qpf (psig, STBOPD, BLPD, BWPD). Sparse: a metric key
   * is absent when no run solved for it, so index it defensively. */
  envelope: Record<string, number[]>;
  /** Per metric: is the supplied target inside the envelope? Sparse in the
   * same way - the key is absent when no target was supplied for that
   * metric, which is NOT the same as false. */
  reachable: Record<string, boolean>;
  best_index: number | null; // index into runs, lowest score
  n_runs: number;
  n_failed: number;
  notes: string[];
}

/** POST /sensitivity/combine - mirror of server.schemas.CombineStarted. The
 * study is a background job; poll it with the id. */
export interface CombineStarted {
  job_id: string;
}

/** GET /sensitivity/combine/{job_id} - mirror of
 * server.schemas.CombineJobStatus. `result` populates when status is done. */
export interface CombineJobStatus {
  job_id: string;
  kind: "sensitivity";
  status: "running" | "done" | "error";
  progress: string | null;
  result: CombineResponse | null;
  error: string | null;
  started_at: string;
  seconds: number;
}

/** One daily-median BHP from an uploaded memory gauge. */
export interface GaugeDay {
  date: string; // YYYY-MM-DD
  bhp: number;
}

export interface GaugeFileMeta {
  filename: string;
  start_date: string;
  end_date: string;
  sample_count: number; // RAW pre-downsample points
  pressure_min: number;
  pressure_max: number;
}

/** POST /gauge/parse - combined server-side parse of all of a well's files. */
export interface GaugeParseResponse {
  files: GaugeFileMeta[];
  daily: GaugeDay[];
  start_date: string;
  end_date: string;
  sample_count: number;
}

export interface IprFitRequest {
  well: string;
  anchor_mode: AnchorMode;
  anchor_date: string | null;
  field_model: FieldModel;
  months: number;
  cap: number;
  /** Memory-gauge daily medians - override test BHP inside coverage. */
  bhp_overrides: GaugeDay[] | null;
}

export interface IprCoeffs {
  res_p: number;
  qmax: number | null;
  qwf: number;
  pwf: number;
  form_wc: number;
  fgor: number;
  r2: number | null;
  num_tests: number;
  most_recent_date: string | null;
  anchor_label: string | null;
  anchor_date: string | null;
  /** "fit" | "floor_fallback" - the latter is max BHP + 50, not a fitted RP; always reported weak. */
  rp_source?: "fit" | "floor_fallback" | string;
}

export interface IprFitResponse {
  well: string;
  coeffs: IprCoeffs;
  seeds: Partial<SimParams>;
  weak: boolean;
}

export interface IprPinResponse {
  status: "none" | "applied" | "stale";
  wt_uid: number | null;
  date_token: string | null;
  entry_user: string | null;
  entry_datetime: string | null;
}

/** POST /wells/{name}/save-ipr - mirror of server.schemas.SaveIprRequest. */
export interface SaveIprRequest {
  qwf_liq: number; // TOTAL LIQUID (BLPD) - the sidebar's qwf verbatim
  pwf: number;
  res_pres: number;
  form_wc: number;
  form_gor: number;
  surf_pres: number | null;
  ken: number | null;
  kth: number | null;
  kdi: number | null;
  /** Event-calibration knobs; server skips values at the 1.0 no-op default
   *  unless a saved override already exists (the friction discipline). */
  nozzle_area_factor: number | null;
  mach_crit: number | null;
  /** Canonical characterization (resvr_bubb / resvr_temp). Sent only when the
   *  engineer moved it off the seeded value, so a save never re-writes the
   *  characterization the server handed out. */
  bubble_point: number | null;
  form_temp: number | null;
  /** Manual-point save: clear any pinned anchor test first, so the well does
   *  not reopen claiming a test the curve was not derived from. */
  unpin: boolean;
  comment: string | null;
  pin_wt_uid: number | null; // null = values-only save (no pinnable anchor)
  pin_date: string | null;
}

export interface SaveIprResponse {
  pinned: boolean;
  pin_skipped: boolean;
  pin_message: string | null;
  n_values: number;
  values_message: string;
}

export interface ClearIprPinResponse {
  cleared: boolean;
  message: string;
}

/** POST /wells/{name}/prop-lock - mirror of server.schemas.PropLockRequest. */
export interface PropLockRequest {
  field: "form_wc" | "form_gor" | "res_pres";
  locked: boolean;
  value: number | null; // pushed only when locking; server re-caps WC at 0.99
}

export interface PropLockResponse {
  ok: boolean;
  message: string;
  field: string;
  locked: boolean; // lock state AFTER the call
  value: number | null;
}

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

export interface BatchRow {
  nozzle: string;
  throat: string;
  qoil_std: number;
  form_wat: number;
  lift_wat: number;
  totl_wat: number;
  psu_solv: number;
  mach_te: number;
  sonic_status: boolean;
  semi: boolean;
  motwr?: number | null;
  molwr?: number | null;
  mofwr?: number | null;
  [key: string]: unknown;
}

export interface BatchStats {
  total: number;
  successful: number;
  success_pct: number;
}

export interface BatchRecommendation {
  nozzle: string;
  throat: string;
  qoil_std: number;
  water_rate: number;
  marginal_ratio: number | null;
  recommendation_type: "optimal" | "best_available";
  theoretical_water_rate: number | null;
  theoretical_oil_rate: number | null;
}

export interface BatchResponse {
  rows: BatchRow[];
  stats: BatchStats;
  recommended: BatchRecommendation | null;
  fit_curve: { x: number[]; y: number[] } | null;
  x_mode: WaterType;
}

// ---------------------------------------------------------------------------
// PF range
// ---------------------------------------------------------------------------

export interface PfRangeRow {
  pump: string; // "12B"
  nozzle: string;
  throat: string;
  power_fluid_pressure: number; // psi
  qoil_std: number;
  form_wat: number;
  lift_wat: number;
  totl_wat: number;
  psu_solv: number;
  mach_te: number;
  sonic_status: boolean;
  [key: string]: unknown;
}

export interface PfRangeResponse {
  rows: PfRangeRow[];
  pressures: number[];
}

// ---------------------------------------------------------------------------
// Pressure profile
// ---------------------------------------------------------------------------

export interface PressureProfileResponse {
  prod: { md: number[]; press: number[] };
  pf: { md: number[]; press: number[] };
  diff: { md: number[]; dp: number[] };
  jpump_md: number;
  metrics: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Well profile
// ---------------------------------------------------------------------------

export interface WellProfileResponse {
  well: string;
  has_survey: boolean;
  md: number[];
  vd: number[];
  hd: number[];
  md_filtered: number[];
  vd_filtered: number[];
  jetpump_md: number | null;
  jetpump_vd: number | null;
  inclination: { md: number[]; deg: number[] } | null;
}

export interface DepthStation {
  md: number;
  tvd: number;
}

export interface DepthLookupResponse {
  well: string;
  has_survey: boolean;
  /** minimum_curvature when the survey carries inclination + azimuth. */
  method: "minimum_curvature" | "chord";
  given: "md" | "tvd";
  md: number;
  tvd: number;
  inclination: number | null;
  azimuth: number | null;
  /** Dogleg severity of the containing segment, deg/100 ft. */
  dls: number | null;
  /** Every MD reaching that TVD - a horizontal well crosses one twice. */
  md_solutions: number[];
  at_station: boolean;
  station_above: DepthStation | null;
  station_below: DepthStation | null;
  station_count: number;
  md_range: number[];
  tvd_range: number[];
  note: string | null;
}

// ---------------------------------------------------------------------------
// Equivalents
// ---------------------------------------------------------------------------

export interface EquivalentRow {
  brand: string;
  nozzle: string;
  throat: string;
  nozzle_dia: number;
  throat_dia: number;
  nozzle_area: number;
  throat_area: number;
  area_ratio_val: number;
  is_reference: boolean;
  [key: string]: unknown;
}

export interface EquivalentsResponse {
  nozzle_no: string;
  area_ratio: string;
  rows: EquivalentRow[];
}

// ---------------------------------------------------------------------------
// JP history
// ---------------------------------------------------------------------------

export interface JpInstallRow {
  date_set: string | null;
  date_pulled: string | null;
  nozzle: string | null;
  throat: string | null;
  tubing_od: number | null;
  circulating: string | null;
  manufacturer: string | null;
  raw_pump: string | null;
  pump_converted: boolean;
  [key: string]: unknown;
}

export interface JpHistoryResponse {
  well: string;
  installs: JpInstallRow[];
  tests: Record<string, unknown>[];
  bhp_daily: { date: string; bhp: number }[];
  current_pump: string | null;
  source: "databricks" | "excel_fallback";
}

// ---------------------------------------------------------------------------
// Well database
// ---------------------------------------------------------------------------

export interface WellDatabaseResponse {
  rows: Record<string, unknown>[];
  source: "databricks" | "csv_fallback";
  missing_surveys: string[];
}

export interface AgingPumpsResponse {
  rows: Record<string, unknown>[];
}

export interface PropHistoryResponse {
  well: string;
  current: Record<string, unknown>[];
  history: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// Well Sort
// ---------------------------------------------------------------------------

export type WellSortMode = "allocated" | "any";

/** Online-table row - server/services/well_sort._ONLINE_COLUMNS. */
export interface WellSortOnlineRow extends Record<string, unknown> {
  well: string;
  pad: string | null;
  reservoir: string | null;
  lift_type: string | null;
  pops_pad: boolean;
  test_date: string | null;
  days_since_test: number | null;
  stale_test: boolean;
  allocated: boolean;
  fallback_used: boolean;
  oil: number | null;
  water: number | null;
  gas: number | null;
  lift_water: number | null;
  lift_gas: number | null;
  total_water: number | null;
  total_gas: number | null;
  esp_hz: number | null;
  esp_amps: number | null;
  wc: number | null; // fraction
  total_wc: number | null; // fraction
  gor: number | null;
  total_gor: number | null;
  bhp: number | null;
  whp: number | null;
  oil_2mo_avg: number | null;
  wat_2mo_avg: number | null;
  oil_dev: number | null; // fraction vs 2-mo avg
  wat_dev: number | null;
  flag_outlier: boolean;
  alloc_vs_info_oil_pct: number | null;
  latest_alloc_date: string | null;
  latest_info_date: string | null;
  prod_xv: number | null; // 1=open 0=closed
  pf_xv: number | null;
  xv_time: string | null; // pre-formatted "MM-DD HH:mm"
  just_restarted: boolean;
}

/** Offline/LTSI row - server/services/well_sort._SHUT_COLUMNS. */
export interface WellSortShutRow extends Record<string, unknown> {
  well: string;
  pad: string | null;
  reservoir: string | null;
  lift_type: string | null;
  pops_pad: boolean;
  shut_in_since: string | null;
  current_code: string | null;
  current_reason: string | null;
  notes: string | null;
  down_hours: number | null;
  last_online_date: string | null;
  last_test_date: string | null;
  oil: number | null;
  water: number | null;
  gas: number | null;
  lift_water: number | null;
  lift_gas: number | null;
  total_water: number | null;
  total_gas: number | null;
  esp_hz: number | null;
  esp_amps: number | null;
  wc: number | null;
  total_wc: number | null;
  gor: number | null;
  total_gor: number | null;
  near_avg_oil: number | null;
  near_avg_water: number | null;
  near_avg_gas: number | null;
  n_tests_near: number | null;
  prod_xv: number | null;
  pf_xv: number | null;
  xv_time: string | null;
}

export interface WellSortTablesResponse {
  online: WellSortOnlineRow[];
  offline: WellSortShutRow[];
  ltsi: WellSortShutRow[];
  all_pads: string[];
  producers: string[];
  xv_available: boolean;
  tests_window_days: number;
  outliers_flagged: number;
  just_restarted: number;
  default_pops_pads: string[];
  pump_limit_presets: Record<string, number>;
  pops_pump_handles: Record<string, "total" | "lift">;
}

export interface WellSortEventRow extends Record<string, unknown> {
  well: string;
  pad: string | null;
  reservoir: string | null;
  started: string | null;
  ended: string | null; // null while ongoing
  days: number;
  max_hrs: number;
  total_hrs: number;
  code: string | null;
  reason: string | null;
  notes: string | null;
  ongoing: boolean;
}

export interface WellSortEventsResponse {
  rows: WellSortEventRow[];
}

export interface MarginalRankedRow extends Record<string, unknown> {
  well: string;
  pad: string | null;
  reservoir: string | null;
  oil: number | null;
  total_water: number | null;
  total_wc: number | null; // fraction
  cum_water: number | null;
  cum_water_pct: number | null; // 0-100
}

export interface MarginalWcResponse {
  marginal_wc: number; // fraction
  well: string;
  pad: string;
  total_field_water: number;
  well_count: number;
  threshold_pct: number;
  marg_idx: number;
  cum_water_at_marginal: number | null;
  rows: MarginalRankedRow[];
}

export interface PadRankedRow extends Record<string, unknown> {
  well: string;
  reservoir: string | null;
  oil: number | null;
  lift_water: number | null;
  total_water: number | null;
  total_wc: number | null;
  wc_pad: number | null; // fraction, on the pad pump's stream
}

export interface PadMarginalWcResponse {
  marginal_wc: number;
  well: string;
  pad: string;
  pad_water: number;
  pump_limit: number;
  headroom: number | null; // null when limit unset
  well_count: number;
  water_basis: "total" | "lift";
  rows: PadRankedRow[];
}

export type TriageOnlineCode = "pops" | "verify_stale" | "keep" | "verify_si" | "si";
export type TriageShutCode =
  | "verify_no_test"
  | "bol"
  | "bol_trial"
  | "verify_form_hist"
  | "leave_shut";

export interface TriageOnlineRow extends WellSortOnlineRow {
  decision_code: TriageOnlineCode;
  why: string;
  wc_vs_marginal: number | null; // fraction delta
  wc_basis: "total" | "form" | null;
  rank: number; // 0=SI 1=verify 2=keep 4=pops
}

export interface TriageShutRow extends WellSortShutRow {
  decision_code: TriageShutCode;
  why: string;
  wc_vs_marginal: number | null;
  wc_basis: "total" | "form" | null;
  near_avg_wc: number | null; // fraction
  near_avg_wc_basis: "total" | "form" | null;
  rank: number; // 0=BOL 1=trial 2=verify 3=leave shut
}

export interface TriageResponse {
  marginal_wc: number;
  well: string;
  pad: string;
  threshold_pct: number;
  raw_worst_wc: number | null;
  raw_worst_well: string | null;
  raw_worst_water: number | null;
  xv_available: boolean;
  online: TriageOnlineRow[];
  shut: TriageShutRow[];
}

// ---------------------------------------------------------------------------
// Suction response history (Solver advanced diagnostic)
// ---------------------------------------------------------------------------

/** One daily (PF pressure, BHP) pair from the field historian. */
export interface ResponseHistoryDay {
  date: string; // YYYY-MM-DD
  ppf: number; // delivered PF pressure (psi)
  bhp: number; // measured suction/BHP (psi)
  era: "current" | "prior"; // relative to the current pump's date_set
  buildup: boolean; // bhp >= reservoir pressure (shut-in/buildup day)
}

/** Measured-response evidence backing the diagnostic, when the server has
 * enough pairs to compute it. */
export interface ResponseHistoryEvidence {
  floor: number | null; // measured suction floor (psi)
  psu_ref: number | null;
  beta: number | null; // dBHP/dPpf slope
  beta_source: string;
  n_pairs: number;
}

export interface ResponseHistoryResponse {
  days: ResponseHistoryDay[];
  era_start: string | null; // ISO date the current pump was set
  pump: string | null; // "14B"
  evidence: ResponseHistoryEvidence | null;
  res_pres: number | null;
}

// ── Scott's Tools (the secret menu) ────────────────────────────────────────

export interface ToolInfo {
  id: string;
  label: string;
  caption: string;
  path: string;
}

export interface ToolCatalogResponse {
  tools: ToolInfo[];
}

export interface DateWindow {
  start: string;
  end: string;
}

export interface PadWatercutPoint {
  date: string | null;
  wc: number | null;
  oil: number | null;
  water: number | null;
}

export interface PadWatercutSeries {
  pad: string;
  points: PadWatercutPoint[];
}

export interface PadWatercutResponse {
  start: string;
  end: string;
  series: PadWatercutSeries[];
}

// Separator Oil Loss - mirrors schemas.SepLossPeriod / SepLossEvent /
// SepOilLossResponse. Every barrel figure is a band (lower, upper).

/** One look-back roll-up. Barrels are a band, never a single number. */
export interface SepLossPeriod {
  label: string; // "Last 24 h"
  days: number;
  hours: number; // valid (separator running) hours in the look-back
  downtime_hours: number;
  flow_avg: number; // BPD, time-weighted
  wc_avg: number; // %, time-weighted
  base_avg: number; // %, the analyzer's own film-corrected plateau
  bbl_upper: number; // meter as read, film-corrected, capped at field oil
  bbl_lower: number; // oil fraction of the leg capped at max_oil_frac
  bopd_upper: number;
  bopd_lower: number;
  pct_field_upper: number | null; // % of field oil production
  pct_field_lower: number | null;
  upset_hours: number;
  events: number;
}

/** One field calendar day (Alaska local) of the loss band. */
export interface SepLossDay {
  date: string; // YYYY-MM-DD
  hours: number; // separator running hours inside the day
  covered_hours: number; // how much of the day the window spans
  bbl_upper: number;
  bbl_lower: number;
  pct_field_upper: number | null; // blank on a day that barely ran
  pct_field_lower: number | null;
  upset_hours: number;
  events: number;
  partial: boolean; // clipped by the window or cut by downtime
}

/** One carry-under excursion, classified by its vessel-level signature. */
export interface SepLossEvent {
  start: string; // ISO 8601, Alaska offset
  end: string;
  hours: number;
  wc_min: number; // %
  wc_avg: number; // %
  flow_avg: number; // BPD
  bbl_upper: number;
  bbl_lower: number;
  level_min: number | null; // controlled level, %
  level_avg: number | null; // %
  level_sp_avg: number | null; // that loop's setpoint, %
  level_dev_avg: number | null; // level - setpoint, points
  // "at setpoint" is the interesting one: level held where it was asked and
  // the water leg ran oil anyway, so separation failed, not level control.
  kind: "level loss" | "off setpoint" | "at setpoint";
  // DataTable rows carry an index signature (same as WellTestRow, EquivalentRow).
  [key: string]: unknown;
}

export interface SepOilLossResponse {
  flow_tag: string; // "MPU_FI_5365", water-leg flow
  wc_tag: string; // "MPU_AI_5317", Red Eye water cut
  level_tag: string | null; // "MPU_LIC_5365CV1", the CONTROLLED level
  level_sp_tag: string | null; // "MPU_LC5365SP1", that loop's setpoint
  days: number;
  start: string | null; // ISO 8601, Alaska offset
  end: string | null;
  field_oil_bopd: number; // ceiling and percent-of-field denominator
  max_oil_frac: number; // oil-fraction cap behind the lower bound
  flow_min_bpd: number; // below this the separator is down; hours excluded
  upset_drop_pts: number; // points below the plateau that count as an upset
  valid_hours: number;
  excluded_hours: number;
  periods: SepLossPeriod[];
  daily: SepLossDay[];
  events: SepLossEvent[];
  /** Parallel arrays keyed by `t` (ISO strings): flow, wc, base, level,
   *  level_sp, oil_upper, cum_upper, cum_lower. Nulls where a tag had none. */
  series: Record<string, (number | null)[] | string[]>;
}

/** The drill-down behind one daily bar: same day at full resolution. */
export interface SepOilLossDayResponse {
  date: string;
  days: number;
  summary: SepLossDay | null;
  events: SepLossEvent[];
  series: Record<string, (number | null)[] | string[]>;
}

// Operator OIW grab samples - mirrors schemas.OiwSampleDay /
// OiwSamplesResponse. Uploaded, parsed, held in page state; never persisted.

/** One Alaska calendar day of grab samples at one location. `bbl` equals
 *  `bopd_mean` - a daily rate held for one day is that many barrels - and
 *  both are the plain unweighted mean of the day's per-sample rates. */
export interface OiwSampleDay {
  date: string; // YYYY-MM-DD, Alaska calendar
  samples: number;
  ppm_mean: number; // ppm oil in water
  ppm_min: number;
  ppm_max: number;
  bopd_mean: number; // ppm x water_rate_bpd / 1e6, BOPD
  bbl: number; // bbl over the day
  location: string;
}

/** POST /tools/sep-oil-loss/samples - one parsed grab-sample workbook.
 *  `notes` carries the water-rate basis and, for every location but V-5317,
 *  the caveat that the samples are downstream of the deoilers. */
export interface OiwSamplesResponse {
  filename: string;
  sheet: string;
  location: string;
  water_rate_bpd: number; // BPD basis the ppm was converted on
  locations_available: string[];
  first_date: string | null; // YYYY-MM-DD
  last_date: string | null;
  sample_count: number; // samples at `location`, not rows in the sheet
  daily: OiwSampleDay[];
  notes: string[];
}

/** Every tool job shares one envelope. `result` shape is per-tool. */
export interface ToolJobStatus {
  job_id: string;
  kind: string;
  status: "running" | "done" | "error";
  progress: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  started_at: string | null;
  seconds: number | null;
}

export interface ToolJobStarted {
  job_id: string;
}

/** Tools return tabular results; columns differ per tool, so rows stay loose. */
export type ToolRow = Record<string, unknown>;

export interface ToolRowsResponse {
  rows: ToolRow[];
  [key: string]: unknown;
}

export interface HarnessCase {
  name: string;
  description: string;
}

export interface HarnessCasesResponse {
  cases: HarnessCase[];
}

export interface HarnessResult {
  name: string;
  description: string;
  passed: boolean;
  summary: string;
  details: Record<string, unknown>;
  error: string | null;
  seconds: number;
}

export interface HarnessRunResult {
  results: HarnessResult[];
  passed: number;
  failed: number;
  total: number;
  seconds: number;
}
