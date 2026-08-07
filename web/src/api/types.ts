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
  as_built_locks: { tubing: boolean; casing: boolean; jpump_tvd: boolean };
  prop_locks: { form_wc: PropLock; form_gor: PropLock; res_pres: PropLock };
  pump: PumpInfo | null;
  pf: PfSeed | null;
  ipr_info: string | null;
  saved_ipr_info: string | null;
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

export type AnchorMode = "recent" | "median" | "specific";

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

/** POST /optimize/run - mirror of server.schemas.OptimizeRunRequest. */
export interface OptimizeRunRequest {
  kind: "pad" | "cfp";
  pad: "S" | "I" | "M" | null;
  offline: string[];
  future: { name: string; match: string }[];
  nozzles: string[];
  throats: string[];
  method: "milp" | "mckp";
  marginal_wc: number | null; // null = auto-derive from the plant budget
  parsimony_bopd: number;
  n_pumps: number | null; // null = pad default
  n_steps: number | null;
  p0_psi: number;
  psi_per_kbpd: number;
  c_pad_pf_psi: number;
  cfp_pads: string[]; // which of B/G/C/J participate (cfp runs)
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
}

export interface PadRunResult {
  pad: string;
  rows: PadRunRow[];
  meta: Record<string, unknown>; // pad_optimize meta contract, JSON-flattened
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
  kind: "pad" | "cfp";
  status: "running" | "done" | "error";
  progress: string | null;
  result: PadRunResult | CfpRunResult | null;
  error: string | null;
  started_at: string;
  seconds: number;
}

export interface OptimizeRunStarted {
  job_id: string;
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
  sg: number;
  suction_psi: number;
  max_header_psi: number | null;
  nameplate: PumpCurveNameplate;
  station: PumpStationCurve;
  pumps: PumpMachineCurve[];
}

/** POST /calibrate - mirror of server.schemas.CalibrateRequest. Only
 * ken/kth/kdi are searched; as-built geometry is never varied. */
export interface CalibrateRequest {
  well: string;
  params: SimParams;
  target_bhp: number;
  test_whp: number | null;
}

export interface CalibrateResponse {
  converged: boolean;
  match_quality: "good" | "fair" | "poor" | "failed";
  bounded: boolean;
  sonic: boolean;
  ken: number;
  kth: number;
  kdi: number;
  target_bhp: number;
  modeled_bhp: number | null;
  bhp_error: number | null;
  iterations: number;
  starts_tried: number;
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
