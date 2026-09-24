import type {
  InstallationFitFold, InstallationFitParam, InstallationFitPrediction, InstallationFitResult,
  InstallationModel, PumpMatchEra, PumpMatchResult, PumpMatchRow, PumpMatchScores,
} from "../api/types";

/** In-sample fitted history, or each test's held-out prediction. */
export type FitView = "fit" | "held_out";

const EMPTY_SCORES: PumpMatchScores = {
  attempted: 0, solved: 0, failed: 0, bhp_count: 0, bhp_bias: null, bhp_rms: null,
  oil_count: 0, oil_mae: null, oil_median_abs_pct: null, pf_count: 0, pf_median_abs_pct: null,
};

export const NOT_HELD_OUT = "Before the first split: this test only trains the held-out checks, so it has no held-out prediction.";

/**
 * The fit result in the history chart's replay shape. One view at a time:
 * each test has exactly one in-sample value and at most one held-out value
 * (it belongs to one fold), so the chart's one-value-per-test lines stay
 * honest. Held-out lines are dashed "prediction" phases; fitted history is
 * the dotted "fit" phase. Failed solves remain rows with no value.
 */
export function fitAsMatch(result: InstallationFitResult, model: InstallationModel, view: FitView): PumpMatchResult {
  const predictions: InstallationFitPrediction[] = (view === "fit"
    ? result.models[model]?.predictions : result.cv.held_predictions[model]) ?? [];
  const byIndex = new Map(predictions.map((p) => [p.index, p]));
  const phase = view === "fit" ? "fit" : "prediction";
  const rows: PumpMatchRow[] = result.rows.map((row, index) => {
    const base: PumpMatchRow = { ...row, phase: null, predicted_bhp: null, predicted_oil: null,
      predicted_pf: null, predicted_liquid: null, sonic: null };
    const p = byIndex.get(index);
    if (p?.message) return { ...base, status: "failed", phase, message: p.message };
    if (p) {
      return { ...base, status: phase, phase, message: null,
        predicted_bhp: p.predicted_bhp ?? null, predicted_oil: p.predicted_oil ?? null,
        predicted_pf: p.predicted_pf ?? null, predicted_liquid: p.predicted_liquid ?? null,
        sonic: p.sonic ?? null };
    }
    const usable = row.status === "replay";
    return { ...base, status: "excluded",
      message: usable ? (view === "held_out" ? NOT_HELD_OUT : "No usable BHP, oil or PF on this test.") : row.message };
  });
  const eras: PumpMatchEra[] = result.eras.map((e) => ({
    installation_id: e.installation_id, date_set: e.date_set, end: e.end, pump: e.pump,
    nozzle: e.nozzle, throat: e.throat, direction: null, manufacturer: null, flags: e.flags,
    training_start: null, training_end: null, training_installation_id: null, training_count: 0,
    training_test_ids: [], prediction_config: null, pump_losses: "clean_reference",
    training_ppf_span: e.ppf_span, input_wc: null, input_gor: null, unavailable: e.unavailable,
    fit_scores: EMPTY_SCORES, prediction_scores: EMPTY_SCORES, replay_scores: EMPTY_SCORES,
  }));
  return {
    well: result.well, physics_model: result.physics_model, snapshot_id: result.snapshot_id,
    as_of: result.as_of, source: result.source, notes: result.notes, eras, rows, well_inputs: {},
    validated_for_sizing: false,
    request: { hydraulics_model: result.request.hydraulics_model, months: result.request.months,
      mode: "all_tests", training_tests: 10, edited_inputs: null, pump_losses: "clean_reference" },
  };
}

/** Well-level parameters first, then the named installation's own. */
export function paramsFor(params: InstallationFitParam[], installationId: string | null): InstallationFitParam[] {
  return params.filter((p) => installationId === null ? p.installation_id === null : p.installation_id === installationId);
}

/**
 * How far an installation's own data moved its parameter from the well level,
 * in prior standard deviations (0 = fully pooled; large = strongly its own).
 * Log-scale parameters (the nozzle-area factor) are compared in log space.
 */
export function shrinkageSigmas(p: InstallationFitParam): number | null {
  if (p.prior_mean === null || p.prior_sd === null || p.prior_sd <= 0) return null;
  const moved = p.physical === "fnz" ? Math.log(p.value) - Math.log(p.prior_mean) : p.value - p.prior_mean;
  return moved / p.prior_sd;
}

export interface ChangeoutRow {
  origin: string;
  bhp: { measured: number; predicted: number; clear: boolean; direction_correct: boolean | null } | null;
  oil: { measured: number; predicted: number; clear: boolean; direction_correct: boolean | null } | null;
}

/** Measured vs held-out predicted change across each pump change. */
export function changeouts(folds: InstallationFitFold[], model: InstallationModel): ChangeoutRow[] {
  return folds.filter((f) => f.kind === "changeout" && f.models[model]?.changeout)
    .map((f) => ({ origin: f.origin, bhp: f.models[model]!.changeout!.bhp, oil: f.models[model]!.changeout!.oil }));
}

/** Direction-correct count over clear changes, for one model. */
export function directionScore(rows: ChangeoutRow[]): { correct: number; clear: number } {
  let correct = 0, clear = 0;
  for (const r of rows) for (const d of [r.bhp, r.oil]) {
    if (d?.clear) {
      clear += 1;
      if (d.direction_correct) correct += 1;
    }
  }
  return { correct, clear };
}

/** The Solver's exclusion keys ("uid:<wt_uid>") as the fit's wt_uid strings. */
export function excludedUids(keys: string[]): string[] {
  return keys.filter((k) => k.startsWith("uid:")).map((k) => k.slice(4)).sort();
}
