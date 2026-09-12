import { DEFAULT_PARAMS, type SimParams, type WellContext } from "../api/types";

const STABLE_INPUTS = ["pres", "form_temp", "jpump_tvd", "tubing_od", "tubing_thickness",
  "casing_od", "casing_thickness", "field_model", "surf_pres", "oil_api", "gas_sg", "wat_sg",
  "rho_pf", "bubble_point", "jpump_direction", "model_as_water"] as const;

const SESSION_ONLY_LABELS = {
  jpump_tvd: "pump depth", tubing_od: "tubing diameter", tubing_thickness: "tubing wall",
  casing_od: "casing diameter", casing_thickness: "casing wall", field_model: "field model",
  oil_api: "oil API", gas_sg: "gas specific gravity", wat_sg: "formation-water specific gravity",
  rho_pf: "power-fluid density", jpump_direction: "circulation direction", model_as_water: "water-pump mode",
} as const;
type SessionOnlyKey = keyof typeof SESSION_ONLY_LABELS;

const same = (a: unknown, b: unknown) => typeof a === "number" && typeof b === "number"
  ? Math.abs(a-b) <= 1e-10 * Math.max(1, Math.abs(a), Math.abs(b)) : a === b;

function oilMax(p: SimParams) {
  const ratio = p.pwf / p.pres;
  return p.qwf * (1-p.form_wc) / (1 - .2*ratio - .8*ratio*ratio);
}

/** Stable dependency changes require a new calibration; composition controls
 * are independent when they preserve the same oil curve. Server verifies hashes. */
export function hasWellModelEdits(params: SimParams, context: WellContext | null) {
  if (!context) return true;
  const saved = { ...DEFAULT_PARAMS, ...context.seeds };
  return !same(oilMax(params), oilMax(saved)) || STABLE_INPUTS.some(k => !same(params[k], saved[k]));
}

/** These model inputs are loaded from characterization/as-built sources and
 * cannot be persisted by Save well inputs or installed-pump calibration. */
export function sessionOnlyWellModelEdits(params: SimParams, context: WellContext | null) {
  if (!context) return [];
  const saved = { ...DEFAULT_PARAMS, ...context.seeds };
  return (Object.keys(SESSION_ONLY_LABELS) as SessionOnlyKey[])
    .filter(key => !same(params[key], saved[key]))
    .map(key => ({ key, label: SESSION_ONLY_LABELS[key] }));
}

/** Restore only unsupported model edits, preserving savable IPR/fluid edits,
 * selected hydraulics and the currently applied pump coefficients. */
export function savedSessionOnlyModelInputs(params: SimParams, context: WellContext | null): Partial<SimParams> {
  if (!context) return {};
  const saved = { ...DEFAULT_PARAMS, ...context.seeds };
  return Object.fromEntries(sessionOnlyWellModelEdits(params, context).map(({ key }) => [key, saved[key]]));
}

export function calibrationInputBlocker(params: SimParams, context: WellContext | null): string | null {
  if (!context) return "Wait for the saved well model to load before calibration.";
  if (context.geometry_issue) return `Resolve the saved survey/pump-depth conflict before calibration: ${context.geometry_issue}`;
  const sessionOnly = sessionOnlyWellModelEdits(params, context);
  if (sessionOnly.length) return `Session-only model edits: ${sessionOnly.map(e => e.label).join(", ")}. Save well inputs cannot persist these settings. Use Restore session-only settings in the save bar before calibrating the saved model. Permanent changes need correction in the well's characterization or as-built source.`;
  return hasWellModelEdits(params, context)
    ? "Save the intended IPR and supported well inputs before calibration. Calibration uses the saved well model."
    : null;
}

/** Compare a fit's server snapshot to saved dependencies, allowing the selected
 * hydraulics to differ while the engineer evaluates another return model. */
export function fitMatchesWellModel(inputs: Record<string, unknown> | null | undefined, context: WellContext | null) {
  if (!inputs || !context?.well_model_inputs) return false;
  return Object.keys(context.well_model_inputs).filter(k => k !== "physics_model")
    .every(k => same(inputs[k], context.well_model_inputs![k]));
}
