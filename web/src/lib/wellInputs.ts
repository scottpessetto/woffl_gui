import { DEFAULT_PARAMS, type PumpMatchWellInputs, type SimParams } from "../api/types";

export const WELL_INPUT_KEYS = ["qwf", "pwf", "pres", "form_wc", "form_gor", "surf_pres", "bubble_point", "form_temp"] as const;

/** The same values drive the historical preview and the database save. */
export function wellInputValues(p: SimParams, seeds: Partial<SimParams> | undefined): PumpMatchWellInputs {
  const characterization = (key: "bubble_point" | "form_temp") => {
    const saved = seeds?.[key];
    return typeof saved === "number" && p[key] !== null && Math.abs(saved - p[key]!) >= 1e-9 ? p[key] : null;
  };
  return {
    qwf_liq: p.qwf, pwf: p.pwf, res_pres: p.pres, form_wc: p.form_wc, form_gor: p.form_gor,
    surf_pres: p.surf_pres, bubble_point: characterization("bubble_point"), form_temp: characterization("form_temp"),
  };
}

export function changedWellInputs(p: SimParams, seeds: Partial<SimParams> | undefined) {
  return WELL_INPUT_KEYS.filter((key) => {
    const before = seeds?.[key] ?? DEFAULT_PARAMS[key];
    const after = p[key];
    return before !== after && (before === null || after === null || Math.abs(before - after) >= 1e-9);
  });
}

export function wellInputProblem(p: SimParams): string | null {
  if (p.model_as_water || p.form_wc > .99) return "Saving an oil-well IPR requires WC at or below 99%.";
  if (p.qwf <= 0 || p.pwf >= p.pres) return "Use a positive IPR rate and anchor BHP below reservoir pressure.";
  return null;
}
