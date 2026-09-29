/**
 * The run tab form's shape and defaults. Pure (no store), so node tests can
 * import it; the optimize store persists one partial form per tab.
 */

/** One run tab's form. Kept per tab (S/I/M/E/CFP) so a pump count or
 *  booster setting chosen on one pad never rides into another pad's run,
 *  and survives leaving the page while a result is on screen. */
export interface RunForm {
  nozzles: string[];
  throats: string[];
  method: "milp" | "mckp";
  strategy: "jpco" | "choke";
  /** true = maximize oil within capacity (no water price). */
  autoLam: boolean;
  /** Manual water price, BOPD given up per 1,000 BPD of machine water. */
  waterPricePerK: number;
  autoSetpoint: boolean;
  manualSetpoint: number;
  /** null = the plant's own default pump count. */
  nPumps: number | null;
  p0: number;
  referencePadPf: Record<string, string>;
  slope: number;
  cPadPf: number;
  cfpPads: string[];
  ePadBuild: "SM25000_26STG" | "SN35000_18STG";
  ePadSuction: number;
  ePadHzMax: number;
  ePadHeaderCap: number;
  ePadAmpLimit: string;
}

/** Mirrors server/schemas.py OptimizeRunRequest defaults. */
export const DEFAULT_RUN_FORM: RunForm = {
  nozzles: ["9", "10", "11", "12", "13", "14", "15"],
  throats: ["A", "B", "C", "D"],
  method: "milp",
  strategy: "jpco",
  autoLam: true,
  waterPricePerK: 20,
  autoSetpoint: true,
  manualSetpoint: 3200,
  nPumps: null,
  p0: 2792,
  referencePadPf: {},
  slope: 13.69,
  cPadPf: 3400,
  cfpPads: ["B", "G", "C", "J"],
  ePadBuild: "SM25000_26STG",
  // E-41 surface-kit rate test: CFP suction at the current-limited maximum
  // rate, and the drive's current limit (woffl/jp_data/E_Pad_Pumps meta).
  ePadSuction: 2704,
  ePadHzMax: 60,
  ePadHeaderCap: 3500,
  ePadAmpLimit: "889",
};

/** A tab's form: stored edits over the defaults. Stored values of the wrong
 *  type (an older saved shape) fall back to the default for that field. */
export function formFor(forms: Record<string, Partial<RunForm>>, runKey: string): RunForm {
  const saved = forms[runKey] ?? {};
  const out = { ...DEFAULT_RUN_FORM };
  for (const key of Object.keys(DEFAULT_RUN_FORM) as (keyof RunForm)[]) {
    const v = saved[key];
    const d = DEFAULT_RUN_FORM[key];
    const ok = v !== undefined && (d === null ? v === null || typeof v === "number"
      : Array.isArray(d) ? Array.isArray(v) : typeof v === typeof d);
    if (ok) (out as Record<string, unknown>)[key] = v;
  }
  return out;
}
