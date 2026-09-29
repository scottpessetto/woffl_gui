/**
 * Pure helpers behind the optimization run screen: the plain-language
 * recommendation, run status, progress/ETA, pre-flight checks and the
 * "inputs changed since this run" diff. No React, so node tests import it.
 */

import type { OptimizeRunRequest, PadRunRow, RunCoverage } from "../../api/types";

// ---------------------------------------------------------------------------
// Run status
// ---------------------------------------------------------------------------

export type RunTone = "good" | "warn" | "bad";

export interface RunStatus {
  tone: RunTone;
  label: string;
  detail: string;
}

/** One status chip from the server's qualification fields. An incomplete
 *  run hides whole-pad feasibility in ``feasible`` (null) and keeps the
 *  subset's verdict in ``modeled_subset_feasible``; both count. */
export function runStatus(meta: Record<string, unknown>, coverage: RunCoverage | undefined): RunStatus {
  const status = typeof meta.recommendation_status === "string" ? meta.recommendation_status : null;
  const infeasible = meta.feasible === false || meta.modeled_subset_feasible === false;
  if (!coverage || !coverage.complete || status === "incomplete_exploratory") {
    return {
      tone: "bad",
      label: "Exploratory",
      detail: infeasible
        ? "Some wells have no usable model, and the modeled wells alone do not meet the operating limits."
        : "Some wells have no usable model, so the totals do not describe the whole pad.",
    };
  }
  if (status === "no_online_wells") return { tone: "warn", label: "No online wells", detail: "Nothing to optimize with the current offline selection." };
  if (infeasible) {
    return {
      tone: "warn",
      label: "Conditional",
      detail: "Every well is modeled, but the plan needs an operating arrangement the plant model does not qualify (see the notes).",
    };
  }
  if (meta.converged === false) return { tone: "warn", label: "Approximate", detail: "The plant pressure balance did not close within tolerance." };
  return { tone: "good", label: "Qualified", detail: "Every online well is modeled and the plan meets the modeled plant limits." };
}

// ---------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------

export interface PlanChange {
  well: string;
  from: string | null;
  to: string | null;
  kind: "replace" | "resize" | "shut_in" | "start";
  gain: number | null;
}

/** The engineer-facing change list: every well whose plan differs from what
 *  is in the ground today. Keeping the installed pump is not a change. */
export function planChanges(rows: PadRunRow[]): PlanChange[] {
  const out: PlanChange[] = [];
  for (const r of rows) {
    const gain = r.modeled_hardware_gain ?? null;
    if (r.pump === null) {
      if (r.outcome === "economic_shut_in" && r.current_pump !== null) {
        out.push({ well: r.well, from: r.current_pump, to: null, kind: "shut_in", gain });
      }
      continue;
    }
    if (r.current_pump === null) {
      out.push({ well: r.well, from: null, to: r.pump, kind: "start", gain });
    } else if (r.pump !== r.current_pump) {
      out.push({ well: r.well, from: r.current_pump, to: r.pump, kind: "resize", gain });
    } else if (r.pump_state === "replacement") {
      out.push({ well: r.well, from: r.current_pump, to: r.pump, kind: "replace", gain });
    }
  }
  // Biggest modeled gain first; unknown gains last, then by name.
  return out.sort((a, b) => (b.gain ?? -Infinity) - (a.gain ?? -Infinity) || a.well.localeCompare(b.well));
}

export function changeText(c: PlanChange): string {
  switch (c.kind) {
    case "shut_in":
      return `shut in (was ${c.from})`;
    case "start":
      return `run ${c.to}`;
    case "replace":
      return `replace ${c.to} with a clean ${c.to}`;
    default:
      return `${c.from} to ${c.to}`;
  }
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

export interface RunProgress {
  fraction: number | null;
  etaSeconds: number | null;
  queued: boolean;
}

/** "trial 4/15 - header ..." / "response surfaces 3/8 - ..." -> fraction and
 *  a naive time-left estimate from the elapsed time. Steps are not equal
 *  work, so the estimate is labelled approximate where it is shown. */
export function runProgress(progress: string | null | undefined, seconds: number | null | undefined): RunProgress {
  const text = progress ?? "";
  const queued = /^queued/.test(text);
  const m = /(\d+)\s*\/\s*(\d+)/.exec(text);
  if (queued || !m) return { fraction: null, etaSeconds: null, queued };
  const step = Number(m[1]);
  const total = Number(m[2]);
  if (!(total > 0) || step < 0) return { fraction: null, etaSeconds: null, queued };
  const fraction = Math.min(1, step / total);
  const elapsed = typeof seconds === "number" && Number.isFinite(seconds) ? seconds : null;
  const etaSeconds = elapsed !== null && step > 0 && step < total ? (elapsed / step) * (total - step) : null;
  return { fraction, etaSeconds, queued };
}

export function fmtDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - 60 * m);
  return s === 0 ? `${m} min` : `${m} min ${s} s`;
}

// ---------------------------------------------------------------------------
// Pre-flight checks
// ---------------------------------------------------------------------------

/** Problems the server would reject (422) or that make a run pointless,
 *  found before anything is sent. Each entry is a sentence. */
export function requestBlockers(req: OptimizeRunRequest): string[] {
  const out: string[] = [];
  const offline = new Set(req.offline);
  const both = (req.required_wells ?? []).filter((w) => offline.has(w));
  if (both.length > 0) {
    out.push(`${both.join(", ")} ${both.length === 1 ? "is" : "are"} both Offline and Required online. Untick one on the readiness board.`);
  }
  const finite = (v: number) => Number.isFinite(v);
  if (req.kind === "pad") {
    if (req.strategy === "jpco" && (req.nozzles.length === 0 || req.throats.length === 0)) {
      out.push("Select at least one nozzle and one throat.");
    }
    if (req.strategy === "choke") {
      const missing = req.future.filter((f) => !f.nozzle || !f.throat).map((f) => f.name);
      if (missing.length > 0) {
        out.push(`A hold-pumps choke run needs a planned pump for ${missing.join(", ")}. Pick one on the readiness board, or remove the future well.`);
      }
    }
    if (req.lambda_bopd_per_bpd !== null && (!finite(req.lambda_bopd_per_bpd) || req.lambda_bopd_per_bpd < 0 || req.lambda_bopd_per_bpd > 10)) {
      out.push("The water price must be between 0 and 10,000 BOPD per 1,000 BPD.");
    }
    if (req.setpoint_psi !== null && (!finite(req.setpoint_psi) || req.setpoint_psi < 1000 || req.setpoint_psi > 5000)) {
      out.push("The header setpoint must be between 1,000 and 5,000 psi.");
    }
    if (req.pad === "E") {
      if (!finite(req.e_pad_suction_psi) || !finite(req.e_pad_max_header_psi) || req.e_pad_suction_psi >= req.e_pad_max_header_psi) {
        out.push("E-Pad booster suction must be below the header cap.");
      }
      if (!finite(req.e_pad_hz_max) || req.e_pad_hz_max < 30 || req.e_pad_hz_max > 60) {
        out.push("E-Pad max speed must be between 30 and 60 Hz.");
      }
    }
  } else {
    if (req.cfp_pads.length === 0) out.push("Select at least one CFP pad.");
    if (!finite(req.p0_psi) || req.p0_psi < 2300 || req.p0_psi > 2880) {
      out.push("The reference discharge must be between 2,300 and 2,880 psi.");
    }
    if (!finite(req.psi_per_kbpd) || req.psi_per_kbpd < 9 || req.psi_per_kbpd > 17.5) {
      out.push("The machine slope must be between 9 and 17.5 psi per 1,000 BPD.");
    }
    for (const [p, v] of Object.entries(req.cfp_pad_pf_psi ?? {})) {
      if (!finite(v) || v < 1000 || v > 5000) out.push(`${p}-Pad PF must be between 1,000 and 5,000 psi.`);
      else if (finite(req.p0_psi) && v > req.p0_psi) out.push(`${p}-Pad PF cannot exceed the reference discharge (${Math.round(req.p0_psi)} psi).`);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Stale result
// ---------------------------------------------------------------------------

const FIELD_LABELS: Partial<Record<keyof OptimizeRunRequest, string>> = {
  strategy: "strategy",
  method: "solver",
  nozzles: "nozzles",
  throats: "throats",
  lambda_bopd_per_bpd: "water price",
  n_pumps: "pumps online",
  setpoint_psi: "header setpoint",
  p0_psi: "reference discharge",
  psi_per_kbpd: "machine slope",
  c_pad_pf_psi: "C-Pad PF",
  cfp_pad_pf_psi: "pad PF",
  cfp_pads: "CFP pads",
  e_pad_build: "booster build",
  e_pad_suction_psi: "booster suction",
  e_pad_hz_max: "max speed",
  e_pad_max_header_psi: "header cap",
  e_pad_amp_limit_a: "amp limit",
};

function listDelta(label: string, before: string[], after: string[]): string | null {
  const b = new Set(before);
  const a = new Set(after);
  const added = after.filter((x) => !b.has(x));
  const removed = before.filter((x) => !a.has(x));
  if (added.length === 0 && removed.length === 0) return null;
  return `${label} ${[...added.map((x) => `+${x}`), ...removed.map((x) => `-${x}`)].join(" ")}`;
}

/** What changed between the request behind the shown result and the one
 *  the form would send now. Empty = the result still describes the form.
 *  ``before`` is null when the result predates request tracking. */
export function requestChanges(before: OptimizeRunRequest | null, after: OptimizeRunRequest): string[] {
  if (before === null) return [];
  const out: string[] = [];
  const off = listDelta("offline", before.offline ?? [], after.offline ?? []);
  if (off) out.push(off);
  const req = listDelta("required", before.required_wells ?? [], after.required_wells ?? []);
  if (req) out.push(req);
  const fut = listDelta(
    "future wells",
    (before.future ?? []).map((f) => `${f.name}${f.nozzle ? ` ${f.nozzle}${f.throat ?? ""}` : ""}`),
    (after.future ?? []).map((f) => `${f.name}${f.nozzle ? ` ${f.nozzle}${f.throat ?? ""}` : ""}`),
  );
  if (fut) out.push(fut);
  for (const [key, label] of Object.entries(FIELD_LABELS) as [keyof OptimizeRunRequest, string][]) {
    // Knobs the run type ignores never make a result stale.
    if (after.kind === "pad" && ["p0_psi", "psi_per_kbpd", "c_pad_pf_psi", "cfp_pad_pf_psi", "cfp_pads"].includes(key)) continue;
    if (after.kind === "cfp" && !["p0_psi", "psi_per_kbpd", "c_pad_pf_psi", "cfp_pad_pf_psi", "cfp_pads", "nozzles", "throats"].includes(key)) continue;
    if (after.kind === "pad" && after.pad !== "E" && key.startsWith("e_pad_")) continue;
    if (after.kind === "pad" && after.strategy === "choke" && ["nozzles", "throats", "method", "lambda_bopd_per_bpd", "setpoint_psi"].includes(key)) continue;
    const b = JSON.stringify(before[key] ?? null);
    const a = JSON.stringify(after[key] ?? null);
    if (a !== b) out.push(label);
  }
  return out;
}
