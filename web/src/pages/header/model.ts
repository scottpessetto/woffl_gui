/**
 * Header page helpers - pure, so node tests can import them.
 *
 * `effective` mirrors server/services/header_study.effective: the relation,
 * IPR and gauge state a well runs with, given the engineer's choice. The
 * table shows it, the run and save send the choice, and the server resolves
 * the same numbers from its own board.
 */

import type {
  HeaderBoard,
  HeaderBoardRow,
  HeaderIpr,
  HeaderRunRequest,
  HeaderRunResult,
  HeaderRunRow,
  HeaderSaveWell,
  HeaderWellChoice,
} from "../../api/types";

export interface HeaderForm {
  pads: string[];
  fitDays: number;
  mode: "scenario" | "event";
  deltas: Record<string, number>;
  eventTime: string;
  preHours: number;
  postHours: number;
  gapHours: number;
  eventWell: string;
  eventWellOil: number | null;
}

export const DEFAULT_HEADER_FORM: HeaderForm = {
  pads: [],
  fitDays: 120,
  mode: "scenario",
  deltas: {},
  eventTime: "",
  preHours: 72,
  postHours: 24,
  gapHours: 6,
  eventWell: "",
  eventWellOil: null,
};

export const DEFAULT_DELTA_PSI = 10;
const MIN_DRAWDOWN_PSI = 300;

export interface RelationView {
  kind: "physics" | "saved" | "measured" | "weak_measured" | "correlation" | "manual" | "none";
  slope: number | null;
  lo: number | null;
  hi: number | null;
  /** For a saved value: what it was derived from. */
  source?: string;
  group: string | null;
  firm: boolean;
  reason?: string;
}

export interface IprView {
  kind: "jp" | "saved" | "fit" | "correlation" | "manual" | "none";
  ipr: HeaderIpr | null;
  presLo: number | null;
  presHi: number | null;
  source?: string;
  group: string | null;
  firm: boolean;
  reason: string | null;
}

export interface EffectiveWell {
  gaugeOk: boolean;
  gaugeBad: boolean;
  online: boolean;
  rel: RelationView;
  ipr: IprView;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const clip = (v: number) => Math.min(Math.max(v, 0), 1.2);

/** Why an IPR anchor cannot be used, mirroring the server's 300 psi rule. */
export function iprProblem(ipr: HeaderIpr | null): string | null {
  if (!ipr || !isNum(ipr.qwf) || !isNum(ipr.pwf) || !isNum(ipr.pres)) return "IPR incomplete";
  if (ipr.qwf <= 0 || ipr.pwf <= 0) return "IPR values must be positive";
  if (ipr.pres - ipr.pwf < MIN_DRAWDOWN_PSI) return "ResP within 300 psi of BHP";
  return null;
}

function relationFor(row: HeaderBoardRow, choice: HeaderWellChoice | undefined, gaugeOk: boolean, corrKey: string | null): RelationView {
  const m = row.measured;
  let kind: string = choice?.relation ?? "auto";
  if (kind === "auto") {
    if (row.saved) kind = "saved";
    else if (gaugeOk && m.status === "measured") kind = "measured";
    else if (corrKey) kind = "correlation";
    else kind = "none";
  }
  const none = (reason: string): RelationView => ({ kind: "none", slope: null, lo: null, hi: null, group: null, firm: false, reason });
  if (kind === "saved" && row.saved) {
    const s = clip(row.saved.slope);
    let lo = s;
    let hi = s;
    if (gaugeOk && isNum(m.slope) && row.saved.source === "measured") {
      lo = Math.min(s, isNum(m.q25) ? clip(m.q25) : s);
      hi = Math.max(s, isNum(m.q75) ? clip(m.q75) : s);
    }
    // Any saved relation is firm: an engineer reviewed it.
    return { kind: "saved", slope: s, lo, hi, source: row.saved.source, group: null, firm: true };
  }
  if (kind === "measured" && isNum(m.slope)) {
    if (!gaugeOk) return none("the BHP gauge is marked bad, so its measured relation is not used");
    const s = clip(m.slope);
    return {
      kind: m.status === "measured" ? "measured" : "weak_measured", slope: s,
      lo: Math.min(s, isNum(m.q25) ? clip(m.q25) : s), hi: Math.max(s, isNum(m.q75) ? clip(m.q75) : s),
      group: null, firm: m.status === "measured",
    };
  }
  if (kind === "correlation" && corrKey && row.corr_options[corrKey]) {
    const o = row.corr_options[corrKey];
    return { kind: "correlation", slope: o.slope, lo: o.lo, hi: o.hi, group: corrKey, firm: false };
  }
  if (kind === "manual" && isNum(choice?.slope) && (choice!.slope as number) >= 0 && (choice!.slope as number) <= 1.5) {
    const s = choice!.slope as number;
    return { kind: "manual", slope: s, lo: s, hi: s, group: null, firm: false };
  }
  return none(`no ${kind} relation for this well`);
}

function iprFor(row: HeaderBoardRow, choice: HeaderWellChoice | undefined, gaugeOk: boolean, iprKey: string | null): IprView {
  let kind: string = choice?.ipr ?? "auto";
  if (kind === "auto") {
    // Saved IPR, then the well's own gauge data (a usable fit of its gauged
    // tests), then its own ResP (saved, else default). Flagged fits only when chosen.
    if (row.saved_ipr) kind = "saved";
    else if (gaugeOk && row.ipr_fit) kind = "fit";
    else if (iprKey) kind = "correlation";
    else kind = "none";
  }
  let ipr: HeaderIpr | null = null;
  let lo: number | null = null;
  let hi: number | null = null;
  let source: string | undefined;
  let group: string | null = null;
  let firm = false;
  if (kind === "saved" && row.saved_ipr) {
    const { qwf, pwf, pres } = row.saved_ipr;
    ipr = { qwf, pwf, pres };
    source = row.saved_ipr.source;
    firm = true; // any saved IPR: an engineer reviewed it
  } else if (kind === "fit" && gaugeOk && (row.ipr_fit || row.ipr_fit_any)) {
    // A usable fit is the well's own data (firm); a flagged one is allowed
    // when chosen but stays conditional until saved.
    ipr = row.ipr_fit ?? row.ipr_fit_any;
    firm = Boolean(row.ipr_fit);
  } else if (kind === "correlation" && iprKey) {
    const v = row.ipr_options[iprKey]?.[gaugeOk ? "gauge" : "nogauge"];
    if (v) {
      ipr = { qwf: v.qwf, pwf: v.pwf, pres: v.pres };
      lo = v.pres_lo;
      hi = v.pres_hi;
      group = iprKey;
      firm = row.pres_basis === "saved"; // the well's own saved ResP
    }
  } else if (kind === "manual") {
    ipr = { qwf: choice?.qwf ?? null, pwf: choice?.pwf ?? null, pres: choice?.pres ?? null };
  }
  const reason = kind === "none" ? "no IPR" : iprProblem(ipr);
  if (reason) return { kind: kind === "none" ? "none" : (kind as IprView["kind"]), ipr: null, presLo: null, presHi: null, source, group, firm: false, reason };
  return {
    kind: kind as IprView["kind"], ipr, presLo: lo ?? (ipr!.pres as number), presHi: hi ?? (ipr!.pres as number),
    source, group, firm, reason: null,
  };
}

/** The gauge verdict with no session choice: saved, else the automatic check. */
export function gaugeBadDefault(row: HeaderBoardRow): boolean {
  return row.gauge_bad_default ?? row.gauge_auto_bad;
}

/** The gauge state, relation and IPR a well runs with (mirrors the server). */
export function effective(row: HeaderBoardRow, choice?: HeaderWellChoice): EffectiveWell {
  const gaugeBad = choice?.gauge_bad ?? gaugeBadDefault(row);
  const gaugeOk = row.has_gauge && !gaugeBad;
  const online = choice?.online ?? (row.age_ok && (!row.looks_down || gaugeBad));
  if (row.lift === "JP") {
    return {
      gaugeOk, gaugeBad, online,
      rel: { kind: "physics", slope: null, lo: null, hi: null, group: null, firm: true },
      ipr: { kind: "jp", ipr: null, presLo: null, presHi: null, group: null, firm: true, reason: null },
    };
  }
  const corrKey = choice?.corr_group && row.corr_options[choice.corr_group] ? choice.corr_group : row.corr_group;
  const iprKey = choice?.ipr_group && row.ipr_options[choice.ipr_group] ? choice.ipr_group : row.ipr_group;
  return { gaugeOk, gaugeBad, online, rel: relationFor(row, choice, gaugeOk, corrKey), ipr: iprFor(row, choice, gaugeOk, iprKey) };
}

export function isOnline(row: HeaderBoardRow, choice?: HeaderWellChoice): boolean {
  return effective(row, choice).online;
}

/** Most specific first: "ESP schrader" before "ESP", "R schrader" before "schrader". */
const specificity = (key: string) => key.split(" ").length;

/** Keep the first key for each displayed value (the more specific group). */
function dedupe(keys: string[], value: (k: string) => string): string[] {
  const seen = new Set<string>();
  return keys.filter((k) => {
    const v = value(k);
    if (seen.has(v)) return false;
    seen.add(v);
    return true;
  });
}

/** This well's lift-type correlation groups, most specific first, without
 *  groups that would give the same slope. */
export function corrGroupsFor(row: HeaderBoardRow): string[] {
  const keys = Object.keys(row.corr_options)
    .filter((k) => row.corr_options[k].same_lift)
    .sort((a, b) => specificity(b) - specificity(a) || a.localeCompare(b));
  return dedupe(keys, (k) => row.corr_options[k].slope.toFixed(2));
}

/** Same-reservoir BHP-ratio groups for a gaugeless well, most specific
 *  first, without groups that would give the same BHP. */
export function iprGroupsFor(row: HeaderBoardRow): string[] {
  const keys = Object.keys(row.ipr_options)
    .filter((k) => row.ipr_options[k].same_reservoir && row.ipr_options[k].nogauge)
    .sort((a, b) => specificity(b) - specificity(a) || a.localeCompare(b));
  return dedupe(keys, (k) => String(row.ipr_options[k].nogauge!.pwf));
}

/** Choices worth sending: anything the engineer set that differs from the default. */
export function choicesForRequest(
  board: HeaderBoard | null,
  choices: Record<string, HeaderWellChoice>,
): HeaderWellChoice[] {
  const rows = new Map((board?.rows ?? []).map((r) => [r.well, r]));
  const out: HeaderWellChoice[] = [];
  for (const c of Object.values(choices)) {
    const row = rows.get(c.well);
    const entry: HeaderWellChoice = { well: c.well };
    let used = false;
    const set = <K extends keyof HeaderWellChoice>(k: K, v: HeaderWellChoice[K]) => {
      entry[k] = v;
      used = true;
    };
    if (c.online !== undefined && c.online !== null && (!row || c.online !== row.online_default)) set("online", c.online);
    if (c.gauge_bad !== undefined && c.gauge_bad !== null && (!row || c.gauge_bad !== gaugeBadDefault(row))) set("gauge_bad", c.gauge_bad);
    if (row?.lift === "JP") {
      if (used) out.push(entry);
      continue;
    }
    if (c.relation && c.relation !== "auto") {
      set("relation", c.relation);
      if (c.relation === "manual") set("slope", c.slope ?? null);
    }
    if (c.corr_group && c.corr_group !== row?.corr_group) set("corr_group", c.corr_group);
    if (c.ipr && c.ipr !== "auto") {
      set("ipr", c.ipr);
      if (c.ipr === "manual") {
        set("qwf", c.qwf ?? null);
        set("pwf", c.pwf ?? null);
        set("pres", c.pres ?? null);
      }
    }
    if (c.ipr_group && c.ipr_group !== row?.ipr_group) set("ipr_group", c.ipr_group);
    if (used) out.push(entry);
  }
  return out.sort((a, b) => a.well.localeCompare(b.well));
}

export function buildRunRequest(
  form: HeaderForm,
  board: HeaderBoard | null,
  choices: Record<string, HeaderWellChoice>,
): HeaderRunRequest {
  const pads = [...form.pads].sort();
  const req: HeaderRunRequest = {
    pads,
    fit_days: form.fitDays,
    mode: form.mode,
    delta_by_pad: Object.fromEntries(pads.map((p) => [p, form.deltas[p] ?? DEFAULT_DELTA_PSI])),
    wells: choicesForRequest(board && board.pads.join(",") === pads.join(",") ? board : null, choices),
  };
  if (form.mode === "event") {
    req.event_time = form.eventTime || null;
    req.pre_hours = form.preHours;
    req.post_hours = form.postHours;
    req.gap_hours = form.gapHours;
    req.event_well = form.eventWell || null;
    req.event_well_oil = form.eventWellOil;
  }
  return req;
}

/** Problems to fix before a run can be sent (shown beside the button). */
export function runBlockers(
  form: HeaderForm,
  board: HeaderBoard | null,
  choices: Record<string, HeaderWellChoice>,
): string[] {
  const out: string[] = [];
  if (!form.pads.length) return ["Pick one or more pads."];
  if (form.mode === "scenario") {
    const vals = form.pads.map((p) => form.deltas[p] ?? DEFAULT_DELTA_PSI);
    if (vals.every((v) => v === 0)) out.push("Every pad's header change is 0 psi.");
    if (vals.some((v) => !Number.isFinite(v) || Math.abs(v) > 300)) out.push("Header changes must be within +/-300 psi.");
  } else if (!form.eventTime) {
    out.push("Pick the event time (for example when the well's power fluid came on).");
  }
  for (const row of board?.rows ?? []) {
    const c = choices[row.well];
    if (!c || row.lift === "JP" || !isOnline(row, c)) continue;
    if (c.relation === "manual" && !(isNum(c.slope) && c.slope >= 0 && c.slope <= 1.5)) {
      out.push(`${row.well}: manual slope must be between 0 and 1.5.`);
    }
    if (c.ipr === "manual") {
      const p = iprProblem({ qwf: c.qwf ?? null, pwf: c.pwf ?? null, pres: c.pres ?? null });
      if (p) out.push(`${row.well}: manual IPR - ${p}.`);
    }
  }
  return out;
}

/** What Save would write for a well, or why it would write nothing. */
export function savePlan(
  row: HeaderBoardRow,
  choice?: HeaderWellChoice,
): { entry: HeaderSaveWell | null; why: string | null } {
  const gaugeChange = choice?.gauge_bad !== undefined && choice.gauge_bad !== null && row.has_gauge &&
    (choice.gauge_bad !== gaugeBadDefault(row) || !row.gauge_saved)
    ? choice.gauge_bad
    : null;
  if (row.lift === "JP") {
    if (gaugeChange !== null) return { entry: { well: row.well, gauge_bad: gaugeChange }, why: null };
    return { entry: null, why: "Jet pumps use the pump model; save their IPR in Solver." };
  }
  const eff = effective(row, choice);
  const { rel, ipr } = eff;
  const entry: HeaderSaveWell = { well: row.well };
  if (rel.kind === "measured") entry.relation = "measured";
  else if (rel.kind === "correlation") {
    entry.relation = "correlation";
    entry.corr_group = rel.group;
  } else if (rel.kind === "manual") {
    entry.relation = "manual";
    entry.slope = rel.slope;
  }
  if (ipr.kind === "fit") entry.ipr = "fit";
  else if (ipr.kind === "correlation" && ipr.ipr) {
    entry.ipr = "correlation";
    entry.ipr_group = ipr.group;
  } else if (ipr.kind === "manual" && ipr.ipr) {
    entry.ipr = "manual";
    entry.qwf = ipr.ipr.qwf;
    entry.pwf = ipr.ipr.pwf;
    entry.pres = ipr.ipr.pres;
  }
  if (gaugeChange !== null) entry.gauge_bad = gaugeChange;
  if (!entry.relation && !entry.ipr && entry.gauge_bad === undefined) {
    if (rel.kind === "saved" && ipr.kind === "saved") return { entry: null, why: "Already saved." };
    if (rel.kind === "weak_measured") return { entry: null, why: "Measured relation is weak - pick a correlation or a manual slope." };
    return { entry: null, why: "Nothing new to save." };
  }
  return { entry, why: null };
}

export type StatusTone = "good" | "fair" | "poor";

export function statusView(status: HeaderRunResult["status"]["status"]): { label: string; tone: StatusTone; text: string } {
  if (status === "complete") {
    return { label: "Firm", tone: "good", text: "Every online well rests on its own data or a saved review." };
  }
  if (status === "conditional") {
    return { label: "Conditional", tone: "fair", text: "Some wells are not reviewed yet: a borrowed correlation, a default ResP, or an unsaved fit or manual value. Save them on the Wells tab." };
  }
  return { label: "Incomplete", tone: "poor", text: "Some online wells have no estimate; the total leaves them out." };
}

const signed = (v: number, dp = 1) => `${v > 0 ? "+" : ""}${v.toFixed(dp)}`;

/** One-line answer: the header change and what it costs. */
export function headline(result: HeaderRunResult): string {
  const deltas = result.pads.map((p) => p.d_header).filter(isNum);
  const same = deltas.length > 0 && deltas.every((d) => Math.abs(d - deltas[0]) < 0.05);
  const pads = result.pads.map((p) => p.pad).join(", ");
  const dh = same
    ? `${signed(deltas[0])} psi header on ${pads}`
    : `Header ${result.pads.map((p) => `${p.pad} ${isNum(p.d_header) ? signed(p.d_header) : "?"}`).join(", ")} psi`;
  const oil = result.totals.d_oil ?? 0;
  const liq = result.totals.d_liq ?? 0;
  return `${dh}: ${signed(oil)} BOPD (${signed(liq, 0)} BLPD) across ${result.totals.modeled} wells`;
}

/** "range -35.5 to -92.9 BOPD" (smaller magnitude first), or null. */
export function rangeText(lo: number | null | undefined, hi: number | null | undefined): string | null {
  if (!isNum(lo) || !isNum(hi) || Math.abs(hi - lo) < 0.05) return null;
  return `${signed(lo)} to ${signed(hi)} BOPD`;
}

/** Wells with the largest oil change. */
export function topMovers(rows: HeaderRunRow[], n = 6): HeaderRunRow[] {
  return rows
    .filter((r) => r.outcome === "modeled" && isNum(r.d_oil) && Math.abs(r.d_oil as number) >= 0.05)
    .sort((a, b) => Math.abs(b.d_oil as number) - Math.abs(a.d_oil as number))
    .slice(0, n);
}

/** Oil change at any header change off the response curve (linear between grid points). */
export function readCurve(grid: number[], ys: number[], x: number): number | null {
  if (!grid.length || grid.length !== ys.length || x < grid[0] || x > grid[grid.length - 1]) return null;
  for (let i = 1; i < grid.length; i++) {
    if (x <= grid[i]) {
      const t = (x - grid[i - 1]) / (grid[i] - grid[i - 1]);
      return ys[i - 1] + t * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

/** Default pad-header change for pads without an entry. */
export function deltaFor(form: HeaderForm, pad: string): number {
  const v = form.deltas[pad];
  return isNum(v) ? v : DEFAULT_DELTA_PSI;
}

// ── client-side Vogel (curves and each card's live impact) ──────────────────

const vogelFactor = (r: number) => {
  const x = Math.min(Math.max(r, 0), 1);
  return 1 - 0.2 * x - 0.8 * x * x;
};

/** Total liquid (BLPD) at ``pwf`` on the Vogel curve through the anchor. */
export function vogelRate(ipr: HeaderIpr, pwf: number): number | null {
  if (!isNum(ipr.qwf) || !isNum(ipr.pwf) || !isNum(ipr.pres) || ipr.pres <= 0) return null;
  const qmax = ipr.qwf / vogelFactor(ipr.pwf / ipr.pres);
  return qmax * vogelFactor(pwf / ipr.pres);
}

/** [liquid, BHP] points of an IPR curve from ResP down to zero BHP. */
export function vogelCurve(ipr: HeaderIpr | null, n = 40): [number, number][] {
  if (!ipr || iprProblem(ipr)) return [];
  const pres = ipr.pres as number;
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const p = pres * (1 - i / n);
    const q = vogelRate(ipr, p);
    if (q !== null) out.push([q, p]);
  }
  return out;
}

export interface WellImpact {
  dOil: number;
  lo: number;
  hi: number;
  dBhp: number;
}

/** A non-JP well's oil change for a header change, with its range - the
 *  same chain the server runs (closed-loop slope, Vogel at current BHP). */
export function wellImpact(row: HeaderBoardRow, choice: HeaderWellChoice | undefined, dHeader: number): WellImpact | null {
  if (row.lift === "JP") return null;
  const eff = effective(row, choice);
  const { rel, ipr } = eff;
  if (rel.slope === null || !ipr.ipr) return null;
  const whpHdr = row.whp_hdr.status === "measured" && isNum(row.whp_hdr.slope) ? clip(row.whp_hdr.slope) || 1 : 1;
  const bhp = eff.gaugeOk && isNum(row.bhp_now) ? row.bhp_now : (ipr.ipr.pwf as number);
  const wc = isNum(row.wc) ? row.wc : 0;
  const dOilFor = (slope: number, pres: number) => {
    const a = { ...ipr.ipr!, pres: Math.max(pres, (ipr.ipr!.pwf as number) + MIN_DRAWDOWN_PSI) };
    const q0 = vogelRate(a, bhp);
    const q1 = vogelRate(a, bhp + slope * whpHdr * dHeader);
    return q0 === null || q1 === null ? 0 : (q1 - q0) * (1 - wc);
  };
  const base = dOilFor(rel.slope, ipr.ipr.pres as number);
  const a = dOilFor(rel.lo ?? rel.slope, ipr.presHi ?? (ipr.ipr.pres as number));
  const b = dOilFor(rel.hi ?? rel.slope, ipr.presLo ?? (ipr.ipr.pres as number));
  const [lo, hi] = Math.abs(a) <= Math.abs(b) ? [a, b] : [b, a];
  return { dOil: base, lo, hi, dBhp: rel.slope * whpHdr * dHeader };
}

/** "ResP 1,800 (default)" / "ResP 2,597 (saved)". */
export function presLabel(row: HeaderBoardRow): string {
  if (!isNum(row.pres_well)) return "ResP -";
  return `ResP ${Math.round(row.pres_well).toLocaleString("en-US")} (${row.pres_basis})`;
}

// ── review status (for the cards) ────────────────────────────────────────────

export interface ReviewState {
  status: "not_saved" | "saved" | "drift";
  notes: string[];
}

/** Rate change since the IPR was saved beyond which it is flagged. */
export const IPR_RATE_DRIFT = 0.25;

/** Has this well been reviewed and saved, and does the saved value still
 *  match what its data says today? Only the header page's own review applies
 *  (relation + IPR); jet pumps are reviewed in Solver. */
export function reviewState(row: HeaderBoardRow): ReviewState {
  const notes: string[] = [];
  const s = row.saved;
  const m = row.measured;
  const gaugeOk = row.has_gauge && !gaugeBadDefault(row);
  if (s && gaugeOk && m.status === "measured" && isNum(m.slope)) {
    const band = Math.max(0.15, isNum(m.q75) && isNum(m.q25) ? m.q75 - m.q25 : 0);
    if (Math.abs(s.slope - m.slope) > band) {
      notes.push(`saved slope ${s.slope.toFixed(2)}, now measures ${m.slope.toFixed(2)} - speed change or pump swap?`);
    }
  }
  const si = row.saved_ipr;
  if (si && isNum(si.qwf) && si.qwf > 0 && isNum(row.liquid)) {
    const move = row.liquid / si.qwf - 1;
    if (Math.abs(move) > IPR_RATE_DRIFT) {
      notes.push(`test rate ${Math.round(row.liquid)} BLPD vs ${Math.round(si.qwf)} when the IPR was saved (${move > 0 ? "+" : ""}${Math.round(move * 100)}%)`);
    }
  }
  if (!s && !si) return { status: "not_saved", notes };
  return { status: notes.length ? "drift" : "saved", notes };
}
