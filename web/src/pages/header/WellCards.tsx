/**
 * Review cards: one card per well, stacked for scrolling - well, test and
 * pressures, gauge / relation / IPR controls and a per-well Save on the left;
 * the pressure trend, the relation fit and the IPR fit on the right. Charts
 * fetch and render only once a card scrolls into view.
 */

import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";

import { useHeaderWell } from "../../api/hooks";
import type { HeaderBoard, HeaderBoardRow, HeaderSaveWell } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import { ACCENT, CRIMSON, GOLD } from "../../charts/theme";
import { Badge, Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useHeaderStore } from "../../state/header";
import { GaugeCell, IprCell, RelationCell } from "./BoardTable";
import { cardIpr, cardSlopes, cardTrend, type IprCurve, type SlopeLine } from "./charts";
import { effective, presLabel, reviewState, savePlan, vogelCurve, wellImpact } from "./model";

const GREEN = "#16a34a";
const NONE_ZOOM = { xAxisIndex: "none" as const, yAxisIndex: "none" as const };

const signed = (v: number | null | undefined, dp = 1) =>
  typeof v === "number" && Number.isFinite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(dp)}` : "-";

/** Becomes true the first time the element is near the viewport, and stays true. */
function useSeen<T extends Element>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (seen || !ref.current) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setSeen(true);
        io.disconnect();
      }
    }, { rootMargin: "600px 0px" });
    io.observe(ref.current);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

/** Does this well need an engineer's look? (Borrowed inputs, bad gauge, no estimate.) */
export function needsReview(row: HeaderBoardRow, eff: ReturnType<typeof effective>): boolean {
  if (row.lift === "JP") return false;
  return eff.gaugeBad || !eff.rel.firm || !eff.ipr.firm || eff.rel.slope === null || !eff.ipr.ipr ||
    reviewState(row).status === "drift";
}

function WellCard({
  row, board, dHeader, onSave, saving, writesOn,
}: {
  row: HeaderBoardRow;
  board: HeaderBoard;
  dHeader: number;
  onSave: (entry: HeaderSaveWell) => void;
  saving: boolean;
  writesOn: boolean;
}) {
  const choice = useHeaderStore((s) => s.choices[row.well]);
  const setChoice = useHeaderStore((s) => s.setChoice);
  const [ref, seen] = useSeen<HTMLDivElement>();
  const detail = useHeaderWell(row.well, board.pads, board.fit_days, seen);
  const eff = effective(row, choice);
  const impact = wellImpact(row, choice, dHeader);
  const plan = savePlan(row, choice);
  const review = reviewState(row);
  const d = detail.data;

  const slopeLines = useMemo<SlopeLine[]>(() => {
    const out: SlopeLine[] = [];
    if (row.measured.slope !== null && eff.gaugeOk) out.push({ label: "measured", value: row.measured.slope, color: ACCENT });
    const cg = eff.rel.group ?? row.corr_group;
    if (cg && row.corr_options[cg]) out.push({ label: `corr ${cg}`, value: row.corr_options[cg].slope, color: GOLD, dashed: true });
    if (row.saved) out.push({ label: "saved", value: row.saved.slope, color: GREEN });
    if (eff.rel.kind === "manual" && eff.rel.slope !== null) out.push({ label: "manual", value: eff.rel.slope, color: CRIMSON });
    return out;
  }, [row, eff.gaugeOk, eff.rel.group, eff.rel.kind, eff.rel.slope]);

  const iprCurves = useMemo<IprCurve[]>(() => {
    const used = eff.ipr.kind;
    const out: IprCurve[] = [];
    const fit = row.ipr_fit ?? row.ipr_fit_any;
    if (fit && eff.gaugeOk) out.push({ label: row.ipr_fit ? "gauge fit" : "gauge fit (flagged)", points: vogelCurve(fit), color: ACCENT, width: used === "fit" ? 2.8 : 1.3 });
    const ig = eff.ipr.group ?? row.ipr_group;
    const opt = ig ? row.ipr_options[ig]?.[eff.gaugeOk ? "gauge" : "nogauge"] : null;
    if (opt) out.push({ label: `well ResP (${row.pres_basis})`, points: vogelCurve(opt), color: GOLD, dashed: true, width: used === "correlation" ? 2.8 : 1.3 });
    if (row.saved_ipr) out.push({ label: "saved", points: vogelCurve(row.saved_ipr), color: GREEN, width: used === "saved" ? 2.8 : 1.3 });
    if (used === "manual" && eff.ipr.ipr) out.push({ label: "manual", points: vogelCurve(eff.ipr.ipr), color: CRIMSON, width: 2.8 });
    return out;
  }, [row, eff.gaugeOk, eff.ipr.group, eff.ipr.kind, eff.ipr.ipr]);

  const now: [number, number] | null =
    row.liquid && (eff.gaugeOk ? row.bhp_now : eff.ipr.ipr?.pwf) ? [row.liquid, (eff.gaugeOk ? row.bhp_now : eff.ipr.ipr?.pwf) as number] : null;
  const trend = useMemo(() => (d ? cardTrend(d) : null), [d]);
  const slopes = useMemo(() => (d ? cardSlopes(d, slopeLines) : null), [d, slopeLines]);
  const ipr = useMemo(() => cardIpr(d, iprCurves, now), [d, iprCurves, now]);
  const fs = row.ipr_fit_stats;

  return (
    <div ref={ref} className={clsx("grid gap-3 rounded-lg border bg-white p-3 xl:grid-cols-[21rem_1fr_1fr_1fr]", eff.online ? "border-slate-200" : "border-slate-200 opacity-60")}>
      <div className="space-y-1.5 text-[12.5px]">
        <div className="flex items-baseline justify-between gap-2">
          <div>
            <span className="text-base font-semibold text-slate-800">{row.well}</span>
            <span className="ml-2 text-xs text-slate-500">{row.pad}-Pad · {row.lift}{row.pump ? ` ${row.pump}` : ""} · {row.reservoir || "?"}</span>
          </div>
          <label className="flex items-center gap-1 text-xs text-slate-600">
            <input type="checkbox" checked={eff.online} onChange={(e) => setChoice(row.well, { online: e.target.checked })} /> online
          </label>
        </div>
        <div className="tabular-nums text-slate-600">
          <span className={row.pres_basis === "saved" ? "text-slate-700" : "text-amber-700"}
            title={row.pres_basis === "saved" ? `Saved ${(row.pres_saved_at ?? "").slice(0, 10)} by ${row.pres_saved_by ?? "?"}` : "No ResP saved in prop_hist: documented default +/-20%. Save an IPR to set this well's ResP."}>
            {presLabel(row)}
          </span>{" · "}
          {fmtNum(row.liquid, 0)} BLPD @ {fmtNum((row.wc ?? 0) * 100, 0)}% WC · {row.test_age_days ?? "?"} d old
          <span className="ml-2">WHP {fmtNum(row.whp_now, 0)} · BHP {row.has_gauge ? fmtNum(row.bhp_now, 0) : "none"}</span>
        </div>
        {(row.down_note || (eff.gaugeBad && row.gauge_note)) && (
          <div className="text-[11px] text-amber-700">{eff.gaugeBad && row.gauge_note ? `Gauge: ${row.gauge_note}` : row.down_note}</div>
        )}
        <div className="flex items-center gap-2"><span className="w-16 text-slate-500">BHP gauge</span><GaugeCell row={row} choice={choice} /></div>
        <div className="flex items-start gap-2">
          <span className="w-16 pt-1 text-slate-500">BHP~WHP</span>
          <div className="space-y-0.5">
            <RelationCell row={row} choice={choice} />
            {row.lift !== "JP" && (
              <div className="text-[11px] text-slate-400">
                measured {row.measured.slope !== null ? fmtNum(row.measured.slope, 2) : "-"} ({row.measured.status.replace("_", " ")},{" "}
                {row.measured.n_fit}/{row.measured.n_days} days)
              </div>
            )}
          </div>
        </div>
        <div className="flex items-start gap-2">
          <span className="w-16 pt-1 text-slate-500">IPR</span>
          <div className="space-y-0.5">
            <IprCell row={row} choice={choice} />
            {row.lift !== "JP" && fs && (
              <div className="text-[11px] text-slate-400">
                gauge fit: {fs.n} tests, {fmtNum(fs.spread, 0)} psi spread{fs.why_not ? ` - ${fs.why_not}` : ""}
              </div>
            )}
          </div>
        </div>
        {impact && (
          <div className="rounded bg-slate-50 px-2 py-1 tabular-nums">
            At {signed(dHeader, 0)} psi: <span className={clsx("font-semibold", impact.dOil < 0 ? "text-red-700" : "text-green-700")}>{signed(impact.dOil)} BOPD</span>
            {Math.abs(impact.hi - impact.lo) > 0.05 && <span className="text-slate-500"> ({signed(impact.lo)} to {signed(impact.hi)})</span>}
            <span className="text-slate-400"> · BHP {signed(impact.dBhp)} psi</span>
          </div>
        )}
        {review.notes.map((n) => <div key={n} className="text-[11px] text-amber-700">Drift: {n}</div>)}
        {row.lift === "JP" && <div className="text-[11px] text-slate-500">Jet pump: solved with its saved pump model in the run. Review its fit in Solver.</div>}
        <div className="flex items-center justify-between gap-2 pt-1">
          <span className="text-[11px] text-slate-500">
            {row.lift === "JP" ? null : review.status === "not_saved" ? (
              <Badge tone="neutral">not saved</Badge>
            ) : (
              <>
                <Badge tone={review.status === "drift" ? "fair" : "good"}>{review.status === "drift" ? "saved - drifted" : "saved"}</Badge>{" "}
                {(row.saved?.at ?? row.saved_ipr?.at ?? "").slice(0, 10)} {(row.saved?.by ?? row.saved_ipr?.by ?? "").split("@")[0]}
              </>
            )}
          </span>
          <Button size="sm" variant="primary" disabled={!plan.entry || !writesOn} busy={saving}
            title={!writesOn ? "Saving is disabled in this environment (read-only)" : (plan.why ?? "Save this well's relation / IPR")}
            onClick={() => plan.entry && onSave(plan.entry)}>
            Save
          </Button>
        </div>
      </div>
      {[
        ["Pressures (BHP left; WHP, header right)", trend],
        ["BHP~WHP: each day's slope", slopes],
        ["IPR: tests, now, curves", ipr],
      ].map(([title, opt]) => (
        <div key={title as string} className="min-w-0">
          <div className="text-[11px] font-medium text-slate-500">{title as string}</div>
          {opt ? (
            <ChartPanel option={opt as never} height={190} zoom={NONE_ZOOM} />
          ) : (
            <div className="flex h-[190px] items-center justify-center text-xs text-slate-400">
              {!seen || detail.isLoading ? "loading..." : detail.error ? "could not load" : "no data"}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function WellCards({
  board, dFor, onSave, saving, writesOn,
}: {
  board: HeaderBoard;
  /** Header change used for each card's impact line, by pad. */
  dFor: (pad: string) => number;
  onSave: (entry: HeaderSaveWell) => void;
  saving: string | null;
  writesOn: boolean;
}) {
  const choices = useHeaderStore((s) => s.choices);
  const [pad, setPad] = useState("all");
  const [lift, setLift] = useState("all");
  const [status, setStatus] = useState<"all" | "review" | "not_saved" | "saved" | "drift">("all");
  const [q, setQ] = useState("");
  const lifts = [...new Set(board.rows.map((r) => r.lift))].sort();
  const rows = board.rows.filter((r) => {
    if (pad !== "all" && r.pad !== pad) return false;
    if (lift === "nonjp" ? r.lift === "JP" : lift !== "all" && r.lift !== lift) return false;
    if (q && !r.well.toLowerCase().includes(q.toLowerCase())) return false;
    if (status === "review" && !needsReview(r, effective(r, choices[r.well]))) return false;
    if ((status === "not_saved" || status === "saved" || status === "drift") && (r.lift === "JP" || reviewState(r).status !== status)) return false;
    return true;
  });
  const chip = (active: boolean) =>
    clsx("rounded-md border px-2 py-0.5", active ? "border-blue-500 bg-blue-50 text-blue-700" : "border-slate-300 bg-white hover:bg-slate-50");

  return (
    <div className="space-y-3">
      <div className="sticky top-0 z-20 flex flex-wrap items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-2 py-1.5 text-xs text-slate-600 shadow-sm">
        {["all", ...board.pads].map((p) => (
          <button key={p} type="button" className={chip(pad === p)} onClick={() => setPad(p)}>{p === "all" ? "All pads" : `${p}-Pad`}</button>
        ))}
        <span className="mx-1 text-slate-300">|</span>
        {["all", "nonjp", ...lifts].map((l) => (
          <button key={l} type="button" className={chip(lift === l)} onClick={() => setLift(l)}>
            {l === "all" ? "All lift" : l === "nonjp" ? "Non-JP" : l}
          </button>
        ))}
        <span className="mx-1 text-slate-300">|</span>
        <select className="h-7 rounded border border-slate-300 bg-white px-1" value={status}
          onChange={(e) => setStatus(e.target.value as typeof status)} title="Review status">
          <option value="all">Any status</option>
          <option value="review">Needs review (borrowed, bad gauge, drifted, no estimate)</option>
          <option value="not_saved">Not saved yet</option>
          <option value="saved">Saved, still matches</option>
          <option value="drift">Saved but drifted</option>
        </select>
        <input className="ml-2 h-7 w-32 rounded border border-slate-300 px-2" placeholder="find well" value={q} onChange={(e) => setQ(e.target.value)} />
        <span className="ml-auto">
          <Badge tone="neutral">{rows.length} wells</Badge>
        </span>
      </div>
      {rows.map((r) => (
        <WellCard key={r.well} row={r} board={board} dHeader={dFor(r.pad)} onSave={onSave} saving={saving === r.well} writesOn={writesOn} />
      ))}
      {!rows.length && <div className="py-6 text-center text-sm text-slate-400">No wells match these filters.</div>}
    </div>
  );
}
