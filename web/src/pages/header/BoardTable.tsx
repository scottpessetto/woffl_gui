/**
 * The wells board: every producer with its gauge state, measured BHP~WHP
 * relation, the relation and IPR it runs with, and what Save would write.
 *
 * Any well without a working gauge can be assigned a BHP~WHP correlation
 * (lift type + reservoir, or lift type) and a reservoir IPR correlation
 * (pad + reservoir, or reservoir). "Default" follows the ladder: saved >
 * measured / gauge fit > correlation. Jet pumps always run the pump model.
 */

import clsx from "clsx";
import { useMemo, useState } from "react";

import type { HeaderBoard, HeaderBoardRow, HeaderWellChoice } from "../../api/types";
import { Badge } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useHeaderStore } from "../../state/header";
import { corrGroupsFor, effective, gaugeBadDefault, iprGroupsFor, presLabel, savePlan } from "./model";

/** The option list plus the group already selected, even if it was deduplicated away. */
const withSelected = (keys: string[], selected: string | null) =>
  selected && !keys.includes(selected) ? [...keys, selected] : keys;

/** What "Default" resolves to for this well's relation, e.g. "measured 0.62". */
function relDefaultLabel(row: HeaderBoardRow, choice?: HeaderWellChoice): string {
  const r = effective(row, { ...(choice ?? { well: row.well }), relation: "auto" }).rel;
  if (r.slope === null) return "none";
  const kind = r.kind === "correlation" ? `correlation ${r.group}` : r.kind.replace("_", " ");
  return `${kind} ${fmtNum(r.slope, 2)}`;
}

/** What "Default" resolves to for this well's IPR. */
function iprDefaultLabel(row: HeaderBoardRow, choice?: HeaderWellChoice): string {
  const e = effective(row, { ...(choice ?? { well: row.well }), ipr: "auto" });
  if (!e.ipr.ipr) return e.ipr.reason ?? "none";
  if (e.ipr.kind === "saved") return "saved IPR";
  if (e.ipr.kind === "fit") return `gauge fit (ResP ${fmtNum(e.ipr.ipr.pres, 0)})`;
  return `${presLabel(row)} @ ${e.gaugeOk ? "gauge BHP" : "ratio BHP"}`;
}

const INPUT = "h-7 rounded border border-slate-300 bg-white px-1.5 text-xs tabular-nums";

function statusTone(status: string): "good" | "fair" | "neutral" {
  return status === "measured" ? "good" : status === "weak" ? "fair" : "neutral";
}

function numOrNull(v: string): number | null {
  if (v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

const range = (lo: number | null, hi: number | null, dp: number) =>
  lo !== null && hi !== null && Math.abs(hi - lo) > 10 ** -dp ? ` [${fmtNum(lo, dp)}-${fmtNum(hi, dp)}]` : "";

export function GaugeCell({ row, choice }: { row: HeaderBoardRow; choice?: HeaderWellChoice }) {
  const setChoice = useHeaderStore((s) => s.setChoice);
  if (!row.has_gauge) return <span className="text-xs text-slate-400">no BHP gauge</span>;
  const eff = effective(row, choice);
  const unsaved = choice?.gauge_bad !== undefined && choice.gauge_bad !== null && choice.gauge_bad !== gaugeBadDefault(row);
  const saved = row.gauge_saved;
  const note = unsaved
    ? "(not saved yet)"
    : saved
      ? `(saved ${(saved.at ?? "").slice(0, 10)}${saved.by ? ` by ${saved.by.split("@")[0]}` : ""})`
      : row.gauge_auto_bad
        ? "(auto-flagged)"
        : "";
  return (
    <label
      className="flex cursor-pointer items-center gap-1.5 text-xs"
      title={
        (row.gauge_note ? `Automatic check: ${row.gauge_note}. ` : "") +
        "Tick when the BHP tag is not a real flowing BHP (dead, flat, reading PF). The well then uses the correlations instead of its gauge."
      }
    >
      <input type="checkbox" checked={eff.gaugeBad} onChange={(e) => setChoice(row.well, { gauge_bad: e.target.checked })} />
      <span className={clsx(eff.gaugeBad ? "font-medium text-red-700" : "text-slate-600")}>Bad gauge</span>
      {note && <span className="text-slate-400">{note}</span>}
    </label>
  );
}

export function RelationCell({ row, choice }: { row: HeaderBoardRow; choice?: HeaderWellChoice }) {
  const setChoice = useHeaderStore((s) => s.setChoice);
  if (row.lift === "JP") {
    return <span className="text-xs text-slate-500" title="Jet pumps are solved with the WOFFL model at both wellhead pressures">pump model</span>;
  }
  const eff = effective(row, choice);
  const view = eff.rel;
  const value =
    choice?.relation === "correlation" ? `corr:${view.group ?? row.corr_group ?? ""}` : (choice?.relation ?? "auto");
  return (
    <div className="flex items-center gap-1">
      <select
        className={clsx(INPUT, "max-w-[11rem]")}
        value={value}
        onChange={(e) => {
          const v = e.target.value;
          if (v.startsWith("corr:")) setChoice(row.well, { relation: "correlation", corr_group: v.slice(5) });
          else setChoice(row.well, { relation: v as HeaderWellChoice["relation"] });
        }}
      >
        <option value="auto">Default: {relDefaultLabel(row, choice)}</option>
        {row.saved && <option value="saved">Saved {fmtNum(row.saved.slope, 2)}</option>}
        {row.measured.slope !== null && eff.gaugeOk && (
          <option value="measured">Measured {fmtNum(row.measured.slope, 2)}{row.measured.status !== "measured" ? " (weak)" : ""}</option>
        )}
        {withSelected(corrGroupsFor(row), view.kind === "correlation" ? view.group : null).map((k) => (
          <option key={k} value={`corr:${k}`}>Correlation {k} {fmtNum(row.corr_options[k].slope, 2)}</option>
        ))}
        <option value="manual">Manual</option>
      </select>
      {choice?.relation === "manual" && (
        <input
          className={clsx(INPUT, "w-16")}
          type="number"
          step={0.05}
          min={0}
          max={1.5}
          value={choice?.slope ?? ""}
          onChange={(e) => setChoice(row.well, { slope: numOrNull(e.target.value) })}
          title="Closed-loop dBHP/dWHP (psi/psi)"
        />
      )}
      <span
        className={clsx("text-xs whitespace-nowrap tabular-nums", view.kind === "none" ? "text-red-600" : view.firm ? "text-slate-700" : "text-amber-700")}
        title={view.reason ?? `Runs with the ${view.kind.replace("_", " ")} relation${view.group ? ` (${view.group})` : ""}${view.source ? `, saved from ${view.source}` : ""}`}
      >
        {view.slope !== null ? `${fmtNum(view.slope, 2)}${range(view.lo, view.hi, 2)}` : "none"}
      </span>
    </div>
  );
}

export function IprCell({ row, choice }: { row: HeaderBoardRow; choice?: HeaderWellChoice }) {
  const setChoice = useHeaderStore((s) => s.setChoice);
  if (row.lift === "JP") {
    return <span className="text-xs text-slate-500" title="Jet pumps use the IPR saved with their pump model (Solver)">Solver IPR</span>;
  }
  const eff = effective(row, choice);
  const view = eff.ipr;
  const value = choice?.ipr === "correlation" ? `ipr:${view.group ?? row.ipr_group ?? ""}` : (choice?.ipr ?? "auto");
  const fitWhy = row.ipr_fit_stats && !row.ipr_fit_stats.usable ? row.ipr_fit_stats.why_not : null;
  return (
    <div className="flex items-center gap-1">
      <select
        className={clsx(INPUT, "max-w-[10rem]")}
        value={value}
        onChange={(e) => {
          const v = e.target.value;
          if (v.startsWith("ipr:")) {
            setChoice(row.well, { ipr: "correlation", ipr_group: v.slice(4) });
            return;
          }
          const ipr = v as HeaderWellChoice["ipr"];
          const seed = ipr === "manual" && !choice?.pres && view.ipr ? { qwf: view.ipr.qwf, pwf: view.ipr.pwf, pres: view.ipr.pres } : {};
          setChoice(row.well, { ipr, ...seed });
        }}
      >
        <option value="auto">Default: {iprDefaultLabel(row, choice)}</option>
        {row.saved_ipr && <option value="saved">Saved IPR (ResP {fmtNum(row.saved_ipr.pres, 0)})</option>}
        {(row.ipr_fit || row.ipr_fit_any) && eff.gaugeOk && (
          <option value="fit">Gauge fit{row.ipr_fit ? "" : " (flagged)"}</option>
        )}
        {eff.gaugeOk ? (
          row.ipr_group && row.ipr_options[row.ipr_group]?.gauge && (
            <option value={`ipr:${row.ipr_group}`}>{presLabel(row)} @ gauge BHP</option>
          )
        ) : (
          withSelected(iprGroupsFor(row), view.kind === "correlation" ? view.group : null).filter((k) => row.ipr_options[k]?.nogauge).map((k) => {
            const o = row.ipr_options[k].nogauge!;
            return (
              <option key={k} value={`ipr:${k}`}>
                {presLabel(row)}, BHP = {(o.pwf / o.pres).toFixed(2)} x ResP ({k})
              </option>
            );
          })
        )}
        <option value="manual">Manual</option>
      </select>
      {choice?.ipr === "manual" ? (
        <>
          {(["qwf", "pwf", "pres"] as const).map((k) => (
            <input
              key={k}
              className={clsx(INPUT, "w-16")}
              type="number"
              placeholder={k === "qwf" ? "BLPD" : k === "pwf" ? "pwf" : "ResP"}
              title={k === "qwf" ? "Anchor liquid rate (BLPD)" : k === "pwf" ? "Anchor flowing BHP (psig)" : "Reservoir pressure (psig)"}
              value={choice?.[k] ?? ""}
              onChange={(e) => setChoice(row.well, { [k]: numOrNull(e.target.value) })}
            />
          ))}
        </>
      ) : (
        <span
          className={clsx("text-xs whitespace-nowrap tabular-nums", view.reason ? "text-red-600" : view.firm ? "text-slate-700" : "text-amber-700")}
          title={
            view.reason ??
            `${view.kind === "correlation" ? `Well ResP (${row.pres_basis}${row.pres_basis === "default" ? " +/-20%" : ""}), ${eff.gaugeOk ? "gauge BHP" : `BHP from ${view.group} BHP/ResP`}` : view.kind === "fit" ? "Gauge-test fit (pseudo-ResP)" : "Saved IPR"}: ` +
              `${fmtNum(view.ipr?.qwf, 0)} BLPD at ${fmtNum(view.ipr?.pwf, 0)} psi, ResP ${fmtNum(view.ipr?.pres, 0)}` +
              (fitWhy ? `. Gauge fit not used: ${fitWhy}` : "")
          }
        >
          {view.reason ? view.reason : `ResP ${fmtNum(view.ipr?.pres, 0)}${range(view.presLo, view.presHi, 0)} / ${fmtNum(view.ipr?.pwf, 0)}`}
        </span>
      )}
    </div>
  );
}

export function BoardTable({
  board,
  selected,
  onToggleSave,
}: {
  board: HeaderBoard;
  selected: Set<string>;
  onToggleSave: (well: string, on: boolean) => void;
}) {
  const choices = useHeaderStore((s) => s.choices);
  const setChoice = useHeaderStore((s) => s.setChoice);
  const [padFilter, setPadFilter] = useState<string>("all");
  const rows = useMemo(
    () => board.rows.filter((r) => padFilter === "all" || r.pad === padFilter),
    [board.rows, padFilter],
  );

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
        <span>Show</span>
        {["all", ...board.pads].map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setPadFilter(p)}
            className={clsx(
              "rounded-md border px-2 py-0.5",
              padFilter === p ? "border-blue-500 bg-blue-50 text-blue-700" : "border-slate-300 bg-white hover:bg-slate-50",
            )}
          >
            {p === "all" ? "All pads" : `${p}-Pad`}
          </button>
        ))}
        <span className="ml-2 text-slate-400">
          Amber = borrowed (correlation) or manual; red = no estimate; [low-high] = the range used. Hover for the basis.
        </span>
      </div>
      <div className="overflow-auto rounded-md border border-slate-200" style={{ maxHeight: "34rem" }}>
        <table className="w-full border-collapse text-[12.5px]">
          <thead className="sticky top-0 z-10 bg-slate-50 text-left text-slate-600">
            <tr>
              {[
                ["Well", ""], ["Lift", "Lift type and reservoir"], ["Online", "Include in the run (defaults off for stale tests or wells that look shut in)"],
                ["Test", "Latest test: liquid (BLPD), water cut, days old"], ["WHP / BHP now", "Historian medians, last 72 h (psig)"],
                ["Gauge", "BHP gauge usable? Automatic check, which you can override"],
                ["Measured BHP~WHP", "Closed-loop within-day slope: median [IQR], fit days / days the WHP moved"],
                ["Relation used", "dBHP/dWHP the run uses [range]"], ["IPR used", "Vogel anchor the run uses: ResP [range] / flowing BHP"],
                ["Saved", "Latest saved header relation"], ["Save", "Include in Save selected"],
              ].map(([label, help]) => (
                <th key={label} title={help} className="border-b border-slate-200 px-2 py-1.5 font-semibold whitespace-nowrap">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const choice = choices[row.well];
              const eff = effective(row, choice);
              const plan = savePlan(row, choice);
              return (
                <tr key={row.well} className={clsx("border-b border-slate-100", !eff.online && "text-slate-400", i % 2 && "bg-slate-50/40")}>
                  <td className="px-2 py-1 whitespace-nowrap">
                    <div className="font-medium">{row.well}</div>
                    {row.down_note && <div className="text-[11px] text-amber-700">{row.down_note}</div>}
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap">
                    {row.lift}{row.pump ? ` ${row.pump}` : ""}
                    <div className="text-[11px] text-slate-400">{row.reservoir || "?"}</div>
                  </td>
                  <td className="px-2 py-1 text-center">
                    <input
                      type="checkbox"
                      checked={eff.online}
                      onChange={(e) => setChoice(row.well, { online: e.target.checked })}
                    />
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap tabular-nums">
                    {fmtNum(row.liquid, 0)} <span className="text-slate-400">@ {fmtNum((row.wc ?? 0) * 100, 0)}%</span>
                    <div className="text-[11px] text-slate-400">{row.test_age_days ?? "?"} d old</div>
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap tabular-nums">
                    {fmtNum(row.whp_now, 0)} / {row.has_gauge ? fmtNum(row.bhp_now, 0) : "-"}
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap"><GaugeCell row={row} choice={choice} /></td>
                  <td className="px-2 py-1 whitespace-nowrap tabular-nums">
                    {row.measured.slope !== null && eff.gaugeOk ? (
                      <>
                        {fmtNum(row.measured.slope, 2)}
                        <span className="text-slate-400"> [{fmtNum(row.measured.q25, 2)}-{fmtNum(row.measured.q75, 2)}]</span>{" "}
                      </>
                    ) : null}
                    <Badge tone={eff.gaugeOk ? statusTone(row.measured.status) : "neutral"} title={`r2 ${fmtNum(row.measured.r2, 2)}, ${row.measured.n_fit} fit days of ${row.measured.n_days}`}>
                      {!eff.gaugeOk ? "no usable gauge" : row.measured.status === "no_data" ? "no data" : `${row.measured.status} ${row.measured.n_fit}/${row.measured.n_days}`}
                    </Badge>
                  </td>
                  <td className="px-2 py-1"><RelationCell row={row} choice={choice} /></td>
                  <td className="px-2 py-1"><IprCell row={row} choice={choice} /></td>
                  <td className="px-2 py-1 whitespace-nowrap text-[11px]">
                    {row.saved ? (
                      <span title={`by ${row.saved.by ?? "?"} at ${row.saved.at ?? "?"}`}>
                        {fmtNum(row.saved.slope, 2)} {row.saved.source}
                        <div className="text-slate-400">{(row.saved.at ?? "").slice(0, 10)}</div>
                      </span>
                    ) : (
                      <span className="text-slate-400">-</span>
                    )}
                  </td>
                  <td className="px-2 py-1 text-center">
                    <input
                      type="checkbox"
                      disabled={!plan.entry}
                      title={plan.why ?? "Save this relation / IPR"}
                      checked={Boolean(plan.entry) && selected.has(row.well)}
                      onChange={(e) => onToggleSave(row.well, e.target.checked)}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
