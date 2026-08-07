/**
 * Sidebar-conditions comparison: modeled vs the selected test as one compact
 * table (mirror of woffl/gui/tabs/jetpump_solver.py:_render_comparison_block),
 * or the four dewatering metrics + mach caption when modeling as 100% water
 * (mirror of the params.model_as_water block in render_tab).
 */

import clsx from "clsx";
import type { ReactNode } from "react";

import type { SolveResult, WellTestRow } from "../../api/types";
import { Card, InfoNote, Metric, Section, WarnNote } from "../../components/ui";
import { fmtNum, fmtSigned } from "../../lib/format";
import { buildComparisonRows, detectWcWashout, GRADE_COLORS } from "../../lib/verdict";

export function ComparisonCard({
  solve,
  compareTest,
  formWc,
  ppfSurf,
  footer,
}: {
  solve: SolveResult | null;
  compareTest: WellTestRow | null;
  formWc: number;
  ppfSurf: number;
  /** Action row rendered at the card's bottom (the calibrate bar). */
  footer?: ReactNode;
}) {
  if (!solve) {
    return (
      <Section title="Modeled vs Actual">
        <Card>
          <InfoNote>No solve yet - results appear once the solver converges.</InfoNote>
        </Card>
      </Section>
    );
  }

  if (solve.dewatering) {
    // mirrors woffl/gui/tabs/jetpump_solver.py:render_tab (model_as_water block)
    return (
      <Section title="Dewatering Solve">
        <Card className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Metric label="Suction Pressure" value={`${fmtNum(solve.psu)} psig`} sub="Dewatering drawdown" />
            <Metric label="Water Rate" value={`${fmtNum(solve.fwat_bwpd)} BWPD`} sub="Formation water lifted" />
            <Metric label="Power Fluid" value={`${fmtNum(solve.qnz_bwpd)} BWPD`} sub="PF rate to drive the pump" />
            <Metric label="PF Surface Pressure" value={`${fmtNum(ppfSurf)} psig`} sub="Supplied at surface (sidebar)" />
          </div>
          <p className="text-xs text-slate-500">
            Throat: {solve.sonic_status ? "Sonic (choked)" : "Subsonic"} (Mach {fmtNum(solve.mach_te, 2)}).
            Total water handled ~ {fmtNum(solve.fwat_bwpd + solve.qnz_bwpd)} BWPD (formation + power fluid).
            Water is near-incompressible, so a water pump typically won't choke the way a gassy oil well does.
          </p>
        </Card>
      </Section>
    );
  }

  const rows = buildComparisonRows(solve, compareTest);
  const washout = detectWcWashout({
    formWc,
    pfRate: compareTest?.lift_wat ?? null,
    producedFluid: compareTest?.total_fluid ?? null,
    modeledPsu: solve.psu,
    actualBhp: compareTest?.bhp ?? null,
    modeledOil: solve.qoil_std,
    actualOil: compareTest?.oil ?? null,
  });

  return (
    <Section title="Modeled vs Actual">
      <Card>
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 text-slate-600">
              <th className="px-2 py-1.5 text-left font-semibold"></th>
              <th className="px-2 py-1.5 text-right font-semibold">Modeled</th>
              <th className="px-2 py-1.5 text-right font-semibold">Actual</th>
              <th className="px-2 py-1.5 text-right font-semibold">Delta</th>
              <th className="px-2 py-1.5 text-right font-semibold">Off by</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const na = row.grade.grade === "na";
              return (
                <tr key={row.label} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1.5 text-left text-slate-600">
                    {row.label} ({row.unit})
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-700">
                    {fmtNum(row.modeled, row.dp)}
                  </td>
                  <td
                    className={clsx(
                      "px-2 py-1.5 text-right tabular-nums",
                      na ? "text-slate-400" : "text-slate-700",
                    )}
                  >
                    {row.actual !== null ? fmtNum(row.actual, row.dp) : "-"}
                  </td>
                  <td
                    className={clsx(
                      "px-2 py-1.5 text-right tabular-nums",
                      na ? "text-slate-400" : "text-slate-700",
                    )}
                  >
                    {row.delta !== null ? fmtSigned(row.delta, row.dp) : "-"}
                  </td>
                  <td
                    className="px-2 py-1.5 text-right font-semibold tabular-nums"
                    style={{ color: GRADE_COLORS[row.grade.grade] }}
                  >
                    {row.grade.errPct !== null ? `${fmtNum(row.grade.errPct, 1)}%` : "-"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-slate-500">
          Throat-entry Mach {fmtNum(solve.mach_te, 3)} ({solve.sonic_status ? "sonic" : "subsonic"})
        </p>
        {solve.sonic_status && (
          <p
            className="mt-1 text-xs text-amber-700"
            title="jetpump_solver returns psu_minimize(tsu, ken, ate, IPR, suction fluid) directly on the choked branch - power-fluid pressure is not one of its arguments. Only the throat-entry area, the entrance loss, the IPR and the free gas at suction can move it."
          >
            Suction is pinned at the choked-flow floor. Power-fluid pressure, kth, kdi and
            wellhead pressure cannot move this BHP - only throat area, ken, the IPR and the
            free gas at suction can.
          </p>
        )}
        {washout && <WarnNote className="mt-2">{washout.reason}</WarnNote>}
        {footer}
      </Card>
    </Section>
  );
}
