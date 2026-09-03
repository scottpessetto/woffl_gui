/**
 * Row A of the Solver workbench: "{well} - {pump}" plus the one-line
 * verdict (MATCH GOOD/FAIR/POOR, WHAT-IF, NO ACTUAL), with the solve
 * status folded in as an inline chip.
 */

import type { PumpInfo, SolveResult, WellTestRow } from "../../api/types";
import { Badge, Card, Spinner } from "../../components/ui";
import { fmtDate, fmtNum } from "../../lib/format";
import type { Grade } from "../../lib/verdict";
import { GRADE_COLORS, matchGrade } from "../../lib/verdict";

function VerdictLine({ grade, text }: { grade: Grade; text: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className="h-2.5 w-2.5 shrink-0 rounded-full"
        style={{ backgroundColor: GRADE_COLORS[grade] }}
      />
      <span className="text-sm text-slate-700">{text}</span>
    </span>
  );
}

export function VerdictBar({
  well,
  nozzle,
  throat,
  contextPump,
  solve,
  compareTest,
  fetching,
}: {
  well: string;
  nozzle: string;
  throat: string;
  contextPump: PumpInfo | null;
  solve: SolveResult | null;
  compareTest: WellTestRow | null;
  fetching: boolean;
}) {
  const pumpDiffers =
    contextPump !== null &&
    contextPump.nozzle_no !== null &&
    contextPump.throat_ratio !== null &&
    (contextPump.nozzle_no !== nozzle || contextPump.throat_ratio !== throat);

  const { grade, errPct } =
    solve && !solve.dewatering
      ? matchGrade(solve.qoil_std, compareTest?.oil ?? null)
      : { grade: "na" as Grade, errPct: null };

  let verdict: React.ReactNode;
  if (pumpDiffers && contextPump) {
    const setPart = contextPump.date_set ? ` (set ${fmtDate(contextPump.date_set)})` : "";
    verdict = (
      <VerdictLine
        grade="na"
        text={
          `WHAT-IF - modeling ${nozzle}${throat}, but the well has ` +
          `${contextPump.nozzle_no}${contextPump.throat_ratio}${setPart}.`
        }
      />
    );
  } else if (solve?.dewatering) {
    verdict = <VerdictLine grade="na" text="DEWATERING - modeled as 100% water (no oil)." />;
  } else if (grade === "na") {
    verdict = <VerdictLine grade="na" text="NO ACTUAL - no comparable test rate for this well." />;
  } else {
    verdict = (
      <span className="inline-flex items-center gap-2">
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: GRADE_COLORS[grade] }}
        />
        <span className="text-sm font-semibold" style={{ color: GRADE_COLORS[grade] }}>
          MATCH {grade.toUpperCase()}
        </span>
        <span className="text-sm text-slate-700">- oil off by {fmtNum(errPct, 0)}%</span>
      </span>
    );
  }

  const against = compareTest ? ` vs the ${fmtDate(compareTest.date)} test` : "";
  const stale = fetching && solve !== null;

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold text-slate-800">
            {well} - {nozzle}
            {throat}
          </h2>
          <div className={stale ? "mt-1 opacity-50" : "mt-1"}>
            {verdict}
            {against && <span className="ml-1 text-sm text-slate-500">{against}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {fetching && solve === null && <Spinner label="Solving" />}
          {stale && <Badge tone="neutral">refreshing</Badge>}
        </div>
      </div>
    </Card>
  );
}
