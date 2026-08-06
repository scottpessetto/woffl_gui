/**
 * Triage view - keep / SI / BOL decisions vs the field marginal WC.
 * Port of well_sort.py:render_triage_tab. Each well's water cut is compared
 * to the field marginal WC: online wells above the line are shut-in (SI)
 * candidates; shut wells below the line are bring-on-line (BOL) candidates.
 * A poor latest test over a healthy recent history is flagged to verify /
 * BOL-trial rather than acted on.
 */

import { Download } from "lucide-react";
import { useMemo, useState } from "react";

import { useTriage } from "../../api/hooks";
import type { TriageOnlineRow, TriageShutRow } from "../../api/types";
import { Badge, Button, type Column, DataTable, ErrorNote, InfoNote, Metric, Spinner } from "../../components/ui";
import { downloadCsv } from "../../lib/csv";
import { fmtNum } from "../../lib/format";
import { useWellSortStore } from "../../state/wellSort";
import {
  ControlRow,
  DecisionBadge,
  decisionLabel,
  flag,
  LabeledSlider,
  num,
  pct,
  pctSigned,
  Toggle,
  txt,
} from "./shared";

// Decision + Why lead every column set; Why gets room to breathe.
function decisionCols<R extends TriageOnlineRow | TriageShutRow>(): Column<R>[] {
  return [
    {
      key: "decision_code",
      label: "Decision",
      render: (r) => <DecisionBadge code={r.decision_code} />,
    },
    {
      key: "why",
      label: "Why",
      render: (r) => (
        <span className="block max-w-md whitespace-normal text-xs leading-snug text-slate-600">
          {r.why}
        </span>
      ),
    },
  ];
}

const ONLINE_COMPACT: Column<TriageOnlineRow>[] = [
  ...decisionCols<TriageOnlineRow>(),
  txt("well", "Well"),
  txt("pad", "Pad"),
  num("oil", "Oil (BOPD)"),
  pct("total_wc", "Total WC (%)", 1, "Latest-test total water cut. Compared to the marginal line."),
  pctSigned("wc_vs_marginal", "WC - Marg (pp)", 1, "Total WC minus the marginal WC, in percentage points. Positive = above the line (SI lean)."),
  num("gor", "GOR (scf/bbl)"),
  num("days_since_test", "Days since"),
  flag("stale_test", "Stale?"),
  flag("flag_outlier", "Outlier?", "Latest test deviates >25% from the 2-month average on oil or water"),
  flag("pops_pad", "POPS?", "Pad has on-pad water separation"),
];

const ONLINE_DETAIL: Column<TriageOnlineRow>[] = [
  ...ONLINE_COMPACT,
  txt("reservoir", "Reservoir"),
  txt("lift_type", "Lift Type"),
  num("water", "Form Water (BWPD)"),
  num("lift_water", "Lift Water (BWPD)"),
  num("total_water", "Total Water (BWPD)"),
  num("gas", "Gas (MCFD)"),
  num("total_gas", "Total Gas (MCFD)"),
  num("oil_2mo_avg", "Oil 2mo avg"),
  pctSigned("oil_dev", "Oil dev vs 2mo (%)"),
  num("bhp", "BHP (psi)"),
  num("whp", "WHP (psi)"),
  num("prod_xv", "Prod XV", 0, "1=open, 0=closed"),
  txt("xv_time", "XV Time"),
  txt("test_date", "Test Date"),
  flag("allocated", "Alloc."),
  flag("fallback_used", "Fallback"),
];

const SHUT_COMPACT: Column<TriageShutRow>[] = [
  ...decisionCols<TriageShutRow>(),
  txt("well", "Well"),
  txt("pad", "Pad"),
  num("oil", "Last Oil (BOPD)", 0, "Oil from the last test on record"),
  pct("total_wc", "Last Total WC (%)"),
  pctSigned("wc_vs_marginal", "WC - Marg (pp)", 1, "Last-test Total WC minus the marginal WC, in percentage points. Negative = below the line (BOL lean)."),
  pct("near_avg_wc", "90-day Hist WC (%)", 1, "History WC averaged over tests within 90 days of the last test - the 'was it healthy recently' signal behind BOL-trial."),
  num("near_avg_oil", "90-day Avg Oil"),
  txt("shut_in_since", "Shut-In Since"),
  txt("current_code", "Code"),
  txt("current_reason", "Reason"),
  txt("last_test_date", "Last Test"),
];

const SHUT_DETAIL: Column<TriageShutRow>[] = [
  ...SHUT_COMPACT,
  txt("reservoir", "Reservoir"),
  txt("lift_type", "Lift Type"),
  num("water", "Last Form Water (BWPD)"),
  num("gas", "Last Gas (MCFD)"),
  num("lift_water", "Last Lift Water (BWPD)"),
  num("total_water", "Last Total Water (BWPD)"),
  num("near_avg_water", "90-day Avg Water"),
  num("n_tests_near", "# Near Tests"),
  txt("notes", "Notes"),
  num("down_hours", "Down hrs", 1),
  txt("last_online_date", "Last Online"),
  num("prod_xv", "Prod XV"),
];

const LEGEND: Array<[string, string]> = [
  ["keep", "online, WC at or below the marginal - worth its water"],
  ["si", "online, WC above marginal on a representative test - shut-in candidate"],
  ["verify_si", "WC above marginal but the latest test looks anomalous - re-test before SI"],
  ["verify_stale", "no recent representative test - re-test before any call"],
  ["bol", "shut, last WC below marginal - worth bringing on"],
  ["bol_trial", "last test poor but recent history was good - BOL to confirm recovery"],
  ["verify_form_hist", "history below the line on formation-basis WC only - unreliable, re-test"],
  ["verify_no_test", "no usable test on record - test before BOL"],
  ["leave_shut", "WC above marginal, history too - water not worth it"],
  ["pops", "on a POPs pad - judge with the per-pad Marginal WC calc, not the field line"],
];

export function TriageView() {
  const popsPads = useWellSortStore((s) => s.popsPads);
  const forceTrue = useWellSortStore((s) => s.forceTrue);

  const [thresholdPct, setThresholdPct] = useState(2.0);
  const [staleDays, setStaleDays] = useState(60);
  const [onlyAction, setOnlyAction] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [subTab, setSubTab] = useState<"online" | "shut">("online");
  const [showLegend, setShowLegend] = useState(false);

  const query = useTriage(thresholdPct, staleDays, popsPads, forceTrue);
  const data = query.data;

  // only_action: online keeps SI + both Verify kinds; shut keeps BOL, trial,
  // and Verify - hiding healthy keep / leave-shut rows (old tab semantics).
  const online = useMemo(() => {
    if (!data) return [];
    const rows = onlyAction ? data.online.filter((r) => r.rank <= 1) : data.online;
    return [...rows].sort((a, b) => a.rank - b.rank || (b.wc_vs_marginal ?? -Infinity) - (a.wc_vs_marginal ?? -Infinity));
  }, [data, onlyAction]);

  const shut = useMemo(() => {
    if (!data) return [];
    const rows = onlyAction ? data.shut.filter((r) => r.rank <= 2) : data.shut;
    return [...rows].sort((a, b) => a.rank - b.rank || (a.wc_vs_marginal ?? Infinity) - (b.wc_vs_marginal ?? Infinity));
  }, [data, onlyAction]);

  if (query.isError) return <ErrorNote error={query.error} />;
  if (!data) return <Spinner label="Computing triage decisions" />;

  const counts = {
    si: data.online.filter((r) => r.rank === 0).length,
    verify: data.online.filter((r) => r.rank === 1).length,
    bol: data.shut.filter((r) => r.rank === 0).length,
    trial: data.shut.filter((r) => r.rank === 1).length,
  };
  const bufferSkipping = data.raw_worst_well !== null && data.raw_worst_well !== data.well;

  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-xs text-slate-500">
        Each well's water cut is compared to the field marginal WC: online wells above the line
        lean SI; shut wells below the line lean BOL. A poor latest test over a healthy recent
        history is flagged to verify / BOL-trial rather than acted on. POPs-pad settings are
        shared with the Wells view.
      </p>

      <ControlRow>
        <LabeledSlider
          label="Marginal WC buffer"
          value={thresholdPct}
          min={0}
          max={10}
          step={0.5}
          onChange={setThresholdPct}
          format={(v) => `${v.toFixed(1)}% of field water`}
          help="Noise buffer on the marginal WC. Skips the worst-WC wells that make up this % of the field's water before reading the marginal - so one tiny 99%-WC stripper doesn't set the line. Typical 1-3%."
        />
        <LabeledSlider
          label="Stale-test threshold"
          value={staleDays}
          min={14}
          max={180}
          step={1}
          onChange={setStaleDays}
          format={(v) => `${v} d`}
          help="Wells whose latest test is older than this are sent to verify."
        />
        <div className="flex items-end gap-4 pb-1">
          <Toggle label="Only wells needing a decision" checked={onlyAction} onChange={setOnlyAction} help="Hide healthy keep / leave-shut wells" />
          <Toggle label="All columns" checked={showAll} onChange={setShowAll} />
          <button
            type="button"
            onClick={() => setShowLegend((v) => !v)}
            className="text-xs text-blue-600 hover:underline"
          >
            {showLegend ? "Hide legend" : "What do the decisions mean?"}
          </button>
        </div>
      </ControlRow>

      {showLegend && (
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <dl className="grid gap-x-6 gap-y-1.5 text-xs md:grid-cols-2">
            {LEGEND.map(([code, text]) => (
              <div key={code} className="flex items-baseline gap-2">
                <dt className="shrink-0 font-medium text-slate-700">{decisionLabel(code)}</dt>
                <dd className="text-slate-500">{text}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-xs text-slate-400">
            LTSI wells (long-term shut-in / mechanical) are out of scope here - see the Wells view.
          </p>
        </div>
      )}

      {!data.xv_available && (
        <InfoNote>
          Safety-valve (XV) status unavailable - classification falls back to the shut-in log only.
        </InfoNote>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <Metric
          label="Marginal WC (cut line)"
          value={`${fmtNum(data.marginal_wc * 100)}%`}
          sub={`set by ${data.well} (${data.pad})`}
        />
        <Metric label="Buffer" value={`${data.threshold_pct.toFixed(1)}% of field water`} sub="skips small high-WC wells" />
        <Metric
          label="Worst single well (no buffer)"
          value={data.raw_worst_wc !== null ? `${fmtNum(data.raw_worst_wc * 100)}%` : "-"}
          sub={
            data.raw_worst_well
              ? `${data.raw_worst_well} | ${fmtNum(data.raw_worst_water)} BWPD`
              : undefined
          }
        />
      </div>
      <p className="text-xs text-slate-500">
        {bufferSkipping
          ? `With a ${data.threshold_pct.toFixed(1)}% buffer the marginal WC is ${fmtNum(data.marginal_wc * 100)}% (set by ${data.well}) instead of ${fmtNum((data.raw_worst_wc ?? 0) * 100)}% from ${data.raw_worst_well} - a small well making only ${fmtNum(data.raw_worst_water)} BWPD. Lower the buffer toward 0 to follow the worst well.`
          : `The ${data.threshold_pct.toFixed(1)}% buffer isn't skipping any wells yet - the marginal WC (${fmtNum(data.marginal_wc * 100)}%) is still set by the worst well ${data.well}. Raise the buffer to skip small high-WC wells.`}
      </p>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="SI candidates" value={fmtNum(counts.si)} tone={counts.si > 0 ? "poor" : undefined} sub="online, above the line" />
        <Metric label="Verify" value={fmtNum(counts.verify)} tone={counts.verify > 0 ? "fair" : undefined} sub="anomalous or stale test" />
        <Metric label="BOL candidates" value={fmtNum(counts.bol)} tone={counts.bol > 0 ? "good" : undefined} sub="shut, below the line" />
        <Metric label="BOL trials" value={fmtNum(counts.trial)} sub="history says recovery likely" />
      </div>

      <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit">
        {(["online", "shut"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setSubTab(t)}
            className={
              subTab === t
                ? "rounded-md bg-blue-600 px-3 py-1 text-sm font-medium text-white"
                : "rounded-md px-3 py-1 text-sm text-slate-600 hover:bg-slate-100"
            }
          >
            {t === "online" ? `Online - SI review (${data.online.length})` : `Shut - BOL review (${data.shut.length})`}
          </button>
        ))}
      </div>

      {subTab === "online" &&
        (online.length === 0 ? (
          <InfoNote>
            {data.online.length === 0
              ? "No online wells."
              : "No online wells above the marginal WC - nothing to review."}
          </InfoNote>
        ) : (
          <TriageTable
            rows={online}
            columns={showAll ? ONLINE_DETAIL : ONLINE_COMPACT}
            csvName="well_sort_triage_online.csv"
          />
        ))}
      {subTab === "shut" &&
        (shut.length === 0 ? (
          <InfoNote>
            {data.shut.length === 0
              ? "No shut-in (offline) wells."
              : "No shut-in wells below the marginal WC - no BOL candidates."}
          </InfoNote>
        ) : (
          <TriageTable
            rows={shut}
            columns={showAll ? SHUT_DETAIL : SHUT_COMPACT}
            csvName="well_sort_triage_shut.csv"
          />
        ))}
    </div>
  );
}

function TriageTable<R extends TriageOnlineRow | TriageShutRow>({
  rows,
  columns,
  csvName,
}: {
  rows: R[];
  columns: Column<R>[];
  csvName: string;
}) {
  const csvCols = [
    { key: "decision_code", label: "Decision" },
    ...columns.slice(1).map((c) => ({ key: c.key, label: c.label })),
  ];
  return (
    <div className="space-y-2">
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r, i) => `${r.well}-${i}`}
        maxHeight="34rem"
        sortable
      />
      <div className="flex items-center justify-between">
        <Badge tone="neutral">{rows.length} rows</Badge>
        <Button variant="ghost" size="sm" onClick={() => downloadCsv(csvName, csvCols, rows)}>
          <span className="inline-flex items-center gap-1.5">
            <Download className="h-3.5 w-3.5" /> Download CSV
          </span>
        </Button>
      </div>
    </div>
  );
}
