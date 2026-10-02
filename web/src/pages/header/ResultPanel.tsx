/**
 * Header-impact result: the answer first (status, headline, range, largest
 * movers), then the response curve (oil change vs header change), pads,
 * coverage, the per-well chart/table and - for an observed event - suggested
 * POPs and predicted vs measured BHP on gauged wells.
 */

import clsx from "clsx";
import { Download } from "lucide-react";
import { useMemo } from "react";

import type { HeaderRunResult, HeaderRunRow } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import { Badge, Button, Card, InfoNote, Metric, Section, WarnNote } from "../../components/ui";
import { downloadCsv } from "../../lib/csv";
import { fmtNum } from "../../lib/format";
import { impactBars, responseCurve, validationChart } from "./charts";
import { headline, rangeText, statusView, topMovers } from "./model";

const SOURCE_TEXT: Record<string, string> = {
  physics: "pump model", measured: "measured", weak_measured: "measured (weak)", correlation: "correlation",
  manual: "manual", jp: "Solver", fit: "gauge fit", saved: "saved",
};

const signed = (v: number | null | undefined, dp = 1) =>
  typeof v === "number" && Number.isFinite(v) ? `${v > 0 ? "+" : ""}${v.toFixed(dp)}` : "-";

function sourceText(row: HeaderRunRow): string {
  const rel = `${SOURCE_TEXT[row.relation_source ?? ""] ?? row.relation_source ?? "-"}${row.relation_group ? ` ${row.relation_group}` : ""}`;
  const ipr = row.ipr_source === "correlation" ? "well ResP" : (SOURCE_TEXT[row.ipr_source ?? ""] ?? row.ipr_source ?? "-");
  const text = `${rel}${row.relation_saved ? " (saved)" : ""} / ${ipr}${row.ipr_saved ? " (saved)" : ""}`;
  // A jet pump on the relation says so; "pump model" already names the other method.
  return row.jp_method === "empirical" ? `BHP~WHP ${text}` : text;
}

/** "Jet pumps: 5 on the BHP~WHP relation, 2 on the WOFFL model (no relation)", or null. */
function jpMethodText(rows: HeaderRunRow[]): string | null {
  const jp = rows.filter((r) => r.lift === "JP" && r.online);
  if (!jp.length) return null;
  const n = (f: (r: HeaderRunRow) => boolean) => jp.filter(f).length;
  const parts = [
    [n((r) => r.jp_method === "empirical"), "on the BHP~WHP relation"],
    [n((r) => r.jp_method !== "empirical" && !r.jp_fallback), "on the WOFFL model"],
    [n((r) => Boolean(r.jp_fallback)), "on the WOFFL model for want of a relation"],
  ] as const;
  return `Jet pumps: ${parts.filter(([k]) => k > 0).map(([k, t]) => `${k} ${t}`).join(", ")}`;
}

const CSV_COLUMNS = [
  { key: "well", label: "Well" }, { key: "pad", label: "Pad" }, { key: "lift", label: "Lift" },
  { key: "online", label: "Online" }, { key: "outcome", label: "Outcome" }, { key: "reason", label: "Reason" },
  { key: "jp_method", label: "JP method" },
  { key: "gauge_bad", label: "Gauge bad" },
  { key: "d_header", label: "Header change (psi)" }, { key: "whp_hdr", label: "dWHP/dHeader" },
  { key: "slope", label: "dBHP/dWHP" }, { key: "relation_source", label: "Relation source" },
  { key: "relation_group", label: "Relation group" }, { key: "ipr_source", label: "IPR source" },
  { key: "ipr_group", label: "IPR group" }, { key: "d_whp", label: "WHP change (psi)" },
  { key: "d_bhp", label: "BHP change (psi)" }, { key: "d_liq", label: "Liquid change (BLPD)" },
  { key: "d_oil", label: "Oil change (BOPD)" }, { key: "d_oil_lo", label: "Oil change low end" },
  { key: "d_oil_hi", label: "Oil change high end" }, { key: "liquid", label: "Test liquid (BLPD)" },
  { key: "wc", label: "WC" }, { key: "pi", label: "PI at BHP (BLPD/psi)" }, { key: "sonic", label: "Sonic" },
];

/** Which wells rest on borrowed inputs, by kind. */
function SoftNote({ rows, soft }: { rows: HeaderRunRow[]; soft: string[] }) {
  const set = new Set(soft);
  const pick = (f: (r: HeaderRunRow) => boolean) => rows.filter((r) => set.has(r.well) && f(r)).map((r) => r.well);
  const unsavedRel = (r: HeaderRunRow) => !r.relation_saved;
  const unsavedIpr = (r: HeaderRunRow) => !r.ipr_saved;
  const groups: [string, string[]][] = [
    ["Borrowed BHP~WHP correlation (not saved)", pick((r) => r.relation_source === "correlation" && unsavedRel(r))],
    ["Weak measured relation (not saved)", pick((r) => r.relation_source === "weak_measured" && unsavedRel(r))],
    ["Manual relation (not saved)", pick((r) => r.relation_source === "manual" && unsavedRel(r))],
    ["Default ResP - none saved for the well", pick((r) => r.ipr_source === "correlation" && r.pres_basis !== "saved" && unsavedIpr(r))],
    ["Flagged gauge fit (not saved)", pick((r) => r.ipr_source === "fit" && !r.ipr_fit_usable && unsavedIpr(r))],
    ["Manual IPR (not saved)", pick((r) => r.ipr_source === "manual" && unsavedIpr(r))],
  ];
  return (
    <InfoNote>
      <div>
        {soft.length} well{soft.length > 1 ? "s are" : " is"} not reviewed yet - the total is conditional on them. Saving a well on
        the Wells tab makes it firm:
      </div>
      <ul className="mt-1 space-y-0.5 text-xs">
        {groups.filter(([, w]) => w.length).map(([label, w]) => (
          <li key={label}><span className="font-medium">{label} ({w.length}):</span> {w.join(", ")}</li>
        ))}
      </ul>
    </InfoNote>
  );
}

export function ResultPanel({ result }: { result: HeaderRunResult }) {
  const status = statusView(result.status.status);
  const movers = useMemo(() => topMovers(result.rows, 8), [result.rows]);
  const bars = useMemo(() => impactBars(result.rows), [result.rows]);
  const deltas = result.pads.map((p) => p.d_header).filter((d): d is number => typeof d === "number");
  const marker = deltas.length && deltas.every((d) => Math.abs(d - deltas[0]) < 0.05) ? Math.round(deltas[0] * 10) / 10 : null;
  const curve = useMemo(() => responseCurve(result, marker), [result, marker]);
  const vChart = useMemo(() => (result.validation ? validationChart(result.validation.rows) : null), [result.validation]);
  const missing = result.rows.filter((r) => r.online && r.outcome === "missing_inputs");
  const modeled = result.rows.filter((r) => r.outcome === "modeled");
  const total = rangeText(result.totals.d_oil_lo, result.totals.d_oil_hi);
  const jpText = jpMethodText(result.rows);

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Badge tone={status.tone} title={status.text}>{status.label}</Badge>
              <span className="text-xs text-slate-500">
                {result.mode === "event" && result.event ? `Observed event ${result.event.time.replace("T", " ")}` : "Scenario"}
              </span>
            </div>
            <div className="mt-1.5 text-lg font-semibold text-slate-800">{headline(result)}</div>
            {total && (
              <div className="mt-0.5 text-sm text-slate-600" title="Every well at the low (or high) end of its slope and reservoir-IPR range at once - an envelope, not a confidence interval">
                Range {total}
              </div>
            )}
            {result.totals.net_oil !== null && (
              <div className="mt-0.5 text-sm text-slate-600">
                Net of {result.event_well ?? "the event well"}'s {fmtNum(result.totals.event_well_oil, 0)} BOPD:{" "}
                <span className="font-semibold">{signed(result.totals.net_oil)} BOPD</span>
              </div>
            )}
            <div className="mt-0.5 text-xs text-slate-500">{status.text}</div>
            {jpText && (
              <div className="mt-0.5 text-xs text-slate-500" title="Set for the run on the Impact form; single wells can be set otherwise on the Wells tab">
                {jpText}.
              </div>
            )}
          </div>
          <Button
            size="sm"
            onClick={() => downloadCsv("header_impact.csv", CSV_COLUMNS, result.rows as unknown as Record<string, unknown>[])}
          >
            <Download className="h-3.5 w-3.5" /> CSV
          </Button>
        </div>

        {movers.length > 0 && (
          <div className="mt-3">
            <div className="text-[11px] font-medium tracking-wide text-slate-500 uppercase">Largest changes</div>
            <ul className="mt-1 grid max-w-4xl gap-x-10 gap-y-0.5 text-sm sm:grid-cols-2">
              {movers.map((r) => (
                <li key={r.well} className="flex justify-between gap-3 tabular-nums">
                  <span>
                    {r.well} <span className="text-xs text-slate-400">{r.lift}{r.sonic ? ", sonic" : ""}</span>
                  </span>
                  <span className={clsx((r.d_oil ?? 0) < 0 ? "text-red-700" : "text-green-700")}>
                    {signed(r.d_oil)} BOPD <span className="text-xs text-slate-400">({signed(r.d_bhp)} psi BHP)</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <div className="flex flex-wrap gap-3">
        <Metric label="Oil change" value={`${signed(result.totals.d_oil)} BOPD`} tone={(result.totals.d_oil ?? 0) < 0 ? "poor" : "good"}
          sub={total ?? undefined} />
        <Metric label="Liquid change" value={`${signed(result.totals.d_liq, 0)} BLPD`} />
        <Metric label="Wells modeled" value={`${result.totals.modeled} / ${result.totals.online}`} sub="online wells" />
        {result.pads.map((p) => (
          <Metric
            key={p.pad}
            label={`${p.pad}-Pad`}
            value={`${signed(p.d_oil)} BOPD`}
            sub={`${signed(p.d_header)} psi; ${fmtNum(p.oil_per_10psi, 1)} BOPD / 10 psi; ${rangeText(p.d_oil_lo, p.d_oil_hi) ?? ""}`}
            title={Object.entries(p.by_lift).map(([k, v]) => `${k}: ${fmtNum(v, 1)} BOPD`).join(", ")}
          />
        ))}
      </div>

      {curve && (
        <Section title="Oil change vs header change">
          <ChartPanel option={curve} height={320} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
          <p className="mt-1 text-xs text-slate-500">
            Same change applied to every selected pad, on today's inputs and choices. Read any header change off the line; the
            shaded band is the range. Jet pumps on the WOFFL model are solved at -30, -10, +10, +20 and +40 psi and
            interpolated; those on the BHP~WHP relation are exact like other wells.
          </p>
        </Section>
      )}

      {missing.length > 0 && (
        <WarnNote>
          <div className="font-medium">No estimate for {missing.length} online well{missing.length > 1 ? "s" : ""} (left out of the total):</div>
          <ul className="mt-1 list-disc pl-5 text-xs">
            {missing.map((r) => <li key={r.well}>{r.well}: {r.reason}</li>)}
          </ul>
          <div className="mt-1 text-xs">Untick them if they are down, mark a bad gauge, or assign a correlation / manual values on the wells board.</div>
        </WarnNote>
      )}
      {result.status.soft.length > 0 && <SoftNote rows={result.rows} soft={result.status.soft} />}

      {result.event && (
        <Section title="Measured header change">
          <div className="flex flex-wrap gap-8">
            <div>
              <table className="text-sm tabular-nums">
                <thead className="text-left text-xs text-slate-500">
                  <tr><th className="pr-6">Pad</th><th className="pr-6">Before</th><th className="pr-6">After</th><th className="pr-6">Change</th><th>Hours (before / after)</th></tr>
                </thead>
                <tbody>
                  {Object.entries(result.event.pads).map(([pad, m]) => (
                    <tr key={pad}>
                      <td className="pr-6">{pad}</td>
                      <td className="pr-6">{fmtNum(m.pre, 1)}</td>
                      <td className="pr-6">{fmtNum(m.post, 1)}</td>
                      <td className="pr-6 font-medium">{signed(m.delta)} psi</td>
                      <td>{m.n_pre} / {m.n_post}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-1 text-xs text-slate-500">
                Medians of hourly header pressure, {result.event.windows.pre_start.replace("T", " ")} to{" "}
                {result.event.windows.pre_end.replace("T", " ")} vs {result.event.windows.post_start.replace("T", " ")} to{" "}
                {result.event.windows.post_end.replace("T", " ")}.
              </div>
            </div>
            <div className="min-w-[16rem]">
              <div className="text-xs font-medium text-slate-600">Possible causes from the downtime log</div>
              {result.event.pops.length ? (
                <ul className="mt-1 space-y-0.5 text-sm">
                  {result.event.pops.map((p) => (
                    <li key={`${p.well}-${p.kind}`}>
                      <span className="font-medium">{p.well}</span> {p.kind} <span className="text-xs text-slate-500">({p.note})</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="mt-1 text-xs text-slate-400">No well on these pads came on or went down within a day or two.</div>
              )}
              <div className="mt-1 text-[11px] text-slate-400">
                The log is daily; a jet pump's power fluid reaches the header before the log calls the well up.
              </div>
            </div>
          </div>
        </Section>
      )}

      {bars && (
        <Section title="Oil change by well">
          <ChartPanel option={bars} height={360} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Section>
      )}

      {result.validation && result.validation.n > 0 && (
        <Section title="Check against the event: predicted vs measured BHP change">
          <div className="mb-2 flex flex-wrap gap-3">
            <Metric label="Gauged wells" value={`${result.validation.n_used} / ${result.validation.n}`} sub={`${result.validation.n_operational} with a well event in the window`} />
            <Metric label="Mean error" value={`${signed(result.validation.bias)} psi`} sub="predicted - measured" />
            <Metric label="Median |error|" value={`${fmtNum(result.validation.median_abs, 1)} psi`} sub={`${result.validation.within_5} within 5 psi`} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {vChart && <ChartPanel option={vChart} height={320} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />}
            <div className="overflow-auto rounded-md border border-slate-200" style={{ maxHeight: 320 }}>
              <table className="w-full text-[12.5px] tabular-nums">
                <thead className="sticky top-0 bg-slate-50 text-left text-slate-600">
                  <tr>
                    <th className="px-2 py-1">Well</th><th className="px-2 py-1" title="Predicted from the header change">Pred dBHP</th>
                    <th className="px-2 py-1">Meas dBHP</th><th className="px-2 py-1">Meas dWHP</th>
                    <th className="px-2 py-1" title="The relation driven by the measured WHP change">Slope x meas dWHP</th>
                  </tr>
                </thead>
                <tbody>
                  {result.validation.rows.map((v) => (
                    <tr key={v.well} className={clsx("border-t border-slate-100", v.operational && "text-slate-400")}
                      title={v.operational ? "BHP moved far more than the header could cause - a well event in the window; excluded from the statistics" : undefined}>
                      <td className="px-2 py-0.5">{v.well}{v.operational ? " *" : ""}</td>
                      <td className="px-2 py-0.5">{signed(v.d_bhp_pred)}</td>
                      <td className="px-2 py-0.5">{signed(v.d_bhp_meas)}</td>
                      <td className="px-2 py-0.5">{signed(v.d_whp_meas)}</td>
                      <td className="px-2 py-0.5">{signed(v.d_bhp_from_meas_whp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="mt-1 text-xs text-slate-500">
            * BHP moved by more than {fmtNum(result.validation.operational_err_psi, 0)} psi (or 3x the header change) beyond the prediction: a
            restart, trip or speed change inside the window, not a test of the relation. Wells with a bad gauge are not checked.
          </div>
        </Section>
      )}

      <Section title="Per-well detail">
        <div className="overflow-auto rounded-md border border-slate-200" style={{ maxHeight: "28rem" }}>
          <table className="w-full text-[12.5px] tabular-nums">
            <thead className="sticky top-0 bg-slate-50 text-left text-slate-600">
              <tr>
                {["Well", "Lift", "Basis (relation / IPR)", "dWHP", "dBHP/dWHP", "dBHP", "dLiquid", "dOil", "dOil range", "Note"].map((h) => (
                  <th key={h} className="px-2 py-1 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...modeled].sort((a, b) => (a.d_oil ?? 0) - (b.d_oil ?? 0)).concat(result.rows.filter((r) => r.outcome !== "modeled")).map((r) => (
                <tr key={r.well} className={clsx("border-t border-slate-100", r.outcome !== "modeled" && "text-slate-400")}>
                  <td className="px-2 py-0.5 font-medium">{r.well}{r.gauge_bad ? <span className="text-[11px] text-red-600"> gauge bad</span> : null}</td>
                  <td className="px-2 py-0.5">{r.lift}{r.pump ? ` ${r.pump}` : ""}</td>
                  <td className="px-2 py-0.5 whitespace-nowrap">{r.outcome === "modeled" ? sourceText(r) : r.outcome.replace("_", " ")}</td>
                  <td className="px-2 py-0.5">{signed(r.d_whp)}</td>
                  <td className="px-2 py-0.5">{r.relation_source === "physics" ? "model" : fmtNum(r.slope, 2)}</td>
                  <td className="px-2 py-0.5">{signed(r.d_bhp)}</td>
                  <td className="px-2 py-0.5">{signed(r.d_liq)}</td>
                  <td className={clsx("px-2 py-0.5 font-medium", (r.d_oil ?? 0) < 0 && "text-red-700")}>{signed(r.d_oil)}</td>
                  <td className="px-2 py-0.5 text-xs text-slate-500 whitespace-nowrap">
                    {r.outcome === "modeled" && r.d_oil_lo !== r.d_oil_hi ? `${signed(r.d_oil_lo)} to ${signed(r.d_oil_hi)}` : ""}
                  </td>
                  <td className="px-2 py-0.5 text-xs">{r.sonic ? "sonic: header does not reach the formation. " : ""}{r.note ?? r.reason ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">{result.assumptions}</p>
      </Section>
    </div>
  );
}
