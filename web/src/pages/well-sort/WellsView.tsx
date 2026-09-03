/**
 * Wells view - the full online / offline / LTSI / 30-day-change picture.
 * Port of the retired Streamlit Well Sort tab. Classification:
 * Online = ProdXV open AND not in vw_shut_in (or in the log but XV shows
 * open - the log lags up to 24 h either way); ProdXV closed forces shut-in.
 */

import { useQueryClient } from "@tanstack/react-query";
import { Download, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";

import { popsQuery, useWellSortEvents, useWellSortTables } from "../../api/hooks";
import { api } from "../../api/client";
import type {
  WellSortEventRow,
  WellSortMode,
  WellSortOnlineRow,
  WellSortShutRow,
} from "../../api/types";
import { Badge, Button, type Column, DataTable, ErrorNote, InfoNote, Metric, Spinner } from "../../components/ui";
import { downloadCsv } from "../../lib/csv";
import { fmtNum } from "../../lib/format";
import { useWellSortStore } from "../../state/wellSort";
import {
  ControlRow,
  flag,
  LabeledSlider,
  num,
  pct,
  pctSigned,
  Toggle,
  txt,
} from "./shared";

type SubTab = "online" | "offline" | "ltsi" | "changes";

// --- column sets (compact default + full detail, ports st.column_config) ---

const ONLINE_COMPACT: Column<WellSortOnlineRow>[] = [
  txt("well", "Well"),
  txt("pad", "Pad"),
  txt("reservoir", "Reservoir"),
  txt("lift_type", "Lift Type"),
  txt("test_date", "Test Date"),
  num("days_since_test", "Days since", 0, "Days between today and the displayed test date"),
  flag("stale_test", "Stale?", "Displayed test older than the stale threshold"),
  num("oil", "Oil (BOPD)"),
  num("water", "Water (BWPD)"),
  num("total_water", "Total Water (BWPD)"),
  pct("total_wc", "Total WC (%)"),
  num("gor", "GOR (scf/bbl)"),
  num("bhp", "BHP (psi)"),
  num("whp", "WHP (psi)"),
  flag("flag_outlier", "Outlier?", "|test - 2-month avg| > 25% on oil or water"),
  num("prod_xv", "Prod XV", 0, "Production safety valve: 1=open, 0=closed"),
  flag("just_restarted", "Just restarted?", "XV shows flowing but vw_shut_in still has it as shut-in"),
  flag("pops_pad", "OnPad Sep?", "Pad has on-pad production separation"),
];

const ONLINE_DETAIL: Column<WellSortOnlineRow>[] = [
  ...ONLINE_COMPACT.slice(0, 7),
  flag("allocated", "Alloc."),
  flag("fallback_used", "Fallback", "No allocated test exists; displayed row is info-only"),
  ...ONLINE_COMPACT.slice(7, 10),
  num("gas", "Gas (MCFD)"),
  num("lift_water", "Lift Water (BWPD)"),
  num("lift_gas", "Lift Gas (MCFD)"),
  num("total_gas", "Total Gas (MCFD)"),
  num("esp_hz", "ESP Hz", 1, "ESP frequency from displayed test (blank for non-ESP wells)"),
  num("esp_amps", "ESP Amps", 0, "ESP motor amps from displayed test"),
  pct("wc", "WC (%)"),
  ...ONLINE_COMPACT.slice(10, 15),
  num("total_gor", "Total GOR"),
  num("oil_2mo_avg", "Oil 2mo avg"),
  num("wat_2mo_avg", "Wat 2mo avg"),
  pctSigned("oil_dev", "Oil dev vs 2mo (%)"),
  pctSigned("wat_dev", "Wat dev vs 2mo (%)"),
  pctSigned("alloc_vs_info_oil_pct", "Info vs Alloc Oil (%)", 0, "Latest info-only oil vs latest allocated. Large values = allocation drift."),
  txt("latest_alloc_date", "Latest Alloc"),
  txt("latest_info_date", "Latest Info"),
  ...ONLINE_COMPACT.slice(15),
  num("pf_xv", "PF XV", 0, "Power-fluid safety valve: 1=open, 0=closed"),
  txt("xv_time", "XV Time", "Timestamp of most recent XV reading"),
];

const SHUT_COMPACT: Column<WellSortShutRow>[] = [
  txt("well", "Well"),
  txt("pad", "Pad"),
  txt("reservoir", "Reservoir"),
  txt("lift_type", "Lift Type"),
  txt("shut_in_since", "Shut-In Since", "Start of current consecutive full-day shut-in streak"),
  txt("current_code", "Code"),
  txt("current_reason", "Reason"),
  num("down_hours", "Down hrs", 1),
  txt("last_online_date", "Last Online"),
  txt("last_test_date", "Last Test", "Absolute-latest test date, any age (not bounded by the tests window)"),
  num("oil", "Oil (BOPD)"),
  num("water", "Water (BWPD)"),
  pct("total_wc", "Total WC (%)"),
  num("near_avg_oil", "Near Avg Oil", 0, "Avg oil rate over tests within 90 days of last test"),
  num("prod_xv", "Prod XV", 0, "1=open, 0=closed"),
];

const SHUT_DETAIL: Column<WellSortShutRow>[] = [
  ...SHUT_COMPACT.slice(0, 8),
  txt("notes", "Notes"),
  ...SHUT_COMPACT.slice(8, 12),
  num("gas", "Gas (MCFD)"),
  pct("wc", "WC (%)"),
  num("gor", "GOR"),
  num("total_gor", "Total GOR"),
  num("lift_water", "Lift Wat (BWPD)"),
  num("lift_gas", "Lift Gas (MCFD)"),
  num("total_water", "Total Wat (BWPD)"),
  num("total_gas", "Total Gas (MCFD)"),
  num("esp_hz", "ESP Hz", 1),
  num("esp_amps", "ESP Amps"),
  ...SHUT_COMPACT.slice(12, 14),
  num("near_avg_water", "Near Avg Wat"),
  num("near_avg_gas", "Near Avg Gas"),
  num("n_tests_near", "# Near Tests", 0, "How many tests in the 90-day near-last window"),
  ...SHUT_COMPACT.slice(14),
  num("pf_xv", "PF XV"),
  txt("xv_time", "XV Time"),
  flag("pops_pad", "OnPad Sep?"),
];

const EVENT_COLUMNS: Column<WellSortEventRow>[] = [
  txt("well", "Well"),
  txt("pad", "Pad"),
  txt("reservoir", "Reservoir"),
  txt("started", "Started"),
  { ...txt("ended", "Ended", "Blank while the event is still ongoing"), render: (r) => (r.ended as string | null) ?? "-" },
  num("days", "Days", 0, "Consecutive days at or above the threshold"),
  num("max_hrs", "Max Hrs", 1, "Peak single-day down hours in this event"),
  num("total_hrs", "Total Hrs", 1, "Sum of down hours across the event"),
  txt("code", "Code"),
  txt("reason", "Reason"),
  txt("notes", "Notes"),
  flag("ongoing", "Ongoing?", "Event extends to the latest log date"),
];

function csvColumns(columns: Column<Record<string, unknown>>[]): { key: string; label: string }[] {
  return columns.map((c) => ({ key: c.key, label: c.label }));
}

export function WellsView() {
  const popsPads = useWellSortStore((s) => s.popsPads);
  const forceTrue = useWellSortStore((s) => s.forceTrue);

  const [mode, setMode] = useState<WellSortMode>("allocated");
  const [staleDays, setStaleDays] = useState(60);
  const [subTab, setSubTab] = useState<SubTab>("online");
  const [allCols, setAllCols] = useState(false);
  const [downHours, setDownHours] = useState(8);
  const [refreshing, setRefreshing] = useState(false);

  const queryClient = useQueryClient();
  const query = useWellSortTables(mode, staleDays, popsPads, forceTrue);
  const events = useWellSortEvents(30, downHours);
  const data = query.data;

  const refresh = async () => {
    setRefreshing(true);
    try {
      await api("/well-sort/refresh", { method: "POST" });
      await queryClient.invalidateQueries({ queryKey: ["well-sort"] });
    } finally {
      setRefreshing(false);
    }
  };

  const benchHref = useMemo(
    () =>
      `/api/well-sort/bench.xlsx?mode=${mode}&stale_days=${staleDays}&${popsQuery(popsPads, forceTrue)}`,
    [mode, staleDays, popsPads, forceTrue],
  );

  if (query.isError) return <ErrorNote error={query.error} />;
  if (!data) return <Spinner label="Loading field status from Databricks" />;

  const counts: Record<SubTab, number> = {
    online: data.online.length,
    offline: data.offline.length,
    ltsi: data.ltsi.length,
    changes: events.data?.rows.length ?? 0,
  };
  const tabLabel: Record<SubTab, string> = {
    online: `Online (${counts.online})`,
    offline: `Offline (${counts.offline})`,
    ltsi: `LTSI (${counts.ltsi})`,
    changes: "30-Day Changes",
  };

  return (
    <div className="space-y-4">
      <ControlRow>
        <Button variant="secondary" size="sm" onClick={refresh} busy={refreshing} title="Clear cache and re-query Databricks">
          <span className="inline-flex items-center gap-1.5">
            <RefreshCw className="h-3.5 w-3.5" /> Refresh data
          </span>
        </Button>
        <label className="block text-xs text-slate-500">
          Display test
          <div className="mt-1 flex rounded-md border border-slate-300 bg-white p-0.5">
            {(["allocated", "any"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={
                  mode === m
                    ? "rounded bg-blue-600 px-2 py-0.5 text-xs font-medium text-white"
                    : "rounded px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-100"
                }
              >
                {m === "allocated" ? "Most recent allocated" : "Most recent (any)"}
              </button>
            ))}
          </div>
        </label>
        <LabeledSlider
          label="Stale-test threshold"
          value={staleDays}
          min={14}
          max={180}
          step={1}
          onChange={setStaleDays}
          format={(v) => `${v} d`}
          help="Flag wells whose most-recent test is older than this."
        />
        <span className="pb-1 text-xs text-slate-400">Tests window: {data.tests_window_days} d</span>
        <div className="ml-auto flex items-end gap-3 pb-0.5">
          <Toggle label="All columns" checked={allCols} onChange={setAllCols} help="Expand from the compact set to every column" />
          <a
            href={benchHref}
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 text-xs font-medium text-slate-700 hover:border-slate-400"
            title="3-sheet MPU_Well_Bench workbook (online / offline / ltsi)"
          >
            <Download className="h-3.5 w-3.5" />
            Bench xlsx
          </a>
        </div>
      </ControlRow>

      {!data.xv_available && (
        <InfoNote>
          Safety-valve (XV) status unavailable - ProdXV/PFXV columns are blank and classification
          falls back to the shut-in log only.
        </InfoNote>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Online Wells" value={fmtNum(counts.online)} />
        <Metric label="Shut-In Wells" value={fmtNum(counts.offline + counts.ltsi)} />
        <Metric label="Outliers flagged" value={fmtNum(data.outliers_flagged)} sub="|test - 2-mo avg| > 25% on oil or water" />
        <Metric label="Just restarted" value={fmtNum(data.just_restarted)} sub="XV open but still in the shut-in log" />
      </div>

      <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit">
        {(Object.keys(tabLabel) as SubTab[]).map((t) => (
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
            {tabLabel[t]}
          </button>
        ))}
      </div>

      {subTab === "online" && (
        <TableSection
          rows={data.online}
          columns={allCols ? ONLINE_DETAIL : ONLINE_COMPACT}
          csvName="well_sort_online.csv"
          csvColumns={csvColumns(ONLINE_DETAIL as Column<Record<string, unknown>>[])}
          empty="No online wells with recent tests."
        />
      )}
      {subTab === "offline" && (
        <TableSection
          rows={data.offline}
          columns={allCols ? SHUT_DETAIL : SHUT_COMPACT}
          csvName="well_sort_offline.csv"
          csvColumns={csvColumns(SHUT_DETAIL as Column<Record<string, unknown>>[])}
          empty="No Offline wells."
        />
      )}
      {subTab === "ltsi" && (
        <TableSection
          rows={data.ltsi}
          columns={allCols ? SHUT_DETAIL : SHUT_COMPACT}
          csvName="well_sort_ltsi.csv"
          csvColumns={csvColumns(SHUT_DETAIL as Column<Record<string, unknown>>[])}
          empty="No LTSI wells."
        />
      )}
      {subTab === "changes" && (
        <div className="space-y-3">
          <ControlRow>
            <LabeledSlider
              label="Min hrs/day to count as down"
              value={downHours}
              min={1}
              max={24}
              step={1}
              onChange={setDownHours}
              format={(v) => `${v} h`}
              help="A day counts as a down day when its total down hours meets this threshold. Lower captures partial-day events (e.g. a 12-hr jet pump changeout)."
            />
            <p className="max-w-xl pb-1 text-xs text-slate-500">
              One row per shut-in event (consecutive days with {">="} {downHours} down hrs)
              overlapping the last 30 days. Ongoing events float to the top.
            </p>
          </ControlRow>
          {events.isError ? (
            <ErrorNote error={events.error} />
          ) : !events.data ? (
            <Spinner label="Loading down events" />
          ) : events.data.rows.length === 0 ? (
            <InfoNote>No shut-in events in the last 30 days.</InfoNote>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Metric label="Events" value={fmtNum(events.data.rows.length)} />
                <Metric label="Still down" value={fmtNum(events.data.rows.filter((r) => r.ongoing).length)} />
                <Metric label="1-day events" value={fmtNum(events.data.rows.filter((r) => r.days === 1).length)} />
                <Metric label="Wells affected" value={fmtNum(new Set(events.data.rows.map((r) => r.well)).size)} />
              </div>
              <TableSection
                rows={events.data.rows}
                columns={EVENT_COLUMNS}
                csvName="well_sort_30day_events.csv"
                csvColumns={csvColumns(EVENT_COLUMNS as Column<Record<string, unknown>>[])}
                empty="No shut-in events in the last 30 days."
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function TableSection<R extends Record<string, unknown>>({
  rows,
  columns,
  csvName,
  csvColumns: csvCols,
  empty,
}: {
  rows: R[];
  columns: Column<R>[];
  csvName: string;
  csvColumns: { key: string; label: string }[];
  empty: string;
}) {
  if (rows.length === 0) return <InfoNote>{empty}</InfoNote>;
  return (
    <div className="space-y-2">
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r, i) => `${String(r.well ?? i)}-${i}`}
        maxHeight="34rem"
        sortable
        pinFirst
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
