/**
 * Well Database - read-only viewer of the mech + reservoir characteristics
 * table, the aging jet pump report, and the per-well prop_hist audit trail.
 */

import { useMemo, useState } from "react";

import { useAgingPumps, usePropHistory, useWellDatabase } from "../api/hooks";
import { Badge, Button, type Column, DataTable, ErrorNote, InfoNote, Metric, Section, Spinner, WarnNote } from "../components/ui";
import { downloadCsv } from "../lib/csv";
import { fmtDate, fmtNum } from "../lib/format";

type DbRow = Record<string, unknown>;

const INPUT_CLS =
  "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 " +
  "outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";
const CHECKBOX_CLS = "h-4 w-4 rounded border-slate-300 accent-blue-600";

/** String cell from a loosely typed row; "-" for null/undefined. */
function str(v: unknown): string {
  return v === null || v === undefined ? "-" : String(v);
}

/** Finite number or null from a loosely typed row. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

const DB_COLUMNS: Column<DbRow>[] = [
  { key: "well", label: "Well" },
  {
    key: "is_sch",
    label: "Field",
    align: "center",
    render: (r) =>
      r.is_sch === true ? <Badge tone="info">Schrader</Badge> : <Badge>Kuparuk</Badge>,
  },
  {
    key: "tvd_estimated",
    label: "TVD Est.",
    align: "center",
    render: (r) => (r.tvd_estimated === true ? <Badge tone="fair">Est.</Badge> : "-"),
  },
  { key: "tubing_od", label: "Tubing OD", align: "right", render: (r) => fmtNum(num(r.tubing_od), 3) },
  { key: "tubing_thickness", label: "Tubing Thick", align: "right", render: (r) => fmtNum(num(r.tubing_thickness), 3) },
  { key: "casing_out_dia", label: "Casing OD", align: "right", render: (r) => fmtNum(num(r.casing_out_dia), 3) },
  { key: "casing_inn_dia", label: "Casing ID", align: "right", render: (r) => fmtNum(num(r.casing_inn_dia), 3) },
  { key: "jp_md", label: "JP MD", align: "right", render: (r) => fmtNum(num(r.jp_md)) },
  { key: "jp_tvd", label: "JP TVD", align: "right", render: (r) => fmtNum(num(r.jp_tvd), 1) },
  { key: "res_pres", label: "Res P (psi)", align: "right", render: (r) => fmtNum(num(r.res_pres)) },
  { key: "form_temp", label: "Temp (degF)", align: "right", render: (r) => fmtNum(num(r.form_temp)) },
  { key: "oil_api", label: "API", align: "right", render: (r) => fmtNum(num(r.oil_api), 1) },
  { key: "gas_sg", label: "Gas SG", align: "right", render: (r) => fmtNum(num(r.gas_sg), 3) },
  { key: "wat_sg", label: "Wat SG", align: "right", render: (r) => fmtNum(num(r.wat_sg), 3) },
  { key: "bubble_point", label: "Pb (psi)", align: "right", render: (r) => fmtNum(num(r.bubble_point)) },
];

/** Aging-pump columns. `flagDays` drives the "Over" flag, mirroring the
 *  Streamlit page's threshold column - it marks rows, it never hides them. */
const agingColumns = (flagDays: number): Column<DbRow>[] => [
  { key: "well", label: "Well" },
  { key: "pump", label: "Pump" },
  { key: "date_set", label: "Date Set", render: (r) => fmtDate(typeof r.date_set === "string" ? r.date_set : null) },
  { key: "days_in_hole", label: "Days In Hole", align: "right", render: (r) => fmtNum(num(r.days_in_hole)) },
  { key: "installs", label: "Installs", align: "right", render: (r) => fmtNum(num(r.installs)) },
  { key: "last_test", label: "Last Test", render: (r) => fmtDate(typeof r.last_test === "string" ? r.last_test : null) },
  {
    key: "last_allocated",
    label: "Last Alloc",
    render: (r) => fmtDate(typeof r.last_allocated === "string" ? r.last_allocated : null),
  },
  {
    key: "online",
    label: "Online",
    align: "center",
    render: (r) => (r.online === true ? <Badge tone="good">Yes</Badge> : <Badge>No</Badge>),
  },
  {
    key: "over",
    label: `Over ${fmtNum(flagDays)} d`,
    align: "center",
    render: (r) => ((num(r.days_in_hole) ?? 0) >= flagDays ? <Badge tone="fair">Yes</Badge> : "-"),
  },
];

/**
 * Save-history stamp for display. prop_hist stores UTC; the server renders
 * `entry_datetime_ak` ("2026-08-03 11:22 AKDT") with the zone labelled, so the
 * engineer never has to guess what "19:22" means. Falls back to the raw UTC
 * stamp, labelled, for a payload that predates the field.
 */
function when(r: DbRow): string {
  const ak = r.entry_datetime_ak;
  if (typeof ak === "string" && ak.length > 0) return ak;
  const v = r.entry_datetime;
  return typeof v === "string" ? `${v.slice(0, 16)} UTC` : "-";
}

/** prop_value cells: integers plain, fractional values to 3 dp. */
function propValue(v: unknown): string {
  const n = num(v);
  if (n !== null) return fmtNum(n, Number.isInteger(n) ? 0 : 3);
  return str(v);
}

const CURRENT_COLUMNS: Column<DbRow>[] = [
  { key: "prop_name", label: "Prop", render: (r) => str(r.prop_name) },
  { key: "prop_value", label: "Value", align: "right", render: (r) => propValue(r.prop_value) },
  { key: "units", label: "Units", render: (r) => str(r.units) },
  { key: "category", label: "Category", render: (r) => str(r.category) },
  { key: "entry_user", label: "By", render: (r) => str(r.entry_user) },
  { key: "entry_datetime", label: "When (AK)", render: (r) => when(r) },
];

const HISTORY_COLUMNS: Column<DbRow>[] = [
  { key: "entry_datetime", label: "When (AK)", render: (r) => when(r) },
  { key: "prop_name", label: "Prop", render: (r) => str(r.prop_name) },
  { key: "prop_value", label: "Value", align: "right", render: (r) => propValue(r.prop_value) },
  { key: "entry_user", label: "By", render: (r) => str(r.entry_user) },
  { key: "comment_text", label: "Comment", render: (r) => str(r.comment_text) },
];

export default function WellDatabasePage() {
  const [filter, setFilter] = useState("");
  const [estOnly, setEstOnly] = useState(false);
  const [knownOnly, setKnownOnly] = useState(true);
  const [onlineOnly, setOnlineOnly] = useState(true);
  const [onlineDays, setOnlineDays] = useState(60);
  const [flagDays, setFlagDays] = useState(365);
  const [propWell, setPropWell] = useState<string | null>(null);

  const dbQuery = useWellDatabase();
  // min_days=0: the threshold FLAGS rows here (the Streamlit page's "Over"
  // column) instead of hiding them - a 334-day pump on an online well is the
  // row an engineer is looking for, and truncating the list server-side made
  // it vanish with no hint that a filter had eaten it.
  const agingQuery = useAgingPumps(knownOnly, onlineOnly, onlineDays, 0);
  const propQuery = usePropHistory(propWell);

  const rows = useMemo(() => dbQuery.data?.rows ?? [], [dbQuery.data]);

  const filtered = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return rows.filter((r) => {
      if (estOnly && r.tvd_estimated !== true) return false;
      return needle.length === 0 || str(r.well).toLowerCase().includes(needle);
    });
  }, [rows, filter, estOnly]);

  const wellNames = useMemo(
    () => rows.map((r) => str(r.well)).filter((w) => w !== "-").sort(),
    [rows],
  );

  const agingCols = useMemo(() => agingColumns(flagDays), [flagDays]);

  const agingStats = useMemo(() => {
    const days = (agingQuery.data?.rows ?? [])
      .map((r) => num(r.days_in_hole))
      .filter((d): d is number => d !== null)
      .sort((a, b) => a - b);
    if (days.length === 0) return null;
    const mid = Math.floor(days.length / 2);
    return {
      tracked: days.length,
      over: days.filter((d) => d >= flagDays).length,
      oldest: days[days.length - 1],
      median: days.length % 2 === 1 ? days[mid] : (days[mid - 1] + days[mid]) / 2,
    };
  }, [agingQuery.data, flagDays]);

  const historySorted = useMemo(() => {
    const history = propQuery.data?.history ?? [];
    // Sort on the raw UTC stamp (the ordering key), never the display string.
    return [...history].sort((a, b) => str(b.entry_datetime).localeCompare(str(a.entry_datetime)));
  }, [propQuery.data]);

  if (dbQuery.isError) {
    return <ErrorNote error={dbQuery.error} />;
  }
  if (!dbQuery.data) {
    return <Spinner label="Loading well database" />;
  }

  const data = dbQuery.data;
  const schraderCount = rows.filter((r) => r.is_sch === true).length;
  const estimatedCount = rows.filter((r) => r.tvd_estimated === true).length;

  return (
    <div className="space-y-6">
      {data.source === "csv_fallback" && (
        <WarnNote>Offline chars (jp_chars.csv) - Databricks unavailable</WarnNote>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Wells" value={fmtNum(rows.length)} />
        <Metric label="Schrader" value={fmtNum(schraderCount)} />
        <Metric label="Kuparuk" value={fmtNum(rows.length - schraderCount)} />
        <Metric label="Estimated TVD" value={fmtNum(estimatedCount)} sub="no survey" />
      </div>

      <Section
        title="Well Database"
        actions={
          <Button
            size="sm"
            onClick={() =>
              downloadCsv(
                "well_database.csv",
                DB_COLUMNS.map((c) => ({ key: c.key, label: c.label })),
                filtered,
              )
            }
          >
            Download CSV
          </Button>
        }
      >
        <div className="mb-2 flex flex-wrap items-center gap-4">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter wells, e.g. MPH"
            className={`${INPUT_CLS} w-56`}
          />
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={estOnly}
              onChange={(e) => setEstOnly(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            Estimated TVD only
          </label>
          <span className="text-xs text-slate-500">
            Showing {fmtNum(filtered.length)} of {fmtNum(rows.length)} wells
          </span>
        </div>
        <DataTable
          columns={DB_COLUMNS}
          rows={filtered}
          rowKey={(r, i) => `${str(r.well)}-${i}`}
          maxHeight="58vh"
        />
        {data.missing_surveys.length > 0 && (
          <WarnNote className="mt-3">
            {fmtNum(data.missing_surveys.length)} wells have estimated JP TVD (no survey)
            <details className="mt-1">
              <summary className="cursor-pointer text-xs">Show wells</summary>
              <span className="text-xs">{data.missing_surveys.join(", ")}</span>
            </details>
          </WarnNote>
        )}
      </Section>

      <Section
        title="Aging jet pumps"
        actions={
          <Button
            size="sm"
            disabled={!agingQuery.data || agingQuery.data.rows.length === 0}
            onClick={() =>
              downloadCsv(
                "aging_pumps.csv",
                agingCols.map((c) => ({ key: c.key, label: c.label })),
                agingQuery.data?.rows ?? [],
              )
            }
          >
            Download CSV
          </Button>
        }
      >
        <div className="mb-2 flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={knownOnly}
              onChange={(e) => setKnownOnly(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            Only wells in the table above
          </label>
          <label
            className="flex cursor-pointer items-center gap-2 text-xs text-slate-600"
            title="Keeps wells with a well test inside the window - allocated or info-only. Allocation lags roughly a month, so requiring an allocated test hides online wells."
          >
            <input
              type="checkbox"
              checked={onlineOnly}
              onChange={(e) => setOnlineOnly(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            Only wells online recently
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-600">
            Online window (days)
            <input
              type="number"
              min={7}
              max={365}
              value={onlineDays}
              disabled={!onlineOnly}
              onChange={(e) => setOnlineDays(Math.max(1, Number(e.target.value) || 60))}
              className={`${INPUT_CLS} w-20 tabular-nums disabled:bg-slate-50 disabled:text-slate-400`}
            />
          </label>
          <label
            className="flex items-center gap-2 text-xs text-slate-600"
            title="Marks pumps at or past this tenure. Every online well stays in the table - this flags, it does not filter."
          >
            Flag pumps older than (days)
            <input
              type="number"
              min={30}
              max={3650}
              step={30}
              value={flagDays}
              onChange={(e) => setFlagDays(Math.max(1, Number(e.target.value) || 365))}
              className={`${INPUT_CLS} w-24 tabular-nums`}
            />
          </label>
        </div>
        {agingStats && (
          <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric label="JP wells tracked" value={fmtNum(agingStats.tracked)} />
            <Metric
              label={`Older than ${fmtNum(flagDays)} d`}
              value={fmtNum(agingStats.over)}
              tone={agingStats.over > 0 ? "fair" : undefined}
            />
            <Metric label="Oldest" value={`${fmtNum(agingStats.oldest)} d`} />
            <Metric label="Median age" value={`${fmtNum(agingStats.median)} d`} />
          </div>
        )}
        {agingQuery.isError ? (
          <ErrorNote error={agingQuery.error} />
        ) : !agingQuery.data ? (
          <Spinner label="Loading aging pumps" />
        ) : (
          <DataTable
            columns={agingCols}
            rows={agingQuery.data.rows}
            rowKey={(r, i) => `${str(r.well)}-${i}`}
            emptyLabel="No pumps match the current filters"
          />
        )}
      </Section>

      <Section title="Save history">
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-600">
            Well
            <select
              value={propWell ?? ""}
              onChange={(e) => setPropWell(e.target.value === "" ? null : e.target.value)}
              className={`${INPUT_CLS} w-44`}
            >
              <option value="">Select a well</option>
              {wellNames.map((w) => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>
          </label>
        </div>
        {propWell === null ? (
          <InfoNote>Pick a well to see everything ever written to prop_hist for it.</InfoNote>
        ) : propQuery.isError ? (
          <ErrorNote error={propQuery.error} />
        ) : !propQuery.data ? (
          <Spinner label={`Loading property history for ${propWell}`} />
        ) : (
          <div className="space-y-4">
            <div>
              <h4 className="mb-1 text-xs font-semibold text-slate-600">Current saved state</h4>
              <DataTable
                columns={CURRENT_COLUMNS}
                rows={propQuery.data.current}
                rowKey={(r, i) => `${str(r.prop_name)}-${i}`}
                emptyLabel={`No saved property rows for ${propWell} yet`}
              />
            </div>
            <div>
              <h4 className="mb-1 text-xs font-semibold text-slate-600">Full history</h4>
              <DataTable
                columns={HISTORY_COLUMNS}
                rows={historySorted}
                rowKey={(r, i) => `${str(r.entry_datetime)}-${str(r.prop_name)}-${i}`}
                emptyLabel="No history rows"
              />
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}
