/**
 * Well Database - read-only viewer of the mech + reservoir characteristics
 * table, the aging jet pump report, and the per-well prop_hist audit trail.
 * Mirrors woffl/gui/well_database_page.py.
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

const AGING_COLUMNS: Column<DbRow>[] = [
  { key: "well", label: "Well" },
  { key: "pump", label: "Pump" },
  { key: "date_set", label: "Date Set", render: (r) => fmtDate(typeof r.date_set === "string" ? r.date_set : null) },
  { key: "days_in_hole", label: "Days In Hole", align: "right", render: (r) => fmtNum(num(r.days_in_hole)) },
  { key: "last_test", label: "Last Test", render: (r) => fmtDate(typeof r.last_test === "string" ? r.last_test : null) },
  {
    key: "online",
    label: "Online",
    align: "center",
    render: (r) => (r.online === true ? <Badge tone="good">Yes</Badge> : <Badge>No</Badge>),
  },
];

/** "YYYY-MM-DD HH:MM" from the UTC "YYYY-MM-DD HH:MM:SS" audit stamp. */
function when(v: unknown): string {
  return typeof v === "string" ? v.slice(0, 16) : "-";
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
  { key: "entry_datetime", label: "When", render: (r) => when(r.entry_datetime) },
];

const HISTORY_COLUMNS: Column<DbRow>[] = [
  { key: "entry_datetime", label: "When", render: (r) => when(r.entry_datetime) },
  { key: "prop_name", label: "Prop", render: (r) => str(r.prop_name) },
  { key: "prop_value", label: "Value", align: "right", render: (r) => propValue(r.prop_value) },
  { key: "entry_user", label: "By", render: (r) => str(r.entry_user) },
  { key: "comment_text", label: "Comment", render: (r) => str(r.comment_text) },
];

export default function WellDatabasePage() {
  const [filter, setFilter] = useState("");
  const [estOnly, setEstOnly] = useState(false);
  const [knownOnly, setKnownOnly] = useState(true);
  const [onlineOnly, setOnlineOnly] = useState(false);
  const [onlineDays, setOnlineDays] = useState(30);
  const [minDays, setMinDays] = useState(365);
  const [propWell, setPropWell] = useState<string | null>(null);

  const dbQuery = useWellDatabase();
  const agingQuery = useAgingPumps(knownOnly, onlineOnly, onlineDays, minDays);
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

  const historySorted = useMemo(() => {
    const history = propQuery.data?.history ?? [];
    return [...history].sort((a, b) => when(b.entry_datetime).localeCompare(when(a.entry_datetime)));
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
                AGING_COLUMNS.map((c) => ({ key: c.key, label: c.label })),
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
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
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
              onChange={(e) => setOnlineDays(Math.max(1, Number(e.target.value) || 30))}
              className={`${INPUT_CLS} w-20 tabular-nums disabled:bg-slate-50 disabled:text-slate-400`}
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-600">
            Min days in hole
            <input
              type="number"
              min={0}
              value={minDays}
              onChange={(e) => setMinDays(Math.max(0, Number(e.target.value) || 0))}
              className={`${INPUT_CLS} w-24 tabular-nums`}
            />
          </label>
        </div>
        {agingQuery.isError ? (
          <ErrorNote error={agingQuery.error} />
        ) : !agingQuery.data ? (
          <Spinner label="Loading aging pumps" />
        ) : (
          <DataTable
            columns={AGING_COLUMNS}
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
                rowKey={(r, i) => `${when(r.entry_datetime)}-${str(r.prop_name)}-${i}`}
                emptyLabel="No history rows"
              />
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}
