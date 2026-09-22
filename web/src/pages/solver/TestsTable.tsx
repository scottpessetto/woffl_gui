/**
 * Full-width well-test history table, with row-click comparison selection
 * added (clicking a row un-syncs the comparison test from the IPR anchor
 * and compares against that row). The Exclude box marks a bad test: it
 * leaves the anchor/comparison dropdowns, the IPR fit and the chart, and
 * stays listed here (greyed) so it can be brought back.
 */

import clsx from "clsx";

import type { WellTestRow } from "../../api/types";
import type { Column } from "../../components/ui";
import { DataTable, Section } from "../../components/ui";
import { fmtDate, fmtNum } from "../../lib/format";

import { testKey } from "./selection";

const COLUMNS: Column<WellTestRow>[] = [
  { key: "date", label: "Test Date", render: (r) => fmtDate(r.date) },
  { key: "oil", label: "Oil (BOPD)", align: "right", render: (r) => fmtNum(r.oil) },
  { key: "water", label: "Water (BWPD)", align: "right", render: (r) => fmtNum(r.water) },
  { key: "total_fluid", label: "Total Fluid (BPD)", align: "right", render: (r) => fmtNum(r.total_fluid) },
  {
    key: "form_wc",
    label: "Form WC (%)",
    align: "right",
    // UNCLAMPED on purpose: a WC outside 0-100% is a data-quality signal.
    render: (r) => (r.form_wc !== null ? fmtNum(r.form_wc * 100, 1) : "-"),
  },
  { key: "bhp", label: "BHP (psi)", align: "right", render: (r) => fmtNum(r.bhp) },
  { key: "fgor", label: "GOR (scf/bbl)", align: "right", render: (r) => fmtNum(r.fgor) },
  { key: "lift_wat", label: "PF Rate (BWPD)", align: "right", render: (r) => fmtNum(r.lift_wat) },
  { key: "whp", label: "Surface Pres (psi)", align: "right", render: (r) => fmtNum(r.whp) },
];

export function TestsTable({
  tests,
  selectedKey,
  onSelect,
  excludedKeys = [],
  onExclude,
}: {
  tests: WellTestRow[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  excludedKeys?: string[];
  onExclude?: (key: string, excluded: boolean) => void;
}) {
  const excluded = new Set(excludedKeys);
  const columns: Column<WellTestRow>[] = onExclude
    ? [
        {
          key: "exclude",
          label: "Exclude",
          render: (r) => (
            <input
              type="checkbox"
              aria-label={`Exclude the ${fmtDate(r.date)} test`}
              title="Bad test: leave it out of the anchor and comparison dropdowns, the IPR fit and the chart"
              checked={excluded.has(testKey(r))}
              // the row click selects a comparison test; the box must not
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => onExclude(testKey(r), e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 accent-blue-600"
            />
          ),
        },
        ...COLUMNS.map((c) => ({
          ...c,
          render: (r: WellTestRow) => (
            <span className={clsx(excluded.has(testKey(r)) && "text-slate-400 line-through")}>{c.render ? c.render(r) : null}</span>
          ),
        })),
      ]
    : COLUMNS;
  const n = tests.filter((t) => excluded.has(testKey(t))).length;
  return (
    <Section title={`Well Test Data (${tests.length} tests${n ? `, ${n} excluded` : ""})`}>
      <DataTable
        columns={columns}
        rows={tests}
        rowKey={(row) => testKey(row)}
        highlightRow={(row) => selectedKey !== null && testKey(row) === selectedKey}
        onRowClick={(row) => { if (!excluded.has(testKey(row))) onSelect(testKey(row)); }}
        emptyLabel="No well tests in the selected window"
      />
    </Section>
  );
}
