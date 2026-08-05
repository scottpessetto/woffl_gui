/**
 * Full-width well-test history table. Mirror of
 * woffl/gui/tabs/jetpump_solver.py:_render_well_test_table with row-click
 * comparison selection added (clicking a row un-syncs the comparison test
 * from the IPR anchor and compares against that row).
 */

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
}: {
  tests: WellTestRow[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <Section title={`Well Test Data (${tests.length} tests)`}>
      <DataTable
        columns={COLUMNS}
        rows={tests}
        rowKey={(row) => testKey(row)}
        highlightRow={(row) => selectedKey !== null && testKey(row) === selectedKey}
        onRowClick={(row) => onSelect(testKey(row))}
        emptyLabel="No well tests in the selected window"
      />
    </Section>
  );
}
