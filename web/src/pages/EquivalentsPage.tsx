/**
 * Pump Equivalents - cross-brand jet pump equivalence table for a chosen
 * nozzle/throat combination. Mirrors woffl/gui/tabs/pump_equivalent.py.
 *
 * Local nozzle/throat state is seeded ONCE from the sidebar params and is
 * independent afterwards - browsing equivalents never disturbs the solver.
 */

import { useState } from "react";

import { useEquivalents } from "../api/hooks";
import type { EquivalentRow } from "../api/types";
import { NOZZLE_OPTIONS, THROAT_OPTIONS } from "../api/types";
import { Badge, Card, type Column, DataTable, ErrorNote, Section, Spinner } from "../components/ui";
import { fmtNum } from "../lib/format";
import { useParamsStore } from "../state/params";

const SELECT_CLS =
  "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 " +
  "outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

const COLUMNS: Column<EquivalentRow>[] = [
  { key: "brand", label: "Brand" },
  { key: "nozzle", label: "Nozzle" },
  { key: "throat", label: "Throat" },
  { key: "nozzle_dia", label: "Nozzle Dia (in)", align: "right", render: (r) => fmtNum(r.nozzle_dia, 4) },
  { key: "throat_dia", label: "Throat Dia (in)", align: "right", render: (r) => fmtNum(r.throat_dia, 4) },
  { key: "nozzle_area", label: "Nozzle Area (in2)", align: "right", render: (r) => fmtNum(r.nozzle_area, 4) },
  { key: "throat_area", label: "Throat Area (in2)", align: "right", render: (r) => fmtNum(r.throat_area, 4) },
  { key: "area_ratio_val", label: "Dia Ratio", align: "right", render: (r) => fmtNum(r.area_ratio_val, 3) },
];

export default function EquivalentsPage() {
  const seedNozzle = useParamsStore((s) => s.params.nozzle_no);
  const seedThroat = useParamsStore((s) => s.params.area_ratio);
  const [nozzle, setNozzle] = useState(seedNozzle);
  const [throat, setThroat] = useState(seedThroat);

  const query = useEquivalents(nozzle, throat);
  const data = query.data;

  return (
    <div className="space-y-4">
      <Section
        title="Pump Equivalents"
        actions={<Badge tone="info">Reference: {nozzle}{throat}</Badge>}
      >
        <div className="flex flex-wrap items-end gap-3">
          <label className="block">
            <span className="text-xs text-slate-500">Nozzle</span>
            <select
              value={nozzle}
              onChange={(e) => setNozzle(e.target.value)}
              className={`${SELECT_CLS} mt-1 block`}
            >
              {NOZZLE_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="text-xs text-slate-500">Throat</span>
            <select
              value={throat}
              onChange={(e) => setThroat(e.target.value)}
              className={`${SELECT_CLS} mt-1 block`}
            >
              {THROAT_OPTIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
        </div>
      </Section>

      {query.isError ? (
        <ErrorNote error={query.error} />
      ) : !data ? (
        <Spinner label="Loading equivalents" />
      ) : (
        <Card>
          <DataTable
            columns={COLUMNS}
            rows={data.rows}
            rowKey={(r) => `${r.brand}-${r.nozzle}-${r.throat}`}
            highlightRow={(r) => r.is_reference}
          />
          <p className="mt-2 text-xs text-slate-500">
            Closest cross-brand equivalents by nozzle diameter, then throat/nozzle diameter
            ratio (Petrie &amp; Smart, 1983).
          </p>
        </Card>
      )}
    </div>
  );
}
