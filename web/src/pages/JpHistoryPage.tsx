/**
 * JP History - production rates and BHP over the well's jet pump install
 * eras, with a pumps-in-hole timeline strip under the main chart and the
 * install history table below. The figure itself is the shared
 * components/HistoryStrip (also rendered on the Solver page) so the two
 * views can never drift.
 */

import { useState } from "react";

import { useJpHistory } from "../api/hooks";
import type { JpInstallRow } from "../api/types";
import { HistoryStrip } from "../components/HistoryStrip";
import { Badge, Card, type Column, DataTable, ErrorNote, InfoNote, Section, Spinner } from "../components/ui";
import { fmtDate, fmtNum, pumpCode } from "../lib/format";
import { useParamsStore } from "../state/params";

const CHECKBOX_CLS = "h-4 w-4 rounded border-slate-300 accent-blue-600";

const INSTALL_COLUMNS: Column<JpInstallRow>[] = [
  { key: "date_set", label: "Date Set", render: (r) => fmtDate(r.date_set) },
  { key: "date_pulled", label: "Date Pulled", render: (r) => fmtDate(r.date_pulled) },
  { key: "pump", label: "Pump", render: (r) => pumpCode(r.nozzle, r.throat) },
  { key: "tubing_od", label: "Tubing OD", align: "right", render: (r) => fmtNum(r.tubing_od, 3) },
  { key: "circulating", label: "Circulation", render: (r) => r.circulating ?? "-" },
  { key: "manufacturer", label: "Manufacturer", render: (r) => r.manufacturer ?? "-" },
  {
    key: "pump_converted",
    label: "Converted",
    align: "center",
    render: (r) =>
      r.pump_converted ? (
        <Badge tone="fair" title={r.raw_pump ?? undefined}>
          Converted
        </Badge>
      ) : (
        "-"
      ),
  },
];

export default function JpHistoryPage() {
  const well = useParamsStore((s) => s.well);
  const [bhpFromZero, setBhpFromZero] = useState(true);
  const [showPf, setShowPf] = useState(false);

  const query = useJpHistory(well);
  const data = query.data;

  if (well === "Custom") {
    return (
      <InfoNote>
        Pick a well in the sidebar to see its jet pump install history and production trend.
      </InfoNote>
    );
  }
  if (query.isError) {
    return <ErrorNote error={query.error} />;
  }
  if (!data) {
    return <Spinner label={`Loading JP history for ${well}`} />;
  }
  if (data.installs.length === 0) {
    return <InfoNote>No jet pump history recorded for {well}</InfoNote>;
  }

  return (
    <div className="space-y-4">
      <Section
        title={`JP History - ${well}`}
        actions={
          <div className="flex items-center gap-2">
            {data.source === "excel_fallback" && <Badge tone="fair">Excel fallback</Badge>}
            <Badge>{fmtNum(data.installs.length)} installs</Badge>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={bhpFromZero}
              onChange={(e) => setBhpFromZero(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            BHP axis from zero
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={showPf}
              onChange={(e) => setShowPf(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            Show PF pressure
          </label>
        </div>
      </Section>

      {data.current_pump && (
        <p className="text-xs text-slate-600">
          Current pump: <span className="font-semibold">{data.current_pump}</span>
        </p>
      )}
      <Card>
        <HistoryStrip data={data} bhpFromZero={bhpFromZero} showPf={showPf} height={560} />
      </Card>

      <Section title="Install history">
        <DataTable
          columns={INSTALL_COLUMNS}
          rows={data.installs}
          rowKey={(r, i) => `${r.date_set ?? "unset"}-${i}`}
        />
      </Section>
    </div>
  );
}
