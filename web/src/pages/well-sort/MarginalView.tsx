/**
 * Marginal WC view - the field-wide cumulative-water-threshold calculator
 * plus the per-pad (POPs) calculator with pump-limit headroom. Port of
 * well_sort.py:render_marginal_wc_tab + _render_pad_marginal_wc_section.
 * Both import buttons write params.marginal_watercut - the cutoff the Batch
 * Run recommender consumes, editable on the Batch Run card itself.
 */

import { Check } from "lucide-react";
import { useMemo, useState } from "react";

import { useMarginalWc, usePadMarginalWc, useWellSortTables } from "../../api/hooks";
import type { MarginalRankedRow, PadRankedRow } from "../../api/types";
import { Badge, Button, type Column, DataTable, InfoNote, Metric, Spinner, WarnNote } from "../../components/ui";
import { fmtNum, fmtSigned } from "../../lib/format";
import { useParamsStore } from "../../state/params";
import { useWellSortStore } from "../../state/wellSort";
import { ControlRow, LabeledSlider, num, NumberBox, pct, ResetButton, txt } from "./shared";

const STALE_DAYS = 60; // the calc always runs on the allocated/60-day table

function rankedColumns(margIdx: number): Column<MarginalRankedRow>[] {
  return [
    {
      key: "rank",
      label: "Rank",
      align: "right",
      render: (r) => fmtNum((r.rank as number | null) ?? null),
    },
    {
      key: "marginal",
      label: "Marginal?",
      align: "center",
      help: "Well at which cumulative water crosses the threshold",
      render: (r) =>
        (r.rank as number) === margIdx + 1 ? (
          <Check className="mx-auto h-3.5 w-3.5 text-blue-600" />
        ) : (
          <span className="text-slate-300">-</span>
        ),
    },
    txt("well", "Well"),
    txt("pad", "Pad"),
    txt("reservoir", "Reservoir"),
    num("oil", "Oil (BOPD)"),
    num("total_water", "Total Water (BWPD)"),
    pct("total_wc", "Total WC (%)", 1),
    num("cum_water", "Cum Water (BWPD)", 0, "Running total of TotalWater from the worst-WC well downward"),
    {
      key: "cum_water_pct",
      label: "Cum %",
      align: "right",
      help: "Cumulative water as a percentage of total field water",
      render: (r) => {
        const v = r.cum_water_pct as number | null;
        return v === null ? "-" : `${fmtNum(v, 1)}%`;
      },
    },
  ];
}

const PAD_LIFT_COLUMNS: Column<PadRankedRow>[] = [
  txt("well", "Well"),
  txt("reservoir", "Reservoir"),
  num("oil", "Oil (BOPD)"),
  num("lift_water", "PF Rate (BWPD)", 0, "Power-fluid water - the stream this pad's pump actually handles"),
  pct("wc_pad", "PF WC (%)", 1, "PF WC = PF water / (PF water + Oil). Marginal WC = max."),
];

const PAD_TOTAL_COLUMNS: Column<PadRankedRow>[] = [
  txt("well", "Well"),
  txt("reservoir", "Reservoir"),
  num("oil", "Oil (BOPD)"),
  num("total_water", "Total Water (BWPD)", 0, "Formation + lift water - the stream this pad's pump handles"),
  pct("total_wc", "Total WC (%)", 1, "Total WC = Total water / (Total water + Oil). Marginal WC = max."),
];

export function MarginalView() {
  const popsPads = useWellSortStore((s) => s.popsPads);
  const forceTrue = useWellSortStore((s) => s.forceTrue);
  const padLimits = useWellSortStore((s) => s.padLimits);
  const setPadLimit = useWellSortStore((s) => s.setPadLimit);
  const resetPadLimit = useWellSortStore((s) => s.resetPadLimit);
  const setParam = useParamsStore((s) => s.set);

  const [thresholdPct, setThresholdPct] = useState(2.0);
  const [pad, setPad] = useState<string | null>(popsPads[0] ?? null);
  const [imported, setImported] = useState<string | null>(null);

  // tables feeds the preset/handles maps (server-echoed engine constants).
  const tables = useWellSortTables("allocated", STALE_DAYS, popsPads, forceTrue);
  const presets = tables.data?.pump_limit_presets ?? {};
  const handles = tables.data?.pops_pump_handles ?? {};

  const field = useMarginalWc(thresholdPct, STALE_DAYS, popsPads, forceTrue);

  const activePad = pad !== null && popsPads.includes(pad) ? pad : popsPads[0] ?? null;
  const preset = activePad !== null ? presets[activePad] ?? 0 : 0;
  const pumpLimit = activePad !== null ? padLimits[activePad] ?? preset : 0;
  const padQuery = usePadMarginalWc(activePad, pumpLimit, STALE_DAYS, popsPads, forceTrue);

  const rankedRows = useMemo(
    () => (field.data ? field.data.rows.map((r, i) => ({ ...r, rank: i + 1 })) : []),
    [field.data],
  );

  const importMarginal = (value: number, label: string) => {
    setParam("marginal_watercut", Number(value.toFixed(3)));
    setImported(label);
  };

  const basis = activePad !== null ? handles[activePad] ?? "total" : "total";
  const basisLabel = basis === "lift" ? "PF water" : "total water";

  return (
    <div className="space-y-5">
      <p className="max-w-3xl text-xs text-slate-500">
        Estimates the marginal water cut by walking down the worst-WC online wells until their
        cumulative water crosses a threshold share of field water. Wells on POPs pads are excluded
        from the field calc - they have their own water handling and don't compete for central
        facility capacity. POPs selections are shared with the Wells view.
      </p>

      <ControlRow>
        <LabeledSlider
          label="Cumulative water threshold"
          value={thresholdPct}
          min={0.5}
          max={10}
          step={0.5}
          onChange={setThresholdPct}
          format={(v) => `${v.toFixed(1)}% of field water`}
          help="Walk down the sorted-by-WC list summing water; the marginal is the first well where the running total crosses this percentage. Higher = stricter."
        />
        {field.data && (
          <Button
            variant="primary"
            size="sm"
            onClick={() => importMarginal(field.data.marginal_wc, "field")}
            title="Sets the Marginal Watercut the Batch Run recommender uses"
          >
            Import {field.data.marginal_wc.toFixed(3)} to Batch Run
          </Button>
        )}
        {imported === "field" && (
          <span className="pb-1 text-xs font-medium text-green-700">
            Imported into the Batch Run Marginal Watercut.
          </span>
        )}
      </ControlRow>

      {field.isError ? (
        <WarnNote>
          No online non-POPs wells with valid Total WC / Total Water - check the Wells view.
        </WarnNote>
      ) : !field.data ? (
        <Spinner label="Computing field marginal WC" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric
              label="Today's Marginal WC"
              value={field.data.marginal_wc.toFixed(3)}
              sub={`set by ${field.data.well} (${field.data.pad}-Pad)`}
              tone="poor"
            />
            <Metric
              label="Field water (non-POPs)"
              value={`${fmtNum(field.data.total_field_water)} BWPD`}
              sub={`across ${field.data.well_count} online wells`}
            />
            <Metric
              label="Water above the line"
              value={`${fmtNum(field.data.cum_water_at_marginal)} BWPD`}
              sub={`${field.data.threshold_pct.toFixed(1)}% threshold`}
            />
            <Metric label="Ranked wells" value={fmtNum(field.data.well_count)} sub="worst WC first" />
          </div>

          <div>
            <p className="mb-1.5 text-sm font-semibold text-slate-700">
              Ranked Online Wells (non-POPs)
            </p>
            <DataTable
              columns={rankedColumns(field.data.marg_idx)}
              rows={rankedRows}
              rowKey={(r) => r.well}
              maxHeight="24rem"
              highlightRow={(r) => (r.rank as number) === field.data.marg_idx + 1}
              sortable
            />
          </div>
        </>
      )}

      <hr className="border-slate-200" />

      {/* ---------------- per-pad calculator ---------------- */}
      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-700">Per-Pad Marginal Water Cut</h3>
          <p className="mt-0.5 max-w-3xl text-xs text-slate-500">
            The pad marginal WC is the single worst online well on the pad pump (plain max, not
            shedding), measured on the stream that pump sees: E/F/M handle the full stream, H/I/S
            handle PF water only. The pump limit only sets headroom.
          </p>
        </div>

        {popsPads.length === 0 ? (
          <InfoNote>No POPs pads configured - add one in the POPs configuration above.</InfoNote>
        ) : (
          <>
            <ControlRow>
              <label className="block text-xs text-slate-500">
                POPs Pad
                <div className="mt-1 flex rounded-md border border-slate-300 bg-white p-0.5">
                  {popsPads.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setPad(p)}
                      className={
                        activePad === p
                          ? "rounded bg-blue-600 px-2.5 py-0.5 text-xs font-medium text-white"
                          : "rounded px-2.5 py-0.5 text-xs text-slate-600 hover:bg-slate-100"
                      }
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </label>
              {activePad !== null && (
                <>
                  <label className="block text-xs text-slate-500">
                    {activePad}-Pad pump limit (BWPD, {basisLabel})
                    <div className="mt-1 flex items-center gap-1.5">
                      <NumberBox
                        value={pumpLimit}
                        min={0}
                        max={200000}
                        step={1000}
                        onCommit={(v) => setPadLimit(activePad, v)}
                      />
                      <ResetButton
                        onClick={() => resetPadLimit(activePad)}
                        title={`Reset to preset (${fmtNum(preset)} BWPD)`}
                      />
                    </div>
                  </label>
                </>
              )}
            </ControlRow>

            {activePad === null ? null : padQuery.isError ? (
              <WarnNote>
                No online wells on {activePad}-Pad with valid WC / {basisLabel}. Check the Wells
                view.
              </WarnNote>
            ) : !padQuery.data ? (
              <Spinner label={`Computing ${activePad}-Pad marginal WC`} />
            ) : (
              <>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Metric
                    label={`${activePad}-Pad ${basis === "lift" ? "PF Water" : "Total Water"}`}
                    value={`${fmtNum(padQuery.data.pad_water)} BWPD`}
                    sub={`across ${padQuery.data.well_count} online wells`}
                  />
                  <Metric
                    label="Pump Limit"
                    value={`${fmtNum(padQuery.data.pump_limit)} BWPD`}
                    sub="editable above"
                  />
                  <Metric
                    label="Headroom"
                    value={
                      padQuery.data.headroom === null
                        ? "-"
                        : `${fmtSigned(padQuery.data.headroom)} BWPD`
                    }
                    tone={
                      padQuery.data.headroom === null
                        ? undefined
                        : padQuery.data.headroom >= 0
                          ? "good"
                          : "poor"
                    }
                    sub={
                      padQuery.data.headroom === null
                        ? "set a pump limit"
                        : padQuery.data.headroom >= 0
                          ? "available to allocate"
                          : `OVER by ${fmtNum(Math.abs(padQuery.data.headroom))} BWPD`
                    }
                  />
                  <Metric
                    label={`${activePad}-Pad Marginal WC`}
                    value={padQuery.data.marginal_wc.toFixed(3)}
                    sub={`set by ${padQuery.data.well}`}
                    tone="poor"
                  />
                </div>

                <ControlRow>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => importMarginal(padQuery.data.marginal_wc, `pad-${activePad}`)}
                    title="Sets the Marginal Watercut the Batch Run recommender uses"
                  >
                    Import {activePad}-Pad Marginal WC ({padQuery.data.marginal_wc.toFixed(3)}) to
                    Batch Run
                  </Button>
                  {imported === `pad-${activePad}` && (
                    <span className="pb-1 text-xs font-medium text-green-700">
                      Imported ({activePad}-Pad, {basisLabel} basis, set by {padQuery.data.well}).
                    </span>
                  )}
                </ControlRow>

                <div>
                  <p className="mb-1.5 text-sm font-semibold text-slate-700">
                    Online wells on {activePad}-Pad
                    <span className="ml-2 font-normal text-slate-400">
                      sorted by {basis === "lift" ? "PF WC" : "Total WC"} desc; top row = marginal
                    </span>
                  </p>
                  <DataTable
                    columns={basis === "lift" ? PAD_LIFT_COLUMNS : PAD_TOTAL_COLUMNS}
                    rows={padQuery.data.rows}
                    rowKey={(r) => r.well}
                    maxHeight="20rem"
                    highlightRow={(r) => r.well === padQuery.data.well}
                    sortable
                  />
                </div>
              </>
            )}
          </>
        )}
      </div>
      <p className="text-xs text-slate-400">
        <Badge tone="neutral">read-only</Badge> Imports only update this session's sidebar value -
        nothing is written to Databricks.
      </p>
    </div>
  );
}
