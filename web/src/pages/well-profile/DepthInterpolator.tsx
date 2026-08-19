/**
 * MD <-> TVD converter for the loaded deviation survey. Type into either box
 * and the other is computed server-side by minimum curvature (SPE 84246) -
 * the circular arc between survey stations, not the chord that straight
 * interpolation gives, which reads shallow through a build section.
 *
 * The box being typed in is the given depth; the other is a read-only
 * readout. Clicking into the readout flips which one is given.
 *
 * A TVD can sit at more than one MD: any lateral that builds past 90 deg
 * crosses the same TVD on the way down and again on the way back up. Every
 * solution is listed; the shallowest drives the readout and the chart marker.
 */

import { useEffect, useState } from "react";
import clsx from "clsx";

import { useDepthLookup } from "../../api/hooks";
import type { DepthLookupResponse } from "../../api/types";
import { Badge, Card, ErrorNote, InfoNote, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useDebounced } from "../../lib/useDebounced";

const INPUT_CLS =
  "h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";
const READOUT_CLS =
  "h-8 w-full rounded-md border border-slate-200 bg-slate-50 px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:bg-white focus:ring-1 focus:ring-blue-200";

/** Survey precision, without dragging a trailing ".0" into the input. */
const asText = (value: number): string => String(Math.round(value * 100) / 100);

export function DepthInterpolator({
  well,
  fieldModel,
  mdRange,
  onResult,
}: {
  well: string;
  fieldModel: string;
  /** [shallowest, deepest] survey MD, feet - shown as the valid range. */
  mdRange: [number, number];
  /** Latest lookup, or null - the page marks it on the charts. */
  onResult: (hit: DepthLookupResponse | null) => void;
}) {
  const [given, setGiven] = useState<"md" | "tvd">("tvd");
  const [mdText, setMdText] = useState("");
  const [tvdText, setTvdText] = useState("");

  const typed = given === "md" ? mdText : tvdText;
  const settled = useDebounced(typed, 300);
  const parsed = Number(settled);
  const ready = settled.trim() !== "" && Number.isFinite(parsed) && parsed >= 0;

  const query = useDepthLookup(well, given, ready ? parsed : null, fieldModel);
  // A cached hit for the OTHER direction is still on hand right after a flip;
  // only one that answers the box being typed in may drive the readout.
  const hit = query.data && query.data.given === given ? query.data : null;

  useEffect(() => onResult(hit), [hit, onResult]);

  const mdShown = given === "md" ? mdText : hit ? asText(hit.md) : "";
  const tvdShown = given === "tvd" ? tvdText : hit ? asText(hit.tvd) : "";

  const focusMd = () => {
    if (given === "md") return;
    setMdText(mdShown);
    setGiven("md");
  };
  const focusTvd = () => {
    if (given === "tvd") return;
    setTvdText(tvdShown);
    setGiven("tvd");
  };

  const extras = hit?.md_solutions.slice(1) ?? [];

  return (
    <Section title="Depth Interpolator">
      <Card className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="text-xs font-medium text-slate-500">Measured depth (ft MD)</span>
            <input
              type="number"
              step={10}
              min={0}
              value={mdShown}
              placeholder={given === "md" ? "" : "computed"}
              onFocus={focusMd}
              onChange={(e) => {
                setGiven("md");
                setMdText(e.target.value);
              }}
              className={clsx(given === "md" ? INPUT_CLS : READOUT_CLS, "mt-1")}
            />
          </label>
          <label className="block">
            <span className="text-xs font-medium text-slate-500">True vertical depth (ft TVD)</span>
            <input
              type="number"
              step={10}
              min={0}
              value={tvdShown}
              placeholder={given === "tvd" ? "" : "computed"}
              onFocus={focusTvd}
              onChange={(e) => {
                setGiven("tvd");
                setTvdText(e.target.value);
              }}
              className={clsx(given === "tvd" ? INPUT_CLS : READOUT_CLS, "mt-1")}
            />
          </label>
        </div>

        {query.isError && <ErrorNote error={query.error} />}

        {hit ? (
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="info">
              {hit.method === "minimum_curvature"
                ? "Minimum curvature"
                : "Straight-line (no angles on file)"}
            </Badge>
            {hit.inclination !== null && <Badge>Inclination: {fmtNum(hit.inclination, 1)} deg</Badge>}
            {hit.azimuth !== null && <Badge>Azimuth: {fmtNum(hit.azimuth, 1)} deg</Badge>}
            {hit.dls !== null && <Badge>DLS: {fmtNum(hit.dls, 2)} deg/100 ft</Badge>}
            {hit.at_station ? (
              <Badge tone="good">On a survey station</Badge>
            ) : (
              hit.station_above &&
              hit.station_below && (
                <Badge>
                  Between stations {fmtNum(hit.station_above.md)} and {fmtNum(hit.station_below.md)} ft MD
                </Badge>
              )
            )}
          </div>
        ) : (
          !query.isError && (
            <p className="text-xs text-slate-500">
              Type a depth in either box. Survey runs {fmtNum(mdRange[0])} to {fmtNum(mdRange[1])} ft MD.
            </p>
          )
        )}

        {extras.length > 0 && (
          <InfoNote>
            {hit?.note} Also reached at {extras.map((m) => `${fmtNum(m)} ft MD`).join(", ")}.
          </InfoNote>
        )}
      </Card>
    </Section>
  );
}
