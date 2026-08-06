/**
 * Interactive chart panel - THE standard mount for every chart in the app.
 *
 * Interaction contract (mirrors plotly's, which the team is used to):
 *   drag          box zoom on the axes listed in `zoom`
 *   shift + drag  pan
 *   ctrl + wheel  wheel zoom at the cursor
 *   double-click  reset zoom (also the reset button)
 *   button        browser fullscreen
 *
 * Implementation: the box brush rides ECharts' toolbox dataZoomSelect
 * feature (toolbox parked offscreen, select cursor armed once per instance
 * by useEChartInstance). Pan/wheel come from injected `inside` dataZoom
 * components on the same axes. Because dataZoom components sharing an axis
 * are linked by ECharts, a multi-grid chart that lists both x axes in
 * `zoom.xAxisIndex` keeps them window-synced natively - no pixel-space
 * mirroring. All injected zooms use filterMode "none" so series data is
 * never dropped, only the window moves.
 *
 * Reset dispatches `restore`, which replays the last option = un-zoomed
 * state with default full-range windows.
 */

import { Maximize2, Minimize2, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { EChartsOption } from "./echarts";
import { type EChart, useEChartInstance } from "./useEChart";

export interface ZoomAxes {
  xAxisIndex: number[] | "all" | "none";
  yAxisIndex: number[] | "all" | "none";
}

const BTN_CLS =
  "pointer-events-auto rounded-md border border-slate-200 bg-white/90 p-1 " +
  "text-slate-400 shadow-sm transition-colors hover:text-slate-700";

/** Count axes of one dimension in an option (axis may be object or array). */
function axisCount(ax: unknown): number {
  if (Array.isArray(ax)) return ax.length;
  return ax ? 1 : 0;
}

/** Resolve a ZoomAxes entry to a concrete index list; null = no zoom. */
function resolveAxes(
  sel: number[] | "all" | "none",
  count: number,
): number[] | null {
  if (sel === "none") return null;
  if (sel === "all") return Array.from({ length: count }, (_, i) => i);
  return sel.length > 0 ? sel : null;
}

export function ChartPanel({
  option,
  height,
  zoom,
}: {
  option: EChartsOption | null;
  height: number;
  /** Which axes zooming (brush, pan, wheel) applies to. */
  zoom: ZoomAxes;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  const armedOption = useMemo<EChartsOption | null>(() => {
    if (!option) return null;
    const xs = resolveAxes(zoom.xAxisIndex, axisCount(option.xAxis));
    const ys = resolveAxes(zoom.yAxisIndex, axisCount(option.yAxis));

    // inside zooms: ctrl-wheel zoom at cursor, shift-wheel pan along x.
    // Plain wheel must keep scrolling the page, so both are modifier-gated.
    // Pan is wheel-based (not drag) because the always-armed box brush owns
    // every drag gesture, and re-toggling the brush cursor per-gesture would
    // stack zoom handlers (see armBoxZoom in useEChart.ts).
    const inside = {
      type: "inside",
      filterMode: "none",
      zoomOnMouseWheel: "ctrl",
      moveOnMouseMove: false,
      preventDefaultMouseMove: false,
    } as const;
    const dataZoom: Record<string, unknown>[] = [];
    if (xs) dataZoom.push({ ...inside, xAxisIndex: xs, moveOnMouseWheel: "shift" });
    if (ys) dataZoom.push({ ...inside, yAxisIndex: ys, moveOnMouseWheel: false });

    return {
      ...option,
      dataZoom,
      // Offscreen toolbox carries the dataZoom feature that powers the
      // drag-a-rectangle brush; its own icons stay hidden.
      toolbox: {
        show: true,
        top: -1000,
        feature: {
          dataZoom: {
            show: true,
            xAxisIndex: xs ?? false,
            yAxisIndex: ys ?? false,
            filterMode: "none",
            brushStyle: {
              color: "rgba(37, 99, 235, 0.08)",
              borderColor: "rgba(37, 99, 235, 0.6)",
              borderWidth: 1,
            },
          },
        },
      },
    };
  }, [option, zoom.xAxisIndex, zoom.yAxisIndex]);

  const onReady = useCallback((chart: EChart) => {
    const zr = chart.getZr();
    // Double-click resets the zoom (plotly muscle memory).
    zr.on("dblclick", () => chart.dispatchAction({ type: "restore" }));
    // A drag start would otherwise leave a stale tooltip frozen on screen.
    zr.on("mousedown", () => chart.dispatchAction({ type: "hideTip" }));
  }, []);

  const { attachRef, getChart } = useEChartInstance(armedOption, true, onReady);

  const resetZoom = useCallback(() => {
    // restore replays the last armed option; the box-select cursor persists
    // across restore (re-arming here would stack another zoom handler).
    getChart()?.dispatchAction({ type: "restore" });
  }, [getChart]);

  const toggleFullscreen = useCallback(() => {
    const el = wrapRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void el.requestFullscreen();
    }
  }, []);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === wrapRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  return (
    <div
      ref={wrapRef}
      className={fullscreen ? "relative flex h-full w-full flex-col bg-white p-4" : "relative"}
    >
      <div className="pointer-events-none absolute bottom-1 left-1 z-10 flex gap-1">
        <button
          type="button"
          title="Reset zoom (or double-click the chart)"
          aria-label="Reset zoom"
          className={BTN_CLS}
          onClick={resetZoom}
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          title={fullscreen ? "Exit full screen" : "Full screen"}
          aria-label={fullscreen ? "Exit full screen" : "Full screen"}
          className={BTN_CLS}
          onClick={toggleFullscreen}
        >
          {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
        <span className="pointer-events-none select-none self-center text-[10px] text-slate-300">
          drag zoom | ctrl-wheel zoom | shift-wheel pan | dbl-click reset
        </span>
      </div>
      <div ref={attachRef} style={fullscreen ? { flex: 1, minHeight: 0 } : { height }} />
    </div>
  );
}
