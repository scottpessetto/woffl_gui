/**
 * Interactive chart panel: drag-a-rectangle zoom (always armed), a reset
 * button, and browser fullscreen - shared by the pump-history strip and the
 * IPR chart.
 *
 * Box zoom rides ECharts' toolbox dataZoomSelect feature: the toolbox is
 * injected into the option but parked offscreen (its icons are redundant -
 * the panel provides its own controls) and the select cursor is re-armed
 * after every setOption by useEChartInstance. Reset dispatches `restore`,
 * which replays the last option = un-zoomed state.
 */

import { Maximize2, Minimize2, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { EChartsOption } from "./echarts";
import { armBoxZoom, useEChartInstance } from "./useEChart";

export interface ZoomAxes {
  xAxisIndex: number[] | "all" | "none";
  yAxisIndex: number[] | "all" | "none";
}

export interface LinkedXAxis {
  /** x axis index that mirrors axis 0's zoom window (the strip). */
  targetAxis: number;
  /** pixel margins of grid 0 (option's grid.left / grid.right). */
  leftPx: number;
  rightPx: number;
}

const BTN_CLS =
  "pointer-events-auto rounded-md border border-slate-200 bg-white/90 p-1 " +
  "text-slate-400 shadow-sm transition-colors hover:text-slate-700";

/** Chart instance type via the action helper's signature. */
type EChart = Parameters<typeof armBoxZoom>[0];

export function ChartPanel({
  option,
  height,
  zoom,
  linkX,
}: {
  option: EChartsOption | null;
  height: number;
  /** Which axes the drag rectangle zooms (per chart geometry). */
  zoom: ZoomAxes;
  /**
   * Multi-grid charts: the toolbox brush only zooms the axes under the
   * brushed grid, so a context strip on its own x axis is mirrored here on
   * every datazoom (window read back via convertFromPixel at the grid
   * margins - public API only).
   */
  linkX?: LinkedXAxis;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const linkXRef = useRef(linkX);
  linkXRef.current = linkX;

  // Offscreen toolbox carries the dataZoom feature that powers the brush.
  const armedOption = useMemo<EChartsOption | null>(() => {
    if (!option) return null;
    return {
      ...option,
      toolbox: {
        show: true,
        top: -1000,
        feature: {
          dataZoom: {
            show: true,
            xAxisIndex: zoom.xAxisIndex,
            yAxisIndex: zoom.yAxisIndex,
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
    chart.on("datazoom", () => {
      const link = linkXRef.current;
      if (!link) return;
      const lo = chart.convertFromPixel({ xAxisIndex: 0 }, link.leftPx + 1);
      const hi = chart.convertFromPixel({ xAxisIndex: 0 }, chart.getWidth() - link.rightPx - 1);
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return;
      const axes: Record<string, unknown>[] = [];
      axes[link.targetAxis] = { min: lo, max: hi };
      for (let i = 0; i < link.targetAxis; i++) axes[i] = axes[i] ?? {};
      // merge setOption: pins the strip axis to the zoomed window; the
      // armed option replayed by `restore` carries the full-range min/max,
      // so reset undoes this automatically.
      chart.setOption({ xAxis: axes });
      armBoxZoom(chart);
    });
  }, []);

  const { attachRef, getChart } = useEChartInstance(armedOption, true, onReady);

  const resetZoom = useCallback(() => {
    const chart = getChart();
    if (!chart) return;
    chart.dispatchAction({ type: "restore" });
    // restore replays the option but drops the active select cursor.
    armBoxZoom(chart);
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
          title="Reset zoom"
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
          drag to zoom
        </span>
      </div>
      <div ref={attachRef} style={fullscreen ? { flex: 1, minHeight: 0 } : { height }} />
    </div>
  );
}
