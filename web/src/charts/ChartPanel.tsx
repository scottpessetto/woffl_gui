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
 * Implementation: the box brush is OWN CODE on zrender mouse events - an
 * overlay rect plus a dataZoom dispatch whose value windows come from
 * chart.convertFromPixel at mouseup, i.e. always the LIVE axis mapping. It
 * deliberately does NOT use the ECharts toolbox dataZoomSelect feature:
 * that feature converts the brush rect through the axis scale it captured
 * when the select cursor was armed. Arm it while the chart is still
 * rendering with empty series (data queries in flight) and every later
 * drag maps pixels through the default empty-axis extent [0, 1000] - the
 * recurring "box zoom lands in the wrong place" bug (x window =
 * grid-fraction x 1000 while y, whose scale object happened to be mutated
 * in place, stayed correct). No setOption or resize heals it, and
 * re-arming stacks zoom handlers instead of replacing them. Percent-window
 * dispatch is no alternative: each component's percent domain is an
 * internal blend of data and nice extents that differs per axis.
 *
 * Pan/wheel come from injected `inside` dataZoom components. X axes share
 * ONE component - ECharts links axes sharing a component, which keeps
 * HistoryStrip's two stacked time axes window-synced natively, and a value
 * window is valid for all of them because they span the same dates. Each Y
 * axis gets its OWN component: stacked grids plot different units, so y
 * windows are per-grid - the brush only zooms the y axes of the grid the
 * drag started in. All injected zooms use filterMode "none" so series data
 * is never dropped, only the window moves.
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

/** Minimum drag extent (px) in BOTH dimensions before a drag counts as a zoom. */
const MIN_DRAG_PX = 6;

/** Count axes of one dimension in an option (axis may be object or array). */
function axisCount(ax: unknown): number {
  if (ax === undefined || ax === null) return 0;
  return Array.isArray(ax) ? ax.length : 1;
}

/** Resolve a ZoomAxes entry to a concrete index list; null = no zoom. */
function resolveAxes(
  sel: number[] | "all" | "none",
  count: number,
): number[] | null {
  if (sel === "none" || count === 0) return null;
  if (sel === "all") return Array.from({ length: count }, (_, i) => i);
  return sel.length > 0 ? sel : null;
}

/** Index of the y axis living on `gridIndex` per the armed option, or -1. */
function yAxisOnGrid(yAxes: { axisIdx: number; gridIdx: number }[], gridIndex: number): number {
  const hit = yAxes.find((a) => a.gridIdx === gridIndex);
  return hit ? hit.axisIdx : -1;
}

/** Index of the grid containing a chart-local pixel point, or -1. */
function gridAt(chart: EChart, gridCount: number, x: number, y: number): number {
  for (let g = 0; g < gridCount; g++) {
    if (chart.containPixel({ gridIndex: g }, [x, y])) return g;
  }
  return -1;
}


export function ChartPanel({
  option,
  height,
  zoom,
  onSelect,
}: {
  option: EChartsOption | null;
  height: number;
  /** Which axes zooming (brush, pan, wheel) applies to. */
  zoom: ZoomAxes;
  /** Category-axis charts where clicking a row picks a detail view (the
   *  sensitivity tornado). Receives the clicked category name. Not a
   *  general event bus - nothing else is forwarded. */
  onSelect?: (name: string) => void;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  // onReady runs once per chart instance; keep the live callback in a ref.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const [fullscreen, setFullscreen] = useState(false);
  // Latest zoom wiring for the (once-attached) brush handlers: which axes
  // zoom, which dataZoom component covers them, and the y-axis -> grid map.
  const brushCfg = useRef<{
    xRefAxis: number | null;
    yAxes: { axisIdx: number; gridIdx: number; compIdx: number }[];
    gridCount: number;
  }>({ xRefAxis: null, yAxes: [], gridCount: 1 });

  const armedOption = useMemo<EChartsOption | null>(() => {
    if (!option) return null;
    const xs = resolveAxes(zoom.xAxisIndex, axisCount(option.xAxis));
    const ys = resolveAxes(zoom.yAxisIndex, axisCount(option.yAxis));

    // inside zooms: ctrl-wheel zoom at cursor, shift-wheel pan along x.
    // Plain wheel must keep scrolling the page, so both are modifier-gated.
    // Plain drag belongs to the box brush below.
    const inside = {
      type: "inside",
      filterMode: "none",
      zoomOnMouseWheel: "ctrl",
      moveOnMouseMove: false,
      preventDefaultMouseMove: false,
    } as const;
    const dataZoom: Record<string, unknown>[] = [];
    if (xs) dataZoom.push({ ...inside, xAxisIndex: xs, moveOnMouseWheel: "shift" });
    const yAxisDefs = Array.isArray(option.yAxis) ? option.yAxis : option.yAxis ? [option.yAxis] : [];
    const yAxes = (ys ?? []).map((axisIdx) => {
      const def: unknown = yAxisDefs[axisIdx];
      const gridIdx =
        def && typeof def === "object" && "gridIndex" in def && typeof def.gridIndex === "number"
          ? def.gridIndex
          : 0;
      const compIdx = dataZoom.length;
      dataZoom.push({ ...inside, yAxisIndex: [axisIdx], moveOnMouseWheel: false });
      return { axisIdx, gridIdx, compIdx };
    });
    brushCfg.current = {
      xRefAxis: xs ? xs[0] : null,
      yAxes,
      gridCount: Math.max(1, axisCount(option.grid)),
    };

    return { ...option, dataZoom };
  }, [option, zoom.xAxisIndex, zoom.yAxisIndex]);

  const onReady = useCallback((chart: EChart) => {
    const zr = chart.getZr();
    // Double-click resets the zoom (plotly muscle memory).
    zr.on("dblclick", () => chart.dispatchAction({ type: "restore" }));

    // Row selection on category charts. Harmless where onSelect is unset.
    chart.on("click", (p: unknown) => {
      if (p === null || typeof p !== "object" || !("name" in p)) return;
      const name = p.name;
      if (typeof name === "string" && name !== "") onSelectRef.current?.(name);
    });

    // ---- box brush ---------------------------------------------------
    // Overlay rect lives inside the chart container (ECharts forces the
    // container to position:relative), so offsetX/Y map 1:1.
    const box = document.createElement("div");
    box.style.cssText =
      "position:absolute;display:none;pointer-events:none;z-index:5;" +
      "border:1px solid rgba(37,99,235,0.6);background:rgba(37,99,235,0.08);";
    chart.getDom().appendChild(box);

    let start: { x: number; y: number; grid: number } | null = null;

    const hideBox = () => {
      start = null;
      box.style.display = "none";
    };

    type ZrMouse = { offsetX: number; offsetY: number; which?: number; event: MouseEvent };

    zr.on("mousedown", (e: ZrMouse) => {
      // A drag start would otherwise leave a stale tooltip frozen on screen.
      chart.dispatchAction({ type: "hideTip" });
      const raw = e.event;
      if ((e.which ?? 1) !== 1 || raw.ctrlKey || raw.metaKey || raw.shiftKey || raw.altKey) return;
      const cfg = brushCfg.current;
      if (cfg.xRefAxis === null && cfg.yAxes.length === 0) return;
      const g = gridAt(chart, cfg.gridCount, e.offsetX, e.offsetY);
      if (g < 0) return;
      start = { x: e.offsetX, y: e.offsetY, grid: g };
    });

    zr.on("mousemove", (e: ZrMouse) => {
      if (!start) return;
      e.event.preventDefault(); // no text selection mid-drag
      box.style.display = "block";
      box.style.left = `${Math.min(start.x, e.offsetX)}px`;
      box.style.top = `${Math.min(start.y, e.offsetY)}px`;
      box.style.width = `${Math.abs(e.offsetX - start.x)}px`;
      box.style.height = `${Math.abs(e.offsetY - start.y)}px`;
    });

    zr.on("mouseup", (e: ZrMouse) => {
      if (!start) return;
      const s = start;
      hideBox();
      if (Math.abs(e.offsetX - s.x) < MIN_DRAG_PX || Math.abs(e.offsetY - s.y) < MIN_DRAG_PX) return;

      // convertFromPixel at mouseup = the LIVE pixel->value mapping; no
      // captured scale, no percent-domain guesswork. Values go per
      // component: the shared x component (same date/value span on every
      // linked axis) and ONLY the y component of the grid dragged in.
      const cfg = brushCfg.current;
      const batch: { dataZoomIndex: number; startValue: number; endValue: number }[] = [];
      if (cfg.xRefAxis !== null) {
        const v0 = chart.convertFromPixel({ xAxisIndex: cfg.xRefAxis }, s.x);
        const v1 = chart.convertFromPixel({ xAxisIndex: cfg.xRefAxis }, e.offsetX);
        batch.push({ dataZoomIndex: 0, startValue: Math.min(v0, v1), endValue: Math.max(v0, v1) });
      }
      const yAxisIdx = yAxisOnGrid(cfg.yAxes, s.grid);
      if (yAxisIdx >= 0) {
        const comp = cfg.yAxes.find((a) => a.axisIdx === yAxisIdx);
        if (comp) {
          const v0 = chart.convertFromPixel({ yAxisIndex: yAxisIdx }, s.y);
          const v1 = chart.convertFromPixel({ yAxisIndex: yAxisIdx }, e.offsetY);
          batch.push({
            dataZoomIndex: comp.compIdx,
            startValue: Math.min(v0, v1),
            endValue: Math.max(v0, v1),
          });
        }
      }
      if (batch.length > 0) chart.dispatchAction({ type: "dataZoom", batch });
    });

    // Cursor left the chart mid-drag: abandon the gesture.
    zr.on("globalout", hideBox);
  }, []);

  const { attachRef, getChart } = useEChartInstance(armedOption, onReady);

  const resetZoom = useCallback(() => {
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
      className={
        fullscreen ? "group relative flex h-full w-full flex-col bg-white p-4" : "group relative"
      }
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
        {/* Hover-only: parked at bottom-left it lands on the x-axis name of
            any half-width chart. The white chip keeps it legible over the
            plot while it is showing. */}
        <span
          className="pointer-events-none select-none self-center rounded bg-white/90 px-1
            text-[10px] text-slate-400 opacity-0 transition-opacity group-hover:opacity-100"
        >
          drag zoom | ctrl-wheel zoom | shift-wheel pan | dbl-click reset
        </span>
      </div>
      <div ref={attachRef} style={fullscreen ? { flex: 1, minHeight: 0 } : { height }} />
    </div>
  );
}
