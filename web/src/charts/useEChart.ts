import { useCallback, useEffect, useRef } from "react";

import { echarts, type EChartsOption } from "./echarts";

export interface EChartInstance {
  /** Attach to the chart container div (callback ref). */
  attachRef: (node: HTMLDivElement | null) => void;
  /** Live chart instance for dispatchAction etc.; null before mount. */
  getChart: () => echarts.ECharts | null;
}

/**
 * Activate the toolbox box-select so plain dragging draws a zoom rect.
 *
 * MUST run EXACTLY ONCE per chart instance: each takeGlobalCursor dispatch
 * stacks another zoom-apply handler inside ECharts 6 (they survive
 * deactivation, restore, and notMerge setOption), and N stacked handlers
 * compound one drag into N nested zooms - the "zooms to the wrong place"
 * bug. Verified on a scratch chart: one arm = exact window; the cursor
 * persists across restore and notMerge setOption, so re-arming is never
 * needed.
 */
function armBoxZoom(chart: echarts.ECharts): void {
  chart.dispatchAction({
    type: "takeGlobalCursor",
    key: "dataZoomSelect",
    dataZoomSelectActive: true,
  });
}

/**
 * Core ECharts mounting hook: creates the chart when the container node
 * attaches (callback ref, so conditional JSX still initializes), disposes on
 * detach, resizes via ResizeObserver, and re-applies `option` on change.
 *
 * `boxZoom` keeps the toolbox dataZoomSelect cursor active so dragging a
 * rectangle zooms - the caller's option must carry a (hidden) toolbox
 * dataZoom feature for the brush to exist (see ChartPanel).
 */
export function useEChartInstance(
  option: EChartsOption | null,
  boxZoom = false,
  onReady?: (chart: echarts.ECharts) => void,
): EChartInstance {
  const chartRef = useRef<echarts.ECharts | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const optionRef = useRef<EChartsOption | null>(option);
  const boxZoomRef = useRef(boxZoom);
  const onReadyRef = useRef(onReady);
  const armedRef = useRef(false);
  optionRef.current = option;
  boxZoomRef.current = boxZoom;
  onReadyRef.current = onReady;

  const maybeArm = useCallback((chart: echarts.ECharts) => {
    if (boxZoomRef.current && !armedRef.current) {
      armBoxZoom(chart);
      armedRef.current = true;
    }
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (option && chart) {
      chart.setOption(option, { notMerge: true });
      maybeArm(chart);
    }
  }, [option, maybeArm]);

  useEffect(
    () => () => {
      observerRef.current?.disconnect();
      observerRef.current = null;
      chartRef.current?.dispose();
      chartRef.current = null;
    },
    [],
  );

  const attachRef = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      if (chartRef.current) return; // same node re-attach (StrictMode)
      // SVG renderer: vector text/lines stay crisp at any devicePixelRatio
      // (Windows 125/150% scaling blurs canvas backing stores) and at any
      // browser zoom - the "plotly-crisp" look. Data volumes here (<= a few
      // thousand points per chart) are well inside SVG's comfort zone.
      const chart = echarts.init(node, undefined, { renderer: "svg" });
      chartRef.current = chart;
      const observer = new ResizeObserver(() => chart.resize());
      observer.observe(node);
      observerRef.current = observer;
      if (optionRef.current) {
        chart.setOption(optionRef.current, { notMerge: true });
        maybeArm(chart);
      }
      onReadyRef.current?.(chart);
    } else {
      observerRef.current?.disconnect();
      observerRef.current = null;
      chartRef.current?.dispose();
      chartRef.current = null;
      armedRef.current = false; // a fresh instance needs its one arming
    }
  }, [maybeArm]);

  const getChart = useCallback(() => chartRef.current, []);

  return { attachRef, getChart };
}


/** Chart instance type for consumers that dispatch actions. */
export type EChart = echarts.ECharts;
