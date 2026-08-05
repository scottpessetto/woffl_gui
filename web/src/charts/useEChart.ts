import { useCallback, useEffect, useRef } from "react";

import { echarts, type EChartsOption } from "./echarts";

export interface EChartInstance {
  /** Attach to the chart container div (callback ref). */
  attachRef: (node: HTMLDivElement | null) => void;
  /** Live chart instance for dispatchAction etc.; null before mount. */
  getChart: () => echarts.ECharts | null;
}

/** Re-arm the toolbox box-select so plain dragging always draws a zoom rect. */
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
  optionRef.current = option;
  boxZoomRef.current = boxZoom;
  onReadyRef.current = onReady;

  useEffect(() => {
    const chart = chartRef.current;
    if (option && chart) {
      chart.setOption(option, { notMerge: true });
      if (boxZoom) armBoxZoom(chart);
    }
  }, [option, boxZoom]);

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
      const chart = echarts.init(node);
      chartRef.current = chart;
      const observer = new ResizeObserver(() => chart.resize());
      observer.observe(node);
      observerRef.current = observer;
      if (optionRef.current) {
        chart.setOption(optionRef.current, { notMerge: true });
        if (boxZoomRef.current) armBoxZoom(chart);
      }
      onReadyRef.current?.(chart);
    } else {
      observerRef.current?.disconnect();
      observerRef.current = null;
      chartRef.current?.dispose();
      chartRef.current = null;
    }
  }, []);

  const getChart = useCallback(() => chartRef.current, []);

  return { attachRef, getChart };
}

/**
 * Plain chart mount - the original hook, kept for simple charts.
 *
 *   const ref = useEChart(option);
 *   return <div ref={ref} className="h-[500px]" />;
 */
export function useEChart(option: EChartsOption | null): (node: HTMLDivElement | null) => void {
  return useEChartInstance(option).attachRef;
}

export { armBoxZoom };
