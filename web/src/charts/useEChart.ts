import { useCallback, useEffect, useRef } from "react";

import { echarts, type EChartsOption } from "./echarts";

export interface EChartInstance {
  /** Attach to the chart container div (callback ref). */
  attachRef: (node: HTMLDivElement | null) => void;
  /** Live chart instance for dispatchAction etc.; null before mount. */
  getChart: () => echarts.ECharts | null;
}

/**
 * Core ECharts mounting hook: creates the chart when the container node
 * attaches (callback ref, so conditional JSX still initializes), disposes on
 * detach, resizes via ResizeObserver, and re-applies `option` on change.
 *
 * `onReady` fires once per created chart instance - the place to attach
 * zrender-level listeners (ChartPanel's box brush, reset, tooltip hygiene).
 */
export function useEChartInstance(
  option: EChartsOption | null,
  onReady?: (chart: echarts.ECharts) => void,
): EChartInstance {
  const chartRef = useRef<echarts.ECharts | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const optionRef = useRef<EChartsOption | null>(option);
  const onReadyRef = useRef(onReady);
  optionRef.current = option;
  onReadyRef.current = onReady;

  useEffect(() => {
    const chart = chartRef.current;
    if (option && chart) {
      chart.setOption(option, { notMerge: true });
    }
  }, [option]);

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


/** Chart instance type for consumers that dispatch actions. */
export type EChart = echarts.ECharts;
