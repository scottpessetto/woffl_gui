import { useCallback, useEffect, useRef } from "react";

import { echarts, type EChartsOption } from "./echarts";

/**
 * Mount an ECharts instance on a div and keep it in sync with `option`.
 *
 * Returns a CALLBACK ref so charts inside conditional JSX (rendered only
 * once data arrives) still initialize: the chart is created when the node
 * attaches and disposed when it detaches. A mount-once effect would miss
 * late-mounting containers entirely.
 *
 *   const ref = useEChart(option);
 *   return <div ref={ref} className="h-[500px]" />;
 */
export function useEChart(option: EChartsOption | null): (node: HTMLDivElement | null) => void {
  const chartRef = useRef<echarts.ECharts | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const optionRef = useRef<EChartsOption | null>(option);
  optionRef.current = option;

  useEffect(() => {
    if (option && chartRef.current) {
      chartRef.current.setOption(option, { notMerge: true });
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

  return useCallback((node: HTMLDivElement | null) => {
    if (node) {
      if (chartRef.current) return; // same node re-attach (StrictMode)
      const chart = echarts.init(node);
      chartRef.current = chart;
      const observer = new ResizeObserver(() => chart.resize());
      observer.observe(node);
      observerRef.current = observer;
      if (optionRef.current) {
        chart.setOption(optionRef.current, { notMerge: true });
      }
    } else {
      observerRef.current?.disconnect();
      observerRef.current = null;
      chartRef.current?.dispose();
      chartRef.current = null;
    }
  }, []);
}
