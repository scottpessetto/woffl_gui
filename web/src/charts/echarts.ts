/**
 * Tree-shaken ECharts bundle: register only what the app renders.
 * Import `echarts` from THIS module everywhere; importing the root
 * "echarts" package would pull the full build into the bundle.
 */

import { BarChart, CustomChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";

echarts.use([
  LineChart,
  ScatterChart,
  BarChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  MarkLineComponent,
  MarkAreaComponent,
  GraphicComponent,
  DataZoomComponent,
  ToolboxComponent,
  TitleComponent,
  SVGRenderer,
]);

export { echarts };

// E2E/diagnostic hook: lets browser automation reach chart instances via
// window.__ECHARTS__.getInstanceByDom(div[_echarts_instance_]). Harmless for
// an internal tool; the bundle already ships the echarts core.
declare global {
  interface Window {
    __ECHARTS__?: typeof echarts;
  }
}
window.__ECHARTS__ = echarts;
export type EChartsOption = echarts.EChartsCoreOption;
