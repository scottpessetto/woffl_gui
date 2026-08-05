/**
 * Single-well view switcher - the port of the old segmented control
 * (single_well_page._VIEWS). Rendered above the routed page on every
 * single-well route; hidden on Well Database.
 */

import clsx from "clsx";
import { NavLink, useLocation } from "react-router-dom";

interface View {
  path: string;
  label: string;
  caption: string;
}

const VIEWS: View[] = [
  { path: "/solver", label: "Solver", caption: "Predicted oil, water, and power-fluid rates for the current pump." },
  { path: "/batch", label: "Batch Run", caption: "Sweep across nozzle/throat combinations to find the best pump." },
  { path: "/pf-range", label: "PF Range", caption: "How does oil rate trade off against power-fluid pressure?" },
  { path: "/pressure-profile", label: "Pressure Profile", caption: "Pressure traverse from surface to suction along the wellbore." },
  { path: "/well-profile", label: "Well Profile", caption: "Deviation survey (MD vs TVD) used by the simulator." },
  { path: "/equivalents", label: "Pump Equivalents", caption: "Other nozzle/throat pairs with similar nozzle and throat areas." },
  { path: "/jp-history", label: "JP History", caption: "Past pumps installed in this well." },
];

export const SINGLE_WELL_PATHS = VIEWS.map((v) => v.path);

export function ViewTabs() {
  const location = useLocation();
  const active = VIEWS.find((v) => location.pathname.startsWith(v.path));
  if (!active) return null;

  return (
    <div className="mb-4">
      <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
        {VIEWS.map((v) => (
          <NavLink
            key={v.path}
            to={v.path}
            className={({ isActive }) =>
              clsx(
                "rounded-md px-3 py-1 text-sm whitespace-nowrap transition-colors",
                isActive
                  ? "bg-blue-600 font-medium text-white"
                  : "text-slate-600 hover:bg-slate-100",
              )
            }
          >
            {v.label}
          </NavLink>
        ))}
      </div>
      <p className="mt-1.5 px-1 text-xs text-slate-500">{active.caption}</p>
    </div>
  );
}
