import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { Spinner } from "./components/ui";
import { Layout } from "./layout/Layout";
import { useSecretMenuListener, useToolsUnlocked } from "./lib/secretMenu";
// The address-bar way in. NOT lazy: it is a redirect, and a chunk fetch
// between the unlock and the hop would be a spinner for no reason.
import ScottUnlock from "./pages/ScottUnlock";

// Route-level code splitting: chart-heavy pages load on demand.
const SolverPage = lazy(() => import("./pages/SolverPage"));
const BatchPage = lazy(() => import("./pages/BatchPage"));
const PfRangePage = lazy(() => import("./pages/PfRangePage"));
const SensitivityPage = lazy(() => import("./pages/SensitivityPage"));
const PressureProfilePage = lazy(() => import("./pages/PressureProfilePage"));
const WellProfilePage = lazy(() => import("./pages/WellProfilePage"));
const EquivalentsPage = lazy(() => import("./pages/EquivalentsPage"));
const JpHistoryPage = lazy(() => import("./pages/JpHistoryPage"));
const WellDatabasePage = lazy(() => import("./pages/WellDatabasePage"));
const WellSortPage = lazy(() => import("./pages/WellSortPage"));
const OptimizePage = lazy(() => import("./pages/OptimizePage"));
// Scott's Tools - the secret menu. Lazy like every other route, so a locked
// app never downloads them.
const ToolsPage = lazy(() => import("./pages/ToolsPage"));
const PadWatercutPage = lazy(() => import("./pages/tools/PadWatercutPage"));
const SepOilLossPage = lazy(() => import("./pages/tools/SepOilLossPage"));
const HeaderImpactPage = lazy(() => import("./pages/tools/HeaderImpactPage"));
const PfScenarioPage = lazy(() => import("./pages/tools/PfScenarioPage"));
const JpWashoutPage = lazy(() => import("./pages/tools/JpWashoutPage"));
const FricTrendPage = lazy(() => import("./pages/tools/FricTrendPage"));
const JpCalibrationPage = lazy(() => import("./pages/tools/JpCalibrationPage"));
const TestHarnessPage = lazy(() => import("./pages/tools/TestHarnessPage"));

export default function App() {
  useSecretMenuListener();
  const toolsUnlocked = useToolsUnlocked();

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/solver" replace />} />
        <Route
          path="/solver"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <SolverPage />
            </Suspense>
          }
        />
        <Route
          path="/batch"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <BatchPage />
            </Suspense>
          }
        />
        <Route
          path="/pf-range"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <PfRangePage />
            </Suspense>
          }
        />
        <Route
          path="/sensitivity"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <SensitivityPage />
            </Suspense>
          }
        />
        <Route
          path="/pressure-profile"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <PressureProfilePage />
            </Suspense>
          }
        />
        <Route
          path="/well-profile"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <WellProfilePage />
            </Suspense>
          }
        />
        <Route
          path="/equivalents"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <EquivalentsPage />
            </Suspense>
          }
        />
        <Route
          path="/jp-history"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <JpHistoryPage />
            </Suspense>
          }
        />
        <Route
          path="/well-sort"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <WellSortPage />
            </Suspense>
          }
        />
        <Route
          path="/well-database"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <WellDatabasePage />
            </Suspense>
          }
        />
        <Route
          path="/optimize"
          element={
            <Suspense fallback={<Spinner label="Loading view" />}>
              <OptimizePage />
            </Suspense>
          }
        />
        {/* The unlock route itself is ALWAYS registered - it cannot sit
            behind the flag it sets. Case-insensitive by default in React
            Router, so /Scott works too. */}
        <Route path="/scott" element={<ScottUnlock />} />
        {/* Locked: /tools/* falls through to the catch-all below, so a
            bookmarked link from a locked browser lands on the Solver rather
            than rendering a page the menu is hiding. */}
        {toolsUnlocked && (
          <Route
            path="/tools"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <ToolsPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/pf-scenario"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <PfScenarioPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/header-impact"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <HeaderImpactPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/jp-calibration"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <JpCalibrationPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/fric-trend"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <FricTrendPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/jp-washout"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <JpWashoutPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/pad-watercut"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <PadWatercutPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/sep-oil-loss"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <SepOilLossPage />
              </Suspense>
            }
          />
        )}
        {toolsUnlocked && (
          <Route
            path="/tools/test-harness"
            element={
              <Suspense fallback={<Spinner label="Loading view" />}>
                <TestHarnessPage />
              </Suspense>
            }
          />
        )}
        <Route path="*" element={<Navigate to="/solver" replace />} />
      </Route>
    </Routes>
  );
}
