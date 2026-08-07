import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { Spinner } from "./components/ui";
import { Layout } from "./layout/Layout";

// Route-level code splitting: chart-heavy pages load on demand.
const SolverPage = lazy(() => import("./pages/SolverPage"));
const BatchPage = lazy(() => import("./pages/BatchPage"));
const PfRangePage = lazy(() => import("./pages/PfRangePage"));
const PressureProfilePage = lazy(() => import("./pages/PressureProfilePage"));
const WellProfilePage = lazy(() => import("./pages/WellProfilePage"));
const EquivalentsPage = lazy(() => import("./pages/EquivalentsPage"));
const JpHistoryPage = lazy(() => import("./pages/JpHistoryPage"));
const WellDatabasePage = lazy(() => import("./pages/WellDatabasePage"));
const WellSortPage = lazy(() => import("./pages/WellSortPage"));
const OptimizePage = lazy(() => import("./pages/OptimizePage"));

export default function App() {
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
        <Route path="*" element={<Navigate to="/solver" replace />} />
      </Route>
    </Routes>
  );
}
