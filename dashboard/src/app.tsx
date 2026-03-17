import { Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout";
import { OverviewPage } from "@/pages/overview";
import { EvalsPage } from "@/pages/evals";
import { VersionsPage } from "@/pages/versions";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/evals" element={<EvalsPage />} />
        <Route path="/versions" element={<VersionsPage />} />
      </Route>
    </Routes>
  );
}
