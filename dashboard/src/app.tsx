import { Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout";
import { OverviewPage } from "@/pages/overview";
import { LossesPage } from "@/pages/losses";
import { ClustersPage } from "@/pages/clusters";
import { GatePage } from "@/pages/gate";
import { HierarchyPage } from "@/pages/hierarchy";
import { GenerationPage } from "@/pages/generation";
import { AblationsPage } from "@/pages/ablations";
import { EvalsPage } from "@/pages/evals";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/losses" element={<LossesPage />} />
        <Route path="/clusters" element={<ClustersPage />} />
        <Route path="/gate" element={<GatePage />} />
        <Route path="/hierarchy" element={<HierarchyPage />} />
        <Route path="/generation" element={<GenerationPage />} />
        <Route path="/evals" element={<EvalsPage />} />
        <Route path="/ablations" element={<AblationsPage />} />
      </Route>
    </Routes>
  );
}
