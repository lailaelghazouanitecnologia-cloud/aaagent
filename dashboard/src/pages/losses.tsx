import { useState, useEffect } from "react";
import { useRuns, useRealtimeMetrics } from "@/lib/hooks";
import { RunSelector } from "@/components/run-selector";
import { LossChart } from "@/components/charts/loss-chart";
import { MetricCard } from "@/components/metric-card";

const INDIVIDUAL_LOSSES = [
  { key: "loss_diffusion", label: "Diffusion", color: "text-accent-blue" },
  { key: "loss_balance", label: "Balance", color: "text-accent-green" },
  { key: "loss_diversity", label: "Diversity", color: "text-accent-amber" },
  { key: "loss_hierarchy", label: "Hierarchy", color: "text-accent-purple" },
];

export function LossesPage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const data = useRealtimeMetrics(selectedRun);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  const latest = data[data.length - 1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Loss Analysis</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      {/* Current values */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MetricCard
          label="Total"
          value={latest?.loss_total?.toFixed(4) ?? "—"}
          color="text-brand-light"
        />
        {INDIVIDUAL_LOSSES.map(({ key, label, color }) => (
          <MetricCard
            key={key}
            label={label}
            value={latest?.[key]?.toFixed(4) ?? "—"}
            color={color}
          />
        ))}
      </div>

      {/* Combined chart */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">All Losses</h3>
        <LossChart data={data} height={350} />
      </div>

      {/* Individual charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {INDIVIDUAL_LOSSES.map(({ key, label }) => (
          <div key={key} className="rounded-lg border border-zinc-800 bg-surface p-4">
            <h3 className="text-sm font-medium text-zinc-400 mb-3">
              L_{label}
            </h3>
            <LossChart data={data} lines={[key]} height={220} />
          </div>
        ))}
      </div>
    </div>
  );
}
