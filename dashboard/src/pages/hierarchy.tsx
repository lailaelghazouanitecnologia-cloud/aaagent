import { useState, useEffect } from "react";
import { useRuns, useRealtimeMetrics } from "@/lib/hooks";
import { RunSelector } from "@/components/run-selector";
import { MetricCard } from "@/components/metric-card";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

export function HierarchyPage() {
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
        <h2 className="text-xl font-semibold">Hierarchy Analysis</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricCard
          label="Coarse-Fine Alignment"
          value={latest?.coarse_fine_alignment?.toFixed(4) ?? "—"}
          color="text-accent-cyan"
          subtitle={
            latest?.coarse_fine_alignment > 0.8
              ? "Strong alignment"
              : latest?.coarse_fine_alignment
                ? "Weak alignment"
                : undefined
          }
          trend={latest?.coarse_fine_alignment > 0.8 ? "up" : "down"}
        />
        <MetricCard
          label="Hierarchy Balance"
          value={latest?.hierarchy_balance?.toFixed(4) ?? "—"}
          color="text-accent-purple"
        />
        <MetricCard
          label="L_hierarchy"
          value={latest?.loss_hierarchy?.toFixed(4) ?? "—"}
          color="text-accent-amber"
        />
      </div>

      {/* Alignment + Balance over time */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">
          Hierarchy Metrics Over Time
        </h3>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="coarse_fine_alignment" stroke="#06b6d4" strokeWidth={2} dot={false} name="Coarse-Fine Alignment" />
              <Line type="monotone" dataKey="hierarchy_balance" stroke="#a855f7" strokeWidth={2} dot={false} name="Hierarchy Balance" />
              <Line type="monotone" dataKey="loss_hierarchy" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="L_hierarchy" strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[350px] flex items-center justify-center text-zinc-600 text-sm">
            Waiting for metrics...
          </div>
        )}
      </div>
    </div>
  );
}
