import { useState, useEffect } from "react";
import { useRuns, useRealtimeMetrics } from "@/lib/hooks";
import { RunSelector } from "@/components/run-selector";
import { MetricCard } from "@/components/metric-card";
import { ClusterHeatmap } from "@/components/charts/cluster-heatmap";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

export function ClustersPage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const data = useRealtimeMetrics(selectedRun);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  const latest = data[data.length - 1];

  // Parse similarity matrix from extra field if available
  const similarityMatrix: number[][] = latest?.extra
    ? (() => {
        try {
          const extra = typeof latest.extra === "string" ? JSON.parse(latest.extra) : latest.extra;
          return extra.centroid_similarity_matrix ?? [];
        } catch {
          return [];
        }
      })()
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Cluster Health</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricCard
          label="Entropy Ratio"
          value={latest?.entropy_ratio?.toFixed(4) ?? "—"}
          subtitle={latest?.entropy_ratio > 0.85 ? "Excellent" : latest?.entropy_ratio > 0.7 ? "Good" : "Low"}
          trend={latest?.entropy_ratio > 0.8 ? "up" : "down"}
          color="text-accent-cyan"
        />
        <MetricCard
          label="Dead Clusters"
          value={latest?.dead_clusters ?? "—"}
          trend={latest?.dead_clusters === 0 ? "up" : "down"}
          color={latest?.dead_clusters === 0 ? "text-accent-green" : "text-accent-red"}
        />
        <MetricCard
          label="Centroid Similarity"
          value={latest?.centroid_similarity?.toFixed(4) ?? "—"}
          color="text-accent-purple"
        />
      </div>

      {/* Entropy + Dead clusters over time */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Cluster Health Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis yAxisId="left" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
            <Legend wrapperStyle={{ fontSize: "12px" }} />
            <Line yAxisId="left" type="monotone" dataKey="entropy_ratio" stroke="#06b6d4" strokeWidth={2} dot={false} name="Entropy Ratio" />
            <Line yAxisId="right" type="stepAfter" dataKey="dead_clusters" stroke="#ef4444" strokeWidth={2} dot={false} name="Dead Clusters" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Heatmap */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">
          Centroid Similarity Matrix
        </h3>
        <ClusterHeatmap matrix={similarityMatrix} width={500} height={500} />
      </div>
    </div>
  );
}
