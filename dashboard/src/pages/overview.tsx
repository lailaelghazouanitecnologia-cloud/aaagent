import { useState, useEffect } from "react";
import { useRuns, useRealtimeMetrics } from "@/lib/hooks";
import { fetchRunSummary } from "@/lib/api";
import { RunSelector } from "@/components/run-selector";
import { MetricCard } from "@/components/metric-card";
import { LossChart } from "@/components/charts/loss-chart";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

export function OverviewPage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [summary, setSummary] = useState<any>(null);
  const data = useRealtimeMetrics(selectedRun);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  useEffect(() => {
    if (selectedRun) {
      fetchRunSummary(selectedRun).then(setSummary);
    }
  }, [selectedRun]);

  const latest = data[data.length - 1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Overview</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Current Step"
          value={latest?.step ?? summary?.last_step ?? "—"}
        />
        <MetricCard
          label="Total Loss"
          value={latest?.loss_total?.toFixed(4) ?? "—"}
          color="text-brand-light"
        />
        <MetricCard
          label="Best Loss"
          value={summary?.best_loss?.toFixed(4) ?? "—"}
          color="text-accent-green"
        />
        <MetricCard
          label="Throughput"
          value={
            latest?.tokens_per_sec
              ? `${(latest.tokens_per_sec / 1000).toFixed(1)}k tok/s`
              : "—"
          }
        />
        <MetricCard
          label="Entropy Ratio"
          value={latest?.entropy_ratio?.toFixed(3) ?? "—"}
          subtitle={
            latest?.entropy_ratio > 0.8
              ? "Healthy"
              : latest?.entropy_ratio
                ? "Low"
                : undefined
          }
          trend={latest?.entropy_ratio > 0.8 ? "up" : "down"}
        />
        <MetricCard
          label="Dead Clusters"
          value={latest?.dead_clusters ?? "—"}
          trend={latest?.dead_clusters === 0 ? "up" : "down"}
          subtitle={latest?.dead_clusters === 0 ? "None" : undefined}
        />
        <MetricCard
          label="Gate Mean"
          value={latest?.gate_mean?.toFixed(4) ?? "—"}
        />
        <MetricCard
          label="GPU Memory"
          value={
            latest?.gpu_memory_gb
              ? `${latest.gpu_memory_gb.toFixed(1)} GB`
              : "—"
          }
        />
      </div>

      {/* Loss curves */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Loss Curves</h3>
        {data.length > 0 ? (
          <LossChart data={data} />
        ) : (
          <div className="h-[320px] flex items-center justify-center text-zinc-600 text-sm">
            Waiting for metrics...
          </div>
        )}
      </div>

      {/* Throughput */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Throughput</h3>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
              <Line type="monotone" dataKey="tokens_per_sec" stroke="#22c55e" strokeWidth={2} dot={false} name="tokens/sec" />
              <Line type="monotone" dataKey="gpu_memory_gb" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="GPU mem (GB)" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-zinc-600 text-sm">
            Waiting for metrics...
          </div>
        )}
      </div>
    </div>
  );
}
