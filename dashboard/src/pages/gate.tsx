import { useState, useEffect } from "react";
import { useRuns, useRealtimeMetrics } from "@/lib/hooks";
import { RunSelector } from "@/components/run-selector";
import { MetricCard } from "@/components/metric-card";
import { GateHistogram, GateTimeSeries } from "@/components/charts/gate-histogram";

export function GatePage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const data = useRealtimeMetrics(selectedRun);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  const latest = data[data.length - 1];

  // Parse gate distribution from extra if available
  const gateDistribution: number[] = (() => {
    if (!latest?.extra) return [];
    try {
      const extra = typeof latest.extra === "string" ? JSON.parse(latest.extra) : latest.extra;
      return extra.gate_distribution ?? [];
    } catch {
      return [];
    }
  })();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Gate Analysis</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Gate Mean"
          value={latest?.gate_mean?.toFixed(4) ?? "—"}
          color="text-brand-light"
        />
        <MetricCard
          label="Gate Std"
          value={latest?.gate_std?.toFixed(4) ?? "—"}
          color="text-accent-purple"
        />
        <MetricCard
          label="Mean/Std Ratio"
          value={
            latest?.gate_mean && latest?.gate_std
              ? (latest.gate_mean / latest.gate_std).toFixed(2)
              : "—"
          }
        />
        <MetricCard
          label="Current Step"
          value={latest?.step ?? "—"}
        />
      </div>

      {/* Gate mean/std over time */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">
          Gate Evolution Over Time
        </h3>
        {data.length > 0 ? (
          <GateTimeSeries data={data} height={300} />
        ) : (
          <div className="h-[300px] flex items-center justify-center text-zinc-600 text-sm">
            Waiting for metrics...
          </div>
        )}
      </div>

      {/* Distribution histogram */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">
          Gate Activation Distribution (Latest Step)
        </h3>
        {gateDistribution.length > 0 ? (
          <GateHistogram distribution={gateDistribution} height={300} />
        ) : (
          <div className="h-[300px] flex items-center justify-center text-zinc-600 text-sm">
            Send gate_distribution in extra field to see histogram.
          </div>
        )}
      </div>
    </div>
  );
}
