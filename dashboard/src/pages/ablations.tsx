import { useState, useEffect, useMemo } from "react";
import { useRuns } from "@/lib/hooks";
import { fetchComparison } from "@/lib/api";
import { LossChart } from "@/components/charts/loss-chart";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
} from "@tanstack/react-table";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const RUN_COLORS = [
  "#6366f1", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ef4444", "#ec4899",
];

interface RunStats {
  run: string;
  steps: number;
  best_loss: number;
  final_loss: number;
  avg_throughput: number;
  best_entropy: number;
  final_dead_clusters: number;
}

const columnHelper = createColumnHelper<RunStats>();

const columns = [
  columnHelper.accessor("run", { header: "Run" }),
  columnHelper.accessor("steps", { header: "Steps" }),
  columnHelper.accessor("best_loss", {
    header: "Best Loss",
    cell: (info) => info.getValue()?.toFixed(4) ?? "—",
  }),
  columnHelper.accessor("final_loss", {
    header: "Final Loss",
    cell: (info) => info.getValue()?.toFixed(4) ?? "—",
  }),
  columnHelper.accessor("avg_throughput", {
    header: "Avg tok/s",
    cell: (info) => info.getValue()?.toFixed(0) ?? "—",
  }),
  columnHelper.accessor("best_entropy", {
    header: "Best Entropy",
    cell: (info) => info.getValue()?.toFixed(4) ?? "—",
  }),
  columnHelper.accessor("final_dead_clusters", {
    header: "Dead Clusters (final)",
    cell: (info) => info.getValue() ?? "—",
  }),
];

export function AblationsPage() {
  const runs = useRuns();
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);

  useEffect(() => {
    if (runs.length && selectedRuns.length === 0) {
      setSelectedRuns(runs.slice(0, 6)); // select first 6 by default
    }
  }, [runs, selectedRuns]);

  useEffect(() => {
    if (selectedRuns.length > 0) {
      fetchComparison(selectedRuns).then(setComparisonData);
    }
  }, [selectedRuns]);

  // Compute table data
  const tableData: RunStats[] = useMemo(() => {
    const grouped: Record<string, any[]> = {};
    for (const d of comparisonData) {
      (grouped[d.run] ??= []).push(d);
    }

    return Object.entries(grouped).map(([run, metrics]) => {
      const losses = metrics.map((m: any) => m.loss_total).filter(Boolean);
      const throughputs = metrics.map((m: any) => m.tokens_per_sec).filter(Boolean);
      const entropies = metrics.map((m: any) => m.entropy_ratio).filter(Boolean);
      const last = metrics[metrics.length - 1];

      return {
        run,
        steps: metrics.length,
        best_loss: losses.length ? Math.min(...losses) : 0,
        final_loss: last?.loss_total ?? 0,
        avg_throughput: throughputs.length ? throughputs.reduce((a: number, b: number) => a + b, 0) / throughputs.length : 0,
        best_entropy: entropies.length ? Math.max(...entropies) : 0,
        final_dead_clusters: last?.dead_clusters ?? 0,
      };
    });
  }, [comparisonData]);

  // Pivot data for overlay chart: merge by step
  const overlayData = useMemo(() => {
    const byStep: Record<number, any> = {};
    for (const d of comparisonData) {
      byStep[d.step] ??= { step: d.step };
      byStep[d.step][`${d.run}_loss`] = d.loss_total;
    }
    return Object.values(byStep).sort((a, b) => a.step - b.step);
  }, [comparisonData]);

  const table = useReactTable({
    data: tableData,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const toggleRun = (run: string) => {
    setSelectedRuns((prev) =>
      prev.includes(run) ? prev.filter((r) => r !== run) : [...prev, run],
    );
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Ablation Comparison</h2>

      {/* Run selection */}
      <div className="flex flex-wrap gap-2">
        {runs.map((run, i) => (
          <button
            key={run}
            onClick={() => toggleRun(run)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              selectedRuns.includes(run)
                ? "border-brand bg-brand/20 text-brand-light"
                : "border-zinc-700 text-zinc-500 hover:border-zinc-500"
            }`}
          >
            {run}
          </button>
        ))}
      </div>

      {/* Comparison table */}
      <div className="rounded-lg border border-zinc-800 bg-surface overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-zinc-800">
                {hg.headers.map((header) => (
                  <th key={header.id} className="text-left px-4 py-2 text-xs text-zinc-500 font-medium">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-zinc-800/50 hover:bg-surface-raised/50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-2 text-zinc-300">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Overlayed loss curves */}
      <div className="rounded-lg border border-zinc-800 bg-surface p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">
          Loss Comparison (Overlay)
        </h3>
        {overlayData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={overlayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              {selectedRuns.map((run, i) => (
                <Line
                  key={run}
                  type="monotone"
                  dataKey={`${run}_loss`}
                  stroke={RUN_COLORS[i % RUN_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  name={run}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[400px] flex items-center justify-center text-zinc-600 text-sm">
            Select runs to compare.
          </div>
        )}
      </div>
    </div>
  );
}
