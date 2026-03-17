import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

interface GateHistogramProps {
  /** Array of gate activation values for a single step */
  distribution?: number[];
  /** Time series of gate mean/std over steps */
  timeSeries?: Array<{ step: number; gate_mean: number; gate_std: number }>;
  height?: number;
}

export function GateHistogram({ distribution, height = 280 }: { distribution: number[]; height?: number }) {
  if (!distribution?.length) {
    return <p className="text-sm text-zinc-500 italic p-4">No gate distribution data.</p>;
  }

  // Bin the distribution into a histogram
  const numBins = 30;
  const min = Math.min(...distribution);
  const max = Math.max(...distribution);
  const binWidth = (max - min) / numBins || 1;

  const bins = Array.from({ length: numBins }, (_, i) => ({
    range: `${(min + i * binWidth).toFixed(2)}`,
    count: 0,
  }));

  for (const val of distribution) {
    const idx = Math.min(Math.floor((val - min) / binWidth), numBins - 1);
    bins[idx].count++;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis dataKey="range" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 10 }} interval="preserveStartEnd" />
        <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px", fontSize: "12px" }}
        />
        <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function GateTimeSeries({ data, height = 280 }: { data: Array<{ step: number; gate_mean: number; gate_std: number }>; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
        <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px", fontSize: "12px" }}
        />
        <Area type="monotone" dataKey="gate_mean" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} strokeWidth={2} />
        <Area type="monotone" dataKey="gate_std" stroke="#a855f7" fill="#a855f7" fillOpacity={0.1} strokeWidth={1.5} strokeDasharray="4 2" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
