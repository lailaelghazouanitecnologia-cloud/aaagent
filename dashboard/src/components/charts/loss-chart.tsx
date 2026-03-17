import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const LOSS_COLORS: Record<string, string> = {
  loss_total: "#6366f1",
  loss_diffusion: "#3b82f6",
  loss_balance: "#22c55e",
  loss_diversity: "#f59e0b",
  loss_hierarchy: "#a855f7",
};

interface LossChartProps {
  data: any[];
  lines?: string[];
  height?: number;
}

export function LossChart({
  data,
  lines = ["loss_total", "loss_diffusion", "loss_balance", "loss_diversity", "loss_hierarchy"],
  height = 320,
}: LossChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
        <XAxis
          dataKey="step"
          stroke="#71717a"
          tick={{ fill: "#a1a1aa", fontSize: 11 }}
        />
        <YAxis
          stroke="#71717a"
          tick={{ fill: "#a1a1aa", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#27272a",
            border: "1px solid #3f3f46",
            borderRadius: "8px",
            fontSize: "12px",
          }}
          labelStyle={{ color: "#a1a1aa" }}
        />
        <Legend
          wrapperStyle={{ fontSize: "12px", color: "#a1a1aa" }}
        />
        {lines.map((key) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={LOSS_COLORS[key] ?? "#71717a"}
            strokeWidth={2}
            dot={false}
            name={key.replace("loss_", "")}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
