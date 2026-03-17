import clsx from "clsx";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  color?: string;
}

export function MetricCard({ label, value, subtitle, trend, color }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-surface p-4">
      <p className="text-xs text-zinc-500 uppercase tracking-wide">{label}</p>
      <p
        className={clsx("text-2xl font-bold mt-1", color ?? "text-zinc-100")}
      >
        {value}
      </p>
      {subtitle && (
        <p
          className={clsx(
            "text-xs mt-1",
            trend === "up" && "text-accent-green",
            trend === "down" && "text-accent-red",
            (!trend || trend === "neutral") && "text-zinc-500",
          )}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}
