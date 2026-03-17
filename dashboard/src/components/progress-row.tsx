interface ProgressRowProps {
  label: string;
  value: string;
  percent: number;
  color?: string;
}

export function ProgressRow({
  label,
  value,
  percent,
  color = "var(--color-accent-blue)",
}: ProgressRowProps) {
  return (
    <div className="progress-row">
      <span className="progress-label">{label}</span>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.min(100, percent)}%`, background: color }}
        />
      </div>
      <span className="progress-value">{value}</span>
    </div>
  );
}
