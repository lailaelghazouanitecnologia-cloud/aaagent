interface EvalRowProps {
  name: string;
  score: number;
  maxScore?: number;
  delta?: number;
  color?: string;
}

export function EvalRow({
  name,
  score,
  maxScore = 100,
  delta,
  color = "var(--color-accent-blue-hi)",
}: EvalRowProps) {
  const pct = (score / maxScore) * 100;
  const barColor = pct > 60 ? color : "var(--color-accent-orange-hi)";

  return (
    <div className="eval-row">
      <span className="eval-name">{name}</span>
      <span className="eval-score">{score.toFixed(1)}</span>
      <div className="eval-bar">
        <div
          className="eval-fill"
          style={{ width: `${Math.min(100, pct)}%`, background: barColor }}
        />
      </div>
      {delta != null && (
        <span
          className="eval-delta"
          style={{
            color:
              delta > 0
                ? "var(--color-accent-green-hi)"
                : delta < 0
                  ? "var(--color-accent-red-hi)"
                  : "var(--color-t3)",
          }}
        >
          {delta > 0 ? "+" : ""}
          {delta.toFixed(1)}
        </span>
      )}
    </div>
  );
}
