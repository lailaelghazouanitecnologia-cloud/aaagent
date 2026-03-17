import { useRef, useEffect } from "react";

interface MetricCellProps {
  label: string;
  value: string | number;
  unit?: string;
  delta?: string;
  deltaDir?: "down" | "up" | "flat";
  sub?: string;
  sparkData?: number[];
  sparkColor?: string;
}

function drawSparkline(canvas: HTMLCanvasElement, data: number[], color: string) {
  const r = canvas.parentElement?.getBoundingClientRect();
  if (!r) return;
  const dpr = devicePixelRatio || 1;
  canvas.width = r.width * dpr;
  canvas.height = r.height * dpr;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  const w = r.width;
  const h = r.height;
  const mn = Math.min(...data);
  const mx = Math.max(...data);
  const range = mx - mn || 1;

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  data.forEach((v, i) => {
    const px = (w / (data.length - 1)) * i;
    const py = h - ((v - mn) / range) * (h - 2) - 1;
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
}

export function MetricCell({
  label,
  value,
  unit,
  delta,
  deltaDir = "flat",
  sub,
  sparkData,
  sparkColor = "#5ec46a",
}: MetricCellProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (sparkData && sparkData.length > 1 && canvasRef.current) {
      drawSparkline(canvasRef.current, sparkData, sparkColor);
    }
  }, [sparkData, sparkColor]);

  return (
    <div className="metric-cell">
      <span className="metric-label">{label}</span>
      <div className="metric-row">
        <span className="metric-value">{value}</span>
        {unit && <span className="metric-unit">{unit}</span>}
        {delta && (
          <span className={`metric-delta ${deltaDir}`}>{delta}</span>
        )}
      </div>
      {sub && <span className="metric-sub">{sub}</span>}
      {sparkData && sparkData.length > 1 && (
        <div className="sparkline">
          <canvas ref={canvasRef} />
        </div>
      )}
    </div>
  );
}
