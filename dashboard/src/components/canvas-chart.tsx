import { useRef, useEffect } from "react";

interface DataSet {
  data: number[];
  color: string;
  width?: number;
  opacity?: number;
  dash?: number[];
  area?: boolean;
  glow?: boolean;
}

interface CanvasChartProps {
  datasets: DataSet[];
  yMin?: number;
  yMax?: number;
  yDecimals?: number;
  height?: number;
  className?: string;
}

function drawChart(
  canvas: HTMLCanvasElement,
  sets: DataSet[],
  opts: { yMin: number; yMax: number; dec: number },
) {
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
  const pad = { t: 6, r: 6, b: 14, l: 32 };
  const cw = w - pad.l - pad.r;
  const ch = h - pad.t - pad.b;
  const { yMin, yMax, dec } = opts;

  // Grid lines
  ctx.strokeStyle = "#141418";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ch / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + cw, y);
    ctx.stroke();
  }

  // Y labels
  ctx.font = '8px "Geist Mono", monospace';
  ctx.fillStyle = "#28282f";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = yMax - ((yMax - yMin) / 4) * i;
    ctx.fillText(v.toFixed(dec), pad.l - 4, pad.t + (ch / 4) * i + 3);
  }

  // Draw datasets
  for (const s of sets) {
    const n = s.data.length;
    if (n < 2) continue;

    const getXY = (i: number) => ({
      x: pad.l + (cw / (n - 1)) * i,
      y: pad.t + ch - ((s.data[i] - yMin) / (yMax - yMin)) * ch,
    });

    // Area fill
    if (s.area) {
      ctx.fillStyle = s.color;
      ctx.globalAlpha = 0.06;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const { x, y } = getXY(i);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.lineTo(pad.l + cw, pad.t + ch);
      ctx.lineTo(pad.l, pad.t + ch);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Glow effect
    if (s.glow) {
      ctx.strokeStyle = s.color;
      ctx.globalAlpha = 0.12;
      ctx.lineWidth = 4;
      ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const { x, y } = getXY(i);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Main line
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width ?? 1.5;
    ctx.globalAlpha = s.opacity ?? 1;
    ctx.setLineDash(s.dash ?? []);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const { x, y } = getXY(i);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }
}

export function CanvasChart({
  datasets,
  yMin = 0,
  yMax = 1,
  yDecimals = 1,
  height = 200,
  className = "",
}: CanvasChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      drawChart(canvasRef.current, datasets, { yMin, yMax, dec: yDecimals });
    }
  }, [datasets, yMin, yMax, yDecimals]);

  // Redraw on resize
  useEffect(() => {
    const handler = () => {
      if (canvasRef.current) {
        drawChart(canvasRef.current, datasets, { yMin, yMax, dec: yDecimals });
      }
    };
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, [datasets, yMin, yMax, yDecimals]);

  return (
    <div className={`chart-body ${className}`} style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
