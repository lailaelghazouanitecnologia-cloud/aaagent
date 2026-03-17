import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";
import { HeatmapRect } from "@visx/heatmap";
import { Tooltip, useTooltip, defaultStyles } from "@visx/tooltip";

interface ClusterHeatmapProps {
  /** Matrix of similarity values, e.g. centroid_similarity[i][j] */
  matrix: number[][];
  width?: number;
  height?: number;
  labels?: string[];
}

export function ClusterHeatmap({
  matrix,
  width = 400,
  height = 400,
  labels,
}: ClusterHeatmapProps) {
  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip,
  } = useTooltip<{ row: number; col: number; value: number }>();

  if (!matrix.length) {
    return (
      <p className="text-sm text-zinc-500 italic p-4">
        No cluster similarity data available yet.
      </p>
    );
  }

  const n = matrix.length;
  const margin = { top: 30, left: 40, right: 10, bottom: 10 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const cellW = innerW / n;
  const cellH = innerH / n;

  const allValues = matrix.flat();
  const colorScale = scaleLinear<string>({
    domain: [Math.min(...allValues), Math.max(...allValues)],
    range: ["#18181b", "#6366f1"],
  });

  const bins = matrix.map((row, i) => ({
    bin: i,
    bins: row.map((val, j) => ({ bin: j, count: val })),
  }));

  return (
    <div className="relative">
      <svg width={width} height={height}>
        <Group top={margin.top} left={margin.left}>
          <HeatmapRect
            data={bins}
            xScale={scaleLinear({ domain: [0, n], range: [0, innerW] })}
            yScale={scaleLinear({ domain: [0, n], range: [0, innerH] })}
            colorScale={colorScale}
            binWidth={cellW}
            binHeight={cellH}
            gap={1}
          >
            {(heatmap) =>
              heatmap.map((heatmapBins) =>
                heatmapBins.map((bin) => (
                  <rect
                    key={`heatmap-rect-${bin.row}-${bin.column}`}
                    x={bin.x}
                    y={bin.y}
                    width={bin.width}
                    height={bin.height}
                    fill={bin.color as string}
                    rx={2}
                    className="cursor-pointer"
                    onMouseEnter={() => {
                      showTooltip({
                        tooltipData: {
                          row: bin.row,
                          col: bin.column,
                          value: matrix[bin.row]?.[bin.column] ?? 0,
                        },
                        tooltipLeft: margin.left + (bin.x ?? 0) + cellW / 2,
                        tooltipTop: margin.top + (bin.y ?? 0),
                      });
                    }}
                    onMouseLeave={hideTooltip}
                  />
                )),
              )
            }
          </HeatmapRect>
          {/* Labels */}
          {labels?.map((label, i) => (
            <text
              key={`label-${i}`}
              x={i * cellW + cellW / 2}
              y={-8}
              textAnchor="middle"
              fontSize={10}
              fill="#71717a"
            >
              {label}
            </text>
          ))}
        </Group>
      </svg>
      {tooltipOpen && tooltipData && (
        <Tooltip
          top={tooltipTop}
          left={tooltipLeft}
          style={{
            ...defaultStyles,
            background: "#27272a",
            border: "1px solid #3f3f46",
            color: "#e4e4e7",
            fontSize: "12px",
            borderRadius: "6px",
            padding: "6px 10px",
          }}
        >
          <div>
            [{tooltipData.row}, {tooltipData.col}]:{" "}
            <strong>{tooltipData.value.toFixed(4)}</strong>
          </div>
        </Tooltip>
      )}
    </div>
  );
}
