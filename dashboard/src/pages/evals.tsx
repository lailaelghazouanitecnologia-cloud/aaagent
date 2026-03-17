import { useState, useEffect, useMemo } from "react";
import { useRuns } from "@/lib/hooks";
import { fetchEvals } from "@/lib/api";
import { RunSelector } from "@/components/run-selector";
import { MetricCard } from "@/components/metric-card";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell,
} from "recharts";
import * as ScrollArea from "@radix-ui/react-scroll-area";

// Dimension colors
const DIM_COLORS: Record<string, string> = {
  coherence: "#818cf8",
  grammar: "#34d399",
  relevance: "#60a5fa",
  creativity: "#f472b6",
  fluency: "#a78bfa",
  completeness: "#fbbf24",
};

const FAILURE_COLORS: Record<string, string> = {
  none: "#22c55e",
  repetition_loop: "#ef4444",
  nonsense: "#f59e0b",
  truncated: "#6366f1",
  copied: "#ec4899",
  off_topic: "#94a3b8",
};

function fmt(v: number | null | undefined, d = 3): string {
  return v != null ? v.toFixed(d) : "—";
}

export function EvalsPage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [evals, setEvals] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);
    fetchEvals(selectedRun)
      .then(setEvals)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedRun]);

  const latest = evals[evals.length - 1];

  // Radar chart data for LLM scores
  const radarData = useMemo(() => {
    if (!latest) return [];
    return Object.keys(DIM_COLORS).map((dim) => ({
      dimension: dim.charAt(0).toUpperCase() + dim.slice(1),
      score: latest[`llm_${dim}`] ?? 0,
      fullMark: 5,
    }));
  }, [latest]);

  // Failure mode distribution
  const failureModes = useMemo(() => {
    if (!latest?.failure_modes) return [];
    try {
      const modes = typeof latest.failure_modes === "string"
        ? JSON.parse(latest.failure_modes)
        : latest.failure_modes;
      return Object.entries(modes).map(([name, count]) => ({
        name,
        count: count as number,
        fill: FAILURE_COLORS[name] ?? "#71717a",
      }));
    } catch {
      return [];
    }
  }, [latest]);

  // Per-category breakdown
  const categoryData = useMemo(() => {
    if (!latest?.by_category) return [];
    try {
      const cats = typeof latest.by_category === "string"
        ? JSON.parse(latest.by_category)
        : latest.by_category;
      return Object.entries(cats).map(([name, data]: [string, any]) => ({
        name,
        distinct_2: data.mean_distinct_2 ?? 0,
        repetition: data.mean_repetition ?? 0,
        keyword_hit: data.mean_keyword_hit ?? 0,
        count: data.count ?? 0,
      }));
    } catch {
      return [];
    }
  }, [latest]);

  // Samples from latest eval
  const samples = useMemo(() => {
    if (!latest?.samples) return [];
    try {
      return typeof latest.samples === "string"
        ? JSON.parse(latest.samples)
        : latest.samples;
    } catch {
      return [];
    }
  }, [latest]);

  // Auto metrics evolution over time
  const autoTimeline = useMemo(() => {
    return evals.map((e) => ({
      step: e.step,
      distinct_2: e.distinct_2,
      repetition: e.repetition_ratio,
      self_bleu: e.self_bleu_4,
      vocab_richness: e.vocab_richness,
      llm_overall: e.llm_overall,
    }));
  }, [evals]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Evaluation Benchmark</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      {loading && (
        <p className="text-sm text-zinc-500">Loading eval data...</p>
      )}

      {!loading && evals.length === 0 && (
        <div className="rounded-lg border border-zinc-800 bg-surface p-8 text-center">
          <p className="text-zinc-500">
            No eval results yet. Run <code className="text-brand-light">python -m eval.agent_eval --checkpoint &lt;path&gt;</code> to generate.
          </p>
          <p className="text-zinc-600 text-xs mt-2">
            Results are sent to <code className="text-zinc-500">POST /api/evals</code>
          </p>
        </div>
      )}

      {latest && (
        <>
          {/* KPI cards — auto metrics */}
          <div>
            <h3 className="text-sm font-medium text-zinc-400 mb-3">Auto Metrics (deterministic)</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <MetricCard
                label="Distinct-2"
                value={fmt(latest.distinct_2)}
                subtitle={latest.distinct_2 > 0.7 ? "Diverse" : latest.distinct_2 > 0.4 ? "Moderate" : "Low"}
                trend={latest.distinct_2 > 0.7 ? "up" : latest.distinct_2 > 0.4 ? "neutral" : "down"}
                color="text-brand-light"
              />
              <MetricCard
                label="Repetition"
                value={fmt(latest.repetition_ratio)}
                subtitle={latest.repetition_ratio < 0.1 ? "Clean" : latest.repetition_ratio < 0.3 ? "Some" : "High"}
                trend={latest.repetition_ratio < 0.1 ? "up" : "down"}
                color={latest.repetition_ratio < 0.1 ? "text-accent-green" : "text-accent-red"}
              />
              <MetricCard
                label="Self-BLEU-4"
                value={fmt(latest.self_bleu_4)}
                subtitle="Lower = more diverse"
                trend={latest.self_bleu_4 < 0.2 ? "up" : "down"}
              />
              <MetricCard
                label="Keyword Hit"
                value={fmt(latest.keyword_hit)}
                subtitle={latest.keyword_hit > 0.8 ? "Strong" : "Weak"}
                trend={latest.keyword_hit > 0.8 ? "up" : "down"}
              />
              <MetricCard
                label="Vocab Richness"
                value={fmt(latest.vocab_richness)}
              />
            </div>
          </div>

          {/* KPI cards — LLM judge */}
          {latest.llm_overall != null && (
            <div>
              <h3 className="text-sm font-medium text-zinc-400 mb-3">LLM Judge (Groq)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard
                  label="Overall Quality"
                  value={`${fmt(latest.llm_overall, 2)}/5`}
                  color={latest.llm_overall >= 3.5 ? "text-accent-green" : latest.llm_overall >= 2.5 ? "text-brand-light" : "text-accent-red"}
                />
                <MetricCard label="Coherence" value={`${fmt(latest.llm_coherence, 2)}/5`} />
                <MetricCard label="Creativity" value={`${fmt(latest.llm_creativity, 2)}/5`} />
                <MetricCard label="Grammar" value={`${fmt(latest.llm_grammar, 2)}/5`} />
              </div>
            </div>
          )}

          {/* Charts row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Radar: LLM dimensions */}
            {radarData.length > 0 && radarData.some((d) => d.score > 0) && (
              <div className="rounded-lg border border-zinc-800 bg-surface p-4">
                <h3 className="text-sm font-medium text-zinc-400 mb-3">LLM Quality Radar</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#3f3f46" />
                    <PolarAngleAxis dataKey="dimension" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                    <PolarRadiusAxis domain={[0, 5]} tick={{ fill: "#71717a", fontSize: 10 }} />
                    <Radar
                      dataKey="score"
                      stroke="#818cf8"
                      fill="#818cf8"
                      fillOpacity={0.25}
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Failure modes bar chart */}
            {failureModes.length > 0 && (
              <div className="rounded-lg border border-zinc-800 bg-surface p-4">
                <h3 className="text-sm font-medium text-zinc-400 mb-3">Failure Modes</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={failureModes}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis dataKey="name" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                    <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {failureModes.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Per-category breakdown */}
          {categoryData.length > 0 && (
            <div className="rounded-lg border border-zinc-800 bg-surface p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Per-Category Breakdown</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={categoryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="name" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <YAxis domain={[0, 1]} stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
                  <Bar dataKey="distinct_2" fill="#818cf8" name="Distinct-2" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="keyword_hit" fill="#34d399" name="Keyword Hit" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="repetition" fill="#ef4444" name="Repetition" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Timeline: metrics over training steps */}
          {autoTimeline.length > 1 && (
            <div className="rounded-lg border border-zinc-800 bg-surface p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Quality Over Training</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={autoTimeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="step" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <YAxis domain={[0, 'auto']} stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
                  <Line type="monotone" dataKey="distinct_2" stroke="#818cf8" strokeWidth={2} dot={false} name="Distinct-2" />
                  <Line type="monotone" dataKey="repetition" stroke="#ef4444" strokeWidth={2} dot={false} name="Repetition" />
                  <Line type="monotone" dataKey="vocab_richness" stroke="#34d399" strokeWidth={2} dot={false} name="Vocab Rich." />
                  {autoTimeline.some((d) => d.llm_overall != null) && (
                    <Line type="monotone" dataKey="llm_overall" stroke="#fbbf24" strokeWidth={2} dot name="LLM Overall" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Sample browser */}
          {samples.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-zinc-400 mb-3">
                Samples ({samples.length})
              </h3>
              <ScrollArea.Root className="h-[400px]">
                <ScrollArea.Viewport className="w-full h-full">
                  <div className="space-y-3">
                    {samples.map((s: any, i: number) => (
                      <div key={i} className="rounded-lg border border-zinc-800 bg-surface p-4">
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                          <span className="text-xs font-mono bg-surface-raised px-2 py-0.5 rounded text-zinc-400">
                            {s.id ?? `#${i + 1}`}
                          </span>
                          <span className="text-xs text-zinc-500">{s.category}</span>
                          {s.metrics?.llm_coherence != null && (
                            <span className="text-xs text-indigo-400">
                              coh:{s.metrics.llm_coherence}
                            </span>
                          )}
                          {s.metrics?.llm_creativity != null && (
                            <span className="text-xs text-pink-400">
                              cre:{s.metrics.llm_creativity}
                            </span>
                          )}
                          {s.metrics?.llm_failure_mode && s.metrics.llm_failure_mode !== "none" && (
                            <span className="text-xs bg-red-900/40 text-red-400 px-1.5 py-0.5 rounded">
                              {s.metrics.llm_failure_mode}
                            </span>
                          )}
                          {s.metrics?.repetition_ratio != null && (
                            <span className="text-xs text-zinc-500">
                              rep:{fmt(s.metrics.repetition_ratio)}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-zinc-500 mb-1">
                          <strong>Prompt:</strong> {s.prompt}
                        </p>
                        <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed bg-zinc-900/50 rounded p-3">
                          {s.generated}
                        </pre>
                        {s.metrics?.llm_reasoning && (
                          <p className="text-xs text-zinc-500 mt-2 italic">
                            {s.metrics.llm_reasoning}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea.Viewport>
                <ScrollArea.Scrollbar
                  orientation="vertical"
                  className="flex select-none touch-none p-0.5 bg-zinc-900 transition-colors w-2 rounded-full"
                >
                  <ScrollArea.Thumb className="flex-1 bg-zinc-700 rounded-full relative" />
                </ScrollArea.Scrollbar>
              </ScrollArea.Root>
            </div>
          )}
        </>
      )}
    </div>
  );
}
