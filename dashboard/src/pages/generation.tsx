import { useState, useEffect } from "react";
import { useRuns } from "@/lib/hooks";
import { RunSelector } from "@/components/run-selector";
import * as ScrollArea from "@radix-ui/react-scroll-area";

interface Sample {
  step: number;
  text: string;
  prompt?: string;
  metrics?: Record<string, number>;
}

export function GenerationPage() {
  const runs = useRuns();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    if (runs.length && !selectedRun) setSelectedRun(runs[0]);
  }, [runs, selectedRun]);

  useEffect(() => {
    if (!selectedRun) return;
    // Fetch generation samples from extra data
    fetch(`/api/runs/${encodeURIComponent(selectedRun)}/metrics?limit=10000`)
      .then((r) => r.json())
      .then((data: any[]) => {
        const parsed: Sample[] = [];
        for (const row of data) {
          if (!row.extra) continue;
          try {
            const extra = typeof row.extra === "string" ? JSON.parse(row.extra) : row.extra;
            if (extra.generated_sample) {
              parsed.push({
                step: row.step,
                text: extra.generated_sample,
                prompt: extra.prompt,
                metrics: extra.sample_metrics,
              });
            }
          } catch { /* skip */ }
        }
        setSamples(parsed);
      });
  }, [selectedRun]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Generation Samples</h2>
        <RunSelector runs={runs} value={selectedRun} onChange={setSelectedRun} />
      </div>

      {samples.length === 0 ? (
        <div className="rounded-lg border border-zinc-800 bg-surface p-8 text-center">
          <p className="text-zinc-500">
            No generation samples yet. Send samples via the <code className="text-brand-light">extra.generated_sample</code> field.
          </p>
        </div>
      ) : (
        <ScrollArea.Root className="h-[calc(100vh-180px)]">
          <ScrollArea.Viewport className="w-full h-full">
            <div className="space-y-4">
              {samples.map((sample, i) => (
                <div key={i} className="rounded-lg border border-zinc-800 bg-surface p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xs font-mono bg-surface-raised px-2 py-0.5 rounded text-zinc-400">
                      Step {sample.step}
                    </span>
                    {sample.metrics &&
                      Object.entries(sample.metrics).map(([k, v]) => (
                        <span key={k} className="text-xs text-zinc-500">
                          {k}: {typeof v === "number" ? v.toFixed(3) : v}
                        </span>
                      ))}
                  </div>
                  {sample.prompt && (
                    <p className="text-xs text-zinc-500 mb-2">
                      <strong>Prompt:</strong> {sample.prompt}
                    </p>
                  )}
                  <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed bg-zinc-900/50 rounded p-3">
                    {sample.text}
                  </pre>
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
      )}
    </div>
  );
}
