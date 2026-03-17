import { useState, useEffect, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

interface Version {
  id: string;
  step: number;
  path: string;
  run: string;
  config: string;
  loss: number | null;
  ppl: number | null;
  entropy: number | null;
  gate_mean: number | null;
  created: string;
  size_mb: number;
  note: string;
}

function fmt(v: number | null, d = 4): string {
  return v != null ? v.toFixed(d) : "—";
}

export function VersionsPage() {
  const [versions, setVersions] = useState<Version[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/versions")
      .then((r) => r.json())
      .then((data) => {
        setVersions(data.versions ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const best = useMemo(() => {
    const withLoss = versions.filter((v) => v.loss != null);
    if (!withLoss.length) return null;
    return withLoss.reduce((a, b) => (a.loss! < b.loss! ? a : b));
  }, [versions]);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const comparisonData = useMemo(() => {
    if (selected.size < 2) return null;
    return versions
      .filter((v) => selected.has(v.id))
      .sort((a, b) => a.step - b.step);
  }, [versions, selected]);

  const timeline = useMemo(() => {
    return versions
      .filter((v) => v.loss != null)
      .sort((a, b) => a.step - b.step)
      .map((v) => ({
        step: v.step,
        loss: v.loss,
        ppl: v.ppl,
        entropy: v.entropy,
        label: v.id,
      }));
  }, [versions]);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Model Versions</h2>

      {loading && (
        <p className="text-sm text-zinc-500">Loading...</p>
      )}

      {!loading && versions.length === 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-8 text-center">
          <p className="text-zinc-500">
            No versions registered. Train a model: <code className="text-cyan-400">z86 train</code>
          </p>
        </div>
      )}

      {versions.length > 0 && (
        <>
          {/* Version table */}
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-x-auto">
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="border-b border-zinc-800 text-xs text-zinc-500 uppercase tracking-wider">
                  <th className="text-left px-4 py-3 w-8"></th>
                  <th className="text-left px-4 py-3">Version</th>
                  <th className="text-right px-4 py-3">Step</th>
                  <th className="text-right px-4 py-3">Loss</th>
                  <th className="text-right px-4 py-3">PPL</th>
                  <th className="text-right px-4 py-3">Entropy</th>
                  <th className="text-left px-4 py-3">Run</th>
                  <th className="text-right px-4 py-3">Size</th>
                  <th className="text-left px-4 py-3">Date</th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => {
                  const isBest = best?.id === v.id;
                  return (
                    <tr
                      key={v.id}
                      onClick={() => toggleSelect(v.id)}
                      className={`cursor-pointer border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors ${
                        selected.has(v.id) ? "bg-zinc-800/40" : ""
                      }`}
                    >
                      <td className="px-4 py-2">
                        <input
                          type="checkbox"
                          checked={selected.has(v.id)}
                          onChange={() => toggleSelect(v.id)}
                          className="accent-blue-500"
                        />
                      </td>
                      <td className={`px-4 py-2 font-medium ${isBest ? "text-green-400" : "text-zinc-200"}`}>
                        {v.id}{isBest && " ★"}
                      </td>
                      <td className="px-4 py-2 text-right text-zinc-200 tabular-nums">{v.step.toLocaleString()}</td>
                      <td className="px-4 py-2 text-right text-zinc-200 tabular-nums">{fmt(v.loss)}</td>
                      <td className="px-4 py-2 text-right text-zinc-400 tabular-nums">{fmt(v.ppl, 2)}</td>
                      <td className="px-4 py-2 text-right text-zinc-400 tabular-nums">{fmt(v.entropy, 3)}</td>
                      <td className="px-4 py-2 text-zinc-500">{v.run}</td>
                      <td className="px-4 py-2 text-right text-zinc-500">{v.size_mb}MB</td>
                      <td className="px-4 py-2 text-zinc-600">{v.created?.slice(0, 10) ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Comparison */}
          {comparisonData && comparisonData.length >= 2 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
              <h3 className="text-sm font-medium text-zinc-400">
                Comparison: {comparisonData.map((v) => v.id).join(" vs ")}
              </h3>
              {["loss", "ppl", "entropy", "gate_mean"].map((metric) => {
                const vals = comparisonData.map((v) => (v as any)[metric] as number | null).filter((x) => x != null);
                if (vals.length < 2) return null;
                const max = Math.max(...vals);
                return (
                  <div key={metric} className="flex items-center gap-3">
                    <span className="font-mono text-xs text-zinc-500 w-20 text-right">{metric}</span>
                    {comparisonData.map((v) => {
                      const val = (v as any)[metric] as number | null;
                      if (val == null) return null;
                      const pct = max > 0 ? (val / max) * 100 : 0;
                      return (
                        <div key={v.id} className="flex-1 flex items-center gap-2">
                          <span className="font-mono text-xs text-zinc-500 w-8">{v.id}</span>
                          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="font-mono text-xs text-zinc-300 w-14 text-right tabular-nums">
                            {val.toFixed(metric === "loss" ? 4 : metric === "ppl" ? 2 : 3)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}

          {/* Loss timeline */}
          {timeline.length > 1 && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
              <h3 className="text-sm font-medium text-zinc-400 mb-3">Loss Over Versions</h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={timeline} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="label" stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <YAxis stroke="#71717a" tick={{ fill: "#a1a1aa", fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#27272a", border: "1px solid #3f3f46", borderRadius: "8px" }} />
                  <Line type="monotone" dataKey="loss" stroke="#e5484d" strokeWidth={1.5} dot={{ r: 3, fill: "#e5484d" }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}
