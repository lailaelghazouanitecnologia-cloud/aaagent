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

  // Comparison data for selected versions
  const comparisonData = useMemo(() => {
    if (selected.size < 2) return null;
    return versions
      .filter((v) => selected.has(v.id))
      .sort((a, b) => a.step - b.step);
  }, [versions, selected]);

  // Timeline data (all versions by step)
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
    <div style={{ padding: "8px 10px", height: "100%", display: "flex", flexDirection: "column", gap: "1px", background: "var(--color-border)" }}>
      {/* Header */}
      <div style={{ background: "var(--color-s1)", padding: "8px 10px" }}>
        <span className="panel-title">Model Versions</span>
      </div>

      {loading && (
        <div style={{ background: "var(--color-s1)", padding: "20px", textAlign: "center" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-t3)" }}>Loading...</span>
        </div>
      )}

      {!loading && versions.length === 0 && (
        <div style={{ background: "var(--color-s1)", padding: "20px", textAlign: "center" }}>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--color-t3)" }}>
            No versions registered. Train a model: <span style={{ color: "var(--color-accent-cyan)" }}>z86 train</span>
          </p>
        </div>
      )}

      {versions.length > 0 && (
        <>
          {/* Version table */}
          <div style={{ background: "var(--color-s1)", padding: "8px 10px", overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: 10 }}>
              <thead>
                <tr style={{ color: "var(--color-text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", fontSize: 8 }}>
                  <th style={{ textAlign: "left", padding: "4px 8px" }}></th>
                  <th style={{ textAlign: "left", padding: "4px 8px" }}>Version</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Step</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Loss</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>PPL</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Entropy</th>
                  <th style={{ textAlign: "left", padding: "4px 8px" }}>Run</th>
                  <th style={{ textAlign: "right", padding: "4px 8px" }}>Size</th>
                  <th style={{ textAlign: "left", padding: "4px 8px" }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {versions.map((v) => {
                  const isBest = best?.id === v.id;
                  return (
                    <tr
                      key={v.id}
                      onClick={() => toggleSelect(v.id)}
                      style={{
                        cursor: "pointer",
                        background: selected.has(v.id) ? "var(--color-muted)" : "transparent",
                        borderBottom: "1px solid var(--color-border)",
                      }}
                    >
                      <td style={{ padding: "4px 8px", width: 20 }}>
                        <input
                          type="checkbox"
                          checked={selected.has(v.id)}
                          onChange={() => toggleSelect(v.id)}
                          style={{ accentColor: "var(--color-accent-blue-hi)" }}
                        />
                      </td>
                      <td style={{ padding: "4px 8px", color: isBest ? "var(--color-accent-green-hi)" : "var(--color-white)", fontWeight: 500 }}>
                        {v.id}{isBest && " ★"}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "var(--color-white)", fontVariantNumeric: "tabular-nums" }}>
                        {v.step.toLocaleString()}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "var(--color-white)", fontVariantNumeric: "tabular-nums" }}>
                        {fmt(v.loss)}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "var(--color-t2)", fontVariantNumeric: "tabular-nums" }}>
                        {fmt(v.ppl, 2)}
                      </td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "var(--color-t2)", fontVariantNumeric: "tabular-nums" }}>
                        {fmt(v.entropy, 3)}
                      </td>
                      <td style={{ padding: "4px 8px", color: "var(--color-t3)" }}>{v.run}</td>
                      <td style={{ padding: "4px 8px", textAlign: "right", color: "var(--color-t3)" }}>{v.size_mb}MB</td>
                      <td style={{ padding: "4px 8px", color: "var(--color-text-sub)" }}>{v.created?.slice(0, 10) ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Comparison: selected versions */}
          {comparisonData && comparisonData.length >= 2 && (
            <div style={{ background: "var(--color-s1)", padding: "8px 10px" }}>
              <span className="panel-title">
                Comparison: {comparisonData.map((v) => v.id).join(" vs ")}
              </span>
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
                {["loss", "ppl", "entropy", "gate_mean"].map((metric) => {
                  const vals = comparisonData.map((v) => (v as any)[metric] as number | null).filter((x) => x != null);
                  if (vals.length < 2) return null;
                  const max = Math.max(...vals);
                  return (
                    <div key={metric} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "var(--color-text-dim)", width: 70, textAlign: "right" }}>
                        {metric}
                      </span>
                      {comparisonData.map((v) => {
                        const val = (v as any)[metric] as number | null;
                        if (val == null) return null;
                        const pct = max > 0 ? (val / max) * 100 : 0;
                        const color = metric === "entropy" ? "var(--color-accent-purple-hi)" : "var(--color-accent-blue-hi)";
                        return (
                          <div key={v.id} style={{ flex: 1, display: "flex", alignItems: "center", gap: 4 }}>
                            <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "var(--color-t3)", width: 24 }}>{v.id}</span>
                            <div style={{ flex: 1, height: 3, background: "var(--color-muted)", borderRadius: 1, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 1 }} />
                            </div>
                            <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: "var(--color-white)", width: 50, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                              {val.toFixed(metric === "loss" ? 4 : metric === "ppl" ? 2 : 3)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Loss timeline chart */}
          {timeline.length > 1 && (
            <div style={{ background: "var(--color-s1)", padding: "8px 10px", flex: 1, minHeight: 200 }}>
              <span className="panel-title">Loss Over Versions</span>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={timeline} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#141418" />
                  <XAxis dataKey="label" stroke="#3e3e4a" tick={{ fill: "#58586a", fontSize: 9, fontFamily: "Geist Mono" }} />
                  <YAxis stroke="#3e3e4a" tick={{ fill: "#58586a", fontSize: 9, fontFamily: "Geist Mono" }} />
                  <Tooltip
                    contentStyle={{
                      background: "#0a0a0c",
                      border: "1px solid #141418",
                      borderRadius: 3,
                      fontFamily: "Geist Mono",
                      fontSize: 10,
                    }}
                  />
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
