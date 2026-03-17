import { Hono } from "hono";
import { cors } from "hono/cors";
import { serveStatic } from "hono/bun";
import { join } from "path";
import {
  insertMetric,
  getRuns,
  getMetrics,
  getLatestMetric,
  getRunSummary,
  getComparisonData,
  type MetricPayload,
} from "./db";

const app = new Hono();
const PORT = parseInt(process.env.DASHBOARD_PORT ?? "3000");

app.use("/api/*", cors());

// --- WebSocket clients ---
const wsClients = new Set<any>();

function broadcast(data: unknown) {
  const msg = JSON.stringify(data);
  for (const ws of wsClients) {
    try {
      ws.send(msg);
    } catch {
      wsClients.delete(ws);
    }
  }
}

// --- API Routes ---

// Ingest metrics from training
app.post("/api/metrics", async (c) => {
  const payload: MetricPayload = await c.req.json();

  if (!payload.run || payload.step == null) {
    return c.json({ error: "run and step are required" }, 400);
  }

  insertMetric(payload);
  broadcast({ type: "metric", data: payload });
  return c.json({ ok: true });
});

// Batch ingest
app.post("/api/metrics/batch", async (c) => {
  const payloads: MetricPayload[] = await c.req.json();
  for (const p of payloads) {
    if (p.run && p.step != null) {
      insertMetric(p);
    }
  }
  broadcast({ type: "batch", count: payloads.length });
  return c.json({ ok: true, count: payloads.length });
});

// List all runs
app.get("/api/runs", (c) => {
  return c.json(getRuns());
});

// Get run summary
app.get("/api/runs/:run/summary", (c) => {
  const run = c.req.param("run");
  return c.json(getRunSummary(run));
});

// Get metrics for a run
app.get("/api/runs/:run/metrics", (c) => {
  const run = c.req.param("run");
  const from = parseInt(c.req.query("from") ?? "0");
  const limit = parseInt(c.req.query("limit") ?? "10000");
  return c.json(getMetrics(run, from, limit));
});

// Get latest metric for a run
app.get("/api/runs/:run/latest", (c) => {
  const run = c.req.param("run");
  return c.json(getLatestMetric(run));
});

// Compare multiple runs
app.get("/api/compare", (c) => {
  const runs = c.req.query("runs")?.split(",") ?? [];
  if (runs.length === 0) return c.json({ error: "provide ?runs=a,b,c" }, 400);
  return c.json(getComparisonData(runs));
});

// Health check
app.get("/api/health", (c) => {
  return c.json({ status: "ok", uptime: process.uptime() });
});

// --- Serve static frontend in production ---
if (process.env.NODE_ENV === "production") {
  const distPath = join(import.meta.dir, "..", "dist");
  app.use("/*", serveStatic({ root: distPath }));
  app.get("*", serveStatic({ root: distPath, path: "index.html" }));
}

// --- Start server with WebSocket support ---
const server = Bun.serve({
  port: PORT,
  fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket upgrade
    if (url.pathname === "/ws") {
      const upgraded = server.upgrade(req);
      if (!upgraded) {
        return new Response("WebSocket upgrade failed", { status: 400 });
      }
      return undefined as any;
    }

    return app.fetch(req, { ip: server.requestIP(req) });
  },
  websocket: {
    open(ws) {
      wsClients.add(ws);
    },
    message(_ws, _message) {
      // Client-to-server messages not needed for now
    },
    close(ws) {
      wsClients.delete(ws);
    },
  },
});

console.log(`Dashboard server running at http://localhost:${server.port}`);
console.log(`WebSocket available at ws://localhost:${server.port}/ws`);
