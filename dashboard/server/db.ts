import { Database } from "bun:sqlite";
import { join } from "path";

const DB_PATH = process.env.DASHBOARD_DB ?? join(import.meta.dir, "..", "metrics.db");

const db = new Database(DB_PATH, { create: true });
db.run("PRAGMA journal_mode = WAL");
db.run("PRAGMA synchronous = NORMAL");

db.run(`
  CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    loss_total REAL,
    loss_diffusion REAL,
    loss_balance REAL,
    loss_diversity REAL,
    loss_hierarchy REAL,
    entropy_ratio REAL,
    dead_clusters INTEGER,
    centroid_similarity REAL,
    gate_mean REAL,
    gate_std REAL,
    coarse_fine_alignment REAL,
    hierarchy_balance REAL,
    tokens_per_sec REAL,
    gpu_memory_gb REAL,
    gpu_utilization REAL,
    extra TEXT
  )
`);

db.run(`CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run, step)`);
db.run(`CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run)`);

const insertStmt = db.prepare(`
  INSERT INTO metrics (
    run, step, loss_total, loss_diffusion, loss_balance, loss_diversity, loss_hierarchy,
    entropy_ratio, dead_clusters, centroid_similarity,
    gate_mean, gate_std, coarse_fine_alignment, hierarchy_balance,
    tokens_per_sec, gpu_memory_gb, gpu_utilization, extra
  ) VALUES (
    $run, $step, $loss_total, $loss_diffusion, $loss_balance, $loss_diversity, $loss_hierarchy,
    $entropy_ratio, $dead_clusters, $centroid_similarity,
    $gate_mean, $gate_std, $coarse_fine_alignment, $hierarchy_balance,
    $tokens_per_sec, $gpu_memory_gb, $gpu_utilization, $extra
  )
`);

export interface MetricPayload {
  run: string;
  step: number;
  losses?: {
    total?: number;
    diffusion?: number;
    balance?: number;
    diversity?: number;
    hierarchy?: number;
  };
  cluster_health?: {
    entropy_ratio?: number;
    dead_clusters?: number;
    centroid_similarity?: number;
  };
  gate?: {
    mean?: number;
    std?: number;
  };
  hierarchy?: {
    coarse_fine_alignment?: number;
    balance?: number;
  };
  throughput?: {
    tokens_per_sec?: number;
    gpu_memory_gb?: number;
    gpu_utilization?: number;
  };
  extra?: Record<string, unknown>;
}

export function insertMetric(m: MetricPayload) {
  insertStmt.run({
    $run: m.run,
    $step: m.step,
    $loss_total: m.losses?.total ?? null,
    $loss_diffusion: m.losses?.diffusion ?? null,
    $loss_balance: m.losses?.balance ?? null,
    $loss_diversity: m.losses?.diversity ?? null,
    $loss_hierarchy: m.losses?.hierarchy ?? null,
    $entropy_ratio: m.cluster_health?.entropy_ratio ?? null,
    $dead_clusters: m.cluster_health?.dead_clusters ?? null,
    $centroid_similarity: m.cluster_health?.centroid_similarity ?? null,
    $gate_mean: m.gate?.mean ?? null,
    $gate_std: m.gate?.std ?? null,
    $coarse_fine_alignment: m.hierarchy?.coarse_fine_alignment ?? null,
    $hierarchy_balance: m.hierarchy?.balance ?? null,
    $tokens_per_sec: m.throughput?.tokens_per_sec ?? null,
    $gpu_memory_gb: m.throughput?.gpu_memory_gb ?? null,
    $gpu_utilization: m.throughput?.gpu_utilization ?? null,
    $extra: m.extra ? JSON.stringify(m.extra) : null,
  });
}

export function getRuns(): string[] {
  return db.query("SELECT DISTINCT run FROM metrics ORDER BY run").all().map((r: any) => r.run);
}

export function getMetrics(run: string, from_step = 0, limit = 10000) {
  return db
    .query(
      "SELECT * FROM metrics WHERE run = ? AND step >= ? ORDER BY step ASC LIMIT ?",
    )
    .all(run, from_step, limit);
}

export function getLatestMetric(run: string) {
  return db
    .query("SELECT * FROM metrics WHERE run = ? ORDER BY step DESC LIMIT 1")
    .get(run);
}

export function getRunSummary(run: string) {
  return db
    .query(`
      SELECT
        run,
        COUNT(*) as total_steps,
        MIN(step) as first_step,
        MAX(step) as last_step,
        MIN(loss_total) as best_loss,
        AVG(tokens_per_sec) as avg_throughput,
        MAX(timestamp) as last_update
      FROM metrics WHERE run = ?
    `)
    .get(run);
}

export function getComparisonData(runs: string[]) {
  const placeholders = runs.map(() => "?").join(",");
  return db
    .query(
      `SELECT * FROM metrics WHERE run IN (${placeholders}) ORDER BY run, step ASC`,
    )
    .all(...runs);
}

export { db };
