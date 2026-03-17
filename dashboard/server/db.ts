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

// --- Evaluation results table ---
db.run(`
  CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    eval_type TEXT NOT NULL DEFAULT 'bench_30',

    -- Auto metrics (no LLM)
    distinct_1 REAL,
    distinct_2 REAL,
    distinct_3 REAL,
    repetition_ratio REAL,
    self_bleu_4 REAL,
    keyword_hit REAL,
    vocab_richness REAL,
    length_compliance REAL,
    banned_violations INTEGER,
    total_tokens INTEGER,
    unique_tokens INTEGER,

    -- LLM judge scores (Groq)
    llm_coherence REAL,
    llm_grammar REAL,
    llm_relevance REAL,
    llm_creativity REAL,
    llm_fluency REAL,
    llm_completeness REAL,
    llm_overall REAL,
    llm_repetition REAL,
    failure_modes TEXT,

    -- Samples + per-item detail (JSON)
    samples TEXT,
    by_category TEXT,

    generation_time_s REAL
  )
`);
db.run(`CREATE INDEX IF NOT EXISTS idx_evals_run_step ON evals(run, step)`);
db.run(`CREATE INDEX IF NOT EXISTS idx_evals_run ON evals(run)`);

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

// --- Eval data functions ---

export interface EvalPayload {
  run: string;
  step: number;
  eval_type?: string;
  auto_metrics?: {
    distinct_1?: number;
    distinct_2?: number;
    distinct_3?: number;
    repetition_ratio?: number;
    self_bleu_4?: number;
    keyword_hit?: number;
    vocab_richness?: number;
    length_compliance?: number;
    banned_violations?: number;
    total_tokens?: number;
    unique_tokens?: number;
  };
  llm_judge?: {
    mean_coherence?: number;
    mean_grammar?: number;
    mean_relevance?: number;
    mean_creativity?: number;
    mean_fluency?: number;
    mean_completeness?: number;
    overall_quality?: number;
    mean_repetition_score?: number;
    failure_modes?: Record<string, number>;
  };
  samples?: any[];
  by_category?: Record<string, any>;
  generation_time_s?: number;
}

const insertEvalStmt = db.prepare(`
  INSERT INTO evals (
    run, step, eval_type,
    distinct_1, distinct_2, distinct_3, repetition_ratio, self_bleu_4,
    keyword_hit, vocab_richness, length_compliance, banned_violations,
    total_tokens, unique_tokens,
    llm_coherence, llm_grammar, llm_relevance, llm_creativity,
    llm_fluency, llm_completeness, llm_overall, llm_repetition,
    failure_modes, samples, by_category, generation_time_s
  ) VALUES (
    $run, $step, $eval_type,
    $distinct_1, $distinct_2, $distinct_3, $repetition_ratio, $self_bleu_4,
    $keyword_hit, $vocab_richness, $length_compliance, $banned_violations,
    $total_tokens, $unique_tokens,
    $llm_coherence, $llm_grammar, $llm_relevance, $llm_creativity,
    $llm_fluency, $llm_completeness, $llm_overall, $llm_repetition,
    $failure_modes, $samples, $by_category, $generation_time_s
  )
`);

export function insertEval(e: EvalPayload) {
  const a = e.auto_metrics ?? {};
  const l = e.llm_judge ?? {};
  insertEvalStmt.run({
    $run: e.run,
    $step: e.step,
    $eval_type: e.eval_type ?? "bench_30",
    $distinct_1: a.distinct_1 ?? null,
    $distinct_2: a.distinct_2 ?? null,
    $distinct_3: a.distinct_3 ?? null,
    $repetition_ratio: a.repetition_ratio ?? null,
    $self_bleu_4: a.self_bleu_4 ?? null,
    $keyword_hit: a.keyword_hit ?? null,
    $vocab_richness: a.vocab_richness ?? null,
    $length_compliance: a.length_compliance ?? null,
    $banned_violations: a.banned_violations ?? null,
    $total_tokens: a.total_tokens ?? null,
    $unique_tokens: a.unique_tokens ?? null,
    $llm_coherence: l.mean_coherence ?? null,
    $llm_grammar: l.mean_grammar ?? null,
    $llm_relevance: l.mean_relevance ?? null,
    $llm_creativity: l.mean_creativity ?? null,
    $llm_fluency: l.mean_fluency ?? null,
    $llm_completeness: l.mean_completeness ?? null,
    $llm_overall: l.overall_quality ?? null,
    $llm_repetition: l.mean_repetition_score ?? null,
    $failure_modes: l.failure_modes ? JSON.stringify(l.failure_modes) : null,
    $samples: e.samples ? JSON.stringify(e.samples) : null,
    $by_category: e.by_category ? JSON.stringify(e.by_category) : null,
    $generation_time_s: e.generation_time_s ?? null,
  });
}

export function getEvals(run: string) {
  return db
    .query("SELECT * FROM evals WHERE run = ? ORDER BY step ASC")
    .all(run);
}

export function getLatestEval(run: string) {
  return db
    .query("SELECT * FROM evals WHERE run = ? ORDER BY step DESC LIMIT 1")
    .get(run);
}

export function getAllEvalRuns(): string[] {
  return db
    .query("SELECT DISTINCT run FROM evals ORDER BY run")
    .all()
    .map((r: any) => r.run);
}

export function getEvalComparison(runs: string[]) {
  const placeholders = runs.map(() => "?").join(",");
  return db
    .query(
      `SELECT run, step, distinct_2, repetition_ratio, self_bleu_4,
              keyword_hit, vocab_richness, llm_overall, llm_coherence,
              llm_creativity, failure_modes, generation_time_s
       FROM evals WHERE run IN (${placeholders}) ORDER BY run, step ASC`,
    )
    .all(...runs);
}

export { db };
