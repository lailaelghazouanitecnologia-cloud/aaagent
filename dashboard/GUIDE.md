# z86.dev — HCLM-D Platform Guide

## 1. Arquitectura General

```
                    ┌──────────────────────────────┐
                    │         z86 CLI               │
                    │  z86 init / train / eval /    │
                    │  versions / generate / serve  │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
    ┌─────────▼──────┐  ┌─────▼──────┐   ┌────────▼────────┐
    │  Training Loop │  │ Eval Agent │   │  Version        │
    │  (PyTorch/GPU) │  │ (auto +    │   │  Registry       │
    │                │  │  LLM judge)│   │  manifest.json  │
    └───────┬────────┘  └─────┬──────┘   └────────┬────────┘
            │                 │                    │
     POST /api/metrics  POST /api/evals    GET /api/versions
            │                 │                    │
    ┌───────▼─────────────────▼────────────────────▼───┐
    │              Bun Server (Hono)                    │
    │  ├─ REST API  → SQLite (WAL)                     │
    │  ├─ WebSocket → broadcast                        │
    │  └─ Static    → Vite build                       │
    └───────────────────┬──────────────────────────────┘
                        │
              ┌─────────▼─────────┐
              │    React SPA      │
              │  Tailwind 4       │
              │  Radix UI         │
              │  Recharts + Canvas│
              └───────────────────┘
```

---

## 2. z86 CLI — Referencia Completa

El CLI se instala con `pip install -e .` y queda disponible como `z86`.

### Comandos

| Comando | Descripción |
|---------|-------------|
| `z86 init` | Setup completo: deps + datos + smoke test |
| `z86 doctor` | Verifica GPU, deps, datos, dashboard, env vars |
| `z86 train` | Entrena modelo (--config, --resume, --name, --dashboard) |
| `z86 versions` | Lista versiones registradas (--detail, --scan) |
| `z86 diff v1 v3` | Compara métricas entre dos versiones |
| `z86 delete v2` | Elimina versión (--keep-file para mantener checkpoint) |
| `z86 eval v3` | Evalúa una versión (--quick, --judge, --dashboard) |
| `z86 generate v3 "prompt"` | Genera texto (--interactive para REPL) |
| `z86 serve v3` | API HTTP de inferencia (--port 8080) |
| `z86 ablation run` | Ejecuta ablaciones A0-A5 (status, compare) |
| `z86 dashboard` | Arranca dashboard (--prod, --port) |

### Flujo típico

```bash
# Primera vez
z86 init                                  # ~30 min (descarga datos)

# Entrenar
z86 train --name "base-run"               # Ctrl+C para parar
z86 train --config fast --name "fast-run"  # Config rápida
z86 train --resume v1 --name "continue"    # Retomar versión

# Gestionar versiones
z86 versions                              # Ver tabla
z86 versions --scan                       # Registrar checkpoints huérfanos
z86 diff v1 v3                            # Comparar métricas

# Evaluar
z86 eval v3                               # Eval completa
z86 eval v3 --quick                       # Sin perplexity (rápido)
z86 eval v3 --judge --dashboard           # Con LLM judge, enviar a dashboard

# Generar
z86 generate v3 "Once upon a time"        # Un sample
z86 generate v3 -i                        # REPL interactivo
z86 serve v3 --port 8080                  # API HTTP

# Dashboard
z86 dashboard                             # Dev (:3000 + :5173)
z86 dashboard --prod                      # Producción (:3000)
```

### Version Registry

Las versiones se almacenan en `checkpoints/manifest.json`:

```json
{
  "versions": [
    {
      "id": "v1",
      "step": 14280,
      "path": "checkpoints/step_14280.pt",
      "run": "base-run",
      "config": "base.yaml",
      "loss": 0.847,
      "ppl": 2.33,
      "entropy": 0.87,
      "gate_mean": 0.51,
      "created": "2026-03-17T14:30:00Z",
      "size_mb": 82
    }
  ]
}
```

- Se auto-registra al hacer Ctrl+C durante training
- `z86 eval` actualiza las métricas automáticamente
- `z86 versions --scan` detecta checkpoints no registrados

---

## 3. Dashboard — Páginas

El dashboard tiene 3 tabs principales:

### Overview (`/`)
- 8 KPI cards: Step, Loss total, Mejor loss, Throughput, Entropy ratio, Dead clusters, Gate mean, GPU memory
- Loss curves (Recharts)
- Throughput en tiempo real (WebSocket)

### Evals (`/evals`)
- **Auto metrics**: Distinct-2, Repetition, Self-BLEU-4, Keyword hit, Vocab richness
- **LLM judge**: Overall quality, Coherence, Grammar, Creativity, Fluency, Completeness
- Radar chart de dimensiones LLM
- Failure modes (bar chart)
- Per-category breakdown (narration, dialogue, instruct, creative, technical, code_doc)
- Timeline de calidad vs training steps
- Browser de samples con scores inline

### Versions (`/versions`)
- Tabla de versiones con selección múltiple
- Comparación visual por métrica (barras)
- Timeline de loss por versión

---

## 4. API Reference

### Training Metrics

| Método | Ruta | Body | Respuesta |
|--------|------|------|-----------|
| `POST` | `/api/metrics` | `MetricPayload` | `{"ok": true}` |
| `POST` | `/api/metrics/batch` | `MetricPayload[]` | `{"ok": true, "count": N}` |
| `GET` | `/api/runs` | — | `["run1", "run2"]` |
| `GET` | `/api/runs/:run/summary` | — | Resumen agregado |
| `GET` | `/api/runs/:run/metrics` | `?from=0&limit=10000` | Array de métricas |
| `GET` | `/api/runs/:run/latest` | — | Última métrica |
| `GET` | `/api/compare` | `?runs=A0,A1,A2` | Métricas combinadas |

### Evaluación

| Método | Ruta | Body | Respuesta |
|--------|------|------|-----------|
| `POST` | `/api/evals` | `EvalPayload` | `{"ok": true}` |
| `GET` | `/api/evals/runs` | — | `["run1", "run2"]` |
| `GET` | `/api/evals/:run` | — | Array de evals |
| `GET` | `/api/evals/:run/latest` | — | Última eval |
| `GET` | `/api/evals/compare` | `?runs=A0,A1` | Comparación |

### Versions

| Método | Ruta | Respuesta |
|--------|------|-----------|
| `GET` | `/api/versions` | `{"versions": [...]}` |

### Sistema

| Método | Ruta | Respuesta |
|--------|------|-----------|
| `GET` | `/api/health` | `{"status":"ok","uptime":N}` |
| `WS` | `/ws` | Real-time metric broadcasts |

### MetricPayload

```typescript
interface MetricPayload {
  run: string;                    // REQUERIDO
  step: number;                   // REQUERIDO
  losses?: {
    total?: number;
    diffusion?: number;
    balance?: number;
    diversity?: number;
    hierarchy?: number;
  };
  cluster_health?: {
    entropy_ratio?: number;       // 0-1, >0.8 = saludable
    dead_clusters?: number;
    centroid_similarity?: number;
  };
  gate?: { mean?: number; std?: number; };
  hierarchy?: { coarse_fine_alignment?: number; balance?: number; };
  throughput?: { tokens_per_sec?: number; gpu_memory_gb?: number; gpu_utilization?: number; };
  extra?: Record<string, unknown>;
}
```

### EvalPayload

```typescript
interface EvalPayload {
  run: string;                    // REQUERIDO
  step: number;                   // REQUERIDO
  eval_type?: string;             // default: "bench_30"
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
    mean_coherence?: number;      // 1-5
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
```

---

## 5. SQLite Schema

```sql
-- Training metrics
CREATE TABLE metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run TEXT NOT NULL, step INTEGER NOT NULL,
  timestamp REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
  loss_total REAL, loss_diffusion REAL, loss_balance REAL,
  loss_diversity REAL, loss_hierarchy REAL,
  entropy_ratio REAL, dead_clusters INTEGER, centroid_similarity REAL,
  gate_mean REAL, gate_std REAL,
  coarse_fine_alignment REAL, hierarchy_balance REAL,
  tokens_per_sec REAL, gpu_memory_gb REAL, gpu_utilization REAL,
  extra TEXT
);

-- Evaluation results
CREATE TABLE evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run TEXT NOT NULL, step INTEGER NOT NULL,
  timestamp REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
  eval_type TEXT NOT NULL DEFAULT 'bench_30',
  distinct_1 REAL, distinct_2 REAL, distinct_3 REAL,
  repetition_ratio REAL, self_bleu_4 REAL,
  keyword_hit REAL, vocab_richness REAL, length_compliance REAL,
  banned_violations INTEGER, total_tokens INTEGER, unique_tokens INTEGER,
  llm_coherence REAL, llm_grammar REAL, llm_relevance REAL,
  llm_creativity REAL, llm_fluency REAL, llm_completeness REAL,
  llm_overall REAL, llm_repetition REAL,
  failure_modes TEXT, samples TEXT, by_category TEXT,
  generation_time_s REAL
);

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

---

## 6. Eval Pipeline

### Auto Metrics (sin LLM)

Evaluación determinista sobre 30 prompts curados en 6 categorías:

| Métrica | Qué mide | Ideal |
|---------|----------|-------|
| Distinct-1/2/3 | Diversidad de n-grams | > 0.7 |
| Repetition ratio | Repetición de 4-grams | < 0.1 |
| Self-BLEU-4 | Similitud entre samples | < 0.2 |
| Keyword hit | Relevancia al prompt | > 0.8 |
| Vocab richness | unique/total tokens | > 0.5 |
| Length compliance | Cumple longitud mínima | > 0.9 |

### LLM Judge (Groq)

6 dimensiones evaluadas por `llama-3.3-70b-versatile`:

| Dimensión | Escala | Descripción |
|-----------|--------|-------------|
| Coherence | 1-5 | Flujo lógico y consistencia |
| Grammar | 1-5 | Corrección gramatical |
| Relevance | 1-5 | Responde al prompt |
| Creativity | 1-5 | Originalidad |
| Fluency | 1-5 | Naturalidad del texto |
| Completeness | 1-5 | Texto completo, no truncado |

Failure modes detectados: `none`, `repetition_loop`, `nonsense`, `truncated`, `copied`, `off_topic`.

### Categorías del benchmark

| Categoría | # Prompts | Ejemplo |
|-----------|-----------|---------|
| narration | 5 | "Write a short story about a forgotten city" |
| dialogue | 5 | "Write a conversation between a cat and a dog" |
| instruction | 5 | "Explain how to make a paper airplane" |
| creative | 5 | "Write a poem about the ocean at night" |
| technical | 5 | "Describe how a neural network learns" |
| code_doc | 5 | "Write a docstring for a sorting function" |

### Ejecutar evaluación

```bash
# Via CLI (recomendado)
z86 eval v3                              # Auto metrics
z86 eval v3 --judge                      # + LLM judge
z86 eval v3 --judge --dashboard          # + Enviar a dashboard

# Via script directo
python scripts/eval_checkpoint.py --checkpoint checkpoints/step_50000.pt --dashboard

# Via agent (Agno)
python -m eval.agent_eval --checkpoint checkpoints/step_50000.pt \
  --dashboard-url http://localhost:3000 --run-name "base-20m"
```

---

## 7. Python Reporter — Uso en Training

```python
from training.dashboard_reporter import DashboardReporter

reporter = DashboardReporter(
    run="base-20m",
    url="http://localhost:3000",   # o env DASHBOARD_URL
    batch_size=10,
)

for step in range(total_steps):
    loss = train_step(batch)

    reporter.report(
        step=step,
        losses={"total": loss.item(), "diffusion": l_diff.item(), ...},
        cluster_health={"entropy_ratio": 0.87, "dead_clusters": 0, ...},
        gate={"mean": 0.51, "std": 0.18},
        throughput={"tokens_per_sec": 48000, "gpu_memory_gb": 12.3},
    )

reporter.close()
```

Comportamiento:
- Thread-safe (Lock interno)
- Async send (background threads)
- Auto-disable si dashboard no responde
- Batching para reducir overhead
- Configurable via `DASHBOARD_URL` env var

---

## 8. Configuración del Proyecto (`config.toml`)

El archivo `config.toml` en la raíz centraliza la configuración de infraestructura:

```toml
[project]
name = "hclm-d"
version = "0.1.0"
seed = 42

[dashboard]
port = 3000
db_path = "metrics.db"
websocket_path = "/ws"

[data]
dataset = "tinystories"
data_dir = "data/tinystories"
tokenizer_path = "data/tokenizer.json"

[checkpoints]
dir = "checkpoints"
manifest = "checkpoints/manifest.json"

[training]
config = "configs/base.yaml"          # Los YAML siguen siendo la fuente primaria

[cloud.runpod]
default_preset = "train-fast"

[cloud.groq]
model = "llama-3.3-70b-versatile"

[wandb]
project = "hclm-d"
```

Los hiperparámetros de entrenamiento viven en `configs/*.yaml`. El TOML cubre infraestructura, dashboard, cloud, y defaults del CLI.

---

## 9. Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DASHBOARD_URL` | `http://localhost:3000` | URL del dashboard (Python reporter) |
| `DASHBOARD_PORT` | `3000` | Puerto del servidor Bun |
| `DASHBOARD_DB` | `./metrics.db` | Ruta SQLite |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU a usar |
| `WANDB_PROJECT` | — | Proyecto W&B (opcional) |
| `WANDB_MODE` | `online` | `disabled` para desactivar |
| `GROQ_API_KEY` | — | API key para LLM judge |
| `RUN_NAME` | — | Nombre del run (set por CLI) |
| `NODE_ENV` | `development` | `production` para static files |

---

## 10. Despliegue (RunPod)

### Flujo completo en un solo pod

```bash
# 1. Setup (primera vez)
cd /workspace && git clone <repo> && cd aaagent
z86 init                              # ~30 min

# 2. Dashboard (background)
z86 dashboard &
export DASHBOARD_URL=http://localhost:3000

# 3. Entrenar
z86 train --name "base-run" --dashboard http://localhost:3000

# 4. Evaluar
z86 eval latest --judge --dashboard

# 5. Ver versiones
z86 versions
```

### GPU recomendada

| GPU | VRAM | Precio/h | Para HCLM-D 20M |
|-----|------|----------|------------------|
| RTX A5000 | 24GB | $0.16 | Suficiente |
| RTX 3090 | 24GB | $0.22 | Buen balance |
| RTX 4090 | 24GB | $0.34 | Más rápida |
| A100 40GB | 40GB | $0.79 | Batch más grande |

### Tips
- Usa volumen persistente en `/workspace` para datos y checkpoints
- Los `.txt` se pueden borrar después del prep (~1.9GB ahorrados)
- Dashboard funciona sin GPU (solo CPU)

---

## 11. Stack Técnico

| Capa | Tecnología |
|------|-----------|
| CLI | Python argparse + ANSI terminal UI |
| Training | PyTorch 2.1+ / bfloat16 / torch.compile |
| Eval (auto) | Python (distinct-n, self-BLEU, repetition) |
| Eval (LLM) | Groq API (llama-3.3-70b-versatile) |
| Eval (agent) | Agno framework |
| Runtime | Bun 1.1+ |
| API | Hono 4.6 |
| DB | SQLite (bun:sqlite, WAL) |
| Real-time | WebSocket (Bun nativo) |
| Frontend | React 19 + Vite 6 |
| CSS | Tailwind 4 (ZARNETTI theme) |
| UI | Radix UI Primitives |
| Charts | Recharts + Canvas nativo |
| Heatmaps | Visx |
| Tables | TanStack React Table |
| Deploy | nginx + systemd + Cloudflare |

---

## 12. Troubleshooting

### `pip install -e .` falla
```bash
pip install ".[dev]"    # sin -e
```

### Tokenización parece colgada (15-45 min sin output)
Es normal. La tokenización de ~2M historias tarda. No hacer Ctrl+C.

### Dashboard no recibe métricas
```bash
curl http://localhost:3000/api/health    # ¿Responde?
echo $DASHBOARD_URL                      # ¿Variable definida?
```

### `z86` command not found
```bash
pip install -e .     # Reinstala para registrar el script
# o ejecutar directamente:
python -m cli.main train
```

### CSS 404 (`index-*.css not found`)
El build de Vite genera archivos con hash. Si el `dist/` está obsoleto:
```bash
cd dashboard && bun install && bun run build
```
En desarrollo, usar `bun run dev` (Vite sirve hot-reload sin necesidad de build).

### GPU al 0% durante prep
Normal. El prep solo usa CPU. GPU se activa al entrenar.
