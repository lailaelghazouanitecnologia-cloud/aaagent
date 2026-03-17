# z86.dev — HCLM-D Training Dashboard

## Guía Completa

---

## 1. Arquitectura

```
Training Loop (Python/GPU)
    │
    │  POST /api/metrics  (HTTP)
    ▼
┌──────────────────────────────────┐
│  Bun Server (Hono)               │
│  ├─ REST API   → SQLite (WAL)   │
│  └─ WebSocket  → broadcast       │
└──────────┬───────────────────────┘
           │
    ┌──────▼──────┐
    │  React SPA  │  ← Vite (dev) / nginx (prod)
    │  Tailwind 4 │
    │  Radix UI   │
    │  Recharts   │
    │  Visx       │
    └─────────────┘

Producción:
  Cloudflare (DNS + edge) → nginx (SSL + proxy) → Bun (:3000)
```

---

## 2. Estructura de Archivos

```
dashboard/
├── server/
│   ├── index.ts              # API + WebSocket (Hono en Bun)
│   └── db.ts                 # SQLite schema, queries, prepared statements
│
├── src/
│   ├── index.html            # Entry HTML (z86.dev, Inter + JetBrains Mono)
│   ├── index.css             # Tailwind 4 + oklch theme + animaciones
│   ├── main.tsx              # React 19 entry
│   ├── app.tsx               # React Router (7 rutas)
│   │
│   ├── lib/
│   │   ├── api.ts            # fetch helpers + connectWS()
│   │   └── hooks.ts          # useRuns, useMetrics, useRealtimeMetrics
│   │
│   ├── components/
│   │   ├── layout.tsx        # Sidebar + Outlet
│   │   ├── metric-card.tsx   # KPI card con trend
│   │   ├── run-selector.tsx  # Radix Select dropdown
│   │   └── charts/
│   │       ├── loss-chart.tsx       # Recharts — loss curves
│   │       ├── cluster-heatmap.tsx  # Visx — similarity matrix
│   │       └── gate-histogram.tsx   # Recharts — histograma + area
│   │
│   └── pages/
│       ├── overview.tsx      # KPIs + loss + throughput (real-time)
│       ├── losses.tsx        # 4 losses individuales + combinada
│       ├── clusters.tsx      # Entropy, dead clusters, heatmap
│       ├── gate.tsx          # Gate mean/std evolution + distribución
│       ├── hierarchy.tsx     # Coarse-fine alignment + balance
│       ├── generation.tsx    # Samples generados por checkpoint
│       └── ablations.tsx     # Tabla comparativa + overlay de runs
│
├── deploy/
│   ├── nginx.conf            # Reverse proxy z86.dev
│   ├── z86-dashboard.service # Systemd unit
│   ├── deploy.sh             # Deploy automatizado por SSH
│   └── cloudflare-setup.md   # Guía DNS + SSL
│
├── package.json
├── tsconfig.json
├── vite.config.ts
├── start.sh                  # Dev/prod launcher
├── .env.example
└── .gitignore

training/
└── dashboard_reporter.py     # Python reporter para el training loop
```

---

## 3. Quick Start (Desarrollo)

### Prerrequisitos

- [Bun](https://bun.sh/) >= 1.1
- Python >= 3.10 (para el reporter)

### Arrancar el dashboard

```bash
cd dashboard
bun install
bun run dev
```

Esto lanza:
- **API + WebSocket**: http://localhost:3000
- **Frontend (Vite HMR)**: http://localhost:5173

### Enviar métricas de prueba

```bash
# Métrica individual
curl -X POST http://localhost:3000/api/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "run": "test-run",
    "step": 1,
    "losses": {"total": 5.23, "diffusion": 5.20, "balance": 0.02, "diversity": 0.005, "hierarchy": 0.005},
    "cluster_health": {"entropy_ratio": 0.75, "dead_clusters": 2, "centroid_similarity": 0.34},
    "gate": {"mean": 0.5, "std": 0.15},
    "hierarchy": {"coarse_fine_alignment": 0.6, "balance": 0.8},
    "throughput": {"tokens_per_sec": 35000, "gpu_memory_gb": 16.2, "gpu_utilization": 0.85}
  }'

# Batch
curl -X POST http://localhost:3000/api/metrics/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"run": "test-run", "step": 2, "losses": {"total": 4.90}},
    {"run": "test-run", "step": 3, "losses": {"total": 4.75}}
  ]'

# Health check
curl http://localhost:3000/api/health
```

---

## 4. API Reference

### Ingestión

| Método | Ruta | Body | Respuesta |
|--------|------|------|-----------|
| `POST` | `/api/metrics` | `MetricPayload` | `{"ok": true}` |
| `POST` | `/api/metrics/batch` | `MetricPayload[]` | `{"ok": true, "count": N}` |

### Consulta

| Método | Ruta | Params | Respuesta |
|--------|------|--------|-----------|
| `GET` | `/api/runs` | — | `["run1", "run2"]` |
| `GET` | `/api/runs/:run/summary` | — | Resumen agregado |
| `GET` | `/api/runs/:run/metrics` | `?from=0&limit=10000` | Array de métricas |
| `GET` | `/api/runs/:run/latest` | — | Última métrica |
| `GET` | `/api/compare` | `?runs=A0,A1,A2` | Métricas combinadas |
| `GET` | `/api/health` | — | `{"status":"ok","uptime":N}` |

### WebSocket

```
ws://localhost:3000/ws     (dev)
wss://z86.dev/ws           (prod)
```

Mensajes del servidor → cliente:
```json
{"type": "metric", "data": { /* MetricPayload */ }}
{"type": "batch", "count": 5}
```

El cliente se reconecta automáticamente cada 2s si se pierde la conexión.

### MetricPayload (schema completo)

```typescript
interface MetricPayload {
  run: string;              // REQUERIDO — nombre del run (ej. "A2-clusters-only")
  step: number;             // REQUERIDO — step actual

  losses?: {
    total?: number;         // Loss total combinada
    diffusion?: number;     // L_diffusion
    balance?: number;       // L_balance (cluster load balancing)
    diversity?: number;     // L_diversity
    hierarchy?: number;     // L_hierarchy (coarse-fine)
  };

  cluster_health?: {
    entropy_ratio?: number;       // 0-1, >0.8 = saludable
    dead_clusters?: number;       // Clusters sin asignaciones
    centroid_similarity?: number; // Similaridad promedio entre centroides
  };

  gate?: {
    mean?: number;   // Media de activaciones del gate
    std?: number;    // Desviación estándar
  };

  hierarchy?: {
    coarse_fine_alignment?: number;  // Score de alineación coarse→fine
    balance?: number;                // Balance score jerárquico
  };

  throughput?: {
    tokens_per_sec?: number;   // Throughput de entrenamiento
    gpu_memory_gb?: number;    // Memoria GPU usada
    gpu_utilization?: number;  // Uso GPU 0-1
  };

  extra?: {
    // Campos extensibles:
    centroid_similarity_matrix?: number[][];  // Para heatmap en /clusters
    gate_distribution?: number[];             // Para histograma en /gate
    generated_sample?: string;                // Para /generation
    prompt?: string;                          // Prompt del sample
    sample_metrics?: Record<string, number>;  // Métricas del sample
    [key: string]: unknown;
  };
}
```

---

## 5. SQLite Schema

```sql
CREATE TABLE metrics (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  run                   TEXT NOT NULL,
  step                  INTEGER NOT NULL,
  timestamp             REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
  -- Losses
  loss_total            REAL,
  loss_diffusion        REAL,
  loss_balance          REAL,
  loss_diversity        REAL,
  loss_hierarchy        REAL,
  -- Cluster health
  entropy_ratio         REAL,
  dead_clusters         INTEGER,
  centroid_similarity   REAL,
  -- Gate
  gate_mean             REAL,
  gate_std              REAL,
  -- Hierarchy
  coarse_fine_alignment REAL,
  hierarchy_balance     REAL,
  -- Throughput
  tokens_per_sec        REAL,
  gpu_memory_gb         REAL,
  gpu_utilization       REAL,
  -- Extensible
  extra                 TEXT  -- JSON string
);

-- Indexes
CREATE INDEX idx_metrics_run_step ON metrics(run, step);
CREATE INDEX idx_metrics_run ON metrics(run);

-- Performance
PRAGMA journal_mode = WAL;         -- Lecturas concurrentes
PRAGMA synchronous = NORMAL;       -- Balance seguridad/velocidad
```

---

## 6. Python Reporter — Uso en Training

### Instalación

El reporter solo requiere `requests`:
```bash
pip install requests
```

### Uso básico

```python
from training.dashboard_reporter import DashboardReporter

# Inicializar (comprueba conectividad automáticamente)
reporter = DashboardReporter(
    run="A2-clusters-only",
    url="http://localhost:3000",  # o DASHBOARD_URL env var
    batch_size=10,                # flush cada 10 métricas
)

for step in range(total_steps):
    loss, l_diff, l_bal, l_div, l_hier = train_step(batch)

    reporter.report(
        step=step,
        losses={
            "total": loss.item(),
            "diffusion": l_diff.item(),
            "balance": l_bal.item(),
            "diversity": l_div.item(),
            "hierarchy": l_hier.item(),
        },
        cluster_health={
            "entropy_ratio": compute_entropy_ratio(assignments),
            "dead_clusters": count_dead_clusters(assignments),
            "centroid_similarity": avg_centroid_sim(codebook),
        },
        gate={
            "mean": gate_values.mean().item(),
            "std": gate_values.std().item(),
        },
        hierarchy={
            "coarse_fine_alignment": alignment_score,
            "balance": hierarchy_balance,
        },
        throughput={
            "tokens_per_sec": tokens_processed / elapsed,
            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu_utilization": get_gpu_util(),
        },
    )

    # Enviar samples generados cada N steps
    if step % 500 == 0:
        sample = generate_sample(model, prompt="The meaning of")
        reporter.report_generation(
            step=step,
            text=sample,
            prompt="The meaning of",
            metrics={"perplexity": compute_ppl(sample)},
        )

    # Enviar datos extra para visualizaciones avanzadas
    if step % 100 == 0:
        reporter.report(
            step=step,
            extra={
                "centroid_similarity_matrix": sim_matrix.tolist(),  # → Heatmap
                "gate_distribution": gate_values.cpu().tolist(),    # → Histograma
            },
        )

# Al terminar
reporter.close()
```

### Comportamiento

- **Thread-safe**: usa `threading.Lock` interno
- **Async send**: envía en background threads (no bloquea training)
- **Auto-disable**: si el dashboard no responde al inicio, se desactiva silenciosamente
- **Batching**: acumula métricas y envía en lotes para reducir overhead
- **Env var**: `DASHBOARD_URL` para configurar sin cambiar código

---

## 7. Páginas del Dashboard

### Overview (`/`)
- **8 KPI cards**: Step actual, Loss total, Mejor loss, Throughput, Entropy ratio, Dead clusters, Gate mean, GPU memory
- **Loss curves**: Todas las losses en un gráfico temporal (Recharts)
- **Throughput**: tokens/sec + GPU memory en tiempo real
- **Real-time**: WebSocket para actualizaciones instantáneas

### Losses (`/losses`)
- **5 KPI cards**: Total + 4 individuales
- **Gráfico combinado**: Todas las losses superpuestas
- **4 gráficos individuales**: L_diffusion, L_balance, L_diversity, L_hierarchy por separado

### Clusters (`/clusters`)
- **3 KPIs**: Entropy ratio (con indicador sano/bajo), Dead clusters, Centroid similarity
- **Gráfico temporal**: Entropy ratio + Dead clusters con doble eje Y
- **Heatmap** (Visx): Matriz de similaridad entre centroides (parsea `extra.centroid_similarity_matrix`)

### Gate (`/gate`)
- **4 KPIs**: Gate mean, Gate std, Ratio mean/std, Step actual
- **Area chart**: Evolución de gate mean + std en el tiempo
- **Histograma**: Distribución de activaciones del gate (parsea `extra.gate_distribution`)

### Hierarchy (`/hierarchy`)
- **3 KPIs**: Coarse-fine alignment (con indicador fuerte/débil), Hierarchy balance, L_hierarchy
- **Gráfico temporal**: Las 3 métricas superpuestas

### Generation (`/generation`)
- **Lista scrollable** (Radix ScrollArea) de samples generados
- Cada sample muestra: step, prompt, texto generado, métricas
- Parsea `extra.generated_sample`, `extra.prompt`, `extra.sample_metrics`

### Ablations (`/ablations`)
- **Toggle buttons** para seleccionar runs a comparar
- **Tabla comparativa** (TanStack Table): Run, Steps, Best/Final Loss, Avg throughput, Best entropy, Dead clusters
- **Overlay chart**: Loss curves de todos los runs seleccionados superpuestas

---

## 8. Despliegue en Producción (z86.dev)

### Prerequisitos del servidor

- VPS con Ubuntu/Debian
- IP pública
- Dominio z86.dev apuntando al servidor en Cloudflare

### Paso 1: Cloudflare DNS

En el panel de Cloudflare para `z86.dev`:

| Tipo | Nombre | Contenido | Proxy | TTL |
|------|--------|-----------|-------|-----|
| A | @ | `<IP_SERVIDOR>` | Proxied | Auto |
| A | www | `<IP_SERVIDOR>` | Proxied | Auto |

Configurar:
- **SSL/TLS**: Full (strict)
- **Always Use HTTPS**: ON
- **Minimum TLS**: 1.2
- **Brotli**: ON

### Paso 2: Origin Certificate

En Cloudflare → SSL/TLS → Origin Server → Create Certificate:
1. Copiar certificado → `/etc/ssl/z86.dev/origin.pem`
2. Copiar clave → `/etc/ssl/z86.dev/origin-key.pem`

### Paso 3: Deploy inicial

```bash
cd dashboard
./deploy/deploy.sh <IP_SERVIDOR> --setup
```

Esto instala Bun, nginx, copia archivos, construye el frontend, configura nginx y systemd.

### Paso 4: Deploys posteriores

```bash
./deploy/deploy.sh <IP_SERVIDOR>
```

### Paso 5: Verificar

```bash
curl https://z86.dev/api/health
# → {"status":"ok","uptime":...}
```

### Conectar el training al dashboard

Desde el pod de RunPod, el training necesita acceso al dashboard:

**Opción A — Mismo pod (recomendado)**:
```bash
# En el pod, arrancar el dashboard
cd /workspace/aaagent/dashboard && bun install && bun run start &

# El reporter usa localhost por defecto
export DASHBOARD_URL=http://localhost:3000
python train.py
```

**Opción B — Dashboard en VPS externo**:
```bash
# El reporter apunta al VPS
export DASHBOARD_URL=https://z86.dev
python train.py
```

### Logs y debugging

```bash
# Logs del dashboard
ssh root@<IP> journalctl -u z86-dashboard -f

# Estado del servicio
ssh root@<IP> systemctl status z86-dashboard

# Nginx logs
ssh root@<IP> tail -f /var/log/nginx/error.log

# Base de datos
ssh root@<IP> sqlite3 /opt/z86-dashboard/data/metrics.db ".tables"
```

---

## 9. Variables de Entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DASHBOARD_PORT` | `3000` | Puerto del servidor Bun |
| `DASHBOARD_DB` | `./metrics.db` | Ruta a la base de datos SQLite |
| `NODE_ENV` | `development` | `production` para servir static files |
| `CF_API_TOKEN` | — | Token de Cloudflare (opcional) |
| `CF_ZONE_ID` | — | Zone ID de Cloudflare (opcional) |
| `CORS_ORIGINS` | `https://z86.dev,http://localhost:5173` | Orígenes CORS permitidos |
| `DASHBOARD_URL` | `http://localhost:3000` | (Python) URL del dashboard |

---

## 10. Stack Técnico

| Capa | Tecnología | Versión |
|------|-----------|---------|
| Runtime | Bun | >= 1.1 |
| API | Hono | 4.6 |
| DB | SQLite (bun:sqlite) | WAL mode |
| Real-time | WebSocket (nativo Bun) | — |
| Frontend | React | 19 |
| Router | React Router | 7 |
| Build | Vite | 6 |
| CSS | Tailwind CSS | 4 (oklch) |
| UI | Radix UI Primitives | — |
| Charts | Recharts | 2.15 |
| Heatmaps | Visx | 3.5 |
| Tablas | TanStack React Table | 8.20 |
| Proxy | nginx | — |
| Proceso | systemd | — |
| DNS/SSL | Cloudflare | Full (strict) |
| Reporter | Python + requests | — |

---

## 11. Seguridad

### Nginx
- Rate limiting: 30 req/s por IP, burst 50
- Headers: HSTS, X-Frame-Options DENY, X-Content-Type-Options nosniff
- SSL: TLS 1.2+ con Cloudflare origin certs

### Systemd
- `NoNewPrivileges=true`
- `ProtectSystem=strict`
- `ProtectHome=true`
- `PrivateTmp=true`
- Solo escritura en `/opt/z86-dashboard/data`

### SQLite
- WAL mode para lecturas concurrentes sin bloqueo
- Prepared statements contra SQL injection

---

## 12. Optimización de Costes en RunPod

### Regla de oro: prep en local, GPU solo para entrenar

La tokenización (prep) usa CPU, no GPU. Hacerla en el pod alquilado es tirar dinero.

**Flujo óptimo:**

```bash
# EN TU PC (gratis, sin prisa):
git clone https://github.com/lailaelghazouanitecnologia-cloud/aaagent.git
cd aaagent
pip install ".[dev]"
python scripts/train.py --config configs/base.yaml --phase prep
# Esperar 15-45 min → genera data/train_tokens.pt y data/val_tokens.pt

# Subir los tokens al volumen de RunPod (rsync, scp, o git LFS)
scp data/train_tokens.pt data/val_tokens.pt root@<POD_IP>:/workspace/aaagent/data/

# EN EL POD (GPU desde el minuto 1):
cd /workspace/aaagent
python scripts/train.py --config configs/base.yaml  # directo a entrenar
```

**Ahorro estimado:** ~$0.10-0.15 por sesión (15-45 min de GPU ociosa evitados).

### Checklist antes de encender el pod

- [ ] `train_tokens.pt` y `val_tokens.pt` generados en local
- [ ] Tokenizer guardado en `data/tokenizer.json`
- [ ] Config del run preparada
- [ ] SSH key configurada en RunPod
- [ ] Saber exactamente qué comandos ejecutar

### GPU recomendada

| GPU | VRAM | Precio/h | Mejor para |
|-----|------|----------|------------|
| RTX A5000 | 24GB | $0.16 | Más barata, suficiente para 20M params |
| RTX 3090 | 24GB | $0.22 | Buen balance, más stock |
| RTX 4090 | 24GB | $0.34 | Más rápida, misma VRAM |
| A100 40GB | 40GB | $0.79 | Solo si necesitas batch más grande |

Para HCLM-D (20M params, batch 64): **cualquier GPU de 24GB sobra**.

---

## 13. Errores Conocidos y Soluciones

### Error: `Cannot import 'setuptools.backends._legacy'`

**Cuándo ocurre:** al hacer `pip install -e ".[dev]"` o `pip install ".[dev]"`

**Causa:** `pyproject.toml` tenía un build-backend obsoleto.

**Solución (ya aplicada en el repo):**
```toml
# ANTES (roto):
build-backend = "setuptools.backends._legacy:_Backend"

# DESPUÉS (correcto):
build-backend = "setuptools.build_meta"
```

Si por alguna razón vuelve a ocurrir:
```bash
sed -i 's|setuptools.backends._legacy:_Backend|setuptools.build_meta|' pyproject.toml
pip install ".[dev]"
```

### Error: `pip install -e` falla con editable check

**Cuándo ocurre:** versiones antiguas de pip + setuptools en contenedores Docker

**Solución:** usar install sin editable:
```bash
pip install ".[dev]"        # sin -e
```

### Error: `destination path 'aaagent' already exists`

**Cuándo ocurre:** al hacer `git clone` cuando el repo ya existe en el pod.

**Solución:**
```bash
cd /workspace/aaagent
git pull origin claude/hclm-d-documentation-uKBwL
```

### La tokenización parece colgada

**Cuándo ocurre:** `Tokenizing train.txt...` sin output durante 15-45 min.

**Es normal.** La tokenización de ~2M historias tarda. Señales de que funciona:
- CPU al 4-10%
- Memory estable ~22%
- GPU al 0% (no la usa)
- Disco activo

**NO hacer Ctrl+C.** Si lo haces, al relanzar retomará desde el tokenizer (ya guardado) pero re-tokenizará desde cero.

### GPU al 0% durante prep

**Es normal.** El prep (descarga + tokenización) solo usa CPU y disco. La GPU se activa cuando empieza el entrenamiento real.

### Warning: `Running pip as root`

**Ignorar.** En contenedores de RunPod todo corre como root. No afecta al funcionamiento.

### Warning: `unauthenticated requests to HF Hub`

**Ignorar.** TinyStories es público. Si quieres evitar rate limits:
```bash
export HF_TOKEN=tu_token_de_huggingface
```
