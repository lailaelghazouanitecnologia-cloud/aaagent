# HCLM-D — Guia rapida

La documentacion completa del proyecto esta en [`dashboard/GUIDE.md`](dashboard/GUIDE.md).

## Inicio rapido

```bash
pip install -e ".[eval]"
z86 init                    # Descarga datos + tokeniza
z86 train --name "mi-run"   # Entrenar
z86 eval latest --judge     # Evaluar con LLM judge
z86 dashboard               # Dashboard en localhost:3000
```

## Datos de entrenamiento

El modelo se entrena con un corpus multilingue de ~50-100M tokens:

| Dominio | % | Dataset | Notas |
|---------|---|---------|-------|
| **Ingles** | 70% | FineWeb-Edu (score >= 2) + TinyStories | Web educativo curado + narrativa simple |
| **Espanol** | 20% | OSCAR 2301 (es) | Corpus web general, filtrado por calidad |
| **Python** | 10% | The Stack v2 (filtrado Python) | Codigo deduplicado, funciones/clases reales |

### Distribucion interna del ingles (dentro del 70%)

- **FineWeb-Edu** (93%): Texto web de alta calidad educativa (HuggingFaceFW/fineweb-edu-score-2)
- **TinyStories** (7%): Narrativa simple, util para bootstrap de estructura y slots

### Preparar datos

```bash
# Corpus por defecto (TinyStories, ingles solamente)
python scripts/train.py --config configs/base.yaml --phase prep

# Corpus multilingue (EN 70% + ES 20% + Python 10%)
python scripts/train.py --config configs/multilingual.yaml --phase prep
```

El pipeline multilingue hace:

1. Descarga cada fuente via HuggingFace datasets (streaming para datasets grandes)
2. Filtra por calidad (largo minimo, dedup basico, archivos con def/class para Python)
3. Anota estructura: `[BLOCK_START]`/`[BLOCK_END]` en parrafos, `[SLOT_START]`/`[SLOT_END]` en dialogos/docstrings
4. Prefija cada documento con tag de idioma: `[LANG_EN]`, `[LANG_ES]`, `[LANG_PY]`
5. Mezcla interleaved (round-robin entre dominios para diversidad)
6. Entrena tokenizer BPE sobre el corpus mezclado (vocab 32,868 = 32,768 BPE + 100 especiales)
7. Tokeniza y guarda como tensores `.pt` (train 95% / val 5%)

### Fuentes y alternativas

| Dominio | Principal | Fallback |
|---------|-----------|----------|
| EN | `HuggingFaceFW/fineweb-edu-score-2` (streaming) | `roneneldan/TinyStories` |
| ES | `oscar-corpus/OSCAR-2301` (language=es) | `wikipedia/20220301.es` |
| Python | `bigcode/the-stack-v2-train-smol-ids` (Python) | `bigcode/the-stack-dedup` |

### Configurar ratios

En `configs/multilingual.yaml`:

```yaml
data:
  dataset: "multilingual"
  data_dir: "data/multilingual"
  tokenizer_path: "data/tokenizer_multilingual.json"
  total_tokens: 50000000      # 50M tokens total
  domain_ratios:
    en: 0.70                  # 35M tokens ingles
    es: 0.20                  # 10M tokens espanol
    python: 0.10              # 5M tokens Python
```

### Budget de tokens recomendado

Para un modelo de 20M parametros:

| Budget | EN (70%) | ES (20%) | PY (10%) | Tiempo aprox (A100) |
|--------|----------|----------|----------|---------------------|
| 50M    | 35M      | 10M      | 5M       | ~3-4h               |
| 100M   | 70M      | 20M      | 10M      | ~6-8h               |

### Tokens especiales por dominio

Cada documento se prefija con un tag de idioma y dominio:

```
[LANG_EN] [DOMAIN_STORY] [BLOCK_START] Once upon a time... [BLOCK_END]
[LANG_ES] [BLOCK_START] Habia una vez... [BLOCK_END]
[LANG_PY] [DOMAIN_CODE] [BLOCK_START] def calculate_area(radius): ... [BLOCK_END]
```

Los tags (IDs 15-24 en el vocabulario) permiten al modelo:
- Condicionar la generacion por idioma
- Especializar clusters por dominio (emergen naturalmente)
- Evaluar per-domain (perplexity EN vs ES vs Python)

## Modelo LLM Judge

El proyecto usa `moonshotai/kimi-k2-instruct-0905` via Groq SDK para evaluacion.
Configurar en `.env`:

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=moonshotai/kimi-k2-instruct-0905
```

## Dashboard

3 tabs: **Overview**, **Evals**, **Versions**.

- Dev: `cd dashboard && bun install && bun run dev`
- Prod: `z86 dashboard --prod`

## RunPod

Ver [`docs/runpod-upload.md`](docs/runpod-upload.md) para subir al pod.

## Configuracion

- Infraestructura: [`config.toml`](config.toml)
- Hiperparametros base: [`configs/base.yaml`](configs/base.yaml)
- Hiperparametros multilingue: [`configs/multilingual.yaml`](configs/multilingual.yaml)
- Variables de entorno: [`.env.example`](.env.example)

## Probar el proyecto

```bash
# 1. Instalar
pip install -e ".[dev,eval]"

# 2. Smoke test (sin datos, sin GPU)
python scripts/smoke_test.py

# 3. Tests unitarios
pytest tests/ -v

# 4. Preparar datos
python scripts/train.py --config configs/multilingual.yaml --phase prep

# 5. Entrenar (config rapida)
python scripts/train.py --config configs/fast.yaml

# 6. Generar texto
python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"

# 7. Evaluar
python scripts/eval.py --checkpoint checkpoints/best.pt --judge
```
