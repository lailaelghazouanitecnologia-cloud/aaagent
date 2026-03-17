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
- Hiperparametros: [`configs/base.yaml`](configs/base.yaml)
- Variables de entorno: [`.env.example`](.env.example)
