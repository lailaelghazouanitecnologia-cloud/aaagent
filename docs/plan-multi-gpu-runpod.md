# Plan: Multi-GPU Training con Volumen Compartido en RunPod

## Resumen

Extender el sistema HCLM-D para soportar entrenamiento distribuido multi-GPU
en RunPod, usando un **Network Volume** compartido y configuraciones en **TOML**.

---

## Arquitectura

```
┌──────────────────────────────────────────────────┐
│            RunPod Network Volume (NFS)           │
│            /runpod-vol  (persistente)            │
│                                                  │
│  📁 data/          ← datasets tokenizados        │
│  📁 checkpoints/   ← checkpoints compartidos     │
│  📁 configs/       ← archivos .toml              │
│  📁 logs/          ← training logs               │
└──────────┬───────────┬───────────┬───────────────┘
           │           │           │
     ┌─────▼───┐ ┌─────▼───┐ ┌────▼────┐
     │  GPU 0  │ │  GPU 1  │ │  GPU 2  │  ... hasta N GPUs
     │  rank=0 │ │  rank=1 │ │  rank=2 │
     │  A100   │ │  A100   │ │  A100   │
     └─────────┘ └─────────┘ └─────────┘
           │           │           │
           └───────────┼───────────┘
                       │
               PyTorch DDP (NCCL)
            sincroniza gradientes
```

### Estrategia: Multi-GPU en 1 Pod (DDP)

- 1 pod con 2x o 4x GPUs (conectadas por NVLink)
- `torchrun --nproc_per_node=N` lanza N procesos
- Cada proceso entrena con un subconjunto del batch
- Los gradientes se sincronizan automáticamente vía NCCL
- El volumen monta datos y checkpoints compartidos

---

## Componentes a Implementar

### 1. Configuración TOML (`configs/distributed.toml`)

Nuevo archivo de configuración en TOML para training distribuido:

```toml
[volume]
name = "hclm-d-vol"
size_gb = 50
mount_path = "/runpod-vol"
region = "US"

[pod]
gpu_type = "NVIDIA A100 80GB PCIe"
gpu_count = 4
cloud_type = "COMMUNITY"  # o "SECURE"
image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
container_disk_gb = 20
template_id = ""  # opcional: usar template preconfigurado

[distributed]
backend = "nccl"
strategy = "ddp"                  # "ddp" o "fsdp"
find_unused_parameters = false
gradient_as_bucket_view = true
broadcast_buffers = true

[sync]
include = ["data/", "configs/", "scripts/"]
exclude = ["*.pyc", "__pycache__", ".git"]

[bootstrap]
repo_url = ""          # se auto-detecta del git remote
branch = "main"
pip_install = true
pre_commands = []      # comandos extra antes del training
post_commands = []     # comandos después del training

[training]
config = "base"        # referencia a configs/base.yaml del modelo
run_name = ""          # auto-generado si vacío
resume = false
wandb = true
dashboard_url = ""

[costs]
budget_limit_usd = 25.0       # alerta cuando se supere
auto_stop_on_complete = true   # parar pod al terminar
```

### 2. Cloud Volume Manager (`cloud/volume.py`)

Nuevo módulo para gestionar Network Volumes de RunPod:

```python
# Funciones principales:
- create_volume(name, size_gb, region) -> VolumeInfo
- list_volumes() -> list[VolumeInfo]
- delete_volume(volume_id)
- get_volume(volume_id) -> VolumeInfo
```

Usa la **GraphQL API** de RunPod (igual que `cloud/runpod.py` actual).

### 3. Distributed Training Setup (`training/distributed.py`)

Nuevo módulo para inicializar DDP:

```python
# Funciones principales:
- setup_distributed() -> DistributedContext
  # Lee RANK, LOCAL_RANK, WORLD_SIZE de env vars (puestas por torchrun)
  # Inicializa process group con backend NCCL

- wrap_model(model, config) -> DistributedDataParallel
  # Envuelve el modelo en DDP

- get_distributed_sampler(dataset) -> DistributedSampler
  # Crea sampler que reparte datos entre GPUs

- cleanup_distributed()
  # Destruye process group

- is_main_process() -> bool
  # Solo rank 0 hace logging, checkpointing, etc.
```

### 4. Modificar Trainer (`training/trainer.py`)

Adaptar el trainer existente (650+ líneas) para soportar DDP:

- **DataLoader**: Usar `DistributedSampler` cuando hay múltiples GPUs
- **Modelo**: Envolver en `DistributedDataParallel`
- **Checkpointing**: Solo `rank=0` guarda checkpoints
- **Logging**: Solo `rank=0` reporta al dashboard/wandb
- **Batch size**: Dividir `batch_size` entre `world_size` (batch efectivo = batch_size)
- **Learning rate**: Escalar linearly con world_size (opcional, configurable)

### 5. Config Loader TOML (`cloud/config.py`)

Utilidad para cargar y validar configs TOML:

```python
- load_distributed_config(path: str) -> DistributedConfig
  # Carga TOML, valida campos, retorna dataclass tipada

- DistributedConfig (dataclass)
  # Tipado fuerte para todos los campos del TOML
```

### 6. Nuevos Comandos CLI (`cli/cmd_cloud.py`)

Extender los comandos existentes de `z86 cloud`:

```
z86 cloud volume create [--config distributed.toml]
    → Crea Network Volume en RunPod

z86 cloud volume list
    → Lista volúmenes existentes

z86 cloud volume sync [--config distributed.toml]
    → Sube data/, configs/, scripts/ al volumen via pod temporal

z86 cloud volume delete <volume_id>
    → Elimina un volumen

z86 cloud train [--config distributed.toml]
    → Crea pod multi-GPU, monta volumen, lanza torchrun
    → Equivale a: volume create + pod create + bootstrap + train

z86 cloud pull [--config distributed.toml]
    → Descarga checkpoints del volumen a local

z86 cloud cost
    → Muestra coste acumulado de pods activos
```

### 7. Bootstrap Script (`scripts/pod_bootstrap.sh`)

Script que se ejecuta al arrancar el pod:

```bash
#!/bin/bash
# 1. Clonar repo (o pull si ya existe)
# 2. pip install -e '.[dev]'
# 3. Ejecutar pre_commands del TOML
# 4. Lanzar torchrun con la config
# 5. Al terminar, ejecutar post_commands
# 6. Si auto_stop_on_complete, parar pod
```

---

## Orden de Implementación

| Paso | Archivo(s) | Descripción |
|------|-----------|-------------|
| 1 | `configs/distributed.toml` | Crear config TOML con todos los parámetros |
| 2 | `cloud/config.py` | Loader TOML → dataclass tipada |
| 3 | `cloud/volume.py` | CRUD de Network Volumes (GraphQL) |
| 4 | `training/distributed.py` | Setup DDP (process group, sampler, wrapper) |
| 5 | `training/trainer.py` | Adaptar trainer para DDP |
| 6 | `scripts/pod_bootstrap.sh` | Script de arranque del pod |
| 7 | `scripts/train.py` | Adaptar para leer distributed config |
| 8 | `cli/cmd_cloud.py` | Nuevos subcomandos (volume, train, pull, cost) |
| 9 | Tests | Tests unitarios para distributed + volume |

---

## Flujo Completo del Usuario

```bash
# 1. Configurar (editar TOML según necesidades)
vim configs/distributed.toml

# 2. Un solo comando para todo
z86 cloud train --config distributed.toml
#    → Crea volumen "hclm-d-vol" (50GB)
#    → Sube datos al volumen
#    → Crea pod 4x A100
#    → Monta volumen en /runpod-vol
#    → Ejecuta bootstrap (clone, install, torchrun)
#    → Muestra logs en tiempo real

# 3. Monitorear
z86 cloud status          # ver pod + coste
z86 dashboard             # métricas en tiempo real

# 4. Descargar resultados
z86 cloud pull            # baja checkpoints a local

# 5. Limpiar
z86 cloud stop <pod_id>   # el volumen persiste
z86 cloud volume delete <vol_id>  # cuando ya no se necesite
```

---

## Costes Estimados (RunPod Community)

| Config | GPU | $/hora | 50k steps (~6h) | 100k steps (~12h) |
|--------|-----|--------|------------------|--------------------|
| 1x A100 | 80GB | ~$1.64 | ~$10 | ~$20 |
| 2x A100 | 80GB | ~$3.28 | ~$10 | ~$20 |
| 4x A100 | 80GB | ~$6.56 | ~$10 | ~$20 |
| Volume | 50GB | ~$0.07/h | +$0.42 | +$0.84 |

> Nota: Más GPUs = menos horas, coste total similar.

---

## Dependencias Nuevas

```toml
# En pyproject.toml
[project.dependencies]
# Ya existente: torch >= 2.1
tomli = ">=2.0"    # parser TOML (Python < 3.11)
# tomllib ya incluido en Python >= 3.11
```

---

## Notas Técnicas

1. **TOML vs YAML**: Los configs de modelo (`base.yaml`, etc.) siguen en YAML.
   Solo la config de distributed/cloud usa TOML. Esto separa las preocupaciones:
   YAML = modelo/training, TOML = infraestructura/cloud.

2. **DDP vs FSDP**: Para 20M params, DDP es suficiente. FSDP se deja como opción
   en el TOML para futuros modelos más grandes donde el modelo no quepa en 1 GPU.

3. **Volumen vs Pod Storage**: El volumen persiste entre sesiones. Los datos se
   suben una vez y se reusan en múltiples entrenamientos.

4. **Solo rank=0 escribe**: Checkpoints, logs, métricas — solo el proceso
   principal escribe al volumen para evitar conflictos.
