# Plan: Sistema Escalable de Training Distribuido en RunPod

## Objetivo

Construir un sistema de training distribuido **escalable** que funcione desde
el modelo actual (20M) hasta modelos futuros de 1B+ parámetros. El sistema
debe abstraer la complejidad de multi-GPU y multi-nodo detrás de una config
TOML y un solo comando CLI.

---

## Niveles de Escala

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ESCALABILIDAD POR NIVELES                       │
├──────────┬──────────────┬───────────────┬──────────────────────────┤
│ Nivel    │ Modelo       │ Estrategia    │ Infraestructura          │
├──────────┼──────────────┼───────────────┼──────────────────────────┤
│ 1 (hoy)  │ 20M params   │ Single GPU    │ 1 pod, 1x A6000/A100    │
│ 2        │ 100-500M     │ DDP           │ 1 pod, 2-8x GPU         │
│ 3        │ 500M-2B      │ FSDP          │ 1 pod, 4-8x GPU         │
│ 4 (futuro)│ 2B+         │ FSDP + Multi  │ N pods, NVLink/InfiniBand│
└──────────┴──────────────┴───────────────┴──────────────────────────┘
```

---

## Arquitectura del Sistema

```
                    ┌──────────────────────┐
                    │   configs/*.toml     │
                    │   (infraestructura)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   z86 cloud train    │
                    │   (orquestador)      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────┐
     │ Volume Manager│ │ Pod Manager  │ │  Bootstrap  │
     │ (cloud/       │ │ (cloud/      │ │  (scripts/  │
     │  volume.py)   │ │  runpod.py)  │ │  bootstrap) │
     └───────────────┘ └──────────────┘ └──────┬──────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │  training/           │
                                    │  distributed.py      │
                                    │                      │
                                    │  ┌─────────────────┐ │
                                    │  │ auto-detect:    │ │
                                    │  │  1 GPU → raw    │ │
                                    │  │  N GPU → DDP    │ │
                                    │  │  big  → FSDP   │ │
                                    │  └─────────────────┘ │
                                    └──────────────────────┘
                                               │
                    ┌──────────────────────────────────────────┐
                    │           RunPod Network Volume          │
                    │           /runpod-vol (NFS)              │
                    │                                          │
                    │   data/         ← datasets               │
                    │   checkpoints/  ← sharded checkpoints    │
                    │   configs/      ← .toml + .yaml          │
                    │   logs/         ← per-rank logs          │
                    └──────────────────────────────────────────┘
```

---

## 1. Configuración TOML

### `configs/distributed.toml` — Config principal de infraestructura

```toml
# ─── Volumen persistente ───
[volume]
name = "hclm-d-vol"
size_gb = 50                     # escalar según dataset
mount_path = "/runpod-vol"
region = "US"

# ─── Pod ───
[pod]
gpu_type = "NVIDIA A100 80GB PCIe"
gpu_count = 4                    # cuántas GPUs por pod
num_pods = 1                     # para multi-nodo futuro
cloud_type = "COMMUNITY"
image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
container_disk_gb = 30

# ─── Estrategia distribuida ───
[distributed]
# "auto" = selecciona según gpu_count y tamaño del modelo
# "ddp"  = DistributedDataParallel (modelo cabe en 1 GPU)
# "fsdp" = FullyShardedDataParallel (modelo NO cabe en 1 GPU)
strategy = "auto"
backend = "nccl"

  [distributed.ddp]
  find_unused_parameters = false
  gradient_as_bucket_view = true
  static_graph = true            # optimización para grafos fijos

  [distributed.fsdp]
  sharding_strategy = "FULL_SHARD"     # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
  mixed_precision = true               # bf16 compute, fp32 params
  activation_checkpointing = false     # activar para modelos muy grandes
  cpu_offload = false                  # offload params a RAM (último recurso)
  auto_wrap_policy = "size_based"      # size_based | transformer_based
  min_num_params = 1_000_000           # para size_based wrapping
  forward_prefetch = true
  backward_prefetch = "BACKWARD_PRE"   # BACKWARD_PRE | BACKWARD_POST
  limit_all_gathers = true

  [distributed.multi_node]
  enabled = false                       # futuro: multi-pod
  master_port = 29500
  rdma = false                         # InfiniBand RDMA

# ─── Scaling del batch/lr ───
[scaling]
# Cómo escalar LR con más GPUs
lr_scaling = "linear"            # "linear" | "sqrt" | "none"
# batch_size en config YAML es PER GPU, el efectivo = batch * world_size
per_gpu_batch = true
# Warmup extra al escalar LR
scaled_warmup_factor = 1.0       # multiplicar warmup steps por este factor

# ─── Checkpointing distribuido ───
[checkpointing]
# "full" = solo rank 0 guarda modelo completo (DDP)
# "sharded" = cada rank guarda su shard (FSDP)
# "auto" = full para DDP, sharded para FSDP
strategy = "auto"
save_on_rank_0_only = true       # para DDP
async_save = false               # guardar en background (futuro)

# ─── Sincronización de archivos ───
[sync]
include = ["data/", "configs/", "scripts/"]
exclude = ["*.pyc", "__pycache__", ".git", "node_modules", "dashboard/"]
# Comprimir antes de subir al volumen
compress = true

# ─── Bootstrap del pod ───
[bootstrap]
repo_url = ""                    # auto-detecta de git remote
branch = "main"
python_version = "3.11"
pip_extras = ["eval"]            # extras de pyproject.toml a instalar
pre_commands = []                # comandos antes del training
post_commands = []               # comandos después del training
# Script custom de bootstrap (sobreescribe el default)
custom_script = ""

# ─── Training ───
[training]
config = "base"                  # referencia a configs/{config}.yaml
run_name = ""                    # auto si vacío: "run_{timestamp}"
resume = false                   # reanudar desde último checkpoint
resume_from = ""                 # checkpoint específico
wandb = true
dashboard_url = ""

# ─── Presupuesto y auto-gestión ───
[budget]
limit_usd = 50.0                 # alerta al superar
hard_limit_usd = 100.0           # auto-stop al superar
auto_stop_on_complete = true     # parar pod al terminar training
auto_terminate_on_complete = false  # destruir pod (volumen persiste)
alert_webhook = ""               # URL para alertas (Slack, Discord, etc.)

# ─── Presets reutilizables ───
[presets.dev]
gpu_type = "NVIDIA RTX A6000"
gpu_count = 1
strategy = "auto"

[presets.train]
gpu_type = "NVIDIA A100 80GB PCIe"
gpu_count = 4
strategy = "auto"

[presets.train-large]
gpu_type = "NVIDIA H100 80GB HBM3"
gpu_count = 8
strategy = "fsdp"

[presets.multi-node]
gpu_type = "NVIDIA H100 80GB HBM3"
gpu_count = 8
num_pods = 4
strategy = "fsdp"
```

---

## 2. Componentes a Implementar

### 2.1 `cloud/config.py` — Loader TOML con dataclasses tipadas

```python
@dataclass
class VolumeConfig:
    name: str
    size_gb: int
    mount_path: str
    region: str

@dataclass
class PodConfig:
    gpu_type: str
    gpu_count: int
    num_pods: int
    cloud_type: str
    image: str
    container_disk_gb: int

@dataclass
class DDPConfig:
    find_unused_parameters: bool
    gradient_as_bucket_view: bool
    static_graph: bool

@dataclass
class FSDPConfig:
    sharding_strategy: str
    mixed_precision: bool
    activation_checkpointing: bool
    cpu_offload: bool
    auto_wrap_policy: str
    min_num_params: int
    forward_prefetch: bool
    backward_prefetch: str
    limit_all_gathers: bool

@dataclass
class DistributedConfig:
    strategy: str               # "auto" | "ddp" | "fsdp"
    backend: str
    ddp: DDPConfig
    fsdp: FSDPConfig

@dataclass
class ScalingConfig:
    lr_scaling: str
    per_gpu_batch: bool
    scaled_warmup_factor: float

@dataclass
class BudgetConfig:
    limit_usd: float
    hard_limit_usd: float
    auto_stop_on_complete: bool
    auto_terminate_on_complete: bool
    alert_webhook: str

@dataclass
class CloudConfig:
    """Raíz de la config TOML."""
    volume: VolumeConfig
    pod: PodConfig
    distributed: DistributedConfig
    scaling: ScalingConfig
    budget: BudgetConfig
    # ... sync, bootstrap, training, checkpointing, presets

def load_cloud_config(path: str) -> CloudConfig: ...
def resolve_preset(config: CloudConfig, preset: str) -> CloudConfig: ...
```

- Usa `tomllib` (Python 3.11+) con fallback a `tomli`
- Validación estricta de tipos y valores permitidos
- `resolve_preset()` fusiona preset con config base

---

### 2.2 `cloud/volume.py` — Network Volume Manager (GraphQL)

```python
class VolumeManager:
    """CRUD de Network Volumes en RunPod."""

    def create(name, size_gb, region) -> VolumeInfo
    def list() -> list[VolumeInfo]
    def get(volume_id) -> VolumeInfo
    def delete(volume_id) -> bool
    def resize(volume_id, new_size_gb) -> VolumeInfo  # escalar volumen
```

Queries GraphQL de RunPod:
- `createNetworkVolume`
- `networkVolumes`
- `deleteNetworkVolume`

---

### 2.3 `training/distributed.py` — Motor de Training Distribuido (CORE)

El componente más importante. Debe ser **agnóstico al modelo** para que funcione
con cualquier modelo futuro.

```python
class DistributedEngine:
    """Abstrae DDP, FSDP y single-GPU detrás de una interfaz unificada."""

    def __init__(self, config: DistributedConfig):
        self.strategy = self._resolve_strategy(config)

    def _resolve_strategy(self, config) -> str:
        """Auto-detect: cuenta GPUs y estima tamaño del modelo."""
        if config.strategy != "auto":
            return config.strategy
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size == 1:
            return "single"
        # Para auto entre DDP y FSDP, se decide en setup() con el modelo
        return "auto"

    def setup(self) -> None:
        """Inicializa process group (NCCL)."""
        # Lee RANK, LOCAL_RANK, WORLD_SIZE de env (puestas por torchrun)
        # Llama a dist.init_process_group()
        # Setea CUDA device al LOCAL_RANK

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Envuelve modelo según estrategia."""
        if self.strategy == "single":
            return model
        elif self.strategy == "ddp":
            return DistributedDataParallel(model, ...)
        elif self.strategy == "fsdp":
            return FullyShardedDataParallel(model, ...)
        elif self.strategy == "auto":
            # Heurística: si model params > gpu_memory * 0.6 → FSDP
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            gpu_mem = torch.cuda.get_device_properties(self.local_rank).total_mem
            if param_bytes > gpu_mem * 0.6:
                return self._wrap_fsdp(model)
            return self._wrap_ddp(model)

    def get_sampler(self, dataset) -> Sampler | None:
        """DistributedSampler o None para single GPU."""

    def scale_lr(self, base_lr: float, config: ScalingConfig) -> float:
        """Escala LR según world_size y estrategia configurada."""
        if config.lr_scaling == "linear":
            return base_lr * self.world_size
        elif config.lr_scaling == "sqrt":
            return base_lr * math.sqrt(self.world_size)
        return base_lr

    def scale_warmup(self, warmup_steps: int, config: ScalingConfig) -> int:
        """Ajusta warmup al escalar LR."""
        return int(warmup_steps * config.scaled_warmup_factor)

    def save_checkpoint(self, model, optimizer, step, path):
        """Guarda checkpoint según estrategia."""
        if not self.is_main_process:
            return
        if self.strategy == "fsdp":
            # FSDP: full_state_dict o sharded
            ...
        else:
            # DDP/single: state_dict normal
            ...

    def load_checkpoint(self, model, optimizer, path):
        """Carga checkpoint según estrategia."""

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    @property
    def local_rank(self) -> int: ...

    def barrier(self): ...

    def cleanup(self): ...

    def log(self, msg, *args):
        """Solo rank 0 loguea."""
        if self.is_main_process:
            logger.info(msg, *args)
```

**FSDP Wrapping Policies:**

```python
def _get_fsdp_wrap_policy(self, model, config):
    """Política de wrapping para FSDP."""
    if config.fsdp.auto_wrap_policy == "transformer_based":
        # Wrappea cada TransformerBlock individualmente
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from model.transformer.block import TransformerBlock
        return transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerBlock},
        )
    else:
        # Size-based: wrappea módulos con > min_num_params
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        return size_based_auto_wrap_policy(
            min_num_params=config.fsdp.min_num_params,
        )
```

---

### 2.4 Adaptar `training/trainer.py` — Integrar DistributedEngine

Cambios mínimos pero críticos al trainer existente:

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config,
                 dist_engine: DistributedEngine | None = None):
        self.dist = dist_engine

        # Device: usa local_rank si distribuido
        if self.dist:
            self.device = torch.device(f"cuda:{self.dist.local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(self.device)

        # Wrap distribuido (después de .to(device), antes de compile)
        if self.dist:
            model = self.dist.wrap_model(model)

        # torch.compile (después de DDP/FSDP wrap)
        if train_cfg.get("compile", True) and self.device.type == "cuda":
            self.model = _try_compile(model)

        # LR scaling
        if self.dist:
            scaling_cfg = config.get("scaling", {})
            lr = self.dist.scale_lr(train_cfg.get("lr", 3e-4), scaling_cfg)
            train_cfg["lr"] = lr

    # DataLoader: usar DistributedSampler
    # → Mover creación del DataLoader al Trainer o pasar sampler

    def save(self):
        """Solo rank 0 guarda (o sharded para FSDP)."""
        if self.dist:
            self.dist.save_checkpoint(self.model, self.optimizer,
                                       self.global_step, path)
        else:
            # original save logic
            ...

    def _log_step(self, loss_output):
        """Solo rank 0 loguea."""
        if self.dist and not self.dist.is_main_process:
            return
        # ... log original ...

    def _report_to_dashboard(self, loss_output):
        if self.dist and not self.dist.is_main_process:
            return
        # ... report original ...
```

---

### 2.5 Adaptar `scripts/train.py` — Entry point distribuido

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dist-config", default=None,
                        help="TOML de config distribuida")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # Setup distribuido (si hay más de 1 GPU)
    dist_engine = None
    if args.dist_config or "WORLD_SIZE" in os.environ:
        from cloud.config import load_cloud_config
        from training.distributed import DistributedEngine

        cloud_cfg = load_cloud_config(args.dist_config) if args.dist_config else None
        dist_engine = DistributedEngine(cloud_cfg.distributed if cloud_cfg else None)
        dist_engine.setup()

    # Cargar config YAML del modelo (no cambia)
    config = load_config(args.config)

    # Build model
    model = HCLMD(ModelConfig.from_dict(config))

    # DataLoader con DistributedSampler si aplica
    sampler = dist_engine.get_sampler(train_dataset) if dist_engine else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        ...
    )

    # Trainer con engine distribuido
    trainer = Trainer(model, train_loader, val_loader, config,
                      dist_engine=dist_engine)
    trainer.train()

    if dist_engine:
        dist_engine.cleanup()
```

**Lanzamiento:**
```bash
# Single GPU (como ahora, no cambia nada)
python scripts/train.py --config configs/base.yaml

# Multi-GPU en el pod
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/base.yaml \
    --dist-config configs/distributed.toml
```

---

### 2.6 `scripts/pod_bootstrap.sh` — Bootstrap escalable

```bash
#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-/runpod-vol/configs/distributed.toml}"
MOUNT="/runpod-vol"
REPO_DIR="$MOUNT/repo"

echo "=== HCLM-D Pod Bootstrap ==="
echo "GPUs disponibles: $(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. Clone/pull repo
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR" && git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# 2. Install
pip install -e '.[eval]' --quiet

# 3. Pre-commands (del TOML)
# Parseados por el orquestador y pasados como env

# 4. Detectar GPUs y lanzar
NUM_GPUS=$(nvidia-smi -L | wc -l)
MODEL_CONFIG="${MODEL_CONFIG:-configs/base.yaml}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "=== Launching distributed training: $NUM_GPUS GPUs ==="
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        scripts/train.py \
        --config "$MODEL_CONFIG" \
        --dist-config "$CONFIG_PATH"
else
    echo "=== Launching single-GPU training ==="
    python scripts/train.py --config "$MODEL_CONFIG"
fi

# 5. Post-commands
# ...

# 6. Auto-stop si configurado
if [ "${AUTO_STOP:-true}" = "true" ]; then
    echo "Training complete. Stopping pod..."
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
```

---

### 2.7 Nuevos comandos CLI (`cli/cmd_cloud.py`)

```
z86 cloud volume create [--config distributed.toml]
z86 cloud volume list
z86 cloud volume sync [--config distributed.toml]
z86 cloud volume delete <volume_id>
z86 cloud volume resize <volume_id> --size <gb>

z86 cloud train [--config distributed.toml] [--preset train]
    → Orquesta: volumen + pod + bootstrap + torchrun
    → Muestra logs en tiempo real (streamed SSH)

z86 cloud pull [--config distributed.toml]
    → Descarga checkpoints/logs del volumen a local

z86 cloud cost
    → Coste acumulado y estimación hasta fin del training

z86 cloud scale <pod_id> --gpus 8
    → Escalar GPUs (crea nuevo pod, migra al mismo volumen)
```

---

### 2.8 Adaptar `training/checkpointing.py` — Checkpoints distribuidos

```python
def save_distributed_checkpoint(model, optimizer, step, path, strategy):
    """Guarda checkpoint compatible con la estrategia distribuida."""
    if strategy == "fsdp":
        # FSDP: obtener full_state_dict (reúne shards en rank 0)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
            if dist.get_rank() == 0:
                torch.save({"model": state, "optimizer": optim_state, "step": step}, path)
    else:
        # DDP: model.module.state_dict() (quitar wrapper DDP)
        raw_model = model.module if hasattr(model, "module") else model
        if dist.get_rank() == 0:
            torch.save({"model": raw_model.state_dict(), ...}, path)

def load_distributed_checkpoint(path, model, optimizer, strategy):
    """Carga checkpoint y distribuye a todos los ranks."""
    if strategy == "fsdp":
        # Cada rank carga su shard
        ...
    else:
        # Broadcast del rank 0
        ...
```

---

## 3. Orden de Implementación

| Paso | Archivo(s) | Prioridad | Descripción |
|------|-----------|-----------|-------------|
| 1 | `configs/distributed.toml` | ALTA | Config TOML con todos los niveles de escala |
| 2 | `cloud/config.py` | ALTA | Loader TOML → dataclasses tipadas + validación |
| 3 | `training/distributed.py` | ALTA | DistributedEngine (DDP + FSDP + auto-detect) |
| 4 | `training/trainer.py` | ALTA | Integrar DistributedEngine en trainer existente |
| 5 | `training/checkpointing.py` | ALTA | Checkpoints distribuidos (DDP + FSDP) |
| 6 | `scripts/train.py` | ALTA | Añadir --dist-config + setup distribuido |
| 7 | `cloud/volume.py` | MEDIA | CRUD Network Volumes (GraphQL) |
| 8 | `scripts/pod_bootstrap.sh` | MEDIA | Script arranque auto-detect GPUs |
| 9 | `cli/cmd_cloud.py` + `cli/main.py` | MEDIA | Nuevos subcomandos (volume, train, pull, cost, scale) |
| 10 | `pyproject.toml` | BAJA | Añadir tomli a dependencias |
| 11 | `tests/test_distributed.py` | BAJA | Tests unitarios con mocks |

---

## 4. Garantías de Escalabilidad

| Aspecto | Hoy (20M) | Futuro (1B+) |
|---------|-----------|--------------|
| **Estrategia** | Auto → DDP | Auto → FSDP |
| **Checkpoints** | Full state en rank 0 | Sharded por rank |
| **Memoria** | Modelo cabe en 1 GPU | FSDP shardea + activation checkpointing |
| **Batch** | 64 per GPU × 4 = 256 efectivo | Configurar grad accumulation |
| **LR** | Scale linear con world_size | Configurable (linear/sqrt/none) |
| **Volumen** | 50GB suficiente | Resize dinámico |
| **GPUs** | 1-4x A100 | 8x H100 multi-nodo |
| **Config** | Mismo TOML, cambiar preset | `--preset train-large` |
| **Código** | 0 cambios | 0 cambios (todo en config) |

---

## 5. Flujo del Usuario (Hoy y Futuro)

```bash
# ─── Hoy: 20M params, 4x A100 ───
z86 cloud train --preset train
# → Crea volumen, pod 4xA100, DDP automático

# ─── Mañana: 500M params, 8x A100 ───
z86 cloud train --preset train-large
# → Mismo comando, FSDP automático, activation checkpointing

# ─── Futuro: 2B+ params, 4 pods × 8x H100 ───
z86 cloud train --preset multi-node
# → Multi-nodo automático (misma interfaz)

# ─── Siempre ───
z86 cloud status          # pods activos + coste
z86 cloud cost            # presupuesto en tiempo real
z86 cloud pull            # descargar checkpoints
z86 cloud scale --gpus 8  # escalar sin parar
```

---

## 6. Dependencias Nuevas

```toml
# pyproject.toml
[project.dependencies]
# existentes...
tomli = { version = ">=2.0", python = "<3.11" }
# tomllib ya incluido en Python 3.11+
```

No se requieren dependencias adicionales — PyTorch 2.1+ ya incluye
`torch.distributed`, DDP y FSDP.

---

## 7. Separación Config: TOML vs YAML

| Archivo | Formato | Qué define |
|---------|---------|------------|
| `configs/base.yaml` | YAML | Arquitectura modelo, hiperparámetros, losses |
| `configs/fast.yaml` | YAML | Variante rápida del modelo |
| `configs/distributed.toml` | TOML | Infraestructura cloud, GPUs, estrategia distribuida |

**Principio:** YAML = "qué entreno", TOML = "dónde y cómo lo ejecuto".
El mismo YAML funciona en 1 GPU local o en 8 H100s. El TOML solo controla
la infraestructura.
