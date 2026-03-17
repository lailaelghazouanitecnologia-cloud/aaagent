# RunPod — Subir proyecto y arrancar

## Opción 1: SSH directo

```bash
# Conectar al pod
ssh USER@ssh.runpod.io -i ~/.ssh/id_ed25519

# Dentro del pod
cd /workspace
git clone https://github.com/YOUR_ORG/aaagent aaagent
cd aaagent
pip install -e ".[eval]"
set -a && source .env && set +a
z86 dashboard > dashboard.log 2>&1 &
z86 train --name "base-run" --dashboard http://localhost:3000
```

## Opcion 2: Script de bootstrap

```bash
# Dentro del pod
cd /workspace/aaagent
bash scripts/pod_bootstrap.sh
```

## Opcion 3: Helper Python (paramiko)

```bash
# Desde tu maquina local — subir proyecto
python scripts/runpod_ssh.py upload \
  --host ssh.runpod.io --user USER --key ~/.ssh/id_ed25519 \
  --local-dir . --remote-dir /workspace/aaagent

# Ejecutar comando remoto
python scripts/runpod_ssh.py exec \
  --host ssh.runpod.io --user USER --key ~/.ssh/id_ed25519 \
  -- "cd /workspace/aaagent && pip install -e '.[eval]' && nvidia-smi"

# Arrancar training
python scripts/runpod_ssh.py exec \
  --host ssh.runpod.io --user USER --key ~/.ssh/id_ed25519 \
  -- "cd /workspace/aaagent && set -a && source .env && set +a && z86 train --name 'base-run'"
```

## Variables de entorno (.env)

Crear `.env` en la raiz del proyecto (ignorado por git):

```bash
GROQ_API_KEY=gsk_...
GROQ_MODEL=moonshotai/kimi-k2-instruct-0905
RUNPOD_API_KEY=rpa_...
CF_API_TOKEN=...
CF_ZONE_ID=...
CUDA_VISIBLE_DEVICES=0
DASHBOARD_URL=http://localhost:3000
```
