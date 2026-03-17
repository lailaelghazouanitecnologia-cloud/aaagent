#!/bin/bash
# ─────────────────────────────────────────────────────────
# RunPod training session — downloads tokens from Vultr
# and starts training immediately.
#
# Usage: scripts/runpod-session.sh <VULTR_IP> [CONFIG]
#
# Examples:
#   scripts/runpod-session.sh 149.28.100.50
#   scripts/runpod-session.sh 149.28.100.50 configs/ablations/flat_baseline.yaml
#   scripts/runpod-session.sh 149.28.100.50 configs/base.yaml --resume checkpoints/step_5000.pt
# ─────────────────────────────────────────────────────────
set -euo pipefail

VULTR_IP="${1:?Usage: $0 <VULTR_IP> [CONFIG] [--resume CHECKPOINT]}"
CONFIG="${2:-configs/base.yaml}"
RESUME_FLAG=""
RESUME_PATH=""

# Parse optional --resume
shift 2 2>/dev/null || shift 1 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_PATH="$2"; RESUME_FLAG="--resume $2"; shift 2 ;;
        *) shift ;;
    esac
done

REPO_DIR="/workspace/aaagent"
DATA_DIR="$REPO_DIR/data"
REMOTE_DATA="/opt/aaagent/data"

echo "══════════════════════════════════════"
echo "  HCLM-D RunPod Training Session"
echo "══════════════════════════════════════"
echo "  Vultr:  $VULTR_IP"
echo "  Config: $CONFIG"
echo ""

# ── 1. Setup repo ──
echo "[1/5] Setting up repo..."
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull origin claude/hclm-d-documentation-uKBwL 2>/dev/null || true
else
    cd /workspace
    git clone https://github.com/lailaelghazouanitecnologia-cloud/aaagent.git
    cd "$REPO_DIR"
    git checkout claude/hclm-d-documentation-uKBwL
fi

# ── 2. Install deps (only if needed) ──
echo "[2/5] Checking dependencies..."
if ! python -c "import model.lm" 2>/dev/null; then
    echo "  Installing Python dependencies..."
    pip install -q ".[dev]"
else
    echo "  ✓ Dependencies already installed"
fi

# ── 3. Download tokens from Vultr ──
echo "[3/5] Downloading tokens from Vultr..."
mkdir -p "$DATA_DIR/tinystories"

# Check which files need downloading
NEED_DOWNLOAD=false
for f in tokenizer.json; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        NEED_DOWNLOAD=true
        break
    fi
done
for f in train_tokens.pt val_tokens.pt; do
    if [ ! -f "$DATA_DIR/tinystories/$f" ]; then
        NEED_DOWNLOAD=true
        break
    fi
done

if [ "$NEED_DOWNLOAD" = true ]; then
    echo "  Downloading tokenizer.json (~200KB)..."
    scp -o StrictHostKeyChecking=no "root@${VULTR_IP}:${REMOTE_DATA}/tokenizer.json" "$DATA_DIR/tokenizer.json"

    echo "  Downloading train_tokens.pt (~2GB, may take 1-2 min)..."
    scp -o StrictHostKeyChecking=no "root@${VULTR_IP}:${REMOTE_DATA}/tinystories/train_tokens.pt" "$DATA_DIR/tinystories/train_tokens.pt"

    echo "  Downloading val_tokens.pt (~20MB)..."
    scp -o StrictHostKeyChecking=no "root@${VULTR_IP}:${REMOTE_DATA}/tinystories/val_tokens.pt" "$DATA_DIR/tinystories/val_tokens.pt"
else
    echo "  ✓ All token files already present"
fi

# Verify
echo "  Verifying files..."
for f in "$DATA_DIR/tokenizer.json" "$DATA_DIR/tinystories/train_tokens.pt" "$DATA_DIR/tinystories/val_tokens.pt"; do
    if [ ! -f "$f" ]; then
        echo "  ✗ MISSING: $f"
        echo "  Make sure Vultr prep completed. Run setup-vultr.sh first."
        exit 1
    fi
done
echo "  ✓ train_tokens.pt: $(du -h $DATA_DIR/tinystories/train_tokens.pt | cut -f1)"
echo "  ✓ val_tokens.pt:   $(du -h $DATA_DIR/tinystories/val_tokens.pt | cut -f1)"
echo "  ✓ tokenizer.json:  $(du -h $DATA_DIR/tokenizer.json | cut -f1)"

# ── 4. GPU check ──
echo "[4/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  ✓ $GPU_NAME ($GPU_MEM)"
else
    echo "  ⚠ nvidia-smi not found. Training will be slow on CPU."
fi

# ── 5. Start training ──
echo "[5/5] Starting training..."
echo "  Config: $CONFIG"
if [ -n "$RESUME_PATH" ]; then
    echo "  Resuming from: $RESUME_PATH"
fi
echo ""
echo "──────────────────────────────────────"
echo "  Training starting now. Monitor at:"
echo "  https://z86.dev (if dashboard configured)"
echo "──────────────────────────────────────"
echo ""

export DASHBOARD_URL="http://${VULTR_IP}:3000"
python scripts/train.py --config "$CONFIG" --phase train $RESUME_FLAG
