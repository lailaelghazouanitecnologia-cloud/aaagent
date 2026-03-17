#!/bin/bash
# ─────────────────────────────────────────────────────────────
# HCLM-D — RunPod Bootstrap Script
#
# Clones (or updates) the repo into a RunPod volume and installs
# dependencies. Run this inside the pod after SSH-ing in.
#
# Usage:
#   bash scripts/pod_bootstrap.sh
#
# Environment (all optional, with defaults):
#   REPO_URL        — Git clone URL (default: current origin)
#   REPO_BRANCH     — Branch to checkout (default: main)
#   MOUNT_PATH      — Volume mount point (default: /workspace)
#   RUN_INIT        — Run z86 init after install? 1/0 (default: 0)
#   START_DASHBOARD — Start dashboard in background? 1/0 (default: 0)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

MOUNT="${MOUNT_PATH:-/workspace}"
REPO="${REPO_URL:-}"
BRANCH="${REPO_BRANCH:-main}"
PROJECT_DIR="${MOUNT}/aaagent"

echo "══════════════════════════════════════════════"
echo "  HCLM-D Pod Bootstrap"
echo "══════════════════════════════════════════════"
echo "  Mount:   ${MOUNT}"
echo "  Project: ${PROJECT_DIR}"
echo ""

# ── 1. Clone or update ──────────────────────────────────────
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "→ Repo exists, pulling latest..."
    cd "${PROJECT_DIR}"
    git fetch origin "${BRANCH}" && git checkout "${BRANCH}" && git pull origin "${BRANCH}"
elif [ -n "${REPO}" ]; then
    echo "→ Cloning repo..."
    git clone -b "${BRANCH}" "${REPO}" "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
else
    echo "→ Using existing directory (no REPO_URL set)"
    cd "${PROJECT_DIR}"
fi

# ── 2. Install Python deps ──────────────────────────────────
echo "→ Installing dependencies..."
pip install -e ".[eval]" -q

# ── 3. Check CUDA ───────────────────────────────────────────
echo "→ Checking GPU..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ── 4. Load .env if present ─────────────────────────────────
if [ -f .env ]; then
    echo "→ Loading .env..."
    set -a && source .env && set +a
fi

# ── 5. Optional: z86 init ───────────────────────────────────
if [ "${RUN_INIT:-0}" = "1" ]; then
    echo "→ Running z86 init..."
    z86 init
fi

# ── 6. Optional: Start dashboard ────────────────────────────
if [ "${START_DASHBOARD:-0}" = "1" ]; then
    echo "→ Starting dashboard in background..."
    z86 dashboard > "${PROJECT_DIR}/dashboard.log" 2>&1 &
    echo "  PID: $!"
    echo "  Log: ${PROJECT_DIR}/dashboard.log"
fi

echo ""
echo "✓ Bootstrap complete. Next steps:"
echo "  cd ${PROJECT_DIR}"
echo "  z86 train --name 'base-run' --dashboard http://localhost:3000"
