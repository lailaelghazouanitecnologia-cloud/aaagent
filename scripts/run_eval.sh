#!/bin/bash
# ─────────────────────────────────────────────────────────────
# HCLM-D Evaluation Runner for RunPod
#
# Usage:
#   export GROQ_API_KEY="gsk_..."
#   bash scripts/run_eval.sh checkpoints/step_10000.pt
#
# Options (env vars):
#   GROQ_API_KEY     — Required for LLM-as-judge
#   GROQ_MODEL       — Default: llama-3.3-70b-versatile
#   DASHBOARD_URL    — Optional, e.g. https://z86.dev
#   EVAL_DEVICE      — Default: cuda
#   EVAL_SEQ_LEN     — Default: 256
#   EVAL_STEPS       — Default: 64
#   EVAL_TEMP        — Default: 0.9
#   WITH_AGENT       — Set to 1 for Agno agent analysis
# ─────────────────────────────────────────────────────────────
set -euo pipefail

CHECKPOINT="${1:?Usage: run_eval.sh <checkpoint_path>}"
DEVICE="${EVAL_DEVICE:-cuda}"
SEQ_LEN="${EVAL_SEQ_LEN:-256}"
STEPS="${EVAL_STEPS:-64}"
TEMP="${EVAL_TEMP:-0.9}"
DASHBOARD="${DASHBOARD_URL:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  HCLM-D Evaluation Pipeline                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Device:      ${DEVICE}"
echo "  Seq length:  ${SEQ_LEN}"
echo "  Steps:       ${STEPS}"
echo "  Temperature: ${TEMP}"
echo "  Groq model:  ${GROQ_MODEL:-llama-3.3-70b-versatile}"
echo ""

# Ensure deps
pip install -q agno groq 2>/dev/null || true

# Build command
CMD="python -m eval.agent_eval \
  --checkpoint ${CHECKPOINT} \
  --seq-len ${SEQ_LEN} \
  --sampling-steps ${STEPS} \
  --temperature ${TEMP} \
  --device ${DEVICE} \
  --output-dir ${OUTPUT_DIR}"

[ -n "${DASHBOARD}" ] && CMD="${CMD} --dashboard-url ${DASHBOARD}"
[ -n "${RUN_NAME}" ] && CMD="${CMD} --run-name ${RUN_NAME}"
[ "${WITH_AGENT:-0}" = "1" ] && CMD="${CMD} --with-agent"

echo "Running: ${CMD}"
echo ""

eval ${CMD}
