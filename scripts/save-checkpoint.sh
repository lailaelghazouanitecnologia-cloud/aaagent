#!/bin/bash
# ─────────────────────────────────────────────────────────
# Save checkpoints back to Vultr after training.
#
# Usage: scripts/save-checkpoint.sh <VULTR_IP>
# ─────────────────────────────────────────────────────────
set -euo pipefail

VULTR_IP="${1:?Usage: $0 <VULTR_IP>}"
REPO_DIR="/workspace/aaagent"
REMOTE_DIR="/opt/aaagent/checkpoints"

echo "══════════════════════════════════════"
echo "  Saving checkpoints to Vultr"
echo "══════════════════════════════════════"

# Create remote dir
ssh -o StrictHostKeyChecking=no "root@${VULTR_IP}" "mkdir -p $REMOTE_DIR"

# Find and upload checkpoints
CKPT_DIR="$REPO_DIR/checkpoints"
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR/*.pt 2>/dev/null)" ]; then
    echo "  Uploading checkpoints..."
    for f in "$CKPT_DIR"/*.pt; do
        BASENAME=$(basename "$f")
        SIZE=$(du -h "$f" | cut -f1)
        echo "  → $BASENAME ($SIZE)"
        scp -o StrictHostKeyChecking=no "$f" "root@${VULTR_IP}:${REMOTE_DIR}/${BASENAME}"
    done
    echo "  ✓ All checkpoints saved"
else
    echo "  ⚠ No checkpoints found in $CKPT_DIR"
fi

echo ""
echo "  Safe to stop the pod now."
echo ""
