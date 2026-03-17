#!/bin/bash
# ─────────────────────────────────────────────────────────
# Setup Vultr CPU server for HCLM-D
# Runs: prep (tokenization) + dashboard
#
# Usage: ssh root@<VULTR_IP> 'bash -s' < scripts/setup-vultr.sh
#   or:  scp this to Vultr and run there
# ─────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="https://github.com/lailaelghazouanitecnologia-cloud/aaagent.git"
INSTALL_DIR="/opt/aaagent"
BRANCH="claude/hclm-d-documentation-uKBwL"

echo "══════════════════════════════════════"
echo "  HCLM-D Vultr Setup"
echo "══════════════════════════════════════"

# ── 1. System dependencies ──
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git unzip curl > /dev/null 2>&1

# ── 2. Install Bun ──
echo "[2/6] Installing Bun..."
if ! command -v bun &> /dev/null; then
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
    echo 'export BUN_INSTALL="$HOME/.bun"' >> ~/.bashrc
    echo 'export PATH="$BUN_INSTALL/bin:$PATH"' >> ~/.bashrc
else
    echo "  Bun already installed: $(bun --version)"
fi

# ── 3. Clone repo ──
echo "[3/6] Cloning repo..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    git pull origin "$BRANCH"
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    git checkout "$BRANCH"
fi

# ── 4. Python deps + prep ──
echo "[4/6] Installing Python dependencies..."
pip install -q ".[dev]"

echo "[4/6] Running data prep (tokenization)..."
echo "       This takes 15-45 minutes. Do NOT interrupt."
python scripts/train.py --config configs/base.yaml --phase prep

# Verify tokens were created
if [ -f "data/tinystories/train_tokens.pt" ] && [ -f "data/tinystories/val_tokens.pt" ]; then
    echo "  ✓ train_tokens.pt: $(du -h data/tinystories/train_tokens.pt | cut -f1)"
    echo "  ✓ val_tokens.pt:   $(du -h data/tinystories/val_tokens.pt | cut -f1)"
    echo "  ✓ tokenizer.json:  $(du -h data/tokenizer.json | cut -f1)"
else
    echo "  ✗ ERROR: Token files not found. Prep may have failed."
    exit 1
fi

# Clean up raw text to save disk
echo "  Cleaning up raw text files..."
rm -f data/tinystories/train.txt data/tinystories/val.txt
echo "  ✓ Freed ~1.9GB"

# ── 5. Setup dashboard ──
echo "[5/6] Setting up dashboard..."
cd "$INSTALL_DIR/dashboard"
bun install --production

# ── 6. Create systemd services ──
echo "[6/6] Creating systemd service..."
cat > /etc/systemd/system/hclm-dashboard.service << 'UNIT'
[Unit]
Description=HCLM-D Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/aaagent/dashboard
ExecStart=/root/.bun/bin/bun server/index.ts
Environment=NODE_ENV=production
Environment=DASHBOARD_PORT=3000
Environment=DASHBOARD_DB=/opt/aaagent/dashboard/metrics.db
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable hclm-dashboard
systemctl start hclm-dashboard

# Wait and verify
sleep 2
if curl -sf http://localhost:3000/api/health > /dev/null; then
    echo "  ✓ Dashboard running at http://localhost:3000"
else
    echo "  ⚠ Dashboard may not have started. Check: journalctl -u hclm-dashboard"
fi

echo ""
echo "══════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════"
echo ""
echo "  Token files ready at:"
echo "    $INSTALL_DIR/data/tinystories/train_tokens.pt"
echo "    $INSTALL_DIR/data/tinystories/val_tokens.pt"
echo "    $INSTALL_DIR/data/tokenizer.json"
echo ""
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo "  Next: configure Cloudflare DNS → z86.dev"
echo "  Then from RunPod, run:"
echo "    scripts/runpod-session.sh <VULTR_IP>"
echo ""
