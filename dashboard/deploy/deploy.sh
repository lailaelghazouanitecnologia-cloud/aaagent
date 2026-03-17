#!/bin/bash
# Deploy HCLM-D Dashboard to z86.dev
# Usage: ./deploy.sh <server-ip> [--setup]
#
# Requires:
#   - SSH access to the server
#   - Cloudflare DNS A record pointing z86.dev → server IP
#   - Bun installed on the server

set -euo pipefail

SERVER_IP="${1:?Usage: ./deploy.sh <server-ip> [--setup]}"
SETUP="${2:-}"
REMOTE_DIR="/opt/z86-dashboard"
SSH="ssh root@${SERVER_IP}"

echo "=== Deploying z86.dev dashboard to ${SERVER_IP} ==="

# --- First-time setup ---
if [ "$SETUP" = "--setup" ]; then
  echo "[1/5] First-time server setup..."
  $SSH << 'SETUP_EOF'
    set -e

    # Install Bun if missing
    if ! command -v bun &> /dev/null; then
      echo "Installing Bun..."
      curl -fsSL https://bun.sh/install | bash
      ln -sf ~/.bun/bin/bun /usr/local/bin/bun
    fi

    # Install nginx if missing
    if ! command -v nginx &> /dev/null; then
      apt-get update && apt-get install -y nginx
    fi

    # Create app directory
    mkdir -p /opt/z86-dashboard/data
    mkdir -p /etc/ssl/z86.dev

    echo "Server setup complete."
SETUP_EOF
fi

# --- Deploy ---
echo "[2/5] Syncing files..."
rsync -avz --delete \
  --exclude node_modules \
  --exclude .git \
  --exclude metrics.db \
  --exclude data/ \
  ./ "root@${SERVER_IP}:${REMOTE_DIR}/"

echo "[3/5] Installing dependencies & building..."
$SSH << EOF
  set -e
  cd ${REMOTE_DIR}
  bun install --frozen-lockfile 2>/dev/null || bun install
  bun run build
EOF

echo "[4/5] Configuring nginx & systemd..."
$SSH << EOF
  set -e
  # Nginx
  cp ${REMOTE_DIR}/deploy/nginx.conf /etc/nginx/sites-available/z86.dev
  ln -sf /etc/nginx/sites-available/z86.dev /etc/nginx/sites-enabled/z86.dev
  nginx -t && systemctl reload nginx

  # Systemd
  cp ${REMOTE_DIR}/deploy/z86-dashboard.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable z86-dashboard
  systemctl restart z86-dashboard
EOF

echo "[5/5] Health check..."
sleep 2
HEALTH=$($SSH "curl -sf http://localhost:3000/api/health" 2>/dev/null || echo '{"status":"error"}')
echo "Health: ${HEALTH}"

if echo "$HEALTH" | grep -q '"ok"'; then
  echo ""
  echo "=== Deploy successful! ==="
  echo "Dashboard live at https://z86.dev"
else
  echo ""
  echo "=== WARNING: Health check failed ==="
  echo "Check logs: ssh root@${SERVER_IP} journalctl -u z86-dashboard -f"
fi
