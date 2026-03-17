#!/bin/bash
# Start the HCLM-D Dashboard
# Usage: ./dashboard/start.sh [dev|prod]

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-dev}"

# Install deps if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  bun install
fi

if [ "$MODE" = "prod" ]; then
  echo "Building frontend..."
  bun run build
  echo "Starting production server on port ${DASHBOARD_PORT:-3000}..."
  NODE_ENV=production bun server/index.ts
else
  echo "Starting development server..."
  echo "  API:       http://localhost:${DASHBOARD_PORT:-3000}"
  echo "  Frontend:  http://localhost:5173"
  bun run dev
fi
