"""z86 dashboard — start the monitoring dashboard."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from cli import ui


def cmd_dashboard(args):
    """Start the dashboard server."""
    ui.logo()

    dash_dir = Path("dashboard")
    if not (dash_dir / "package.json").exists():
        ui.err("Dashboard not found at dashboard/")
        return 1

    if not shutil.which("bun"):
        ui.err("Bun not installed. Install: curl -fsSL https://bun.sh/install | bash")
        return 1

    # Check node_modules
    if not (dash_dir / "node_modules").exists():
        ui.step("Installing dashboard dependencies")
        result = subprocess.run(["bun", "install"], cwd=str(dash_dir))
        if result.returncode != 0:
            ui.err("bun install failed")
            return 1
        ui.ok("Dependencies installed")

    env = os.environ.copy()
    if args.port:
        env["DASHBOARD_PORT"] = str(args.port)

    port = args.port or 3000

    if args.prod:
        ui.step(f"Starting dashboard (production) on :{port}")

        # Build first
        ui.info("Building frontend...")
        result = subprocess.run(["bun", "run", "build"], cwd=str(dash_dir))
        if result.returncode != 0:
            ui.err("Build failed")
            return 1

        env["NODE_ENV"] = "production"
        cmd = ["bun", "run", "start"]
    else:
        ui.step(f"Starting dashboard (dev) on :{port}")
        ui.info(f"Server:   http://localhost:{port}")
        ui.info(f"Client:   http://localhost:5173")
        ui.info(f"WebSocket: ws://localhost:{port}/ws")
        cmd = ["bun", "run", "dev"]

    print()

    try:
        return subprocess.run(cmd, cwd=str(dash_dir), env=env).returncode
    except KeyboardInterrupt:
        print()
        ui.warn("Dashboard stopped")
        return 0
