"""z86 train — start or resume training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cli import ui
from cli.registry import Registry


def cmd_train(args):
    """Start training with optional config and resume."""
    ui.logo()

    # Resolve config path
    config = args.config
    if not config.endswith(".yaml"):
        # Allow short names: "base", "fast", "flat_baseline"
        candidates = [
            f"configs/{config}.yaml",
            f"configs/ablations/{config}.yaml",
        ]
        config = next((c for c in candidates if Path(c).exists()), config)

    if not Path(config).exists():
        ui.err(f"Config not found: {config}")
        return 1

    # Check data exists
    if not Path("data/tinystories/train_tokens.pt").exists():
        ui.err("Training data not found. Run: z86 init")
        return 1

    # Build command
    cmd = [sys.executable, "scripts/train.py", "--config", config]

    # Resume from version
    if args.resume:
        reg = Registry()
        v = reg.get(args.resume)
        if v:
            cmd += ["--resume", v.path]
            ui.info(f"Resuming from {v.id} (step {v.step})")
        elif Path(args.resume).exists():
            cmd += ["--resume", args.resume]
            ui.info(f"Resuming from {args.resume}")
        else:
            ui.err(f"Version or checkpoint not found: {args.resume}")
            return 1

    # Set run name via env
    env = os.environ.copy()
    if args.name:
        env["RUN_NAME"] = args.name

    # Dashboard URL
    if args.dashboard:
        env["DASHBOARD_URL"] = args.dashboard

    ui.step(f"Training · {config}")
    ui.info(f"Run name: {args.name or 'default'}")
    if args.dashboard:
        ui.info(f"Dashboard: {args.dashboard}")

    print()

    # Execute training (foreground, inherit stdio)
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        print()
        ui.warn("Training interrupted")

        # Auto-register last checkpoint
        reg = Registry()
        ckpt_dir = Path("checkpoints")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                latest = ckpts[-1]
                rel = str(latest)
                # Check if already registered
                existing = [v for v in reg.list_all() if v.path == rel]
                if not existing:
                    # Extract step from filename
                    try:
                        step_num = int(latest.stem.split("_")[1])
                    except (IndexError, ValueError):
                        step_num = 0

                    v = reg.register(
                        step=step_num,
                        path=rel,
                        run=args.name or "default",
                        config=config,
                        note="auto-saved on interrupt",
                    )
                    ui.ok(f"Registered {v.id} (step {v.step}, {v.size_mb:.0f}MB)")

        return 130  # SIGINT
