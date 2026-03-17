"""z86 train — start or resume training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from cli import ui
from cli.registry import Registry


def _read_version_tag(config_path: str) -> tuple[str, str]:
    """Extract version tag and note from a config file."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        v = cfg.get("version", {})
        return v.get("tag", ""), v.get("note", "")
    except Exception:
        return "", ""


def cmd_train(args):
    """Start training with optional config and resume."""
    ui.logo()

    # Resolve config: --config flag > active version > base.yaml
    config = args.config

    if config is None:
        # No --config: use active version if set
        from cli.active import get_active
        active = get_active()
        if active:
            _key, config, _tag = active
            ui.info(f"Using active version: {_key} [{_tag}]")
        else:
            config = "configs/base.yaml"

    if not config.endswith(".yaml"):
        # Allow short names: "base", "fast", "v4_hier_boost"
        candidates = [
            f"configs/{config}.yaml",
            f"configs/ablations/{config}.yaml",
            f"configs/versions/{config}.yaml",
        ]
        config = next((c for c in candidates if Path(c).exists()), config)

    if not Path(config).exists():
        ui.err(f"Config not found: {config}")
        return 1

    # Check data exists
    if not Path("data/tinystories/train_tokens.pt").exists():
        ui.err("Training data not found. Run: z86 init")
        return 1

    # Read version tag from config
    tag, note = _read_version_tag(config)

    # Build command
    cmd = [sys.executable, "scripts/train.py", "--config", config]

    # Resume from version
    if args.resume:
        reg = Registry()
        v = reg.get(args.resume)
        if v:
            cmd += ["--resume", v.path]
            ui.info(f"Resuming from {v.id} [{v.tag}] (step {v.step})")
        elif Path(args.resume).exists():
            cmd += ["--resume", args.resume]
            ui.info(f"Resuming from {args.resume}")
        else:
            ui.err(f"Version or checkpoint not found: {args.resume}")
            return 1

    # Set run name via env
    env = os.environ.copy()
    run_name = args.name or tag or "default"
    env["RUN_NAME"] = run_name

    # Dashboard URL
    if args.dashboard:
        env["DASHBOARD_URL"] = args.dashboard

    ui.step(f"Training · {config}")
    if tag:
        ui.info(f"Tag: {tag}")
    if note:
        ui.info(f"Note: {note}")
    ui.info(f"Run name: {run_name}")
    if args.dashboard:
        ui.info(f"Dashboard: {args.dashboard}")

    print()

    # Execute training (foreground, inherit stdio)
    try:
        result = subprocess.run(cmd, env=env)

        # On success: auto-register with tag
        if result.returncode == 0:
            _auto_register(config, tag, note, run_name)

        return result.returncode
    except KeyboardInterrupt:
        print()
        ui.warn("Training interrupted")
        _auto_register(config, tag, note or "interrupted", run_name)
        return 130  # SIGINT


def _auto_register(config: str, tag: str, note: str, run_name: str):
    """Register the latest checkpoint as a version."""
    reg = Registry()
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return

    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return

    latest = ckpts[-1]
    rel = str(latest)

    # Check if already registered
    existing = [v for v in reg.list_all() if v.path == rel]
    if existing:
        return

    # Extract step from filename
    try:
        step_num = int(latest.stem.split("_")[1])
    except (IndexError, ValueError):
        step_num = 0

    v = reg.register(
        step=step_num,
        path=rel,
        run=run_name,
        config=config,
        tag=tag,
        note=note,
    )
    ui.ok(f"Registered {v.id} [{v.tag or '—'}] (step {v.step}, {v.size_mb:.0f}MB)")
