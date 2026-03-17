"""z86 eval — evaluate a model version."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli import ui
from cli.registry import Registry


def cmd_eval(args):
    """Run evaluation on a version or checkpoint."""
    ui.logo()
    reg = Registry()

    # Resolve version to checkpoint path
    checkpoint = _resolve_checkpoint(args.version, reg)
    if not checkpoint:
        return 1

    ui.step(f"Evaluating {args.version or checkpoint}")

    cmd = [sys.executable, "scripts/eval_checkpoint.py", "--checkpoint", checkpoint]

    if args.config:
        cmd += ["--config", args.config]

    if args.quick:
        cmd += ["--quick"]
        ui.info("Quick mode — skipping perplexity")

    if args.judge:
        ui.info("LLM judge enabled (requires GROQ_API_KEY)")

    if args.dashboard:
        cmd += ["--dashboard"]
        ui.info("Results will be sent to dashboard")

    if args.prompts:
        cmd += ["--prompts"] + args.prompts

    print()

    result = subprocess.run(cmd)

    # Update version metadata if we have results
    if result.returncode == 0:
        v = reg.get(args.version) if args.version else None
        if v:
            # Try to read the output JSON
            output_path = Path(f"results/eval_step_{v.step}.json")
            if output_path.exists():
                import json
                data = json.loads(output_path.read_text())
                ppl_data = data.get("perplexity", {})
                health = data.get("cluster_health", {})
                gate = data.get("gate", {})
                reg.update(
                    v.id,
                    ppl=ppl_data.get("ppl"),
                    entropy=health.get("entropy_ratio"),
                    gate_mean=gate.get("mean"),
                )
                ui.ok(f"Updated {v.id} with eval metrics")

    return result.returncode


def _resolve_checkpoint(version_id: str | None, reg: Registry) -> str | None:
    """Resolve a version id or path to a checkpoint file."""
    if version_id is None:
        # Use latest
        v = reg.latest()
        if v:
            ui.info(f"Using latest: {v.id} (step {v.step})")
            return v.path
        ui.err("No versions found. Train first: z86 train")
        return None

    # Try as version id
    v = reg.get(version_id)
    if v:
        return v.path

    # Try as special names
    if version_id == "latest":
        v = reg.latest()
        if v:
            return v.path
    elif version_id == "best":
        v = reg.best()
        if v:
            return v.path

    # Try as file path
    if Path(version_id).exists():
        return version_id

    ui.err(f"Not found: {version_id} (not a version id, 'latest', 'best', or file path)")
    return None
