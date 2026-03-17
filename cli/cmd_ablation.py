"""z86 ablation — run and compare ablation studies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli import ui


ABLATIONS = [
    ("A0", "flat_autoregressive", "Standard AR baseline"),
    ("A1", "flat_baseline", "Diffusion + flat embedding"),
    ("A2", "clusters_only", "Diffusion + fine clusters, no hierarchy"),
    ("A3", "no_gate", "No gating mechanism"),
    ("A4", "no_meta", "No meta-optimizer"),
    ("A5", "full", "Full HCLM-D (everything enabled)"),
]


def cmd_ablation(args):
    """Run or inspect ablation studies."""
    if args.sub == "run":
        return _run(args)
    elif args.sub == "status":
        return _status(args)
    elif args.sub == "compare":
        return _compare(args)
    else:
        ui.err(f"Unknown subcommand: {args.sub}")
        ui.info("Usage: z86 ablation {run|status|compare}")
        return 1


def _run(args):
    """Run all (or specific) ablation configs."""
    ui.logo()
    ui.step("Running ablation studies")

    targets = ABLATIONS
    if args.only:
        targets = [(i, n, d) for i, n, d in ABLATIONS if i.lower() in args.only or n in args.only]
        if not targets:
            ui.err(f"No matching ablations: {args.only}")
            return 1

    results = []
    for tag, name, desc in targets:
        config = f"configs/ablations/{name}.yaml"
        if not Path(config).exists():
            ui.warn(f"Config not found: {config} — skipping {tag}")
            results.append((tag, name, "skip"))
            continue

        ui.step(f"{tag}: {desc}")
        ui.info(f"Config: {config}")

        ret = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", config],
        ).returncode

        status = "pass" if ret == 0 else "fail"
        results.append((tag, name, status))

        if status == "pass":
            ui.ok(f"{tag} completed")
        else:
            ui.err(f"{tag} failed (exit {ret})")

    # Summary
    ui.header("Ablation Summary")
    headers = ["TAG", "CONFIG", "STATUS"]
    rows = []
    for tag, name, status in results:
        s = f"{ui.C.GREEN}PASS{ui.C.RST}" if status == "pass" else (
            f"{ui.C.RED}FAIL{ui.C.RST}" if status == "fail" else
            f"{ui.C.YELLOW}SKIP{ui.C.RST}"
        )
        rows.append([tag, name, s])
    ui.table(headers, rows)
    return 0


def _status(args):
    """Show which ablation checkpoints exist."""
    ui.logo()
    ui.step("Ablation status")

    headers = ["TAG", "CONFIG", "EXISTS", "STEP"]
    rows = []

    for tag, name, desc in ABLATIONS:
        config = f"configs/ablations/{name}.yaml"
        exists = Path(config).exists()

        # Check for checkpoints
        ckpt_dir = Path("checkpoints")
        ckpts = list(ckpt_dir.glob(f"*{name}*")) if ckpt_dir.exists() else []

        config_str = f"{ui.C.GREEN}✓{ui.C.RST}" if exists else f"{ui.C.RED}✗{ui.C.RST}"
        ckpt_str = f"{len(ckpts)} ckpt" if ckpts else "—"

        rows.append([tag, name, config_str, ckpt_str])

    ui.table(headers, rows)
    return 0


def _compare(args):
    """Compare ablation results (delegates to dashboard or local)."""
    ui.logo()
    ui.step("Ablation comparison")
    ui.info("Use the dashboard ablations page for visual comparison:")
    ui.info("  z86 dashboard")
    ui.info("  Open http://localhost:3000/ablations")
    return 0
