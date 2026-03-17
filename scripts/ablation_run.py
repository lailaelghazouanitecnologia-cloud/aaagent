#!/usr/bin/env python3
"""Run all ablation configs sequentially."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ABLATION_CONFIGS = [
    "configs/ablations/flat_autoregressive.yaml",  # A0
    "configs/ablations/flat_baseline.yaml",          # A1
    "configs/ablations/clusters_only.yaml",          # A2
    "configs/ablations/no_gate.yaml",                # A3
    "configs/ablations/no_meta.yaml",                # A4
    "configs/ablations/full.yaml",                   # A5
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(__file__).parent.parent
    results = {}

    for config_path in ABLATION_CONFIGS:
        full_path = project_root / config_path
        if not full_path.exists():
            logging.warning("Config not found: %s, skipping", config_path)
            continue

        name = Path(config_path).stem
        logging.info("=" * 60)
        logging.info("Starting ablation: %s", name)
        logging.info("Config: %s", config_path)
        logging.info("=" * 60)

        cmd = [
            sys.executable, str(project_root / "scripts" / "train.py"),
            "--config", str(full_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
            )
            results[name] = {
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }

            if result.returncode != 0:
                logging.error("Ablation %s failed:\n%s", name, result.stderr[-500:])
            else:
                logging.info("Ablation %s completed successfully", name)

        except Exception as e:
            logging.error("Ablation %s crashed: %s", name, e)
            results[name] = {"returncode": -1, "success": False, "error": str(e)}

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("ABLATION SUMMARY")
    logging.info("=" * 60)
    for name, result in results.items():
        status = "OK" if result["success"] else "FAIL"
        logging.info("  %s: %s", name, status)


if __name__ == "__main__":
    main()
