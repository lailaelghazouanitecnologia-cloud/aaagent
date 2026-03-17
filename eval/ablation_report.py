"""Cross-run comparison tables for ablation studies."""

from __future__ import annotations

import json
from pathlib import Path


def generate_ablation_report(
    results_dir: str = "results",
    output_path: str = "results/ablation_report.md",
) -> str:
    """Generate a markdown comparison table from ablation run results.

    Args:
        results_dir: Directory containing per-run result JSON files.
        output_path: Where to save the report.

    Returns:
        Markdown string with the comparison table.
    """
    results_path = Path(results_dir)
    runs = {}

    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            runs[json_file.stem] = data

    if not runs:
        return "No results found."

    # Build comparison table
    headers = ["Run", "Val Loss", "NLL", "BPB", "Entropy Ratio", "Dead Clusters", "Gate Mean"]
    lines = [
        "# Ablation Report\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for name, data in runs.items():
        row = [
            name,
            f"{data.get('val_loss', 'N/A'):.4f}" if isinstance(data.get('val_loss'), (int, float)) else "N/A",
            f"{data.get('nll', 'N/A'):.4f}" if isinstance(data.get('nll'), (int, float)) else "N/A",
            f"{data.get('bpb', 'N/A'):.4f}" if isinstance(data.get('bpb'), (int, float)) else "N/A",
            f"{data.get('entropy_ratio', 'N/A'):.3f}" if isinstance(data.get('entropy_ratio'), (int, float)) else "N/A",
            str(data.get("dead_clusters", "N/A")),
            f"{data.get('gate_mean', 'N/A'):.3f}" if isinstance(data.get('gate_mean'), (int, float)) else "N/A",
        ]
        lines.append("| " + " | ".join(row) + " |")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report
