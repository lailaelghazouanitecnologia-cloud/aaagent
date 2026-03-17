"""Metrics collector — gathers, stores, and compares per-version metrics.

Creates structured metrics under metrics/{tag}/:
    summary.json   — single-file overview (loss, ppl, cluster health, gate, hierarchy)
    samples.json   — generated texts + per-sample auto metrics
    judge.json     — LLM judge scores (if GROQ_API_KEY available)
    history.json   — training loss curve snapshots (appended over time)
    compare.json   — auto-generated when comparing two versions
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRICS_ROOT = Path("metrics")


@dataclass
class VersionMetrics:
    """Complete metrics snapshot for a model version."""
    tag: str
    version_id: str
    step: int
    config: str
    timestamp: str

    # Core performance
    val_loss: float | None = None
    nll: float | None = None
    ppl: float | None = None
    bpb: float | None = None

    # Structural health
    alpha: float | None = None
    beta: float | None = None
    gate_mean: float | None = None
    gate_std: float | None = None
    gate_pct_floor: float | None = None
    entropy_ratio: float | None = None
    dead_clusters: int | None = None
    centroid_similarity: float | None = None
    hierarchy_coherence: float | None = None
    hierarchy_balance: float | None = None

    # Router
    router_entropy: float | None = None
    router_temperature: float | None = None

    # Auto metrics (corpus-level)
    mean_distinct_2: float | None = None
    mean_repetition: float | None = None
    self_bleu_4: float | None = None
    vocab_richness: float | None = None

    # LLM judge (aggregate)
    judge_overall: float | None = None
    judge_coherence: float | None = None
    judge_grammar: float | None = None
    judge_fluency: float | None = None
    judge_creativity: float | None = None
    judge_failure_modes: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def metrics_dir(tag: str) -> Path:
    """Get or create the metrics directory for a version tag."""
    d = METRICS_ROOT / _safe_name(tag)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(tag: str) -> str:
    """Sanitize tag for use as directory name."""
    return tag.replace("+", "_plus_").replace(" ", "_").replace("/", "_")


def save_summary(tag: str, metrics: VersionMetrics) -> Path:
    """Save the summary metrics for a version."""
    d = metrics_dir(tag)
    path = d / "summary.json"
    path.write_text(json.dumps(metrics.to_dict(), indent=2))
    logger.info("Saved metrics summary: %s", path)
    return path


def save_samples(tag: str, samples: list[dict]) -> Path:
    """Save generated samples with per-sample metrics."""
    d = metrics_dir(tag)
    path = d / "samples.json"
    path.write_text(json.dumps(samples, indent=2, default=str))
    logger.info("Saved %d samples: %s", len(samples), path)
    return path


def save_judge(tag: str, judge_results: dict) -> Path:
    """Save LLM judge evaluation results."""
    d = metrics_dir(tag)
    path = d / "judge.json"
    path.write_text(json.dumps(judge_results, indent=2, default=str))
    logger.info("Saved judge results: %s", path)
    return path


def append_history(tag: str, step: int, losses: dict) -> Path:
    """Append a training loss snapshot to the history."""
    d = metrics_dir(tag)
    path = d / "history.json"

    history = []
    if path.exists():
        history = json.loads(path.read_text())

    history.append({
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **losses,
    })

    path.write_text(json.dumps(history, indent=2))
    return path


def load_summary(tag: str) -> VersionMetrics | None:
    """Load the summary metrics for a version."""
    d = METRICS_ROOT / _safe_name(tag)
    path = d / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VersionMetrics(**{k: v for k, v in data.items()
                            if k in VersionMetrics.__dataclass_fields__})


def load_judge(tag: str) -> dict | None:
    """Load LLM judge results for a version."""
    d = METRICS_ROOT / _safe_name(tag)
    path = d / "judge.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_versions_with_metrics() -> list[str]:
    """List all tags that have metrics saved."""
    if not METRICS_ROOT.exists():
        return []
    return sorted(
        d.name for d in METRICS_ROOT.iterdir()
        if d.is_dir() and (d / "summary.json").exists()
    )


def compare_versions(tag_a: str, tag_b: str) -> dict[str, Any]:
    """Compare two versions and return a structured diff."""
    a = load_summary(tag_a)
    b = load_summary(tag_b)

    if a is None or b is None:
        missing = tag_a if a is None else tag_b
        return {"error": f"No metrics for '{missing}'. Run: z86 eval {missing}"}

    da = a.to_dict()
    db = b.to_dict()

    # Metrics where lower is better
    lower_better = {"val_loss", "nll", "ppl", "bpb", "dead_clusters",
                    "centroid_similarity", "mean_repetition", "self_bleu_4",
                    "gate_pct_floor"}

    comparison = {
        "version_a": {"tag": tag_a, "step": a.step},
        "version_b": {"tag": tag_b, "step": b.step},
        "metrics": {},
    }

    all_keys = sorted(set(da.keys()) | set(db.keys()))
    skip = {"tag", "version_id", "config", "timestamp", "step", "judge_failure_modes"}

    for key in all_keys:
        if key in skip:
            continue
        va = da.get(key)
        vb = db.get(key)
        if va is None and vb is None:
            continue
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            comparison["metrics"][key] = {"a": va, "b": vb}
            continue

        delta = vb - va
        is_lower_better = key in lower_better
        improved = (delta < 0) if is_lower_better else (delta > 0)

        comparison["metrics"][key] = {
            "a": va,
            "b": vb,
            "delta": round(delta, 6),
            "pct": round((delta / abs(va)) * 100, 2) if va != 0 else None,
            "improved": improved,
        }

    # Save comparison
    d = metrics_dir(f"compare_{_safe_name(tag_a)}_vs_{_safe_name(tag_b)}")
    path = d / "compare.json"
    path.write_text(json.dumps(comparison, indent=2))

    return comparison
