"""z86 eval — evaluate a model version, save rich metrics to metrics/{tag}/.

Supports:
    z86 eval                      # eval latest version
    z86 eval hier-boost           # by tag
    z86 eval v4                   # by id
    z86 eval v4 --judge           # include LLM judge
    z86 eval v4 --quick           # skip perplexity
    z86 eval --compare v3 v4      # compare two versions
"""

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

    # ── Compare mode ──
    if args.compare:
        return _compare(args.compare, reg)

    # Resolve version to checkpoint path
    v, checkpoint = _resolve(args.version, reg)
    if not checkpoint:
        return 1

    tag = v.tag if v else (args.version or "unknown")
    version_id = v.id if v else ""
    step = v.step if v else 0
    config_path = v.config if v else (args.config or "configs/base.yaml")

    ui.step(f"Evaluating [{tag}] — {checkpoint}")

    # Use the new metrics pipeline
    cmd = [
        sys.executable, "-c",
        _eval_script(checkpoint, config_path, tag, version_id, step,
                     quick=args.quick, skip_judge=not args.judge),
    ]

    if args.quick:
        ui.info("Quick mode — skipping perplexity")
    if args.judge:
        ui.info("LLM judge enabled (requires GROQ_API_KEY)")

    print()
    result = subprocess.run(cmd)

    # Update registry with metrics from summary.json
    if result.returncode == 0 and v:
        _update_registry_from_metrics(reg, v, tag)

    return result.returncode


def _compare(tags: list[str], reg: Registry):
    """Compare two versions using metrics/."""
    if len(tags) < 2:
        ui.err("Need two versions to compare: z86 eval --compare v3 v4")
        return 1

    tag_a, tag_b = tags[0], tags[1]

    # Resolve tags to actual tag strings
    va = reg.get(tag_a)
    vb = reg.get(tag_b)
    real_a = va.tag if va and va.tag else tag_a
    real_b = vb.tag if vb and vb.tag else tag_b

    try:
        from metrics.collector import compare_versions, load_summary

        comparison = compare_versions(real_a, real_b)
        if "error" in comparison:
            ui.err(comparison["error"])
            return 1

        ui.header(f"[{real_a}] vs [{real_b}]")

        headers = ["METRIC", real_a, real_b, "DELTA", ""]
        rows = []

        # Define display groups
        groups = {
            "Performance": ["val_loss", "nll", "ppl", "bpb"],
            "Structure": ["alpha", "beta", "entropy_ratio", "dead_clusters",
                         "centroid_similarity", "router_entropy"],
            "Gate": ["gate_mean", "gate_std", "gate_pct_floor"],
            "Hierarchy": ["hierarchy_coherence", "hierarchy_balance"],
            "Generation": ["mean_distinct_2", "mean_repetition", "self_bleu_4",
                          "vocab_richness"],
            "LLM Judge": ["judge_overall", "judge_coherence", "judge_grammar",
                         "judge_fluency", "judge_creativity"],
        }

        for group_name, keys in groups.items():
            has_data = any(k in comparison["metrics"] for k in keys)
            if not has_data:
                continue
            rows.append([f"{ui.C.CYAN}── {group_name} ──{ui.C.RST}", "", "", "", ""])
            for key in keys:
                m = comparison["metrics"].get(key)
                if m is None:
                    continue
                va_s = f"{m['a']:.4f}" if isinstance(m.get('a'), float) else str(m.get('a', '—'))
                vb_s = f"{m['b']:.4f}" if isinstance(m.get('b'), float) else str(m.get('b', '—'))

                if isinstance(m.get("delta"), (int, float)):
                    improved = m.get("improved", False)
                    color = ui.C.GREEN if improved else ui.C.RED
                    delta_s = f"{color}{m['delta']:+.4f}{ui.C.RST}"
                    pct_s = f"{ui.C.GRAY}({m['pct']:+.1f}%){ui.C.RST}" if m.get('pct') is not None else ""
                else:
                    delta_s = "—"
                    pct_s = ""

                rows.append([key, va_s, vb_s, delta_s, pct_s])

        ui.table(headers, rows, col_widths=[24, 12, 12, 14, 10])
        return 0

    except ImportError as e:
        ui.err(f"Missing dependency: {e}")
        return 1


def _resolve(version_id: str | None, reg: Registry):
    """Resolve a version id/tag/path to (Version, checkpoint_path)."""
    if version_id is None:
        v = reg.latest()
        if v:
            ui.info(f"Using latest: {v.id} [{v.tag}] (step {v.step})")
            return v, v.path
        ui.err("No versions found. Train first: z86 train")
        return None, None

    v = reg.get(version_id)
    if v:
        return v, v.path

    if version_id == "latest":
        v = reg.latest()
        return (v, v.path) if v else (None, None)
    if version_id == "best":
        v = reg.best()
        return (v, v.path) if v else (None, None)

    if Path(version_id).exists():
        return None, version_id

    ui.err(f"Not found: {version_id}")
    return None, None


def _eval_script(checkpoint, config, tag, version_id, step, quick, skip_judge):
    """Generate inline Python script for evaluation."""
    return f"""\
import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from scripts.train import load_config
from model.config import ModelConfig
from model.lm import HCLMD
from training.checkpointing import load_checkpoint
from metrics.evaluate import run_full_eval
import torch
from torch.utils.data import DataLoader
from pathlib import Path

config = load_config("{config}")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = ModelConfig.from_dict(config)
model = HCLMD(model_config)
load_checkpoint("{checkpoint}", model, device=device)
model = model.to(device)

# Data
data_dir = Path(config.get("data", {{}}).get("data_dir", "data/tinystories"))
seq_len = config.get("model", {{}}).get("max_seq_len", 512)
batch_size = config.get("training", {{}}).get("batch_size", 64)

val_loader = None
val_path = data_dir / "val_tokens.pt"
if val_path.exists():
    from data.dataset import HCLMDataset, collate_fn
    val_tokens = torch.load(val_path, weights_only=True)
    val_dataset = HCLMDataset(val_tokens, seq_len=seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

tokenizer = None
tok_path = config.get("data", {{}}).get("tokenizer_path", "data/tokenizer.json")
if Path(tok_path).exists():
    from data.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(tok_path)

vm = run_full_eval(
    model, config,
    tag="{tag}",
    version_id="{version_id}",
    step={step},
    device=device,
    val_loader=val_loader,
    tokenizer=tokenizer,
    skip_perplexity={quick},
    skip_judge={skip_judge},
)

print()
print("=" * 60)
print(f"  METRICS SAVED: metrics/{tag}/")
print("=" * 60)
d = vm.to_dict()
for k, v in d.items():
    if k not in ("tag", "version_id", "config", "timestamp"):
        print(f"  {{k}}: {{v}}")
"""


def _update_registry_from_metrics(reg, v, tag):
    """Update registry entry from saved metrics."""
    try:
        from metrics.collector import load_summary
        vm = load_summary(tag)
        if vm is None:
            return
        updates = {}
        if vm.ppl is not None:
            updates["ppl"] = vm.ppl
        if vm.entropy_ratio is not None:
            updates["entropy"] = vm.entropy_ratio
        if vm.gate_mean is not None:
            updates["gate_mean"] = vm.gate_mean
        if vm.nll is not None:
            updates["loss"] = vm.nll
        if updates:
            reg.update(v.id, **updates)
            ui.ok(f"Updated {v.id} [{v.tag}] with metrics from metrics/{tag}/")
    except Exception:
        pass
