#!/usr/bin/env python3
"""Comprehensive checkpoint evaluation with dashboard reporting.

Runs ALL eval modules on a checkpoint and optionally sends results to dashboard.

Usage:
    # Full eval on checkpoint
    python scripts/eval_checkpoint.py --checkpoint checkpoints/step_10000.pt

    # With dashboard reporting
    python scripts/eval_checkpoint.py --checkpoint checkpoints/step_10000.pt --dashboard

    # Quick mode (skip slow perplexity)
    python scripts/eval_checkpoint.py --checkpoint checkpoints/step_10000.pt --quick

    # Custom prompts
    python scripts/eval_checkpoint.py --checkpoint checkpoints/step_10000.pt \
        --prompts "Once upon a time" "The little girl" "There was a"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.lm import HCLMD
from model.embedding.composite import CompositeEmbedding
from data.dataset import HCLMDataset, collate_fn
from training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "There was a big",
    "One day the",
    "She looked at the",
]


def load_model_and_data(args, config):
    """Load model from checkpoint and prepare data loaders."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    ckpt = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    step = ckpt.get("step", 0)
    logger.info("Loaded checkpoint: %s (step %d)", args.checkpoint, step)

    # Data
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/tinystories"))
    seq_len = config.get("model", {}).get("max_seq_len", 512)
    batch_size = config.get("training", {}).get("batch_size", 64)

    val_loader = None
    val_path = data_dir / "val_tokens.pt"
    if val_path.exists():
        val_tokens = torch.load(val_path, weights_only=True)
        val_dataset = HCLMDataset(val_tokens, seq_len=seq_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    else:
        logger.warning("No validation data at %s", val_path)

    # Tokenizer
    tokenizer = None
    tok_path = data_cfg.get("tokenizer_path", "data/tokenizer.json")
    if Path(tok_path).exists():
        from data.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(tok_path)

    return model, val_loader, tokenizer, step, device


def eval_perplexity(model, val_loader, device):
    """Compute NLL, perplexity, bits-per-byte."""
    from eval.perplexity import compute_nll

    logger.info("Computing perplexity (this may take a minute)...")
    t0 = time.time()
    results = compute_nll(model, val_loader, device=device, n_samples=5)
    elapsed = time.time() - t0
    logger.info("  NLL: %.4f | PPL: %.2f | BPB: %.4f (%.1fs)",
                results["nll"], results["ppl"], results["bpb"], elapsed)
    return results


def eval_cluster_health(model, val_loader, device):
    """Evaluate cluster usage, dead clusters, centroid diversity."""
    from eval.cluster_health import full_cluster_health

    if not isinstance(model.embedding, CompositeEmbedding):
        return {"skipped": "flat embedding"}

    logger.info("Evaluating cluster health...")

    # Run a few batches to get routing stats
    all_fine_weights = []
    max_batches = 50

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            _ = model(input_ids, attn_mask)
            info = model.embedding.get_routing_info()
            fw = info.get("fine_weights")
            if fw is not None:
                all_fine_weights.append(fw.cpu())

    if not all_fine_weights:
        return {"error": "no routing weights collected"}

    combined_weights = torch.cat(all_fine_weights, dim=0)
    fine_centroids = model.embedding.fine_centroids.centroids.detach().cpu()

    health = full_cluster_health(combined_weights, fine_centroids)
    logger.info("  Entropy ratio: %.4f | Dead clusters: %d | Mean cosine: %.4f",
                health["entropy_ratio"], health["dead_clusters"], health["mean_cosine"])
    return health


def eval_gate(model, val_loader, device):
    """Analyze gate activation distribution."""
    from eval.gate_analysis import analyze_gate

    if not isinstance(model.embedding, CompositeEmbedding):
        return {"skipped": "flat embedding"}

    logger.info("Analyzing gate activations...")
    results = analyze_gate(model, val_loader, device=device, max_batches=50)
    if "error" not in results:
        logger.info("  Gate mean: %.4f | std: %.4f | %%floor: %.1f%% | %%>0.5: %.1f%%",
                    results["mean"], results["std"],
                    results["pct_at_floor"] * 100, results["pct_above_half"] * 100)
    return results


def eval_hierarchy(model):
    """Evaluate coarse-fine alignment."""
    from eval.hierarchy_metrics import coarse_fine_alignment

    if not isinstance(model.embedding, CompositeEmbedding):
        return {"skipped": "flat embedding"}
    if model.embedding.coarse_centroids is None:
        return {"skipped": "no hierarchy"}

    logger.info("Evaluating hierarchy alignment...")
    fine_c = model.embedding.fine_centroids.centroids.detach().cpu()
    coarse_c = model.embedding.coarse_centroids.centroids.detach().cpu()
    results = coarse_fine_alignment(fine_c, coarse_c)
    logger.info("  Coherence: %.4f | Balance: %.4f | Counts: %s",
                results["coherence"], results["balance_score"],
                results["counts_per_coarse"])
    return results


def eval_learned_params(model):
    """Report alpha, beta values."""
    if not isinstance(model.embedding, CompositeEmbedding):
        return {"skipped": "flat embedding"}

    alpha = model.embedding.alpha.item()
    beta = model.embedding.beta.item()
    logger.info("  Alpha: %.6f | Beta: %.6f", alpha, beta)
    return {"alpha": alpha, "beta": beta}


def eval_generation(model, tokenizer, prompts, device):
    """Generate text samples."""
    from eval.generation import generate_samples

    if tokenizer is None:
        logger.warning("No tokenizer — skipping generation")
        return []

    logger.info("Generating %d samples...", len(prompts))
    results = generate_samples(
        model, tokenizer, prompts,
        seq_len=128,
        sampling_steps=32,
        temperature=0.8,
        device=device,
    )

    for i, r in enumerate(results):
        text = r["generated"][:200]
        logger.info("  [%d] Prompt: %s", i + 1, r["prompt"])
        logger.info("       Output: %s", text)

    return results


def send_to_dashboard(step, all_results, run_name):
    """Send evaluation results to dashboard."""
    try:
        from training.dashboard_reporter import DashboardReporter
        reporter = DashboardReporter(run=run_name)
        if not reporter._enabled:
            logger.warning("Dashboard not reachable — results saved locally only")
            return False

        # Losses from perplexity
        ppl = all_results.get("perplexity", {})

        # Cluster health
        health = all_results.get("cluster_health", {})
        cluster_health = None
        if "entropy_ratio" in health:
            cluster_health = {
                "entropy_ratio": health["entropy_ratio"],
                "dead_clusters": health.get("dead_clusters", 0),
                "centroid_similarity": health.get("mean_cosine", 0),
            }

        # Gate
        gate_data = all_results.get("gate", {})
        gate = None
        if "mean" in gate_data:
            gate = {"mean": gate_data["mean"], "std": gate_data["std"]}

        # Hierarchy
        hier = all_results.get("hierarchy", {})
        hierarchy = None
        if "coherence" in hier:
            hierarchy = {
                "coarse_fine_alignment": hier["coherence"],
                "balance": hier["balance_score"],
            }

        # Extra
        extra = {}
        params = all_results.get("learned_params", {})
        if "alpha" in params:
            extra["alpha"] = params["alpha"]
            extra["beta"] = params["beta"]
        if ppl:
            extra["eval_nll"] = ppl.get("nll")
            extra["eval_ppl"] = ppl.get("ppl")
            extra["eval_bpb"] = ppl.get("bpb")

        reporter.report(
            step=step,
            cluster_health=cluster_health,
            gate=gate,
            hierarchy=hierarchy,
            extra=extra if extra else None,
        )

        # Send generation samples
        for sample in all_results.get("generation", []):
            reporter.report_generation(
                step=step,
                text=sample["generated"],
                prompt=sample["prompt"],
            )

        reporter.close()
        logger.info("Results sent to dashboard (run=%s)", run_name)
        return True
    except Exception as e:
        logger.warning("Failed to send to dashboard: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Comprehensive HCLM-D checkpoint evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    parser.add_argument("--dashboard", action="store_true", help="Send results to dashboard")
    parser.add_argument("--run-name", type=str, default="base-20m", help="Dashboard run name")
    parser.add_argument("--quick", action="store_true", help="Skip slow perplexity eval")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom generation prompts")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    from scripts.train import load_config
    config = load_config(args.config)

    model, val_loader, tokenizer, step, device = load_model_and_data(args, config)

    prompts = args.prompts or DEFAULT_PROMPTS
    all_results = {"step": step, "checkpoint": args.checkpoint}

    print(f"\n{'='*60}")
    print(f"  HCLM-D Evaluation — Step {step}")
    print(f"{'='*60}\n")

    # 1. Learned parameters (instant)
    print("--- Learned Parameters ---")
    all_results["learned_params"] = eval_learned_params(model)

    # 2. Cluster health
    if val_loader:
        print("\n--- Cluster Health ---")
        all_results["cluster_health"] = eval_cluster_health(model, val_loader, device)

    # 3. Gate analysis
    if val_loader:
        print("\n--- Gate Analysis ---")
        all_results["gate"] = eval_gate(model, val_loader, device)

    # 4. Hierarchy
    print("\n--- Hierarchy ---")
    all_results["hierarchy"] = eval_hierarchy(model)

    # 5. Perplexity (slow)
    if val_loader and not args.quick:
        print("\n--- Perplexity ---")
        all_results["perplexity"] = eval_perplexity(model, val_loader, device)
    elif args.quick:
        logger.info("Skipping perplexity (--quick mode)")

    # 6. Generation
    print("\n--- Generation ---")
    all_results["generation"] = eval_generation(model, tokenizer, prompts, device)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    summary_lines = [f"  Step: {step}"]

    ppl = all_results.get("perplexity", {})
    if ppl:
        summary_lines.append(f"  NLL: {ppl['nll']:.4f} | PPL: {ppl['ppl']:.2f} | BPB: {ppl['bpb']:.4f}")

    params = all_results.get("learned_params", {})
    if "alpha" in params:
        summary_lines.append(f"  Alpha: {params['alpha']:.6f} | Beta: {params['beta']:.6f}")

    health = all_results.get("cluster_health", {})
    if "entropy_ratio" in health:
        summary_lines.append(
            f"  Entropy ratio: {health['entropy_ratio']:.4f} | "
            f"Dead clusters: {health['dead_clusters']} | "
            f"Centroid sim: {health['mean_cosine']:.4f}"
        )

    gate = all_results.get("gate", {})
    if "mean" in gate:
        summary_lines.append(
            f"  Gate: {gate['mean']:.4f} ± {gate['std']:.4f} | "
            f"Floor: {gate['pct_at_floor']*100:.1f}% | >0.5: {gate['pct_above_half']*100:.1f}%"
        )

    hier = all_results.get("hierarchy", {})
    if "coherence" in hier:
        summary_lines.append(
            f"  Hierarchy coherence: {hier['coherence']:.4f} | "
            f"Balance: {hier['balance_score']:.4f}"
        )

    for line in summary_lines:
        print(line)
    print()

    # Save JSON
    output_path = args.output or f"results/eval_step_{step}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable generation results for JSON
    json_results = {k: v for k, v in all_results.items()}
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # Dashboard
    if args.dashboard:
        send_to_dashboard(step, all_results, args.run_name)


if __name__ == "__main__":
    main()
