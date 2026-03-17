"""Full evaluation pipeline that populates metrics/{tag}/.

Runs all evaluation modules and saves structured results.
Optionally runs LLM judge if GROQ_API_KEY is set.

Usage:
    from metrics.evaluate import run_full_eval
    run_full_eval(model, config, tag="hier-boost", step=10000, device="cuda")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.lm import HCLMD
from model.embedding.composite import CompositeEmbedding
from metrics.collector import (
    VersionMetrics, save_summary, save_samples, save_judge,
)

logger = logging.getLogger(__name__)


def run_full_eval(
    model: HCLMD,
    config: dict,
    tag: str,
    version_id: str = "",
    step: int = 0,
    device: str = "cuda",
    val_loader: DataLoader | None = None,
    tokenizer=None,
    skip_perplexity: bool = False,
    skip_judge: bool = False,
    n_gen_samples: int = 10,
) -> VersionMetrics:
    """Run complete evaluation and save to metrics/{tag}/.

    Returns the populated VersionMetrics object.
    """
    model.eval()

    vm = VersionMetrics(
        tag=tag,
        version_id=version_id,
        step=step,
        config=config.get("_source", "unknown"),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # ── 1. Learned params ──
    if isinstance(model.embedding, CompositeEmbedding):
        vm.alpha = round(model.embedding.alpha.item(), 6)
        vm.beta = round(model.embedding.beta.item(), 6)
        logger.info("Alpha: %.6f | Beta: %.6f", vm.alpha, vm.beta)

    # ── 2. Cluster health ──
    if val_loader and isinstance(model.embedding, CompositeEmbedding):
        _eval_clusters(model, val_loader, device, vm)

    # ── 3. Gate analysis ──
    if val_loader and isinstance(model.embedding, CompositeEmbedding):
        _eval_gate(model, val_loader, device, vm)

    # ── 4. Hierarchy metrics ──
    if isinstance(model.embedding, CompositeEmbedding):
        _eval_hierarchy(model, vm)

    # ── 5. Perplexity ──
    if val_loader and not skip_perplexity:
        _eval_perplexity(model, val_loader, device, vm)

    # ── 6. Generation + auto metrics ──
    samples = []
    if tokenizer:
        samples = _eval_generation(model, tokenizer, device, vm, n_samples=n_gen_samples)

    # ── 7. LLM judge ──
    if samples and not skip_judge and os.environ.get("GROQ_API_KEY"):
        _eval_judge(tag, samples, vm)
    elif not os.environ.get("GROQ_API_KEY"):
        logger.info("GROQ_API_KEY not set — skipping LLM judge")

    # ── Save everything ──
    save_summary(tag, vm)
    if samples:
        save_samples(tag, samples)

    logger.info("Evaluation complete for [%s] at step %d", tag, step)
    return vm


def _eval_clusters(model, val_loader, device, vm: VersionMetrics):
    """Evaluate cluster health metrics."""
    from eval.cluster_health import full_cluster_health
    import math

    logger.info("Evaluating cluster health...")
    all_fine_weights = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 50:
                break
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            _ = model(input_ids, attn_mask)
            info = model.embedding.get_routing_info()
            fw = info.get("fine_weights")
            if fw is not None:
                all_fine_weights.append(fw.cpu())

    if not all_fine_weights:
        return

    combined = torch.cat(all_fine_weights, dim=0)
    fine_c = model.embedding.fine_centroids.centroids.detach().cpu()
    health = full_cluster_health(combined, fine_c)

    vm.entropy_ratio = round(health["entropy_ratio"], 4)
    vm.dead_clusters = health["dead_clusters"]
    vm.centroid_similarity = round(health["mean_cosine"], 4)

    # Router entropy (bits)
    eps = 1e-8
    with torch.no_grad():
        h = -(combined * (combined + eps).log2()).sum(dim=-1).mean().item()
    max_h = math.log2(combined.shape[-1])
    vm.router_entropy = round(h, 4)

    logger.info("  Entropy ratio: %.4f | Dead: %d | Centroid sim: %.4f | H_router: %.2f/%.2f",
                vm.entropy_ratio, vm.dead_clusters, vm.centroid_similarity, h, max_h)


def _eval_gate(model, val_loader, device, vm: VersionMetrics):
    """Evaluate gate activation distribution."""
    from eval.gate_analysis import analyze_gate

    if model.embedding.gate is None:
        return

    logger.info("Analyzing gate...")
    results = analyze_gate(model, val_loader, device=device, max_batches=50)
    if "error" in results:
        return

    vm.gate_mean = round(results["mean"], 4)
    vm.gate_std = round(results["std"], 4)
    vm.gate_pct_floor = round(results["pct_at_floor"], 4)
    logger.info("  Gate: %.4f ± %.4f | Floor: %.1f%%",
                vm.gate_mean, vm.gate_std, vm.gate_pct_floor * 100)


def _eval_hierarchy(model, vm: VersionMetrics):
    """Evaluate hierarchy alignment."""
    from eval.hierarchy_metrics import coarse_fine_alignment

    if model.embedding.coarse_centroids is None:
        return

    logger.info("Evaluating hierarchy...")
    fine_c = model.embedding.fine_centroids.centroids.detach().cpu()
    coarse_c = model.embedding.coarse_centroids.centroids.detach().cpu()
    results = coarse_fine_alignment(fine_c, coarse_c)

    vm.hierarchy_coherence = round(results["coherence"], 4)
    vm.hierarchy_balance = round(results["balance_score"], 4)
    logger.info("  Coherence: %.4f | Balance: %.4f", vm.hierarchy_coherence, vm.hierarchy_balance)


def _eval_perplexity(model, val_loader, device, vm: VersionMetrics):
    """Compute perplexity metrics."""
    from eval.perplexity import compute_nll

    logger.info("Computing perplexity...")
    t0 = time.time()
    results = compute_nll(model, val_loader, device=device, n_samples=5)
    elapsed = time.time() - t0

    vm.nll = round(results["nll"], 4)
    vm.ppl = round(results["ppl"], 2)
    vm.bpb = round(results["bpb"], 4)
    logger.info("  NLL: %.4f | PPL: %.2f | BPB: %.4f (%.1fs)",
                vm.nll, vm.ppl, vm.bpb, elapsed)


def _eval_generation(model, tokenizer, device, vm: VersionMetrics,
                     n_samples: int = 10) -> list[dict]:
    """Generate samples and compute auto metrics."""
    from eval.generation import generate_samples
    from eval.auto_metrics import compute_sample_metrics, aggregate_metrics

    prompts = [
        "Once upon a time",
        "The little girl",
        "There was a big",
        "One day the",
        "She looked at the",
        "He went to the",
        "The cat and the",
        "In a small village",
        "The sun was",
        "They played in the",
    ][:n_samples]

    logger.info("Generating %d samples...", len(prompts))
    gen_results = generate_samples(
        model, tokenizer, prompts,
        seq_len=128, sampling_steps=32, temperature=0.8, device=device,
    )

    # Auto metrics per sample
    samples = []
    for r in gen_results:
        auto = compute_sample_metrics(r["generated"], r["prompt"])
        samples.append({
            "prompt": r["prompt"],
            "generated": r["generated"],
            "auto_metrics": auto,
        })

    # Corpus-level
    texts = [s["generated"] for s in samples]
    agg = aggregate_metrics([s["auto_metrics"] for s in samples], texts)

    vm.mean_distinct_2 = agg.get("mean_distinct_2")
    vm.mean_repetition = agg.get("mean_repetition_ratio")
    vm.self_bleu_4 = agg.get("self_bleu_4")
    vm.vocab_richness = agg.get("mean_vocab_richness")

    logger.info("  Distinct-2: %.4f | Repetition: %.4f | Self-BLEU: %.4f",
                vm.mean_distinct_2 or 0, vm.mean_repetition or 0, vm.self_bleu_4 or 0)

    model.train()
    return samples


def _eval_judge(tag: str, samples: list[dict], vm: VersionMetrics):
    """Run LLM judge evaluation."""
    from eval.llm_judge import judge_batch, aggregate_judge_scores

    logger.info("Running LLM judge on %d samples...", len(samples))
    judge_input = [{"prompt": s["prompt"], "generated": s["generated"]} for s in samples]
    results = judge_batch(judge_input, delay=0.5)

    agg = aggregate_judge_scores(results)
    vm.judge_overall = agg.get("overall_quality")
    vm.judge_coherence = agg.get("mean_coherence")
    vm.judge_grammar = agg.get("mean_grammar")
    vm.judge_fluency = agg.get("mean_fluency")
    vm.judge_creativity = agg.get("mean_creativity")
    vm.judge_failure_modes = agg.get("failure_modes")

    # Save detailed judge results
    detailed = {
        "aggregate": agg,
        "per_sample": [
            {
                "prompt": s["prompt"],
                "coherence": r.coherence,
                "grammar": r.grammar,
                "fluency": r.fluency,
                "creativity": r.creativity,
                "relevance": r.relevance,
                "completeness": r.completeness,
                "repetition_score": r.repetition_score,
                "failure_mode": r.failure_mode,
                "reasoning": r.reasoning,
            }
            for s, r in zip(samples, results) if r.success
        ],
    }
    save_judge(tag, detailed)

    logger.info("  Judge overall: %.2f | Coherence: %.2f | Grammar: %.2f",
                vm.judge_overall or 0, vm.judge_coherence or 0, vm.judge_grammar or 0)
