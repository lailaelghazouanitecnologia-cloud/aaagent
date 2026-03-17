"""Agno-based evaluation agent for HCLM-D.

Orchestrates the full evaluation pipeline:
  1. Load checkpoint → generate 30 samples
  2. Run automatic metrics (no LLM)
  3. Run LLM-as-judge via Groq
  4. Aggregate & report to dashboard

Requires:
  pip install agno groq
  export GROQ_API_KEY=...

Usage:
  python -m eval.agent_eval --checkpoint checkpoints/step_10000.pt
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agno agent definition
# ---------------------------------------------------------------------------

def build_eval_agent():
    """Build the Agno evaluation agent with Groq as the LLM backend."""
    from agno.agent import Agent
    from agno.models.groq import Groq

    model_id = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    agent = Agent(
        name="HCLM-D Evaluator",
        model=Groq(id=model_id),
        description="Evaluates HCLM-D text generation quality using structured rubrics.",
        instructions=[
            "You are an expert evaluator of language model outputs.",
            "You evaluate children's story text on: coherence, grammar, relevance, "
            "creativity, fluency, completeness, and repetition.",
            "Always return structured JSON scores.",
            "Detect failure modes: repetition_loop, nonsense, truncated, copied, off_topic.",
        ],
        markdown=False,
        show_tool_calls=False,
    )
    return agent


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    config_path: str = "configs/base.yaml",
    seq_len: int = 256,
    sampling_steps: int = 64,
    temperature: float = 0.9,
    device: str = "cuda",
    dashboard_url: str | None = None,
    run_name: str | None = None,
    output_dir: str = "eval_results",
) -> dict[str, Any]:
    """Run the complete evaluation pipeline.

    Returns a dict with all metrics (auto + LLM judge).
    """
    import torch
    import yaml

    from data.tokenizer import get_tokenizer
    from eval.auto_metrics import aggregate_metrics, compute_sample_metrics
    from eval.bench_dataset import BENCH, get_prompt_texts
    from eval.generation import generate_samples
    from eval.llm_judge import aggregate_judge_scores, judge_batch
    from model.config import ModelConfig
    from model.lm import HCLMD

    # ── 1. Load model ────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    config = ModelConfig.from_dict(raw_cfg)
    model = HCLMD(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = get_tokenizer()
    step = ckpt.get("step", 0)
    run_name = run_name or Path(checkpoint_path).stem

    logger.info(f"Model loaded — step {step}, device {device}")

    # ── 2. Generate 30 samples ───────────────────────────────────────────
    logger.info("Generating 30 samples from benchmark prompts...")
    prompts = get_prompt_texts()
    t0 = time.time()
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        seq_len=seq_len,
        sampling_steps=sampling_steps,
        temperature=temperature,
        device=device,
    )
    gen_time = time.time() - t0
    logger.info(f"Generation complete in {gen_time:.1f}s")

    # ── 3. Automatic metrics (no LLM) ───────────────────────────────────
    logger.info("Computing automatic metrics...")
    per_sample_metrics = []
    texts = []
    for bench_item, sample in zip(BENCH, samples):
        text = sample["generated"]
        texts.append(text)
        m = compute_sample_metrics(
            text=text,
            prompt=sample["prompt"],
            required_keywords=bench_item.required_keywords,
            banned_patterns=bench_item.banned_patterns,
            min_tokens=bench_item.min_tokens,
            max_tokens=bench_item.max_tokens,
        )
        m["id"] = bench_item.id
        m["category"] = bench_item.category
        per_sample_metrics.append(m)

    auto_summary = aggregate_metrics(per_sample_metrics, texts)
    auto_summary["generation_time_s"] = round(gen_time, 2)
    logger.info(f"Auto metrics: distinct_2={auto_summary['mean_distinct_2']}, "
                f"repetition={auto_summary['mean_repetition_ratio']}")

    # ── 4. LLM-as-judge via Groq ────────────────────────────────────────
    llm_summary: dict[str, Any] = {}
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        logger.info("Running LLM-as-judge (Groq)...")
        categories = [b.category for b in BENCH]
        judge_results = judge_batch(
            samples=samples,
            categories=categories,
            api_key=groq_key,
        )
        llm_summary = aggregate_judge_scores(judge_results)

        # Per-sample LLM scores
        for i, (jr, sm) in enumerate(zip(judge_results, per_sample_metrics)):
            if jr.success:
                sm["llm_coherence"] = jr.coherence
                sm["llm_grammar"] = jr.grammar
                sm["llm_relevance"] = jr.relevance
                sm["llm_creativity"] = jr.creativity
                sm["llm_fluency"] = jr.fluency
                sm["llm_completeness"] = jr.completeness
                sm["llm_failure_mode"] = jr.failure_mode
                sm["llm_reasoning"] = jr.reasoning

        logger.info(f"LLM judge: overall_quality={llm_summary.get('overall_quality', 'N/A')}")
    else:
        logger.warning("GROQ_API_KEY not set — skipping LLM-as-judge")
        llm_summary = {"skipped": True, "reason": "GROQ_API_KEY not set"}

    # ── 5. Assemble final report ─────────────────────────────────────────
    report = {
        "checkpoint": checkpoint_path,
        "step": step,
        "run": run_name,
        "config": config_path,
        "generation_params": {
            "seq_len": seq_len,
            "sampling_steps": sampling_steps,
            "temperature": temperature,
        },
        "auto_metrics": auto_summary,
        "llm_judge": llm_summary,
        "samples": [
            {
                "id": BENCH[i].id,
                "category": BENCH[i].category,
                "prompt": samples[i]["prompt"],
                "generated": samples[i]["generated"],
                "metrics": per_sample_metrics[i],
            }
            for i in range(len(samples))
        ],
    }

    # ── 6. Save results ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_{run_name}_step{step}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    # ── 7. Report to dashboard (optional) ────────────────────────────────
    if dashboard_url:
        _report_to_dashboard(report, dashboard_url, run_name, step)

    return report


def _report_to_dashboard(
    report: dict, url: str, run: str, step: int
) -> None:
    """Send evaluation summary to the HCLM-D dashboard."""
    from training.dashboard_reporter import DashboardReporter

    reporter = DashboardReporter(run=run, url=url, batch_size=1)
    if not reporter._enabled:
        logger.warning("Dashboard not reachable, skipping report")
        return

    auto = report["auto_metrics"]
    llm = report["llm_judge"]

    reporter.report(
        step=step,
        extra={
            "eval_auto": {
                "distinct_1": auto["mean_distinct_1"],
                "distinct_2": auto["mean_distinct_2"],
                "repetition": auto["mean_repetition_ratio"],
                "self_bleu_4": auto["self_bleu_4"],
                "keyword_hit": auto["mean_keyword_hit"],
                "vocab_richness": auto["mean_vocab_richness"],
            },
            "eval_llm": {
                k: v for k, v in llm.items()
                if k not in ("failure_modes",) and not isinstance(v, dict)
            } if not llm.get("skipped") else {"skipped": True},
            "eval_type": "bench_30",
        },
    )

    # Report individual samples for the generation page
    for s in report["samples"][:5]:  # Top 5 samples
        reporter.report_generation(
            step=step,
            text=s["generated"],
            prompt=s["prompt"],
            metrics=s["metrics"],
        )

    reporter.close()
    logger.info("Eval metrics sent to dashboard")


# ---------------------------------------------------------------------------
# Agno agent wrapper — for interactive use
# ---------------------------------------------------------------------------

def run_with_agent(checkpoint_path: str, **kwargs) -> dict[str, Any]:
    """Run evaluation via the Agno agent for richer analysis.

    The agent:
    1. Runs the standard pipeline
    2. Analyzes the results
    3. Produces a natural-language summary with recommendations
    """
    # Run the pipeline first
    report = run_evaluation(checkpoint_path, **kwargs)

    # Then ask the agent to analyze
    agent = build_eval_agent()

    analysis_prompt = _build_analysis_prompt(report)
    response = agent.run(analysis_prompt)

    report["agent_analysis"] = response.content

    # Save updated report
    output_dir = kwargs.get("output_dir", "eval_results")
    run_name = report["run"]
    step = report["step"]
    out_path = os.path.join(output_dir, f"eval_{run_name}_step{step}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Agent analysis complete")
    return report


def _build_analysis_prompt(report: dict) -> str:
    """Build a prompt for the Agno agent to analyze eval results."""
    auto = report["auto_metrics"]
    llm = report.get("llm_judge", {})

    prompt = f"""Analyze these HCLM-D evaluation results (step {report['step']}):

## Automatic Metrics
- Distinct-1: {auto['mean_distinct_1']}, Distinct-2: {auto['mean_distinct_2']}
- Repetition ratio: {auto['mean_repetition_ratio']}
- Self-BLEU-4 (diversity): {auto['self_bleu_4']}
- Keyword hit rate: {auto['mean_keyword_hit']}
- Vocab richness: {auto['mean_vocab_richness']}
- Banned violations: {auto['total_banned_violations']}
- Length compliance: {auto['mean_length_compliance']}

## Per-Category Breakdown
{json.dumps(auto.get('by_category', {}), indent=2)}

## Vocabulary Stats
{json.dumps(auto.get('vocab', {}), indent=2)}
"""

    if not llm.get("skipped"):
        prompt += f"""
## LLM Judge Scores
- Overall quality: {llm.get('overall_quality', 'N/A')}/5
- Coherence: {llm.get('mean_coherence', 'N/A')}/5
- Grammar: {llm.get('mean_grammar', 'N/A')}/5
- Creativity: {llm.get('mean_creativity', 'N/A')}/5
- Fluency: {llm.get('mean_fluency', 'N/A')}/5
- Failure modes: {json.dumps(llm.get('failure_modes', {}))}
"""

    prompt += """
## Sample Outputs (first 3)
"""
    for s in report["samples"][:3]:
        prompt += f"\nPrompt: {s['prompt']}\nGenerated: {s['generated'][:200]}...\n"

    prompt += """
Provide:
1. Overall assessment (1 paragraph)
2. Top 3 strengths
3. Top 3 weaknesses
4. Specific recommendations for next training steps
5. Risk of mode collapse or training failure (low/medium/high)

Be concise and actionable.
"""
    return prompt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HCLM-D Evaluation Agent")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/base.yaml", help="Model config")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dashboard-url", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument(
        "--with-agent", action="store_true",
        help="Run Agno agent for natural-language analysis",
    )

    args = parser.parse_args()

    kwargs = dict(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        seq_len=args.seq_len,
        sampling_steps=args.sampling_steps,
        temperature=args.temperature,
        device=args.device,
        dashboard_url=args.dashboard_url,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    if args.with_agent:
        report = run_with_agent(**kwargs)
        if "agent_analysis" in report:
            print("\n" + "=" * 60)
            print("AGENT ANALYSIS")
            print("=" * 60)
            print(report["agent_analysis"])
    else:
        report = run_evaluation(**kwargs)

    # Print summary
    auto = report["auto_metrics"]
    llm = report.get("llm_judge", {})
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — step {report['step']}")
    print("=" * 60)
    print(f"  Distinct-2:        {auto['mean_distinct_2']}")
    print(f"  Repetition:        {auto['mean_repetition_ratio']}")
    print(f"  Self-BLEU-4:       {auto['self_bleu_4']}")
    print(f"  Keyword hit:       {auto['mean_keyword_hit']}")
    print(f"  Vocab richness:    {auto['mean_vocab_richness']}")
    if not llm.get("skipped"):
        print(f"  LLM quality:       {llm.get('overall_quality', 'N/A')}/5")
        print(f"  Failure modes:     {llm.get('failure_modes', {})}")
    print(f"\n  Full report: {args.output_dir}/eval_{report['run']}_step{report['step']}.json")


if __name__ == "__main__":
    main()
