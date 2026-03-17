"""LLM-as-Judge evaluation via Groq API (used by the Agno agent).

Sends each generated sample to an LLM with a structured rubric
and parses back numerical scores + reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert evaluator of children's story text quality.
You will receive a PROMPT and the model's GENERATED continuation.
Score the generation on each dimension (1-5 integer) and detect failure modes.

Return ONLY valid JSON with this exact schema:
{
  "coherence": <1-5>,
  "grammar": <1-5>,
  "relevance": <1-5>,
  "creativity": <1-5>,
  "fluency": <1-5>,
  "completeness": <1-5>,
  "repetition_score": <0.0-1.0>,
  "failure_mode": "<none|repetition_loop|nonsense|truncated|copied|off_topic>",
  "reasoning": "<1-2 sentence explanation>"
}

Scoring guide:
- coherence: Does the text make logical sense as a whole? (5=perfect, 1=nonsensical)
- grammar: Is the text grammatically correct? (5=perfect, 1=broken)
- relevance: Does the continuation follow from the prompt? (5=directly relevant, 1=unrelated)
- creativity: Is the output original and interesting? (5=very creative, 1=generic/copied)
- fluency: Does it read naturally? (5=natural, 1=robotic/broken)
- completeness: Does the text feel complete, not cut off? (5=complete arc, 1=mid-sentence stop)
- repetition_score: 0.0=no repetition, 1.0=entirely repetitive loops
- failure_mode: The primary failure category, or "none" if acceptable quality
"""


def _build_user_msg(prompt: str, generated: str, category: str) -> str:
    return (
        f"Category: {category}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"GENERATED:\n{generated}\n\n"
        "Evaluate and return JSON only."
    )


# ---------------------------------------------------------------------------
# Groq API caller
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    coherence: int = 1
    grammar: int = 1
    relevance: int = 1
    creativity: int = 1
    fluency: int = 1
    completeness: int = 1
    repetition_score: float = 1.0
    failure_mode: str = "nonsense"
    reasoning: str = ""
    raw_response: str = ""
    success: bool = False


def call_groq(
    prompt: str,
    generated: str,
    category: str = "general",
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int = 3,
    timeout: int = 30,
) -> JudgeResult:
    """Call Groq API to judge a single sample. Returns parsed JudgeResult."""
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return JudgeResult(reasoning="API key missing")

    model = model or GROQ_MODEL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _build_user_msg(prompt, generated, category)},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                GROQ_API_URL, headers=headers, json=payload, timeout=timeout
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Groq rate-limited, retrying in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_judge_response(content)

        except requests.RequestException as e:
            logger.warning(f"Groq request failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return JudgeResult(reasoning="All retries exhausted")


def _parse_judge_response(raw: str) -> JudgeResult:
    """Parse the JSON response from the LLM judge."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return JudgeResult(
            coherence=int(data.get("coherence", 1)),
            grammar=int(data.get("grammar", 1)),
            relevance=int(data.get("relevance", 1)),
            creativity=int(data.get("creativity", 1)),
            fluency=int(data.get("fluency", 1)),
            completeness=int(data.get("completeness", 1)),
            repetition_score=float(data.get("repetition_score", 1.0)),
            failure_mode=str(data.get("failure_mode", "unknown")),
            reasoning=str(data.get("reasoning", "")),
            raw_response=raw,
            success=True,
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse judge response: {e}\nRaw: {raw[:300]}")
        return JudgeResult(reasoning=f"Parse error: {e}", raw_response=raw)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def judge_batch(
    samples: list[dict[str, str]],
    categories: list[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    delay: float = 0.5,
) -> list[JudgeResult]:
    """Evaluate a batch of samples. Adds delay between calls to respect rate limits."""
    results = []
    for i, sample in enumerate(samples):
        cat = categories[i] if categories else "general"
        result = call_groq(
            prompt=sample["prompt"],
            generated=sample["generated"],
            category=cat,
            api_key=api_key,
            model=model,
        )
        results.append(result)
        if i < len(samples) - 1:
            time.sleep(delay)
    return results


def aggregate_judge_scores(results: list[JudgeResult]) -> dict[str, Any]:
    """Aggregate JudgeResult list into summary statistics."""
    successful = [r for r in results if r.success]
    n = len(successful)
    if n == 0:
        return {"error": "No successful judge evaluations", "n_failed": len(results)}

    dims = ["coherence", "grammar", "relevance", "creativity", "fluency", "completeness"]
    scores: dict[str, Any] = {"n_evaluated": n, "n_failed": len(results) - n}

    for dim in dims:
        vals = [getattr(r, dim) for r in successful]
        scores[f"mean_{dim}"] = round(sum(vals) / n, 2)
        scores[f"min_{dim}"] = min(vals)
        scores[f"max_{dim}"] = max(vals)

    rep_vals = [r.repetition_score for r in successful]
    scores["mean_repetition_score"] = round(sum(rep_vals) / n, 3)

    # Failure mode distribution
    modes: dict[str, int] = {}
    for r in successful:
        modes[r.failure_mode] = modes.get(r.failure_mode, 0) + 1
    scores["failure_modes"] = modes

    # Overall quality score (mean of all dimensions)
    all_means = [scores[f"mean_{d}"] for d in dims]
    scores["overall_quality"] = round(sum(all_means) / len(all_means), 2)

    return scores
