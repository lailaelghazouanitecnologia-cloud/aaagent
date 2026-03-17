"""Automatic text quality metrics — no LLM required.

These complement the LLM-as-judge scores with deterministic,
reproducible measurements.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def distinct_n(tokens: list[str], n: int) -> float:
    """Fraction of unique n-grams among all n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_ratio(text: str, window: int = 10) -> float:
    """Sliding-window repetition: fraction of windows that are duplicates."""
    words = text.lower().split()
    if len(words) < window:
        return 0.0
    windows = [tuple(words[i : i + window]) for i in range(len(words) - window + 1)]
    unique = len(set(windows))
    return 1.0 - unique / len(windows)


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in the generated text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def banned_pattern_violations(text: str, patterns: list[str]) -> int:
    """Count how many banned patterns appear in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if p.lower() in text_lower)


def length_compliance(n_tokens: int, min_t: int, max_t: int) -> float:
    """1.0 if within range, decays linearly outside."""
    if min_t <= n_tokens <= max_t:
        return 1.0
    if n_tokens < min_t:
        return max(0.0, n_tokens / min_t)
    # over max
    return max(0.0, 1.0 - (n_tokens - max_t) / max_t)


def sentence_count(text: str) -> int:
    """Rough sentence count via punctuation splitting."""
    return max(1, len(re.split(r'[.!?]+', text.strip())) - 1)


def vocab_richness(tokens: list[str]) -> float:
    """Type-token ratio (TTR)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_sample_metrics(
    text: str,
    prompt: str,
    required_keywords: list[str] | None = None,
    banned_patterns: list[str] | None = None,
    min_tokens: int = 20,
    max_tokens: int = 200,
) -> dict[str, Any]:
    """Compute all automatic metrics for a single generated sample."""
    # Strip the prompt prefix from the generated text if present
    continuation = text
    if text.startswith(prompt):
        continuation = text[len(prompt):]

    words = continuation.split()
    n_tokens = len(words)

    return {
        "n_tokens": n_tokens,
        "distinct_1": round(distinct_n(words, 1), 4),
        "distinct_2": round(distinct_n(words, 2), 4),
        "distinct_3": round(distinct_n(words, 3), 4),
        "repetition_ratio": round(repetition_ratio(continuation), 4),
        "keyword_hit": round(keyword_hit_rate(continuation, required_keywords or []), 4),
        "banned_violations": banned_pattern_violations(continuation, banned_patterns or []),
        "length_compliance": round(length_compliance(n_tokens, min_tokens, max_tokens), 4),
        "vocab_richness": round(vocab_richness(words), 4),
        "sentence_count": sentence_count(continuation),
    }


# ---------------------------------------------------------------------------
# Corpus-level (across all 30 samples) metrics
# ---------------------------------------------------------------------------

def self_bleu_approx(texts: list[str], n: int = 4) -> float:
    """Approximate self-BLEU: average pairwise n-gram overlap.

    Lower is better (more diverse). Exact self-BLEU is expensive;
    this uses Jaccard on n-gram sets as a fast proxy.
    """
    if len(texts) < 2:
        return 0.0

    ngram_sets = []
    for t in texts:
        words = t.lower().split()
        grams = set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        ngram_sets.append(grams)

    total = 0.0
    pairs = 0
    for i in range(len(ngram_sets)):
        for j in range(i + 1, len(ngram_sets)):
            if not ngram_sets[i] or not ngram_sets[j]:
                continue
            intersection = len(ngram_sets[i] & ngram_sets[j])
            union = len(ngram_sets[i] | ngram_sets[j])
            total += intersection / union if union > 0 else 0.0
            pairs += 1

    return round(total / max(pairs, 1), 4)


def corpus_vocab_coverage(texts: list[str]) -> dict[str, Any]:
    """Vocabulary statistics across all samples."""
    all_words: list[str] = []
    for t in texts:
        all_words.extend(t.lower().split())

    counter = Counter(all_words)
    total = len(all_words)
    unique = len(counter)

    # Top-heaviness: fraction of tokens from the top 20 words
    top20 = sum(c for _, c in counter.most_common(20))
    top20_frac = top20 / total if total > 0 else 0.0

    return {
        "total_tokens": total,
        "unique_tokens": unique,
        "ttr": round(unique / total, 4) if total > 0 else 0.0,
        "top20_concentration": round(top20_frac, 4),
    }


def aggregate_metrics(
    sample_metrics: list[dict[str, Any]],
    texts: list[str],
) -> dict[str, Any]:
    """Aggregate per-sample metrics + corpus-level stats into a summary."""
    n = len(sample_metrics)
    if n == 0:
        return {}

    def _mean(key: str) -> float:
        vals = [s[key] for s in sample_metrics if key in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for sm in sample_metrics:
        cat = sm.get("category", "unknown")
        categories.setdefault(cat, []).append(sm)

    category_scores: dict[str, dict[str, float]] = {}
    for cat, items in categories.items():
        category_scores[cat] = {
            "mean_distinct_2": round(sum(i["distinct_2"] for i in items) / len(items), 4),
            "mean_repetition": round(sum(i["repetition_ratio"] for i in items) / len(items), 4),
            "mean_keyword_hit": round(sum(i["keyword_hit"] for i in items) / len(items), 4),
            "count": len(items),
        }

    return {
        "n_samples": n,
        "mean_distinct_1": _mean("distinct_1"),
        "mean_distinct_2": _mean("distinct_2"),
        "mean_distinct_3": _mean("distinct_3"),
        "mean_repetition_ratio": _mean("repetition_ratio"),
        "mean_keyword_hit": _mean("keyword_hit"),
        "total_banned_violations": sum(s["banned_violations"] for s in sample_metrics),
        "mean_length_compliance": _mean("length_compliance"),
        "mean_vocab_richness": _mean("vocab_richness"),
        "mean_sentence_count": _mean("sentence_count"),
        "self_bleu_4": self_bleu_approx(texts, n=4),
        "vocab": corpus_vocab_coverage(texts),
        "by_category": category_scores,
    }
