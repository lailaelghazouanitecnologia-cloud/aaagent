"""LLM Judge — evaluates cluster quality and suggests reorganizations.

Supports multiple backends:
  - groq: Groq API (fast, cheap — recommended)
  - anthropic: Claude API
  - openai: OpenAI API
  - local: Local model via transformers
  - mock: Deterministic mock for testing
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are analyzing the cluster hierarchy of a language model.
The model groups tokens into {n_fine} fine clusters, then groups those into {n_coarse} coarse groups.

A GOOD hierarchy has:
- Fine clusters that group semantically related tokens (nouns together, verbs together, etc.)
- Coarse groups that represent broader categories (content words vs function words, etc.)
- Balanced sizes (no cluster should have 0 tokens or absorb everything)
- Clear separation (clusters should be distinguishable from each other)

A BAD hierarchy has:
- Random-looking clusters mixing unrelated tokens
- Uniform/collapsed coarse routing (all fine clusters in one group)
- Empty clusters or mega-clusters
- No semantic coherence

{cluster_description}

Respond with EXACTLY this JSON format:
{{
  "quality_score": <float 0-10>,
  "coherence": <float 0-10>,
  "balance": <float 0-10>,
  "separation": <float 0-10>,
  "suggested_merges": [[<fine_id_1>, <fine_id_2>], ...],
  "suggested_splits": [<fine_id>, ...],
  "reassign": [{{\"fine_id\": <id>, \"from_coarse\": <id>, \"to_coarse\": <id>}}, ...],
  "reasoning": "<brief explanation>"
}}"""


@dataclass
class JudgmentResult:
    """Result from LLM evaluation."""
    quality_score: float  # 0-10 overall
    coherence: float  # 0-10 semantic coherence
    balance: float  # 0-10 size balance
    separation: float  # 0-10 cluster separation
    suggested_merges: list[tuple[int, int]]  # fine cluster pairs to merge
    suggested_splits: list[int]  # fine clusters to split
    reassign: list[dict]  # fine clusters to move between coarse groups
    reasoning: str
    raw_response: str = ""


class LLMJudge:
    """Evaluates cluster hierarchy quality using an LLM."""

    def __init__(
        self,
        backend: str = "groq",
        model: str = "moonshotai/kimi-k2-instruct-0905",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.6,
    ):
        self.backend = backend
        self.model_name = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    def evaluate(self, cluster_description: str, n_fine: int = 64, n_coarse: int = 8) -> JudgmentResult:
        """Evaluate cluster quality via LLM."""
        prompt = EVALUATION_PROMPT.format(
            n_fine=n_fine,
            n_coarse=n_coarse,
            cluster_description=cluster_description,
        )

        if self.backend == "mock":
            return self._mock_evaluate(cluster_description, n_fine, n_coarse)
        elif self.backend == "groq":
            return self._groq_evaluate(prompt)
        elif self.backend == "anthropic":
            return self._anthropic_evaluate(prompt)
        elif self.backend == "openai":
            return self._openai_evaluate(prompt)
        elif self.backend == "local":
            return self._local_evaluate(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _parse_response(self, text: str) -> JudgmentResult:
        """Parse LLM JSON response into JudgmentResult."""
        # Extract JSON from response (may have markdown fences)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            logger.warning("Could not parse LLM response, using defaults")
            return JudgmentResult(
                quality_score=5.0, coherence=5.0, balance=5.0, separation=5.0,
                suggested_merges=[], suggested_splits=[], reassign=[],
                reasoning="parse_failed", raw_response=text,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LLM response")
            return JudgmentResult(
                quality_score=5.0, coherence=5.0, balance=5.0, separation=5.0,
                suggested_merges=[], suggested_splits=[], reassign=[],
                reasoning="json_parse_failed", raw_response=text,
            )

        merges = [tuple(m) for m in data.get("suggested_merges", [])]
        return JudgmentResult(
            quality_score=float(data.get("quality_score", 5.0)),
            coherence=float(data.get("coherence", 5.0)),
            balance=float(data.get("balance", 5.0)),
            separation=float(data.get("separation", 5.0)),
            suggested_merges=merges,
            suggested_splits=data.get("suggested_splits", []),
            reassign=data.get("reassign", []),
            reasoning=data.get("reasoning", ""),
            raw_response=text,
        )

    def _groq_evaluate(self, prompt: str) -> JudgmentResult:
        """Evaluate using Groq API (fast inference)."""
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")

        client = Groq(api_key=self.api_key) if self.api_key else Groq()
        # Collect streamed response
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            top_p=1,
            stream=False,
        )
        text = completion.choices[0].message.content
        return self._parse_response(text)

    def _anthropic_evaluate(self, prompt: str) -> JudgmentResult:
        """Evaluate using Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return self._parse_response(text)

    def _openai_evaluate(self, prompt: str) -> JudgmentResult:
        """Evaluate using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return self._parse_response(text)

    def _local_evaluate(self, prompt: str) -> JudgmentResult:
        """Evaluate using a local model (transformers pipeline)."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("pip install transformers")

        pipe = pipeline("text-generation", model=self.model_name, max_new_tokens=self.max_tokens)
        result = pipe(prompt)
        text = result[0]["generated_text"][len(prompt):]
        return self._parse_response(text)

    def _mock_evaluate(self, description: str, n_fine: int, n_coarse: int) -> JudgmentResult:
        """Deterministic mock for testing — analyzes description heuristically."""
        # Simple heuristic: count how many clusters have tokens, check balance
        lines = description.split("\n")
        cluster_lines = [l for l in lines if l.strip().startswith("Cluster")]
        n_with_tokens = sum(1 for l in cluster_lines if "0 tokens" not in l)
        utilization = n_with_tokens / max(n_fine, 1)

        # Check coarse balance from group lines
        group_lines = [l for l in lines if l.strip().startswith("Group")]
        group_sizes = []
        for l in group_lines:
            parts = l.split("fine clusters")
            if parts:
                try:
                    size = int(parts[0].split()[-1])
                    group_sizes.append(size)
                except (ValueError, IndexError):
                    pass

        balance = 5.0
        if group_sizes:
            mean_size = sum(group_sizes) / len(group_sizes)
            variance = sum((s - mean_size) ** 2 for s in group_sizes) / len(group_sizes)
            balance = max(0, 10 - variance)  # Less variance = higher balance

        quality = utilization * 6 + balance * 0.4
        return JudgmentResult(
            quality_score=min(10, quality),
            coherence=utilization * 7,
            balance=balance,
            separation=5.0,
            suggested_merges=[],
            suggested_splits=[],
            reassign=[],
            reasoning=f"mock: {n_with_tokens}/{n_fine} clusters used, balance={balance:.1f}",
            raw_response="mock",
        )
