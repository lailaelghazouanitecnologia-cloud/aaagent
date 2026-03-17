"""A* search over hierarchy configurations.

State: current centroid/routing configuration
Actions: merge, split, reassign, rebalance
g(n): cumulative action cost
h(n): LLM quality score (inverted: 10 - score)
f(n) = g(n) + h(n), minimize

The search finds a sequence of actions that maximizes
the LLM's quality assessment of the resulting cluster hierarchy.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from hierarchy.analyzer import ClusterAnalyzer, HierarchySnapshot
from hierarchy.llm_judge import LLMJudge, JudgmentResult
from hierarchy.actions import (
    Action, ActionType, HierarchyModifier,
)

logger = logging.getLogger(__name__)


@dataclass(order=True)
class SearchNode:
    """Node in the A* search tree."""
    f_score: float
    g_cost: float = field(compare=False)
    h_heuristic: float = field(compare=False)
    actions: list[Action] = field(compare=False, default_factory=list)
    judgment: Optional[JudgmentResult] = field(compare=False, default=None)
    snapshot_hash: int = field(compare=False, default=0)


@dataclass
class SearchResult:
    """Result of A* hierarchy search."""
    best_actions: list[Action]
    best_score: float
    initial_score: float
    improvement: float
    nodes_explored: int
    time_seconds: float
    judgment: Optional[JudgmentResult] = None


class HierarchySearch:
    """A* search over hierarchy configurations.

    Given a model, explores modifications (merge/split/reassign/rebalance)
    and uses an LLM judge to evaluate each configuration. Returns the
    best sequence of actions found.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        judge: Optional[LLMJudge] = None,
        max_nodes: int = 20,
        max_depth: int = 3,
        beam_width: int = 5,
        device: str = "cpu",
    ):
        self.model = model
        self.analyzer = ClusterAnalyzer(model, tokenizer, device)
        self.judge = judge or LLMJudge(backend="mock")
        self.modifier = HierarchyModifier(model)
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.device = device

    def search(self, step: int = 0) -> SearchResult:
        """Run A* search to find best hierarchy configuration.

        Returns the best sequence of actions and the improvement achieved.
        """
        start_time = time.monotonic()

        # Snapshot initial state
        initial_snapshot = self.analyzer.analyze(step=step)
        initial_judgment = self._evaluate(initial_snapshot)
        initial_score = initial_judgment.quality_score

        logger.info(
            "Hierarchy search starting at step %d, initial score: %.2f",
            step, initial_score,
        )

        # Save initial model state for rollback
        initial_state = self._save_state()

        # Priority queue: (f_score, node)
        start_node = SearchNode(
            f_score=self._f(0.0, initial_score),
            g_cost=0.0,
            h_heuristic=10.0 - initial_score,
            actions=[],
            judgment=initial_judgment,
        )

        open_set: list[SearchNode] = [start_node]
        best_node = start_node
        nodes_explored = 0

        while open_set and nodes_explored < self.max_nodes:
            current = heapq.heappop(open_set)
            nodes_explored += 1

            # Update best
            if current.judgment and current.judgment.quality_score > best_node.judgment.quality_score:
                best_node = current

            # Don't expand beyond max depth
            if len(current.actions) >= self.max_depth:
                continue

            # Generate candidate actions
            candidates = self._generate_candidates(current)

            for action in candidates[:self.beam_width]:
                # Restore to initial state, then apply all actions up to this point
                self._restore_state(initial_state)
                for a in current.actions:
                    self.modifier.apply(a)
                # Apply new action
                success = self.modifier.apply(action)
                if not success:
                    continue

                # Evaluate new state
                new_snapshot = self.analyzer.analyze(step=step)
                new_judgment = self._evaluate(new_snapshot)
                new_g = current.g_cost + action.cost
                new_h = 10.0 - new_judgment.quality_score

                child = SearchNode(
                    f_score=self._f(new_g, new_judgment.quality_score),
                    g_cost=new_g,
                    h_heuristic=new_h,
                    actions=current.actions + [action],
                    judgment=new_judgment,
                )

                heapq.heappush(open_set, child)

        # Restore initial state, then apply best sequence
        self._restore_state(initial_state)
        if best_node.actions:
            for a in best_node.actions:
                self.modifier.apply(a)
            logger.info(
                "Hierarchy search: applying %d actions, score %.2f → %.2f",
                len(best_node.actions), initial_score, best_node.judgment.quality_score,
            )
        else:
            logger.info("Hierarchy search: no improvement found")

        elapsed = time.monotonic() - start_time
        return SearchResult(
            best_actions=best_node.actions,
            best_score=best_node.judgment.quality_score if best_node.judgment else initial_score,
            initial_score=initial_score,
            improvement=best_node.judgment.quality_score - initial_score if best_node.judgment else 0.0,
            nodes_explored=nodes_explored,
            time_seconds=elapsed,
            judgment=best_node.judgment,
        )

    def _evaluate(self, snapshot: HierarchySnapshot) -> JudgmentResult:
        """Evaluate a snapshot using the LLM judge."""
        description = self.analyzer.format_for_llm(snapshot)
        n_fine = len(snapshot.fine_profiles)
        n_coarse = len(snapshot.coarse_groups)
        return self.judge.evaluate(description, n_fine, n_coarse)

    def _f(self, g_cost: float, quality_score: float) -> float:
        """Compute f = g + h. Lower is better.

        h = 10 - quality_score (want to maximize quality).
        """
        return g_cost + (10.0 - quality_score)

    def _generate_candidates(self, node: SearchNode) -> list[Action]:
        """Generate candidate actions based on current judgment."""
        candidates = []
        judgment = node.judgment

        if judgment is None:
            # Fallback: just try rebalance
            candidates.append(Action(ActionType.REBALANCE, {}, cost=2.0))
            return candidates

        # From LLM suggestions
        for a_id, b_id in judgment.suggested_merges[:3]:
            candidates.append(Action(
                ActionType.MERGE_FINE,
                {"cluster_a": a_id, "cluster_b": b_id},
                cost=1.0,
            ))

        for cluster_id in judgment.suggested_splits[:3]:
            candidates.append(Action(
                ActionType.SPLIT_FINE,
                {"cluster_id": cluster_id},
                cost=1.5,
            ))

        for r in judgment.reassign[:3]:
            candidates.append(Action(
                ActionType.REASSIGN,
                {"fine_id": r["fine_id"], "to_coarse": r["to_coarse"]},
                cost=0.5,
            ))

        # Always consider rebalance and reset
        candidates.append(Action(ActionType.REBALANCE, {}, cost=2.0))
        candidates.append(Action(ActionType.RESET_EMPTY, {"threshold": 0.01}, cost=1.0))

        return candidates

    def _save_state(self) -> dict[str, torch.Tensor]:
        """Save model centroid/router state for rollback."""
        state = {}
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["centroid", "router", "alpha", "beta"]):
                state[name] = param.data.clone()
        return state

    def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore model state from saved snapshot."""
        for name, param in self.model.named_parameters():
            if name in state:
                param.data.copy_(state[name])
