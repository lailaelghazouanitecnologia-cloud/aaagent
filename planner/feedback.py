"""Plan execution feedback — connects VM results to training loss.

After the VM executes a plan, the feedback module:
1. Checks if execution succeeded
2. Compares predicted vs actual complexity/steps/depth
3. Produces loss signals for L_plan and L_meta_transition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vm.executor import ExecutionResult
from planner.plan_parser import ExecutionPlan


@dataclass
class PlanFeedback:
    """Feedback from plan execution for training."""
    success: bool
    predicted_steps: int
    actual_steps: int
    predicted_complexity: str
    actual_complexity: str  # Inferred from execution
    predicted_depth: int
    actual_depth: int
    execution_time_ms: float
    error: Optional[str] = None


class FeedbackCollector:
    """Collects execution feedback for training."""

    def __init__(self):
        self._feedbacks: list[PlanFeedback] = []

    def collect(
        self,
        plan: ExecutionPlan,
        result: ExecutionResult,
    ) -> PlanFeedback:
        """Collect feedback from a plan execution."""
        # Infer actual complexity from execution stats
        if result.steps_executed <= 2:
            actual_complexity = "low"
        elif result.steps_executed <= 8:
            actual_complexity = "mid"
        else:
            actual_complexity = "high"

        feedback = PlanFeedback(
            success=result.success,
            predicted_steps=plan.expected_steps,
            actual_steps=result.steps_executed,
            predicted_complexity=plan.expected_complexity,
            actual_complexity=actual_complexity,
            predicted_depth=plan.expected_depth,
            actual_depth=0,  # Computed from block tree
            execution_time_ms=result.time_ms,
            error=result.error,
        )

        self._feedbacks.append(feedback)
        return feedback

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert collected feedbacks to tensors for loss computation."""
        if not self._feedbacks:
            return {}

        n = len(self._feedbacks)
        complexity_map = {"low": 0, "mid": 1, "high": 2}

        success_labels = torch.tensor(
            [1.0 if f.success else 0.0 for f in self._feedbacks],
            device=device,
        )
        predicted_steps = torch.tensor(
            [float(f.predicted_steps) for f in self._feedbacks],
            device=device,
        )
        actual_steps = torch.tensor(
            [float(f.actual_steps) for f in self._feedbacks],
            device=device,
        )
        predicted_depth = torch.tensor(
            [float(f.predicted_depth) for f in self._feedbacks],
            device=device,
        )
        actual_depth = torch.tensor(
            [float(f.actual_depth) for f in self._feedbacks],
            device=device,
        )
        actual_complexity = torch.tensor(
            [complexity_map.get(f.actual_complexity, 0) for f in self._feedbacks],
            dtype=torch.long,
            device=device,
        )

        return {
            "success_labels": success_labels,
            "predicted_steps": predicted_steps,
            "actual_steps": actual_steps,
            "predicted_depth": predicted_depth,
            "actual_depth": actual_depth,
            "actual_complexity": actual_complexity,
        }

    def reset(self) -> None:
        self._feedbacks.clear()

    @property
    def n_collected(self) -> int:
        return len(self._feedbacks)

    @property
    def success_rate(self) -> float:
        if not self._feedbacks:
            return 0.0
        return sum(1 for f in self._feedbacks if f.success) / len(self._feedbacks)
