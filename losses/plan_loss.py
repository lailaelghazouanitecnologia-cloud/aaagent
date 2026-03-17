"""Plan loss: feedback from VM execution into training.

L_plan measures whether the model's generated plans actually
execute correctly. This creates a feedback loop:
  model generates plan → VM executes → result quality → loss signal

Components:
  L_plan_success: Binary CE on whether the plan succeeded
  L_plan_accuracy: MSE between predicted and actual result
  L_meta_transition: Quality of meta-state predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanLoss(nn.Module):
    """Combined plan loss: complexity prediction + step count prediction."""

    def __init__(self, n_complexity_classes: int = 3):
        super().__init__()
        self.n_complexity_classes = n_complexity_classes

    def forward(
        self,
        complexity_logits: torch.Tensor,
        complexity_labels: torch.Tensor,
        pred_steps: torch.Tensor,
        true_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute plan loss.

        Args:
            complexity_logits: [N, n_classes] logits for complexity prediction
            complexity_labels: [N] class labels
            pred_steps: [N] predicted step counts
            true_steps: [N] actual step counts

        Returns:
            Scalar loss tensor.
        """
        l_complexity = F.cross_entropy(complexity_logits, complexity_labels)
        l_steps = F.mse_loss(pred_steps, true_steps)
        return l_complexity + l_steps


def plan_execution_loss(
    plan_success_logits: torch.Tensor,
    plan_success_labels: torch.Tensor,
) -> torch.Tensor:
    """Binary CE on plan execution success.

    Args:
        plan_success_logits: [N] logits for plan success prediction
        plan_success_labels: [N] binary labels (1=success, 0=failure)

    Returns:
        Scalar loss tensor.
    """
    if plan_success_logits.numel() == 0:
        return torch.tensor(0.0, device=plan_success_logits.device)

    return F.binary_cross_entropy_with_logits(
        plan_success_logits, plan_success_labels.float()
    )


def plan_result_loss(
    predicted_result: torch.Tensor,
    actual_result: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted and actual plan results.

    Args:
        predicted_result: [N, dim] predicted result embeddings
        actual_result: [N, dim] actual result embeddings

    Returns:
        Scalar loss tensor.
    """
    if predicted_result.numel() == 0:
        return torch.tensor(0.0, device=predicted_result.device)

    return F.mse_loss(predicted_result, actual_result)


def meta_transition_loss(
    predicted_depth: torch.Tensor,
    actual_depth: torch.Tensor,
    predicted_steps: torch.Tensor,
    actual_steps: torch.Tensor,
    predicted_complexity: torch.Tensor,
    actual_complexity: torch.Tensor,
) -> torch.Tensor:
    """Loss for meta-state predictions.

    The meta-state should correctly predict:
    - depth: how many levels of templates needed
    - steps: how many VM steps needed
    - complexity: how hard the task is

    Args:
        predicted_*: [N] predictions from the model
        actual_*: [N] ground truth from execution

    Returns:
        Scalar loss tensor.
    """
    device = predicted_depth.device
    loss = torch.tensor(0.0, device=device)
    n = 0

    if predicted_depth.numel() > 0:
        loss = loss + F.mse_loss(predicted_depth.float(), actual_depth.float())
        n += 1

    if predicted_steps.numel() > 0:
        loss = loss + F.mse_loss(predicted_steps.float(), actual_steps.float())
        n += 1

    if predicted_complexity.numel() > 0:
        loss = loss + F.cross_entropy(predicted_complexity, actual_complexity)
        n += 1

    return loss / max(n, 1)
