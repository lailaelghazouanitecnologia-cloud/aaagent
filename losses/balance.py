"""Cluster usage balance loss.

Penalizes uneven cluster usage across a batch to prevent cluster collapse.
"""

from __future__ import annotations

import torch


def balance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
    """Penalize uneven cluster usage across a batch.

    Computes the negative entropy of the average routing distribution.
    High entropy = even usage (good). Low entropy = collapse (bad).

    Args:
        routing_weights: Soft assignment probabilities, shape [B, S, K].

    Returns:
        Scalar loss (lower = more balanced).
    """
    # Average routing distribution across batch and sequence
    avg_probs = routing_weights.mean(dim=(0, 1))  # [K]

    # Uniform target
    K = avg_probs.size(0)
    uniform = torch.ones_like(avg_probs) / K

    # KL divergence from uniform (penalizes deviation from even usage)
    eps = 1e-10
    loss = torch.sum(avg_probs * (torch.log(avg_probs + eps) - torch.log(uniform + eps)))

    return loss
