"""Masked token prediction loss (cross-entropy over masked positions only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss over masked positions only.

    L_diffusion = -(1/|masked|) Σ_{i ∈ masked} log p_θ(x₀[i] | xₜ)

    Args:
        logits: Model output logits, shape [B, S, V].
        targets: Ground truth token IDs, shape [B, S].
        mask: Boolean mask, True where tokens are masked, shape [B, S].

    Returns:
        Scalar loss value.
    """
    # Flatten
    B, S, V = logits.shape
    logits_flat = logits.view(-1, V)  # [B*S, V]
    targets_flat = targets.view(-1)  # [B*S]
    mask_flat = mask.view(-1)  # [B*S]

    # Compute per-token cross-entropy
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*S]

    # Only average over masked positions
    n_masked = mask_flat.sum().clamp(min=1)
    loss = (ce * mask_flat.float()).sum() / n_masked

    return loss
