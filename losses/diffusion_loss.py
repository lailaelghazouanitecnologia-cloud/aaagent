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
    # Only compute cross-entropy on masked positions to save memory
    B, S, V = logits.shape
    mask_flat = mask.view(-1)  # [B*S]
    masked_idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [N_masked]

    if masked_idx.numel() == 0:
        return logits.sum() * 0.0  # no masked tokens, return zero grad-able loss

    logits_masked = logits.view(-1, V)[masked_idx]  # [N_masked, V]
    targets_masked = targets.view(-1)[masked_idx]  # [N_masked]

    loss = F.cross_entropy(logits_masked, targets_masked)

    return loss
