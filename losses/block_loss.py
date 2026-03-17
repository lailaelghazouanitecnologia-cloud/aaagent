"""Block loss: CE over masked block regions or block boundary prediction.

Two modes:
1. Block region CE: reconstruct entire masked blocks (higher-level than token loss)
2. Block boundary prediction: BCE on boundary detection logits

L_block = -(1/|masked_blocks|) Σ_{b ∈ masked_blocks}
          (1/|tokens_in_b|) Σ_{t ∈ b} log p_θ(x_t | context)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockLoss(nn.Module):
    """Block-level loss function.

    Supports two modes:
    - Boundary prediction: BCE between predicted and true block boundaries
    - Region CE: cross-entropy over tokens within masked blocks
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        boundary_logits: torch.Tensor,
        boundary_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute block boundary prediction loss.

        Args:
            boundary_logits: [batch, seq_len] raw logits for boundary prediction
            boundary_labels: [batch, seq_len] float, 1.0 = boundary, 0.0 = not

        Returns:
            Scalar loss tensor.
        """
        return F.binary_cross_entropy_with_logits(boundary_logits, boundary_labels)


# Keep function API for backward compatibility
def block_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_mask: torch.Tensor,
    block_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Legacy function API for block region CE. See BlockLoss class."""
    if block_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    batch_size = logits.size(0)
    n_blocks = block_boundaries.size(0)

    total_loss = torch.tensor(0.0, device=logits.device)
    n_masked_blocks = 0

    for b in range(batch_size):
        for i in range(n_blocks):
            if not block_mask[b, i]:
                continue

            start, end = block_boundaries[i]
            start, end = start.item(), end.item()
            if start >= end or start >= logits.size(1):
                continue

            end = min(end, logits.size(1))

            block_logits = logits[b, start:end]
            block_targets = targets[b, start:end]

            block_ce = F.cross_entropy(block_logits, block_targets)
            total_loss = total_loss + block_ce
            n_masked_blocks += 1

    return total_loss / max(n_masked_blocks, 1)
