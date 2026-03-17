"""Block loss: CE over masked block regions.

L_block = -(1/|masked_blocks|) Σ_{b ∈ masked_blocks}
          (1/|tokens_in_b|) Σ_{t ∈ b} log p_θ(x_t | context)

This loss trains the model to reconstruct entire masked blocks.
Higher-level than token loss — forces structural understanding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def block_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_mask: torch.Tensor,
    block_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Compute CE loss over masked blocks.

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        block_mask: [batch, n_blocks] bool, True = block is masked
        block_boundaries: [n_blocks, 2] (start, end) positions

    Returns:
        Scalar loss tensor.
    """
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

            block_logits = logits[b, start:end]     # [block_len, vocab]
            block_targets = targets[b, start:end]   # [block_len]

            block_ce = F.cross_entropy(block_logits, block_targets)
            total_loss = total_loss + block_ce
            n_masked_blocks += 1

    return total_loss / max(n_masked_blocks, 1)
