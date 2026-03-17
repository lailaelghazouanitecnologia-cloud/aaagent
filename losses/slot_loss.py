"""Slot loss: CE over masked slots within templates.

L_slot = -(1/|masked_slots|) Σ_{s ∈ masked_slots} log p_θ(slot_s | context)

This loss trains the model to correctly fill template slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_positions: torch.Tensor,
) -> torch.Tensor:
    """Compute CE loss over masked template slots.

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len] ground truth token IDs
        slot_mask: [batch, n_slots] bool, True = slot is masked
        slot_positions: [n_slots] token positions of slot centers

    Returns:
        Scalar loss tensor.
    """
    if slot_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    batch_size = logits.size(0)
    total_loss = torch.tensor(0.0, device=logits.device)
    n_masked = 0

    for b in range(batch_size):
        for s in range(slot_mask.size(1)):
            if slot_mask[b, s]:
                pos = slot_positions[s].item()
                if 0 <= pos < logits.size(1):
                    loss_s = F.cross_entropy(
                        logits[b, pos].unsqueeze(0),
                        targets[b, pos].unsqueeze(0),
                    )
                    total_loss = total_loss + loss_s
                    n_masked += 1

    return total_loss / max(n_masked, 1)
