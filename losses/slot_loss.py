"""Slot loss: CE over masked slots within templates.

L_slot = -(1/|masked_slots|) Σ_{s ∈ masked_slots} log p_θ(slot_s | context)

This loss trains the model to correctly fill template slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotLoss(nn.Module):
    """Cross-entropy loss over masked template slots.

    Accepts full logits/targets and a slot_mask indicating which
    positions are template slots. Only computes loss on masked slots.
    """

    def __init__(self, vocab_size: int = 0):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CE loss over masked template slots.

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len] ground truth token IDs
            mask: [batch, seq_len] bool, diffusion mask
            slot_mask: [batch, seq_len] bool, True = position is a template slot

        Returns:
            Scalar loss tensor.
        """
        # Combine: only compute on positions that are both masked AND slot
        active = mask & slot_mask
        if active.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Flatten and select
        flat_logits = logits.reshape(-1, logits.size(-1))  # [B*S, V]
        flat_targets = targets.reshape(-1)                  # [B*S]
        flat_active = active.reshape(-1)                    # [B*S]

        selected_logits = flat_logits[flat_active]
        selected_targets = flat_targets[flat_active]

        return F.cross_entropy(selected_logits, selected_targets)


# Keep function API for backward compatibility
def slot_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    slot_mask: torch.Tensor,
    slot_positions: torch.Tensor,
) -> torch.Tensor:
    """Legacy function API. See SlotLoss class for preferred usage."""
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
