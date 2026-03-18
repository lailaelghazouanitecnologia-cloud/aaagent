"""Combined loss for Za v7.

L_total = L_diffusion
        + λ₁·L_balance + λ₂·L_diversity + λ₃·L_hierarchy
        + λ₄·L_slot + λ₅·L_block

Phases control which losses are active:
  Phase 0-1: L_diffusion + structural (balance/diversity/hierarchy)
  Phase 2+:  + L_slot + L_block (via loss_multipliers)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from losses.diffusion_loss import diffusion_loss
from losses.balance import balance_loss
from losses.diversity import diversity_loss
from losses.hierarchy import hierarchy_loss


@dataclass
class LossOutput:
    """Container for all loss components."""

    total: torch.Tensor
    diffusion: torch.Tensor
    balance: torch.Tensor
    diversity: torch.Tensor
    hierarchy: torch.Tensor
    slot: torch.Tensor
    block: torch.Tensor


class CombinedLoss:
    """Combined loss function for Za v7.

    L_total = L_diffusion
            + λ_balance · L_balance
            + λ_diversity · L_diversity
            + λ_hierarchy · L_hierarchy
            + λ_slot · L_slot      (Phase 2+)
            + λ_block · L_block    (Phase 2+)
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_diversity: float = 0.001,
        lambda_hierarchy: float = 0.01,
        lambda_slot: float = 0.5,
        lambda_block: float = 0.3,
    ):
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_slot = lambda_slot
        self.lambda_block = lambda_block

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        routing_weights: torch.Tensor | None = None,
        fine_centroids: torch.Tensor | None = None,
        coarse_centroids: torch.Tensor | None = None,
        fine_to_coarse_weights: torch.Tensor | None = None,
        coarse_routing_weights: torch.Tensor | None = None,
        loss_multipliers: dict[str, float] | None = None,
        # Multi-level masking (Phase 2+)
        slot_mask: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        block_boundaries: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute combined loss.

        Args:
            logits: Model output [B, S, V].
            targets: Ground truth tokens [B, S].
            mask: Masking indicator [B, S].
            routing_weights: Fine router outputs [B, S, K].
            fine_centroids: Fine centroid matrix [K, D].
            coarse_centroids: Coarse centroid matrix [M, D].
            fine_to_coarse_weights: Coarse router weights for hierarchy loss.
            coarse_routing_weights: Coarse router outputs [B, S, M].
            loss_multipliers: Dynamic multipliers from warmup.
            slot_mask: Slot-level mask [B, n_slots] (True=masked).
            block_mask: Block-level mask [B, n_blocks] (True=masked).
            block_boundaries: Block boundaries [n_blocks, 2].

        Returns:
            LossOutput with all components.
        """
        device = logits.device
        mult = loss_multipliers or {}
        zero = torch.tensor(0.0, device=device)

        # Effective lambdas = base × dynamic multiplier
        eff_balance = self.lambda_balance * mult.get("balance", 1.0)
        eff_diversity = self.lambda_diversity * mult.get("diversity", 1.0)
        eff_hierarchy = self.lambda_hierarchy * mult.get("hierarchy", 1.0)
        eff_slot = self.lambda_slot * mult.get("slot", 0.0)
        eff_block = self.lambda_block * mult.get("block", 0.0)

        # Core diffusion loss (always computed)
        l_diff = diffusion_loss(logits, targets, mask)

        # Auxiliary losses (only if structural components are active)
        l_bal = zero
        l_div = zero
        l_hier = zero
        l_slot = zero
        l_block = zero

        if routing_weights is not None and eff_balance > 0:
            l_bal = balance_loss(routing_weights)
            if coarse_routing_weights is not None:
                l_bal = l_bal + balance_loss(coarse_routing_weights)

        if fine_centroids is not None and eff_diversity > 0:
            l_div = diversity_loss(fine_centroids)
            if coarse_centroids is not None:
                l_div = l_div + diversity_loss(coarse_centroids)

        if (
            fine_centroids is not None
            and coarse_centroids is not None
            and fine_to_coarse_weights is not None
            and eff_hierarchy > 0
        ):
            l_hier = hierarchy_loss(fine_centroids, coarse_centroids, fine_to_coarse_weights)

        # Slot loss: CE over positions that are both masked and inside slots
        if slot_mask is not None and eff_slot > 0 and slot_mask.any():
            try:
                from losses.slot_loss import SlotLoss
                _sl = SlotLoss()
                l_slot = _sl(logits, targets, mask, slot_mask)
            except Exception:
                pass

        # Block loss: CE over positions within masked blocks
        if block_mask is not None and block_boundaries is not None and eff_block > 0 and block_mask.any():
            try:
                from losses.block_loss import block_loss as _block_loss_fn
                l_block = _block_loss_fn(logits, targets, mask, block_boundaries)
            except Exception:
                pass

        # Total
        total = (
            l_diff
            + eff_balance * l_bal
            + eff_diversity * l_div
            + eff_hierarchy * l_hier
            + eff_slot * l_slot
            + eff_block * l_block
        )

        return LossOutput(
            total=total,
            diffusion=l_diff,
            balance=l_bal,
            diversity=l_div,
            hierarchy=l_hier,
            slot=l_slot,
            block=l_block,
        )
