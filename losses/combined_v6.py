"""Combined loss for Za v6.

L = L_token + λ₁·L_slot + λ₂·L_block
  + λ₃·L_balance + λ₄·L_diversity + λ₅·L_hierarchy
  + λ₆·L_plan + λ₇·L_compose
  + λ₈·L_template + λ₉·L_hash

Phases control which losses are active:
  Phase 0-1: L_token only (+ balance/diversity/hierarchy)
  Phase 2:   + L_slot, L_block
  Phase 3:   + L_plan
  Phase 4:   + L_compose, L_hash
  Phase 5:   + L_template, full meta-loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from losses.diffusion_loss import diffusion_loss
from losses.balance import balance_loss
from losses.diversity import diversity_loss
from losses.hierarchy import hierarchy_loss
from losses.slot_loss import slot_loss
from losses.block_loss import block_loss
from losses.template_loss import (
    template_selection_loss,
    hash_consistency_loss,
    compose_quality_loss,
)
from losses.plan_loss import plan_execution_loss, meta_transition_loss


@dataclass
class V6LossOutput:
    """Container for all v6 loss components."""
    total: torch.Tensor
    # Core diffusion losses
    diffusion: torch.Tensor       # L_token (standard LLaDA)
    slot: torch.Tensor            # L_slot (masked slots)
    block: torch.Tensor           # L_block (masked blocks)
    # Structural losses (from v4)
    balance: torch.Tensor
    diversity: torch.Tensor
    hierarchy: torch.Tensor
    # Plan/VM losses
    plan: torch.Tensor
    compose: torch.Tensor
    # Template/hash losses
    template: torch.Tensor
    hash_consistency: torch.Tensor


class CombinedLossV6:
    """Combined loss function for Za v6.

    Extends CombinedLoss with multi-level masking losses, plan losses,
    and template/hash losses. Losses are phased in during training.
    """

    def __init__(
        self,
        # v4 lambdas
        lambda_balance: float = 0.01,
        lambda_diversity: float = 0.001,
        lambda_hierarchy: float = 0.01,
        # v6 lambdas
        lambda_slot: float = 0.5,
        lambda_block: float = 0.3,
        lambda_plan: float = 0.1,
        lambda_compose: float = 0.05,
        lambda_template: float = 0.1,
        lambda_hash: float = 0.01,
    ):
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_slot = lambda_slot
        self.lambda_block = lambda_block
        self.lambda_plan = lambda_plan
        self.lambda_compose = lambda_compose
        self.lambda_template = lambda_template
        self.lambda_hash = lambda_hash

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        # Structural info (v4)
        routing_weights: Optional[torch.Tensor] = None,
        fine_centroids: Optional[torch.Tensor] = None,
        coarse_centroids: Optional[torch.Tensor] = None,
        fine_to_coarse_weights: Optional[torch.Tensor] = None,
        loss_multipliers: Optional[dict[str, float]] = None,
        # Multi-level masking info (v6)
        slot_mask: Optional[torch.Tensor] = None,
        slot_positions: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
        block_boundaries: Optional[torch.Tensor] = None,
        # Plan/VM info (v6)
        plan_success_logits: Optional[torch.Tensor] = None,
        plan_success_labels: Optional[torch.Tensor] = None,
        # Template/hash info (v6)
        template_logits: Optional[torch.Tensor] = None,
        template_targets: Optional[torch.Tensor] = None,
        block_embeddings: Optional[torch.Tensor] = None,
        hash_labels: Optional[torch.Tensor] = None,
        composed_embeddings: Optional[torch.Tensor] = None,
        compose_targets: Optional[torch.Tensor] = None,
        # Phase control
        active_phase: int = 0,
    ) -> V6LossOutput:
        """Compute combined v6 loss.

        Active losses depend on training phase:
          Phase 0-1: L_token + structural
          Phase 2:   + L_slot + L_block
          Phase 3:   + L_plan
          Phase 4-5: + L_compose + L_template + L_hash
        """
        device = logits.device
        mult = loss_multipliers or {}
        zero = torch.tensor(0.0, device=device)

        # Effective lambdas with dynamic multipliers
        eff_balance = self.lambda_balance * mult.get("balance", 1.0)
        eff_diversity = self.lambda_diversity * mult.get("diversity", 1.0)
        eff_hierarchy = self.lambda_hierarchy * mult.get("hierarchy", 1.0)

        # === Phase 0+: Core diffusion loss (always active) ===
        l_diff = diffusion_loss(logits, targets, mask)

        # === Phase 0+: Structural losses ===
        l_bal = zero
        l_div = zero
        l_hier = zero

        if routing_weights is not None and eff_balance > 0:
            l_bal = balance_loss(routing_weights)

        if fine_centroids is not None and eff_diversity > 0:
            l_div = diversity_loss(fine_centroids)
            if coarse_centroids is not None:
                l_div = l_div + diversity_loss(coarse_centroids)

        if (fine_centroids is not None and coarse_centroids is not None
                and fine_to_coarse_weights is not None and eff_hierarchy > 0):
            l_hier = hierarchy_loss(fine_centroids, coarse_centroids, fine_to_coarse_weights)

        # === Phase 2+: Multi-level masking losses ===
        l_slot = zero
        l_block = zero

        if active_phase >= 2:
            if slot_mask is not None and slot_positions is not None:
                l_slot = slot_loss(logits, targets, slot_mask, slot_positions)

            if block_mask is not None and block_boundaries is not None:
                l_block = block_loss(logits, targets, block_mask, block_boundaries)

        # === Phase 3+: Plan/VM losses ===
        l_plan = zero

        if active_phase >= 3:
            if plan_success_logits is not None and plan_success_labels is not None:
                l_plan = plan_execution_loss(plan_success_logits, plan_success_labels)

        # === Phase 4+: Compose, template, hash losses ===
        l_compose = zero
        l_template = zero
        l_hash = zero

        if active_phase >= 4:
            if composed_embeddings is not None and compose_targets is not None:
                l_compose = compose_quality_loss(composed_embeddings, compose_targets)

            if template_logits is not None and template_targets is not None:
                l_template = template_selection_loss(template_logits, template_targets)

            if block_embeddings is not None and hash_labels is not None:
                l_hash = hash_consistency_loss(block_embeddings, hash_labels)

        # === Total ===
        total = (
            l_diff
            + eff_balance * l_bal
            + eff_diversity * l_div
            + eff_hierarchy * l_hier
            + self.lambda_slot * l_slot
            + self.lambda_block * l_block
            + self.lambda_plan * l_plan
            + self.lambda_compose * l_compose
            + self.lambda_template * l_template
            + self.lambda_hash * l_hash
        )

        return V6LossOutput(
            total=total,
            diffusion=l_diff,
            slot=l_slot,
            block=l_block,
            balance=l_bal,
            diversity=l_div,
            hierarchy=l_hier,
            plan=l_plan,
            compose=l_compose,
            template=l_template,
            hash_consistency=l_hash,
        )
