"""Combined loss: L_total = L_diffusion + λ₁L_balance + λ₂L_diversity + λ₃L_hierarchy."""

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


class CombinedLoss:
    """Combined loss function for HCLM-D.

    L_total = L_diffusion + λ₁·L_balance + λ₂·L_diversity + λ₃·L_hierarchy
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_diversity: float = 0.001,
        lambda_hierarchy: float = 0.01,
    ):
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity
        self.lambda_hierarchy = lambda_hierarchy

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
    ) -> LossOutput:
        """Compute combined loss.

        Args:
            logits: Model output, shape [B, S, V].
            targets: Ground truth tokens, shape [B, S].
            mask: Masking indicator, shape [B, S].
            routing_weights: Fine router outputs, shape [B, S, K].
            fine_centroids: Fine centroid matrix, shape [K, D].
            coarse_centroids: Coarse centroid matrix, shape [M, D].
            fine_to_coarse_weights: Coarse router weights for hierarchy loss.
            coarse_routing_weights: Coarse router outputs [B, S, M] for balance.
            loss_multipliers: Dynamic multipliers from warmup curriculum.
                Keys: 'hierarchy', 'balance', 'diversity'. Values multiply
                the base lambdas. None = no scaling (multiplier 1.0).

        Returns:
            LossOutput with all components.
        """
        device = logits.device
        mult = loss_multipliers or {}

        # Effective lambdas = base × dynamic multiplier
        eff_balance = self.lambda_balance * mult.get("balance", 1.0)
        eff_diversity = self.lambda_diversity * mult.get("diversity", 1.0)
        eff_hierarchy = self.lambda_hierarchy * mult.get("hierarchy", 1.0)

        # Core diffusion loss (always computed)
        l_diff = diffusion_loss(logits, targets, mask)

        # Auxiliary losses (only if structural components are active)
        l_bal = torch.tensor(0.0, device=device)
        l_div = torch.tensor(0.0, device=device)
        l_hier = torch.tensor(0.0, device=device)

        if routing_weights is not None and eff_balance > 0:
            l_bal = balance_loss(routing_weights)
            # Also penalize uniform coarse routing — without this the coarse
            # router has no direct gradient encouraging specialization.
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

        # Total
        total = (
            l_diff
            + eff_balance * l_bal
            + eff_diversity * l_div
            + eff_hierarchy * l_hier
        )

        return LossOutput(
            total=total,
            diffusion=l_diff,
            balance=l_bal,
            diversity=l_div,
            hierarchy=l_hier,
        )
