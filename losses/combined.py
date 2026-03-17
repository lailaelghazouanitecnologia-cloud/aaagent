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
    ) -> LossOutput:
        """Compute combined loss.

        Args:
            logits: Model output, shape [B, S, V].
            targets: Ground truth tokens, shape [B, S].
            mask: Masking indicator, shape [B, S].
            routing_weights: Fine router outputs, shape [B, S, K].
            fine_centroids: Fine centroid matrix, shape [K, D].
            coarse_centroids: Coarse centroid matrix, shape [M, D].
            fine_to_coarse_weights: Coarse router weights over fine clusters.

        Returns:
            LossOutput with all components.
        """
        device = logits.device

        # Core diffusion loss (always computed)
        l_diff = diffusion_loss(logits, targets, mask)

        # Auxiliary losses (only if structural components are active)
        l_bal = torch.tensor(0.0, device=device)
        l_div = torch.tensor(0.0, device=device)
        l_hier = torch.tensor(0.0, device=device)

        if routing_weights is not None and self.lambda_balance > 0:
            l_bal = balance_loss(routing_weights)

        if fine_centroids is not None and self.lambda_diversity > 0:
            l_div = diversity_loss(fine_centroids)
            if coarse_centroids is not None:
                l_div = l_div + diversity_loss(coarse_centroids)

        if (
            fine_centroids is not None
            and coarse_centroids is not None
            and fine_to_coarse_weights is not None
            and self.lambda_hierarchy > 0
        ):
            l_hier = hierarchy_loss(fine_centroids, coarse_centroids, fine_to_coarse_weights)

        # Total
        total = (
            l_diff
            + self.lambda_balance * l_bal
            + self.lambda_diversity * l_div
            + self.lambda_hierarchy * l_hier
        )

        return LossOutput(
            total=total,
            diffusion=l_diff,
            balance=l_bal,
            diversity=l_div,
            hierarchy=l_hier,
        )
