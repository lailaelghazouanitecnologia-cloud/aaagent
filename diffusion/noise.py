"""Forward process: masking with t ~ U[0, 1]."""

from __future__ import annotations

import torch


class DiffusionForwardProcess:
    """Forward diffusion process: corrupt clean sequences by masking.

    Given clean tokens x₀, sample t ~ U[0, 1] and replace each token
    independently with [MASK] with probability t.
    """

    def __init__(self, mask_token_id: int = 0):
        self.mask_token_id = mask_token_id

    def __call__(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to clean sequences.

        Args:
            x0: Clean token IDs, shape [B, S].
            t: Masking ratios per example, shape [B] or [B, 1].
               If None, samples t ~ U[0, 1].
            attention_mask: 1 = real token, 0 = pad. Shape [B, S].

        Returns:
            xt: Masked token IDs, shape [B, S].
            mask: Boolean mask (True = masked), shape [B, S].
            t: Masking ratios used, shape [B].
        """
        B, S = x0.shape
        device = x0.device

        # Sample t if not provided
        if t is None:
            t = torch.rand(B, device=device)

        t = t.view(B, 1)  # [B, 1]

        # Per-position masking
        rand = torch.rand(B, S, device=device)
        mask = rand < t  # [B, S]

        # Don't mask padding
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        # Apply
        xt = x0.clone()
        xt[mask] = self.mask_token_id

        return xt, mask, t.squeeze(-1)
