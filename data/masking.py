"""Diffusion masking logic for LLaDA-style training.

Forward process: Given a clean sequence x₀, sample t ~ U[0, 1] and independently
replace each token with [MASK] with probability t.
"""

from __future__ import annotations

import torch


class DiffusionMasker:
    """Applies random masking for masked diffusion training.

    For each sequence, samples a masking ratio t ~ U[0, 1] and masks each
    token independently with probability t. This is the core of the LLaDA
    forward process.
    """

    def __init__(
        self,
        mask_token_id: int = 0,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ):
        self.mask_token_id = mask_token_id
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to a single sequence.

        Args:
            input_ids: Clean token IDs, shape [seq_len].
            attention_mask: 1 = real token, 0 = padding. Shape [seq_len].

        Returns:
            masked_ids: Token IDs with masking applied, shape [seq_len].
            mask: Boolean tensor, True where tokens are masked, shape [seq_len].
        """
        # Sample masking ratio t ~ U[min_ratio, max_ratio]
        t = torch.empty(1).uniform_(self.min_ratio, self.max_ratio).item()

        # Create mask: each position independently masked with probability t
        mask = torch.rand_like(input_ids.float()) < t

        # Don't mask padding positions
        if attention_mask is not None:
            mask = mask & (attention_mask.bool())

        # Apply masking
        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id

        return masked_ids, mask

    def mask_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to a batch, with per-example masking ratios.

        Args:
            input_ids: Shape [batch_size, seq_len].
            attention_mask: Shape [batch_size, seq_len].

        Returns:
            masked_ids: Shape [batch_size, seq_len].
            mask: Boolean tensor, shape [batch_size, seq_len].
        """
        batch_size, seq_len = input_ids.shape

        # Per-example masking ratios: t_i ~ U[min, max]
        t = torch.empty(batch_size, 1, device=input_ids.device).uniform_(
            self.min_ratio, self.max_ratio
        )

        # Per-position random values
        rand = torch.rand(batch_size, seq_len, device=input_ids.device)
        mask = rand < t

        # Don't mask padding
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id

        return masked_ids, mask
