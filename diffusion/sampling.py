"""Reverse process: iterative unmasking for text generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffusionSampler:
    """Iterative unmasking sampler for masked diffusion models.

    Generates text by starting from a fully masked sequence and progressively
    unmasking tokens over S steps, selecting the most confident predictions first.
    """

    def __init__(
        self,
        mask_token_id: int = 0,
        sampling_steps: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence",
    ):
        self.mask_token_id = mask_token_id
        self.sampling_steps = sampling_steps
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.remasking_strategy = remasking_strategy

    @torch.no_grad()
    def sample(
        self,
        model,
        prompt_ids: torch.Tensor | None = None,
        seq_len: int = 512,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate a sequence via iterative unmasking.

        Args:
            model: The HCLM-D model.
            prompt_ids: Optional prompt token IDs, shape [1, prompt_len].
                       These tokens remain unmasked throughout.
            seq_len: Total sequence length to generate.
            device: Device to run on.

        Returns:
            Generated token IDs, shape [1, seq_len].
        """
        if device is None:
            device = next(model.parameters()).device

        # Initialize: fully masked sequence
        x = torch.full((1, seq_len), self.mask_token_id, dtype=torch.long, device=device)

        # Place prompt tokens (unmasked)
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = prompt_ids.size(1)
            x[0, :prompt_len] = prompt_ids[0]

        # Track which positions are "locked" (prompt or already confidently unmasked)
        is_prompt = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        if prompt_len > 0:
            is_prompt[0, :prompt_len] = True

        # Number of tokens to unmask per step
        n_to_unmask_total = seq_len - prompt_len
        tokens_per_step = max(1, n_to_unmask_total // self.sampling_steps)

        for step in range(self.sampling_steps):
            # Find currently masked positions
            is_masked = (x == self.mask_token_id) & ~is_prompt
            n_masked = is_masked.sum().item()

            if n_masked == 0:
                break

            # Get model predictions
            logits = model(x)  # [1, S, V]

            # Apply temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Top-k filtering
            if self.top_k > 0:
                top_k_vals, _ = logits.topk(self.top_k, dim=-1)
                logits[logits < top_k_vals[..., -1:]] = float("-inf")

            # Get probabilities and predictions for masked positions
            probs = F.softmax(logits, dim=-1)  # [1, S, V]
            confidence, predictions = probs.max(dim=-1)  # [1, S]

            # Only consider masked positions
            confidence[~is_masked] = -1.0

            # Determine how many to unmask this step
            n_unmask = min(tokens_per_step, n_masked)
            if step == self.sampling_steps - 1:
                n_unmask = n_masked  # Unmask everything on last step

            # Select most confident masked positions
            _, top_indices = confidence.view(-1).topk(n_unmask)

            # Unmask selected positions
            for idx in top_indices:
                pos = idx.item()
                if self.temperature > 0:
                    # Sample from distribution
                    token = torch.multinomial(probs[0, pos], 1).item()
                else:
                    token = predictions[0, pos].item()
                x[0, pos] = token

        return x

    def get_schedule(self, n_masked: int) -> list[int]:
        """Compute how many tokens to unmask at each step.

        Returns a list of length sampling_steps with the number of tokens
        to unmask at each step (decreasing schedule).
        """
        schedule = []
        remaining = n_masked
        for step in range(self.sampling_steps):
            if remaining <= 0:
                break
            # Linear schedule
            n = max(1, remaining // (self.sampling_steps - step))
            schedule.append(n)
            remaining -= n
        return schedule
