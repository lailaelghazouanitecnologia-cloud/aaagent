"""Step schedules for diffusion generation."""

from __future__ import annotations

import math

import torch


def linear_schedule(n_steps: int, n_tokens: int) -> list[int]:
    """Linear unmasking schedule: equal tokens per step."""
    per_step = max(1, n_tokens // n_steps)
    schedule = []
    remaining = n_tokens
    for i in range(n_steps):
        n = min(per_step, remaining)
        if i == n_steps - 1:
            n = remaining
        schedule.append(n)
        remaining -= n
        if remaining <= 0:
            break
    return schedule


def cosine_schedule(n_steps: int, n_tokens: int) -> list[int]:
    """Cosine unmasking schedule: starts slow, speeds up, then slows down."""
    # Cumulative fraction at each step using cosine
    fractions = []
    for i in range(n_steps):
        t = (i + 1) / n_steps
        frac = 0.5 * (1 - math.cos(math.pi * t))
        fractions.append(frac)

    schedule = []
    prev = 0
    for frac in fractions:
        target = int(frac * n_tokens)
        n = max(0, target - prev)
        schedule.append(n)
        prev = target

    # Ensure all tokens are accounted for
    total = sum(schedule)
    if total < n_tokens:
        schedule[-1] += n_tokens - total

    return schedule


def get_masking_ratios(n_steps: int) -> torch.Tensor:
    """Get masking ratios for each denoising step (1.0 → 0.0).

    Returns:
        Tensor of shape [n_steps] with masking ratios from high to low.
    """
    return torch.linspace(1.0, 0.0, n_steps + 1)[:-1]
