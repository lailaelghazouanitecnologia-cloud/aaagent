"""Optimizer and learning rate scheduler configuration."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: nn.Module, config: dict) -> AdamW:
    """Build AdamW optimizer with weight decay.

    Args:
        model: The model to optimize.
        config: Training configuration dict.

    Returns:
        Configured AdamW optimizer.
    """
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 0.1)
    betas = tuple(config.get("betas", [0.9, 0.95]))
    eps = config.get("eps", 1e-8)

    # Separate parameters: apply weight decay only to 2D+ params (not biases, norms)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> LambdaLR | None:
    """Build learning rate scheduler.

    Args:
        optimizer: The optimizer.
        config: Training configuration dict.

    Returns:
        LR scheduler or None.
    """
    schedule = config.get("lr_schedule", "cosine")
    warmup_steps = config.get("warmup_steps", 1000)
    total_steps = config.get("total_steps", 100000)

    if schedule == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(step, 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    return None
