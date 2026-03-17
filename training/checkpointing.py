"""Save/load model + optimizer + training state."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    **extra,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model.
        optimizer: The optimizer.
        step: Current training step.
        path: Save path.
        **extra: Additional state to save.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    state.update(extra)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved: %s (step %d)", path, step)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Checkpoint path.
        model: Model to load state into.
        optimizer: Optional optimizer to restore.
        device: Device to map tensors to.

    Returns:
        Full checkpoint dict (including step, extra state).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    logger.info("Model loaded from %s", path)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Optimizer state restored")

    return checkpoint
