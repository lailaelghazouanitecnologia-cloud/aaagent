"""Save/load model + optimizer + training state.

Handles torch.compile transparently: compiled models save clean keys,
and checkpoints with ``_orig_mod.`` prefixes are stripped on load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    """Return the underlying module if *model* was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove ``_orig_mod.`` prefix that torch.compile adds to keys."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    cleaned = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    logger.info("Stripped '%s' prefix from %d state_dict keys", prefix, len(cleaned))
    return cleaned


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    **extra,
) -> None:
    """Save a training checkpoint.

    Automatically unwraps torch.compile so saved keys never contain
    the ``_orig_mod.`` prefix.

    Args:
        model: The model (compiled or not).
        optimizer: The optimizer.
        step: Current training step.
        path: Save path.
        **extra: Additional state to save.
    """
    raw = _unwrap_compiled(model)
    state = {
        "model": raw.state_dict(),
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

    Handles legacy checkpoints saved from compiled models by stripping
    the ``_orig_mod.`` key prefix when present.

    Args:
        path: Checkpoint path.
        model: Model to load state into (compiled or not).
        optimizer: Optional optimizer to restore.
        device: Device to map tensors to.

    Returns:
        Full checkpoint dict (including step, extra state).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    state_dict = checkpoint["model"]
    state_dict = _strip_compiled_prefix(state_dict)

    target = _unwrap_compiled(model)
    target.load_state_dict(state_dict)
    logger.info("Model loaded from %s", path)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Optimizer state restored")

    return checkpoint
