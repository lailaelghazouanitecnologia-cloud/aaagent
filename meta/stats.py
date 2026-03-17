"""Per-group gradient statistics for meta-optimizer."""

from __future__ import annotations

import torch


class GradientStats:
    """Tracks gradient statistics per parameter group for the meta-optimizer."""

    def __init__(self):
        self._stats: dict[str, dict[str, float]] = {}

    def update(self, name: str, param: torch.Tensor) -> None:
        """Update statistics for a parameter group.

        Args:
            name: Parameter group name.
            param: Parameter tensor (must have .grad set).
        """
        if param.grad is None:
            return

        grad = param.grad.detach()
        self._stats[name] = {
            "grad_norm": grad.norm().item(),
            "grad_mean": grad.mean().item(),
            "grad_std": grad.std().item(),
            "grad_max": grad.abs().max().item(),
            "param_norm": param.detach().norm().item(),
        }

    def get_features(self, name: str) -> torch.Tensor:
        """Get feature vector for a parameter group.

        Returns:
            Feature tensor of shape [5] containing gradient statistics.
        """
        if name not in self._stats:
            return torch.zeros(5)

        s = self._stats[name]
        return torch.tensor([
            s["grad_norm"],
            s["grad_mean"],
            s["grad_std"],
            s["grad_max"],
            s["param_norm"],
        ])

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._stats.clear()
