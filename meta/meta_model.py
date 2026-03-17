"""Small MLP meta-model: gradient stats → learning rate multipliers."""

from __future__ import annotations

import torch
import torch.nn as nn


class MetaModel(nn.Module):
    """Small MLP that predicts per-group learning rate multipliers.

    Input: gradient statistics per parameter group (norm, mean, std, max, param_norm).
    Output: scalar multiplier α_k for each group.
    """

    def __init__(self, n_features: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive multiplier
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict learning rate multiplier from gradient statistics.

        Args:
            features: Gradient statistics, shape [n_features].

        Returns:
            Scalar multiplier (positive), shape [1].
        """
        return self.net(features)


class MetaOptimizer:
    """Meta-optimizer that modulates per-group gradient magnitude.

    Applied every N steps. Trained via simple online regression on loss deltas.
    """

    def __init__(
        self,
        param_groups: dict[str, list[torch.nn.Parameter]],
        meta_lr: float = 1e-4,
        hidden_dim: int = 128,
        update_every: int = 100,
    ):
        self.param_groups = param_groups
        self.update_every = update_every
        self.meta_model = MetaModel(n_features=5, hidden_dim=hidden_dim)
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        self._prev_loss: float | None = None
        self._step = 0

    def should_update(self, step: int) -> bool:
        """Check if meta-model should be updated at this step."""
        return step > 0 and step % self.update_every == 0

    def get_multipliers(self, grad_stats) -> dict[str, float]:
        """Compute LR multipliers for each parameter group.

        Args:
            grad_stats: GradientStats instance with current statistics.

        Returns:
            Dict mapping group name to multiplier.
        """
        multipliers = {}
        for name in self.param_groups:
            features = grad_stats.get_features(name)
            with torch.no_grad():
                mult = self.meta_model(features).item()
            multipliers[name] = mult
        return multipliers

    def update(self, current_loss: float, grad_stats) -> None:
        """Update meta-model based on loss delta.

        Args:
            current_loss: Current training loss.
            grad_stats: Current gradient statistics.
        """
        if self._prev_loss is None:
            self._prev_loss = current_loss
            return

        # Target: loss should decrease. Delta > 0 means loss increased (bad).
        delta = current_loss - self._prev_loss

        # Simple regression: push multipliers to reduce loss delta
        self.meta_optimizer.zero_grad()

        total_pred = torch.tensor(0.0, requires_grad=True)
        for name in self.param_groups:
            features = grad_stats.get_features(name)
            mult = self.meta_model(features)
            total_pred = total_pred + mult

        # Loss: encourage multipliers that correlate with loss decrease
        meta_loss = delta * total_pred
        meta_loss.backward()
        self.meta_optimizer.step()

        self._prev_loss = current_loss
