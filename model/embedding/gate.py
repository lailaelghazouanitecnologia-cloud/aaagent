"""Learned gate with floor for controlling structural influence.

g(x) = max(σ(G · e_local), g_min)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingGate(nn.Module):
    """Gate that controls how much structural (cluster + hierarchy) information
    is mixed into the base embedding.

    g(x) = max(σ(G · e_local), g_min)

    The gate floor (g_min) prevents the gate from permanently shutting off
    structural information.
    """

    def __init__(self, embed_dim: int, gate_min: float = 0.1):
        super().__init__()
        self.gate_min = gate_min
        self.linear = nn.Linear(embed_dim, embed_dim)

        # Initialize gate to output ~0.5 after sigmoid
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        e_local: torch.Tensor,
        override_value: float | None = None,
    ) -> torch.Tensor:
        """Compute gate values.

        Args:
            e_local: Local embeddings, shape [..., embed_dim].
            override_value: If set, returns this constant instead of learned gate.
                           Used during structural warmup (Phase 0-1).

        Returns:
            Gate values, shape [..., embed_dim]. Values in [g_min, 1.0].
        """
        if override_value is not None:
            return torch.full_like(e_local, override_value)

        raw = torch.sigmoid(self.linear(e_local))
        return torch.clamp(raw, min=self.gate_min)
