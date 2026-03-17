"""Learned gate with floor for controlling structural influence.

g(x) = max(σ(G · e_local), g_min)

During structural warmup, a scalar ``gate_scale`` blends the gate output
from zero towards the fully learned value.  Because the scale is passed as
a *tensor* (not a Python float), ``torch.compile`` does **not** create a
new graph guard for every distinct warmup value — eliminating the
``cache_size_limit`` recompilation storm.
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
        gate_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute gate values.

        Args:
            e_local: Local embeddings, shape [..., embed_dim].
            gate_scale: Scalar tensor in [0, 1] that scales the learned gate.
                        ``None`` or ``1.0`` → full learned gate (normal training).
                        ``0.0`` → gate output is zero (Phase 0 warmup).
                        Intermediate values blend linearly (Phase 1 ramp).

        Returns:
            Gate values, shape [..., embed_dim]. Values in [0, 1.0].
        """
        raw = torch.sigmoid(self.linear(e_local))
        g = torch.clamp(raw, min=self.gate_min)

        if gate_scale is not None:
            g = g * gate_scale

        return g
