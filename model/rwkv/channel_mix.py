"""RWKV-7 ChannelMix: FFN with token-shift and receptance gate.

ChannelMix(x):
  x_shifted = lerp(x[i], x[i-1], mu)
  k = W_k @ x_shifted
  v = SiLU(W_v @ x_shifted) * (W_gate @ x_shifted)  # SwiGLU
  r = sigmoid(W_r @ x_shifted)                        # receptance
  out = r * (W_o @ v)

This is essentially SwiGLU FFN with token-shift and an R-gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMix(nn.Module):
    """RWKV-7 ChannelMix: SwiGLU FFN with token-shift and R-gate."""

    def __init__(self, embed_dim: int, d_ff: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Token-shift mix factor
        self.mu = nn.Parameter(torch.zeros(embed_dim))

        # Receptance gate
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)

        # SwiGLU: two parallel projections up, then gate, then project down
        self.W_v = nn.Linear(embed_dim, d_ff, bias=False)
        self.W_gate = nn.Linear(embed_dim, d_ff, bias=False)
        self.W_o = nn.Linear(d_ff, embed_dim, bias=False)

    def _token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift and mix with previous token."""
        mix = torch.sigmoid(self.mu)
        x_shifted = torch.zeros_like(x)
        x_shifted[:, 0, :] = x[:, 0, :]
        x_shifted[:, 1:, :] = mix * x[:, :-1, :] + (1 - mix) * x[:, 1:, :]
        return x_shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            out: [B, S, D]
        """
        xs = self._token_shift(x)

        # Receptance gate
        r = torch.sigmoid(self.W_r(xs))

        # SwiGLU
        v = F.silu(self.W_v(xs)) * self.W_gate(xs)

        # Output with R-gate
        return r * self.W_o(v)
