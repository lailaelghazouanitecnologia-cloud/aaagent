"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    FFN(x) = (Swish(xW₁) ⊙ xV) W₂

    SwiGLU uses a gated linear unit with Swish (SiLU) activation,
    which has been shown to improve Transformer quality.
    """

    def __init__(self, embed_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, d_ff, bias=False)
        self.v = nn.Linear(embed_dim, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [..., embed_dim].

        Returns:
            Output tensor, shape [..., embed_dim].
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.v(x)))


class StandardFFN(nn.Module):
    """Standard FFN with GELU activation (fallback)."""

    def __init__(self, embed_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))
