"""Pre-norm Transformer block with RMSNorm."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.transformer.attention import MultiHeadSelfAttention
from model.transformer.ffn import SwiGLUFFN, StandardFFN


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        norm: str = "rmsnorm",
        causal: bool = False,
    ):
        super().__init__()

        # Normalization
        if norm == "rmsnorm":
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        # Attention
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout, causal)

        # FFN
        if activation == "swiglu":
            self.ffn = SwiGLUFFN(embed_dim, d_ff, dropout)
        else:
            self.ffn = StandardFFN(embed_dim, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input, shape [batch_size, seq_len, embed_dim].
            attention_mask: Padding mask, shape [batch_size, seq_len].

        Returns:
            Output, shape [batch_size, seq_len, embed_dim].
        """
        # Pre-norm attention + residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x
