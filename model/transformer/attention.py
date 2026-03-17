"""Multi-head self-attention (bidirectional for diffusion, optional causal for AR baseline)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    For masked diffusion (default): bidirectional — every token attends to all others.
    For autoregressive baseline: causal — tokens only attend to earlier positions.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.causal = causal

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, embed_dim].
            attention_mask: Padding mask, shape [batch_size, seq_len].
                           1 = attend, 0 = ignore.

        Returns:
            Output tensor, shape [batch_size, seq_len, embed_dim].
        """
        B, S, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, S, S]

        # Causal mask (for autoregressive baseline)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, 1, S]
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, H, S, D_h]
        out = out.transpose(1, 2).reshape(B, S, D)

        return self.out_proj(out)
