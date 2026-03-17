"""Multi-head self-attention (bidirectional for diffusion, optional causal for AR baseline).

Uses Flash Attention (via PyTorch SDPA) when available, with automatic fallback
to manual matmul attention for compatibility.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Detect Flash Attention support once at import time
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
_FLASH_WARNED = False


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    For masked diffusion (default): bidirectional — every token attends to all others.
    For autoregressive baseline: causal — tokens only attend to earlier positions.

    Automatically uses Flash Attention (SDPA) when available for O(N) memory
    instead of O(N²). Falls back to manual matmul if SDPA is not supported.
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
        self.dropout_p = dropout

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

        if _SDPA_AVAILABLE:
            out = self._sdpa_attention(q, k, v, attention_mask, S)
        else:
            out = self._manual_attention(q, k, v, attention_mask, S)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor:
        """Flash Attention via PyTorch scaled_dot_product_attention."""
        global _FLASH_WARNED
        if not _FLASH_WARNED:
            logger.info("Using Flash Attention (SDPA)")
            _FLASH_WARNED = True

        attn_mask = None

        # Build combined mask if needed
        if self.causal and attention_mask is not None:
            # Causal + padding mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
            )
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            attn_mask = ~(causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask)
        elif self.causal:
            # Causal only — SDPA has native is_causal support
            return F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        elif attention_mask is not None:
            # Padding mask only: [B, S] → [B, 1, 1, S] boolean (True = attend)
            attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor:
        """Fallback: manual scaled dot-product attention."""
        global _FLASH_WARNED
        if not _FLASH_WARNED:
            logger.warning("Flash Attention not available, using manual matmul (slower)")
            _FLASH_WARNED = True

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)
