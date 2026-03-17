"""Stack of N Transformer blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.transformer.block import TransformerBlock, RMSNorm


class TransformerBackbone(nn.Module):
    """Stack of N pre-norm Transformer blocks with final normalization."""

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        norm: str = "rmsnorm",
        causal: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm=norm,
                causal=causal,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        if norm == "rmsnorm":
            self.final_norm = RMSNorm(embed_dim)
        else:
            self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings, shape [batch_size, seq_len, embed_dim].
            attention_mask: Padding mask, shape [batch_size, seq_len].

        Returns:
            Output representations, shape [batch_size, seq_len, embed_dim].
        """
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)
