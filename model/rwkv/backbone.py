"""Bidirectional RWKV backbone: forward stack + backward stack + merge.

RWKV is naturally unidirectional. For masked diffusion we need bidirectional
context. Solution: two independent RWKV stacks (forward and backward) with
a learned merge projection.

  RWKV forward (→): processes left-to-right
  RWKV backward (←): processes right-to-left (input reversed)
  Merge: h[i] = W_merge · [state_fwd[i] ; state_bwd[i]]
         Learned projection from 2D → D

Optionally supports weight sharing between forward and backward stacks
to reduce parameter count (~47M → ~33M).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.rwkv.block import RWKVBlock


class RWKVBackbone(nn.Module):
    """Bidirectional RWKV backbone with forward + backward stacks and merge."""

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        share_weights: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.share_weights = share_weights

        # Forward stack
        self.forward_layers = nn.ModuleList([
            RWKVBlock(embed_dim, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Backward stack (shared or independent)
        if share_weights:
            self.backward_layers = self.forward_layers
        else:
            self.backward_layers = nn.ModuleList([
                RWKVBlock(embed_dim, n_heads, d_ff)
                for _ in range(n_layers)
            ])

        # Merge projection: [fwd ; bwd] → D
        self.merge = nn.Linear(embed_dim * 2, embed_dim, bias=True)

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [B, S, D]
            attention_mask: Padding mask [B, S] (1=real, 0=pad). Applied after
                            merge by zeroing padded positions.
        Returns:
            Merged bidirectional representations [B, S, D]
        """
        # Forward pass (left → right)
        h_fwd = x
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd)

        # Backward pass (right → left)
        h_bwd = x.flip(dims=[1])  # Reverse sequence
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd)
        h_bwd = h_bwd.flip(dims=[1])  # Flip back to original order

        # Merge: concatenate and project
        h_merged = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, S, 2D]
        h = self.merge(h_merged)  # [B, S, D]

        # Final norm
        h = self.final_norm(h)

        # Zero out padded positions
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)

        return h
