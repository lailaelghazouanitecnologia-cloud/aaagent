"""Single RWKV-7 block: LayerNorm → TimeMix → Residual → LayerNorm → ChannelMix → Residual.

Following RWKV-7 convention: PreLN with LayerNorm (not RMSNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.rwkv.time_mix import TimeMix
from model.rwkv.channel_mix import ChannelMix


class RWKVBlock(nn.Module):
    """One RWKV-7 block.

    h' = h + TimeMix(LayerNorm(h))
    h'' = h' + ChannelMix(LayerNorm(h'))
    """

    def __init__(self, embed_dim: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.time_mix = TimeMix(embed_dim, n_heads)
        self.channel_mix = ChannelMix(embed_dim, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            [B, S, D]
        """
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x
