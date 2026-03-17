"""Learned positional encoding, added after composite embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional encoding.

    Added AFTER the composite embedding, so the router never sees position.
    This keeps cluster assignment position-independent.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.encoding = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.encoding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positional encoding for a given sequence length.

        Args:
            seq_len: Length of the sequence.
            device: Device to place the tensor on.

        Returns:
            Positional encoding, shape [1, seq_len, embed_dim].
        """
        positions = torch.arange(seq_len, device=device)
        return self.encoding(positions).unsqueeze(0)
