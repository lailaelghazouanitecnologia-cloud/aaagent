"""E_local: Standard token embedding lookup with mask token support."""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalEmbedding(nn.Module):
    """Standard embedding lookup table.

    Maps token IDs to dense vectors. Supports a special [MASK] token for
    masked diffusion training.
    """

    def __init__(self, vocab_size: int, embed_dim: int, mask_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mask_token_id = mask_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Initialize with scaled normal
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Access embedding weight matrix (for weight tying)."""
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings for token IDs.

        Args:
            x: Token IDs, shape [batch_size, seq_len].

        Returns:
            Embeddings, shape [batch_size, seq_len, embed_dim].
        """
        return self.embedding(x)
