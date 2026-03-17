"""Output projection head (optionally weight-tied with E_local)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Language model head: projects Transformer output to vocabulary logits.

    Supports weight tying with the input embedding matrix (E_local).
    """

    def __init__(self, embed_dim: int, vocab_size: int, tie_weights: bool = True):
        super().__init__()
        self.tie_weights = tie_weights

        if not tie_weights:
            self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        else:
            # Weight will be set externally via set_weight
            self.proj = None

        self._tied_weight: torch.Tensor | None = None

    def set_weight(self, weight: torch.Tensor) -> None:
        """Set the tied weight matrix from the embedding layer."""
        self._tied_weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to vocabulary logits.

        Args:
            x: Hidden states, shape [batch_size, seq_len, embed_dim].

        Returns:
            Logits, shape [batch_size, seq_len, vocab_size].
        """
        if self.tie_weights:
            assert self._tied_weight is not None, "Tied weight not set. Call set_weight()."
            return torch.matmul(x, self._tied_weight.T)
        else:
            return self.proj(x)
