"""Fine router and coarse router for cluster assignment.

Fine router: p(x) = softmax(R_fine · e_local / τ) — routes tokens to K fine clusters.
Coarse router: q(x) = softmax(R_coarse · e_cluster / τ) — routes to M coarse clusters (bottom-up).

Temperature τ can be annealed externally to sharpen routing over time.
Accepts τ as a scalar *tensor* to avoid ``torch.compile`` recompilation guards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineRouter(nn.Module):
    """Routes token embeddings to K fine-grained clusters.

    p(x) = softmax(R_fine · e_local / τ)
    """

    def __init__(self, embed_dim: int, n_clusters: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_clusters, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        e_local: torch.Tensor,
        temperature: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Compute soft assignment probabilities over fine clusters.

        Args:
            e_local: Local embeddings, shape [..., embed_dim].
            temperature: Softmax temperature scalar (tensor or float).
                         None defaults to 1.0 (no scaling).

        Returns:
            Probabilities, shape [..., n_clusters].
        """
        logits = self.linear(e_local)
        if temperature is not None:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)


class CoarseRouter(nn.Module):
    """Routes fine cluster embeddings to M coarse clusters (bottom-up).

    q(x) = softmax(R_coarse · e_cluster / τ)

    Operates on the output of the fine cluster mixing, NOT on e_local directly.
    This enforces that coarse clusters are genuine abstractions of fine clusters.
    """

    def __init__(self, embed_dim: int, n_clusters: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_clusters, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self,
        e_cluster: torch.Tensor,
        temperature: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Compute soft assignment probabilities over coarse clusters.

        Args:
            e_cluster: Fine cluster output, shape [..., embed_dim].
            temperature: Softmax temperature scalar (tensor or float).
                         None defaults to 1.0 (no scaling).

        Returns:
            Probabilities, shape [..., n_coarse_clusters].
        """
        logits = self.linear(e_cluster)
        if temperature is not None:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)
