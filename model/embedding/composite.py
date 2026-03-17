"""Composite embedding: local + gate * (α·cluster + β·hierarchy) + positional.

z₀(x) = e_local + g(x) ⊙ (α·e_cluster + β·e_hier) + e_pos
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.embedding.local import LocalEmbedding
from model.embedding.router import FineRouter, CoarseRouter
from model.embedding.clusters import ClusterCentroids
from model.embedding.gate import EmbeddingGate
from model.embedding.positional import LearnedPositionalEncoding


class FlatEmbedding(nn.Module):
    """Standard flat embedding (baseline): just E_local + positional."""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, mask_token_id: int = 0):
        super().__init__()
        self.local_embedding = LocalEmbedding(vocab_size, embed_dim, mask_token_id)
        self.positional = LearnedPositionalEncoding(max_seq_len, embed_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.local_embedding.weight

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape [batch_size, seq_len].

        Returns:
            Embeddings, shape [batch_size, seq_len, embed_dim].
        """
        e_local = self.local_embedding(x)
        e_pos = self.positional(x.size(1), device=x.device)
        return e_local + e_pos

    def get_routing_info(self) -> dict:
        """Return empty routing info for compatibility."""
        return {}


class CompositeEmbedding(nn.Module):
    """Full hierarchical clustered embedding.

    z₀(x) = e_local + g(x) ⊙ (α·e_cluster + β·e_hier) + e_pos

    Components:
        - E_local: Standard lookup
        - Fine router + centroids: Soft cluster assignment (K clusters)
        - Coarse router + centroids: Hierarchical abstraction (M clusters, bottom-up)
        - Gate: Controls structural influence with floor
        - Positional encoding: Added last, position-independent routing
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        fine_clusters: int = 64,
        coarse_clusters: int = 8,
        gate_min: float = 0.1,
        alpha_init: float = 0.1,
        beta_init: float = 0.05,
        use_gate: bool = True,
        use_hierarchy: bool = True,
        mask_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gate = use_gate
        self.use_hierarchy = use_hierarchy

        # Core embedding
        self.local_embedding = LocalEmbedding(vocab_size, embed_dim, mask_token_id)
        self.positional = LearnedPositionalEncoding(max_seq_len, embed_dim)

        # Fine clusters
        self.fine_router = FineRouter(embed_dim, fine_clusters)
        self.fine_centroids = ClusterCentroids(fine_clusters, embed_dim)

        # Coarse clusters (optional, bottom-up from fine)
        if use_hierarchy and coarse_clusters > 0:
            self.coarse_router = CoarseRouter(embed_dim, coarse_clusters)
            self.coarse_centroids = ClusterCentroids(coarse_clusters, embed_dim)
        else:
            self.coarse_router = None
            self.coarse_centroids = None

        # Gate (optional)
        if use_gate:
            self.gate = EmbeddingGate(embed_dim, gate_min)
        else:
            self.gate = None

        # Learnable mixing scalars
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # Cached routing info from last forward pass (for loss computation)
        self._cached_fine_weights: torch.Tensor | None = None
        self._cached_coarse_weights: torch.Tensor | None = None
        self._cached_gate_values: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.local_embedding.weight

    def forward(
        self,
        x: torch.Tensor,
        gate_scale: torch.Tensor | None = None,
        router_temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute composite embedding.

        Args:
            x: Token IDs, shape [batch_size, seq_len].
            gate_scale: Scalar tensor in [0, 1] that scales the learned gate.
                        Passed as a tensor to avoid torch.compile recompilation.
            router_temperature: Scalar tensor for softmax temperature.
                                Passed as a tensor to avoid torch.compile recompilation.

        Returns:
            Composite embeddings, shape [batch_size, seq_len, embed_dim].
        """
        # Local embedding
        e_local = self.local_embedding(x)

        # Fine cluster assignment
        fine_weights = self.fine_router(e_local, temperature=router_temperature)  # [..., K]
        e_cluster = self.fine_centroids(fine_weights)  # [..., D]
        self._cached_fine_weights = fine_weights.detach()

        # Coarse (hierarchical) cluster assignment — bottom-up from fine
        if self.use_hierarchy and self.coarse_router is not None:
            coarse_weights = self.coarse_router(e_cluster, temperature=router_temperature)  # [..., M]
            e_hier = self.coarse_centroids(coarse_weights)  # [..., D]
            self._cached_coarse_weights = coarse_weights.detach()
        else:
            e_hier = torch.zeros_like(e_local)
            self._cached_coarse_weights = None

        # Structural component
        structural = self.alpha * e_cluster + self.beta * e_hier

        # Gate
        if self.gate is not None:
            g = self.gate(e_local, gate_scale=gate_scale)
            self._cached_gate_values = g.detach()
            z = e_local + g * structural
        else:
            self._cached_gate_values = None
            z = e_local + structural

        # Positional encoding (added after, position-independent routing)
        e_pos = self.positional(x.size(1), device=x.device)
        z = z + e_pos

        return z

    def get_routing_info(self) -> dict:
        """Return routing information for monitoring and loss computation.

        Returns both model components and cached per-batch routing tensors
        from the last forward pass.
        """
        info = {
            "fine_centroids": self.fine_centroids,
            "coarse_centroids": self.coarse_centroids,
            "alpha": self.alpha,
            "beta": self.beta,
            # Cached tensors from last forward pass (detached)
            "fine_weights": self._cached_fine_weights,
            "coarse_weights": self._cached_coarse_weights,
            "gate_values": self._cached_gate_values,
        }
        if self.gate is not None:
            info["gate"] = self.gate
        return info
