"""Centroid matrices for fine (K=64) and coarse (M=8) clusters."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClusterCentroids(nn.Module):
    """Learnable centroid (prototype) vectors for soft clustering.

    Each centroid is a d-dimensional vector. Token embeddings are mixed
    with centroids via soft routing weights.
    """

    def __init__(self, n_clusters: int, embed_dim: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim

        # Centroid matrix: [n_clusters, embed_dim]
        self.centroids = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.02)

    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of centroids using routing weights.

        e_cluster = Σᵢ pᵢ(x) · Cᵢ

        Args:
            routing_weights: Soft assignment probabilities, shape [..., n_clusters].

        Returns:
            Cluster embedding, shape [..., embed_dim].
        """
        # routing_weights: [..., K], centroids: [K, D]
        # Result: [..., D]
        return torch.matmul(routing_weights, self.centroids)

    def pairwise_cosine_similarity(self) -> torch.Tensor:
        """Compute pairwise cosine similarity between centroids.

        Returns:
            Similarity matrix, shape [n_clusters, n_clusters].
        """
        normed = self.centroids / (self.centroids.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(normed, normed.T)
