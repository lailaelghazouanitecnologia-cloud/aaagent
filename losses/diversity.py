"""Centroid diversity loss.

Penalizes high cosine similarity between cluster centroids.
"""

from __future__ import annotations

import torch


def diversity_loss(centroids: torch.Tensor) -> torch.Tensor:
    """Penalize high cosine similarity between centroids.

    Encourages centroids to be diverse (spread out in embedding space).

    Args:
        centroids: Centroid matrix, shape [K, D].

    Returns:
        Scalar loss (lower = more diverse).
    """
    # Normalize centroids
    normed = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)

    # Pairwise cosine similarity
    sim = torch.matmul(normed, normed.T)  # [K, K]

    # Zero out diagonal (self-similarity)
    K = centroids.size(0)
    mask = ~torch.eye(K, device=centroids.device, dtype=torch.bool)
    off_diag = sim[mask]

    # Penalize high similarity (mean of squared off-diagonal similarities)
    loss = (off_diag ** 2).mean()

    return loss
