"""Hierarchy consistency loss.

Encourages coarse clusters to be genuine abstractions of fine clusters.
"""

from __future__ import annotations

import torch


def hierarchy_loss(
    fine_centroids: torch.Tensor,
    coarse_centroids: torch.Tensor,
    fine_to_coarse_weights: torch.Tensor,
) -> torch.Tensor:
    """Encourage coarse centroids to be weighted means of fine centroids.

    Each coarse centroid should be close to the weighted average of the fine
    centroids that route to it, ensuring the hierarchy is meaningful.

    Args:
        fine_centroids: Fine centroid matrix, shape [K, D].
        coarse_centroids: Coarse centroid matrix, shape [M, D].
        fine_to_coarse_weights: How much each fine centroid contributes to
                                each coarse centroid, shape [K, M] (softmax).

    Returns:
        Scalar loss.
    """
    # Compute expected coarse centroids from fine centroids
    # fine_to_coarse_weights: [K, M], fine_centroids: [K, D]
    # expected: [M, D] = weights^T @ fine_centroids
    weights_norm = fine_to_coarse_weights / (fine_to_coarse_weights.sum(dim=0, keepdim=True) + 1e-8)
    expected_coarse = torch.matmul(weights_norm.T, fine_centroids)  # [M, D]

    # MSE between actual and expected coarse centroids
    loss = torch.mean((coarse_centroids - expected_coarse) ** 2)

    return loss
