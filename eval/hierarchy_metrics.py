"""Coarse-fine alignment scores for hierarchy evaluation."""

from __future__ import annotations

import torch


def coarse_fine_alignment(
    fine_centroids: torch.Tensor,
    coarse_centroids: torch.Tensor,
) -> dict[str, float]:
    """Measure how well coarse centroids align with groups of fine centroids.

    For each coarse centroid, find the nearest fine centroids and measure
    how much of the fine centroid space it "explains."

    Args:
        fine_centroids: Shape [K, D].
        coarse_centroids: Shape [M, D].

    Returns:
        Dict with alignment metrics.
    """
    K, D = fine_centroids.shape
    M = coarse_centroids.size(0)

    # Normalize
    fine_normed = fine_centroids / (fine_centroids.norm(dim=-1, keepdim=True) + 1e-8)
    coarse_normed = coarse_centroids / (coarse_centroids.norm(dim=-1, keepdim=True) + 1e-8)

    # Similarity: [K, M]
    sim = torch.matmul(fine_normed, coarse_normed.T)

    # Hard assignment: each fine centroid to nearest coarse centroid
    assignments = sim.argmax(dim=-1)  # [K]

    # Count fine centroids per coarse cluster
    counts = torch.zeros(M, device=fine_centroids.device)
    for m in range(M):
        counts[m] = (assignments == m).sum().float()

    # Balance: how evenly are fine centroids distributed?
    expected = K / M
    balance_score = 1.0 - (counts - expected).abs().mean().item() / expected

    # Coherence: mean similarity of fine centroids to their assigned coarse centroid
    coherence = sim[torch.arange(K), assignments].mean().item()

    return {
        "balance_score": balance_score,
        "coherence": coherence,
        "counts_per_coarse": counts.tolist(),
        "min_count": counts.min().item(),
        "max_count": counts.max().item(),
    }
