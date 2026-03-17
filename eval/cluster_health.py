"""Cluster health metrics: usage entropy, diversity, dead clusters."""

from __future__ import annotations

import math

import torch


def cluster_usage_entropy(routing_weights: torch.Tensor) -> float:
    """Compute entropy of average cluster usage distribution.

    High entropy = even usage (healthy). Low entropy = collapse.

    Args:
        routing_weights: Shape [B, S, K].

    Returns:
        Entropy value. Max entropy = log(K).
    """
    avg_probs = routing_weights.mean(dim=(0, 1))  # [K]
    avg_probs = avg_probs.clamp(min=1e-10)
    entropy = -(avg_probs * avg_probs.log()).sum().item()
    return entropy


def max_entropy(n_clusters: int) -> float:
    """Maximum possible entropy for K clusters (uniform distribution)."""
    return math.log(n_clusters)


def dead_cluster_count(routing_weights: torch.Tensor, threshold: float = 0.001) -> int:
    """Count clusters that receive less than threshold average probability.

    Args:
        routing_weights: Shape [B, S, K].
        threshold: Minimum average probability to be considered alive.

    Returns:
        Number of dead clusters.
    """
    avg_probs = routing_weights.mean(dim=(0, 1))  # [K]
    return (avg_probs < threshold).sum().item()


def centroid_diversity(centroids: torch.Tensor) -> dict[str, float]:
    """Compute pairwise cosine similarity statistics between centroids.

    Args:
        centroids: Shape [K, D].

    Returns:
        Dict with mean, max, and min off-diagonal cosine similarity.
    """
    normed = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(normed, normed.T)

    K = centroids.size(0)
    mask = ~torch.eye(K, device=centroids.device, dtype=torch.bool)
    off_diag = sim[mask]

    return {
        "mean_cosine": off_diag.mean().item(),
        "max_cosine": off_diag.max().item(),
        "min_cosine": off_diag.min().item(),
    }


def full_cluster_health(
    routing_weights: torch.Tensor,
    centroids: torch.Tensor,
) -> dict[str, float]:
    """Compute all cluster health metrics.

    Args:
        routing_weights: Shape [B, S, K].
        centroids: Shape [K, D].

    Returns:
        Dict with all health metrics.
    """
    K = centroids.size(0)
    entropy = cluster_usage_entropy(routing_weights)
    max_ent = max_entropy(K)

    metrics = {
        "usage_entropy": entropy,
        "max_entropy": max_ent,
        "entropy_ratio": entropy / max_ent if max_ent > 0 else 0.0,
        "dead_clusters": dead_cluster_count(routing_weights),
    }
    metrics.update(centroid_diversity(centroids))
    return metrics
