"""Hierarchy actions — atomic operations on cluster configurations.

Each action modifies the model's centroids/routing in a specific way.
Actions are reversible and composable for A* search.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(Enum):
    MERGE_FINE = "merge_fine"      # Merge two fine clusters into one
    SPLIT_FINE = "split_fine"      # Split a fine cluster into two
    REASSIGN = "reassign"          # Move fine cluster to different coarse group
    REBALANCE = "rebalance"        # Rebalance coarse groups by size
    RESET_EMPTY = "reset_empty"    # Reinitialize empty clusters


@dataclass
class Action:
    """A single hierarchy modification."""
    action_type: ActionType
    params: dict[str, Any]
    cost: float = 1.0  # g-cost for A*

    def __repr__(self):
        return f"Action({self.action_type.value}, {self.params})"


class HierarchyModifier:
    """Applies actions to model centroids and routers."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.embedding = model.embedding

    @torch.no_grad()
    def apply(self, action: Action) -> bool:
        """Apply an action to the model. Returns True if successful."""
        handlers = {
            ActionType.MERGE_FINE: self._merge_fine,
            ActionType.SPLIT_FINE: self._split_fine,
            ActionType.REASSIGN: self._reassign,
            ActionType.REBALANCE: self._rebalance,
            ActionType.RESET_EMPTY: self._reset_empty,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return False
        return handler(action.params)

    @torch.no_grad()
    def _merge_fine(self, params: dict) -> bool:
        """Merge two fine clusters: average their centroids, redirect routing.

        After merge, cluster_b's centroid becomes a copy of the averaged centroid,
        and cluster_b's router weights are zeroed to redirect traffic.
        """
        a, b = params["cluster_a"], params["cluster_b"]
        centroids = self.embedding.fine_centroids.centroids  # [K, D]
        K = centroids.shape[0]
        if a >= K or b >= K:
            return False

        # Average centroids
        merged = (centroids[a] + centroids[b]) / 2.0
        centroids[a] = merged
        centroids[b] = merged  # Both point to same — effectively merged

        # Adjust fine router: boost a, suppress b
        router_weight = self.embedding.fine_router.linear.weight  # [K, D]
        router_weight[a] = (router_weight[a] + router_weight[b]) / 2.0
        router_weight[b] *= 0.01  # Near-zero so nothing routes here

        return True

    @torch.no_grad()
    def _split_fine(self, params: dict) -> bool:
        """Split a fine cluster by perturbing its centroid.

        Finds the least-used cluster and repurposes it as the split target.
        The original cluster keeps centroid + noise, the new one gets centroid - noise.
        """
        source = params["cluster_id"]
        centroids = self.embedding.fine_centroids.centroids  # [K, D]
        K = centroids.shape[0]
        if source >= K:
            return False

        # Find least-used cluster (smallest router weight norm)
        router_weight = self.embedding.fine_router.linear.weight  # [K, D]
        norms = router_weight.norm(dim=-1)
        norms[source] = float("inf")  # Don't pick the source
        target = norms.argmin().item()

        # Split with perturbation
        noise = torch.randn_like(centroids[source]) * 0.1 * centroids[source].norm()
        centroids[target] = centroids[source] - noise
        centroids[source] = centroids[source] + noise

        # Copy router weights with perturbation
        router_weight[target] = router_weight[source] + torch.randn_like(router_weight[source]) * 0.01

        return True

    @torch.no_grad()
    def _reassign(self, params: dict) -> bool:
        """Nudge the coarse router to assign a fine cluster to a different coarse group.

        Modifies the coarse router bias to prefer the target coarse group
        for embeddings similar to the specified fine cluster's centroid.
        """
        fine_id = params["fine_id"]
        to_coarse = params["to_coarse"]

        fine_centroids = self.embedding.fine_centroids.centroids
        coarse_router = getattr(self.embedding, "coarse_router", None)
        if coarse_router is None:
            return False

        K = fine_centroids.shape[0]
        M = coarse_router.linear.weight.shape[0]
        if fine_id >= K or to_coarse >= M:
            return False

        # Get the fine cluster's centroid
        fine_centroid = fine_centroids[fine_id]  # [D]

        # Adjust coarse router weights to favor to_coarse for this direction
        # Increase dot product between coarse_router[to_coarse] and fine_centroid
        coarse_w = coarse_router.linear.weight  # [M, D]
        direction = fine_centroid / (fine_centroid.norm() + 1e-8)

        # Boost target, suppress others slightly
        boost = 0.1 * direction.norm()
        coarse_w[to_coarse] += boost * direction

        return True

    @torch.no_grad()
    def _rebalance(self, params: dict) -> bool:
        """Rebalance coarse groups by running k-means on fine centroids.

        This is the nuclear option: completely reassign fine→coarse
        mapping based on geometric proximity.
        """
        fine_centroids = self.embedding.fine_centroids.centroids  # [K, D]
        coarse_centroids_mod = getattr(self.embedding, "coarse_centroids", None)
        coarse_router = getattr(self.embedding, "coarse_router", None)
        if coarse_centroids_mod is None or coarse_router is None:
            return False

        coarse_centroids = coarse_centroids_mod.centroids  # [M, D]
        K, D = fine_centroids.shape
        M = coarse_centroids.shape[0]

        # Run k-means (3 iterations) on fine centroids
        # Initialize from current coarse centroids
        centers = coarse_centroids.clone()

        for _ in range(3):
            # Assign fine clusters to nearest coarse center
            dists = torch.cdist(fine_centroids, centers)  # [K, M]
            assignments = dists.argmin(dim=-1)  # [K]

            # Update centers
            for m in range(M):
                mask = assignments == m
                if mask.sum() > 0:
                    centers[m] = fine_centroids[mask].mean(dim=0)

        # Write back
        coarse_centroids_mod.centroids.copy_(centers)

        # Update coarse router to match new assignments
        # Set router weights so that fine centroid → assigned coarse has high dot product
        new_router_w = torch.zeros(M, D, device=fine_centroids.device)
        for m in range(M):
            mask = assignments == m
            if mask.sum() > 0:
                new_router_w[m] = fine_centroids[mask].mean(dim=0)
        # Normalize
        new_router_w = new_router_w / (new_router_w.norm(dim=-1, keepdim=True) + 1e-8)
        coarse_router.linear.weight.copy_(new_router_w)

        return True

    @torch.no_grad()
    def _reset_empty(self, params: dict) -> bool:
        """Reinitialize clusters that have near-zero usage.

        Copies the centroid of the most-used cluster with added noise.
        """
        threshold = params.get("threshold", 0.01)
        centroids = self.embedding.fine_centroids.centroids  # [K, D]
        router_weight = self.embedding.fine_router.linear.weight  # [K, D]

        norms = router_weight.norm(dim=-1)
        max_idx = norms.argmax().item()
        max_centroid = centroids[max_idx].clone()

        reset_count = 0
        for k in range(centroids.shape[0]):
            if norms[k] < threshold * norms.max():
                noise = torch.randn_like(max_centroid) * 0.1 * max_centroid.norm()
                centroids[k] = max_centroid + noise
                router_weight[k] = router_weight[max_idx] + torch.randn_like(router_weight[max_idx]) * 0.01
                reset_count += 1

        return reset_count > 0
