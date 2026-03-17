"""Tests for centroid initialization and diversity."""

import torch
import pytest

from model.embedding.clusters import ClusterCentroids


class TestClusterCentroids:
    def test_forward_shape(self):
        centroids = ClusterCentroids(n_clusters=64, embed_dim=384)
        weights = torch.randn(2, 16, 64).softmax(dim=-1)
        out = centroids(weights)
        assert out.shape == (2, 16, 384)

    def test_initialization(self):
        centroids = ClusterCentroids(n_clusters=64, embed_dim=384)
        # Centroids should be initialized with small values
        assert centroids.centroids.abs().max() < 1.0

    def test_pairwise_cosine(self):
        centroids = ClusterCentroids(n_clusters=64, embed_dim=384)
        sim = centroids.pairwise_cosine_similarity()
        assert sim.shape == (64, 64)
        # Diagonal should be ~1
        diag = sim.diag()
        assert torch.allclose(diag, torch.ones_like(diag), atol=0.01)

    def test_weighted_sum(self):
        """Verify output is a proper weighted sum of centroids."""
        centroids = ClusterCentroids(n_clusters=4, embed_dim=8)
        # One-hot weight: should select a single centroid
        weights = torch.zeros(1, 1, 4)
        weights[0, 0, 2] = 1.0
        out = centroids(weights)
        expected = centroids.centroids[2].unsqueeze(0).unsqueeze(0)
        assert torch.allclose(out, expected)
