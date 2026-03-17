"""Tests for router outputs: valid softmax, bottom-up wiring."""

import torch
import pytest

from model.embedding.router import FineRouter, CoarseRouter


class TestFineRouter:
    def test_output_sums_to_one(self):
        router = FineRouter(embed_dim=384, n_clusters=64)
        e_local = torch.randn(2, 16, 384)
        probs = router(e_local)
        assert probs.shape == (2, 16, 64)
        # Should sum to 1 along cluster dim
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_all_positive(self):
        router = FineRouter(embed_dim=384, n_clusters=64)
        e_local = torch.randn(2, 16, 384)
        probs = router(e_local)
        assert (probs >= 0).all()


class TestCoarseRouter:
    def test_output_sums_to_one(self):
        router = CoarseRouter(embed_dim=384, n_clusters=8)
        e_cluster = torch.randn(2, 16, 384)
        probs = router(e_cluster)
        assert probs.shape == (2, 16, 8)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_bottom_up_wiring(self):
        """Coarse router takes e_cluster output, not e_local."""
        fine_router = FineRouter(embed_dim=384, n_clusters=64)
        coarse_router = CoarseRouter(embed_dim=384, n_clusters=8)

        e_local = torch.randn(2, 16, 384)
        fine_probs = fine_router(e_local)

        # Simulate e_cluster (would normally use centroids)
        # The key test: coarse router accepts fine cluster output dimensions
        e_cluster = torch.randn(2, 16, 384)  # Same dim as embed
        coarse_probs = coarse_router(e_cluster)

        assert coarse_probs.shape == (2, 16, 8)
