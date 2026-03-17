"""Tests for all loss terms."""

import torch
import pytest

from losses.diffusion_loss import diffusion_loss
from losses.balance import balance_loss
from losses.diversity import diversity_loss
from losses.hierarchy import hierarchy_loss
from losses.combined import CombinedLoss


class TestDiffusionLoss:
    def test_basic(self):
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)
        loss = diffusion_loss(logits, targets, mask)
        assert loss.shape == ()
        assert loss > 0

    def test_no_masked_tokens(self):
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        mask = torch.zeros(2, 16, dtype=torch.bool)
        loss = diffusion_loss(logits, targets, mask)
        assert loss == 0.0

    def test_only_masked_contribute(self):
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        mask_full = torch.ones(2, 16, dtype=torch.bool)
        mask_partial = torch.zeros(2, 16, dtype=torch.bool)
        mask_partial[:, :8] = True
        # Losses should differ since different tokens contribute
        loss_full = diffusion_loss(logits, targets, mask_full)
        loss_partial = diffusion_loss(logits, targets, mask_partial)
        assert loss_full != loss_partial


class TestBalanceLoss:
    def test_uniform_is_zero(self):
        # Perfectly uniform routing → loss should be 0 (or near 0)
        K = 64
        weights = torch.ones(4, 32, K) / K
        loss = balance_loss(weights)
        assert loss < 1e-5

    def test_collapsed_is_high(self):
        # All tokens route to cluster 0 → high loss
        K = 64
        weights = torch.zeros(4, 32, K)
        weights[:, :, 0] = 1.0
        loss = balance_loss(weights)
        assert loss > 1.0


class TestDiversityLoss:
    def test_orthogonal_is_low(self):
        # Orthogonal centroids → low diversity loss
        centroids = torch.eye(8, 384)[:8]
        loss = diversity_loss(centroids)
        assert loss < 0.01

    def test_identical_is_high(self):
        # Identical centroids → high diversity loss
        centroids = torch.randn(1, 384).expand(8, -1).clone()
        loss = diversity_loss(centroids)
        assert loss > 0.9


class TestHierarchyLoss:
    def test_basic(self):
        fine = torch.randn(64, 384)
        coarse = torch.randn(8, 384)
        weights = torch.randn(64, 8).softmax(dim=-1)
        loss = hierarchy_loss(fine, coarse, weights)
        assert loss.shape == ()
        assert loss >= 0


class TestCombinedLoss:
    def test_all_components(self):
        combined = CombinedLoss()
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        output = combined(logits, targets, mask)
        assert output.total > 0
        assert output.diffusion > 0

    def test_flat_model(self):
        """Combined loss works without structural components."""
        combined = CombinedLoss()
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.bool)

        output = combined(logits, targets, mask)
        # Without structural components, only diffusion loss contributes
        assert output.balance == 0.0
        assert output.diversity == 0.0
        assert output.hierarchy == 0.0
