"""Tests for single training step: forward + loss + backward."""

import torch
import pytest

from model.config import ModelConfig
from model.lm import HCLMD
from data.masking import DiffusionMasker
from losses.combined import CombinedLoss


class TestTrainingStep:
    @pytest.fixture
    def setup(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            fine_clusters=8, coarse_clusters=4,
        )
        model = HCLMD(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        masker = DiffusionMasker(mask_token_id=0)
        criterion = CombinedLoss()
        return model, optimizer, masker, criterion

    def test_single_step(self, setup):
        model, optimizer, masker, criterion = setup

        # Create dummy batch
        input_ids = torch.randint(1, 256, (4, 16))
        attention_mask = torch.ones_like(input_ids)

        # Mask
        masked_ids, mask = masker.mask_batch(input_ids, attention_mask)

        # Forward
        logits = model(masked_ids, attention_mask)

        # Loss
        loss_output = criterion(logits, input_ids, mask)

        # Backward
        optimizer.zero_grad()
        loss_output.total.backward()
        optimizer.step()

        # Verify gradients flowed
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients computed"

    def test_loss_decreases(self, setup):
        """Verify loss can decrease over a few steps (basic sanity)."""
        model, optimizer, masker, criterion = setup

        input_ids = torch.randint(1, 256, (4, 16))
        attention_mask = torch.ones_like(input_ids)

        losses = []
        for _ in range(10):
            masked_ids, mask = masker.mask_batch(input_ids, attention_mask)
            logits = model(masked_ids, attention_mask)
            loss_output = criterion(logits, input_ids, mask)

            optimizer.zero_grad()
            loss_output.total.backward()
            optimizer.step()
            losses.append(loss_output.total.item())

        # Loss should generally decrease (not strictly, but average of last 3 < first 3)
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, f"Loss not decreasing: {avg_first:.4f} -> {avg_last:.4f}"

    def test_flat_model_step(self):
        """Training step works for flat (baseline) model too."""
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            embedding_type="flat",
        )
        model = HCLMD(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        masker = DiffusionMasker(mask_token_id=0)
        criterion = CombinedLoss()

        input_ids = torch.randint(1, 256, (4, 16))
        attention_mask = torch.ones_like(input_ids)
        masked_ids, mask = masker.mask_batch(input_ids, attention_mask)

        logits = model(masked_ids, attention_mask)
        loss_output = criterion(logits, input_ids, mask)

        optimizer.zero_grad()
        loss_output.total.backward()
        optimizer.step()

        assert loss_output.total.item() > 0
