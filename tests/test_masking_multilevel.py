"""Tests for Za v6 multi-level masking."""

import torch
import pytest

from diffusion.masking_multilevel import MultiLevelMasker, MaskLevel


B, T = 4, 64


class TestMultiLevelMasker:
    def test_default_init(self):
        masker = MultiLevelMasker()
        assert masker.p_token > 0
        assert masker.p_span > 0

    def test_apply_masks(self):
        masker = MultiLevelMasker(p_token=0.4, p_span=0.3, p_slot=0.2, p_block=0.1)
        tokens = torch.randint(0, 1000, (B, T))
        mask_ratio = 0.5

        masked, mask, levels = masker.apply(tokens, mask_ratio)
        assert masked.shape == (B, T)
        assert mask.shape == (B, T)
        assert levels.shape == (B, T)

        # Some positions should be masked
        assert mask.any()
        # Mask ratio should be approximately correct (within tolerance)
        actual_ratio = mask.float().mean().item()
        assert 0.1 < actual_ratio < 0.9  # Loose bound

    def test_token_only_masking(self):
        masker = MultiLevelMasker(p_token=1.0, p_span=0.0, p_slot=0.0, p_block=0.0)
        tokens = torch.randint(0, 1000, (B, T))
        masked, mask, levels = masker.apply(tokens, 0.3)

        # All masks should be token-level
        masked_levels = levels[mask]
        if masked_levels.numel() > 0:
            assert (masked_levels == MaskLevel.TOKEN.value).all()

    def test_span_masking(self):
        masker = MultiLevelMasker(
            p_token=0.0, p_span=1.0, p_slot=0.0, p_block=0.0,
            mean_span_len=4, max_span_len=8,
        )
        tokens = torch.randint(0, 1000, (B, T))
        masked, mask, levels = masker.apply(tokens, 0.3)

        # Should have some masked positions
        assert mask.any()

    def test_zero_ratio(self):
        masker = MultiLevelMasker()
        tokens = torch.randint(0, 1000, (B, T))
        masked, mask, levels = masker.apply(tokens, 0.0)
        assert not mask.any()

    def test_full_ratio(self):
        masker = MultiLevelMasker()
        tokens = torch.randint(0, 1000, (B, T))
        masked, mask, levels = masker.apply(tokens, 1.0)
        assert mask.all()

    def test_with_block_boundaries(self):
        masker = MultiLevelMasker(p_token=0.0, p_span=0.0, p_slot=0.0, p_block=1.0)
        tokens = torch.randint(0, 1000, (B, T))
        # Block boundaries at every 8 tokens
        boundaries = torch.zeros(B, T, dtype=torch.bool)
        boundaries[:, ::8] = True

        masked, mask, levels = masker.apply(tokens, 0.3, block_boundaries=boundaries)
        assert mask.any()
