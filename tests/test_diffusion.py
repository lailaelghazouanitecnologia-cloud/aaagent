"""Tests for masking + unmasking roundtrip."""

import torch
import pytest

from data.masking import DiffusionMasker
from diffusion.noise import DiffusionForwardProcess


class TestDiffusionMasker:
    def test_mask_shape(self):
        masker = DiffusionMasker(mask_token_id=0)
        input_ids = torch.randint(1, 100, (16,))
        masked_ids, mask = masker.mask(input_ids)
        assert masked_ids.shape == input_ids.shape
        assert mask.shape == input_ids.shape
        assert mask.dtype == torch.bool

    def test_masked_tokens_have_mask_id(self):
        masker = DiffusionMasker(mask_token_id=0)
        input_ids = torch.randint(1, 100, (64,))
        masked_ids, mask = masker.mask(input_ids)
        assert (masked_ids[mask] == 0).all()

    def test_unmasked_tokens_preserved(self):
        masker = DiffusionMasker(mask_token_id=0)
        input_ids = torch.randint(1, 100, (64,))
        masked_ids, mask = masker.mask(input_ids)
        assert (masked_ids[~mask] == input_ids[~mask]).all()

    def test_batch_masking(self):
        masker = DiffusionMasker(mask_token_id=0)
        input_ids = torch.randint(1, 100, (4, 32))
        attention_mask = torch.ones_like(input_ids)
        masked_ids, mask = masker.mask_batch(input_ids, attention_mask)
        assert masked_ids.shape == (4, 32)
        assert mask.shape == (4, 32)

    def test_respects_padding(self):
        masker = DiffusionMasker(mask_token_id=0)
        input_ids = torch.randint(1, 100, (4, 32))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -8:] = 0  # Last 8 positions are padding
        _, mask = masker.mask_batch(input_ids, attention_mask)
        # Padding positions should never be masked
        assert not mask[:, -8:].any()


class TestForwardProcess:
    def test_forward_shape(self):
        process = DiffusionForwardProcess(mask_token_id=0)
        x0 = torch.randint(1, 100, (4, 32))
        xt, mask, t = process(x0)
        assert xt.shape == (4, 32)
        assert mask.shape == (4, 32)
        assert t.shape == (4,)

    def test_t_range(self):
        process = DiffusionForwardProcess(mask_token_id=0)
        x0 = torch.randint(1, 100, (100, 32))
        _, _, t = process(x0)
        assert (t >= 0).all() and (t <= 1).all()

    def test_full_masking(self):
        """With t=1.0, everything should be masked."""
        process = DiffusionForwardProcess(mask_token_id=0)
        x0 = torch.randint(1, 100, (2, 32))
        t = torch.ones(2)
        xt, mask, _ = process(x0, t=t)
        assert mask.all()
        assert (xt == 0).all()

    def test_no_masking(self):
        """With t=0.0, nothing should be masked."""
        process = DiffusionForwardProcess(mask_token_id=0)
        x0 = torch.randint(1, 100, (2, 32))
        t = torch.zeros(2)
        xt, mask, _ = process(x0, t=t)
        assert not mask.any()
        assert (xt == x0).all()
