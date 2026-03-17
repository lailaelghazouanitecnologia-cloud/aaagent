"""Tests for full forward pass and parameter count verification."""

import torch
import pytest

from model.config import ModelConfig
from model.lm import HCLMD


class TestHCLMD:
    def test_forward_hierarchical(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            fine_clusters=8, coarse_clusters=4,
        )
        model = HCLMD(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_flat(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            embedding_type="flat",
        )
        model = HCLMD(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)

    def test_param_count(self):
        config = ModelConfig(
            vocab_size=8192, embed_dim=384, max_seq_len=512,
            n_layers=8, n_heads=6, d_ff=1536,
            fine_clusters=64, coarse_clusters=8,
        )
        model = HCLMD(config)
        counts = model.count_parameters()

        # Total should be roughly 18-20M
        total = counts["total"]
        assert 15_000_000 < total < 25_000_000, f"Expected ~20M params, got {total:,}"

    def test_weight_tying(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            tie_weights=True,
        )
        model = HCLMD(config)
        # Head should use embedding weight
        assert model.head._tied_weight is model.embedding.weight

    def test_gate_override(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
        )
        model = HCLMD(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x, gate_override=0.0)
        assert logits.shape == (2, 16, 256)

    def test_attention_mask(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
        )
        model = HCLMD(config)
        x = torch.randint(0, 256, (2, 16))
        mask = torch.ones(2, 16)
        mask[:, -4:] = 0
        logits = model(x, attention_mask=mask)
        assert logits.shape == (2, 16, 256)

    def test_causal_mode(self):
        config = ModelConfig(
            vocab_size=256, embed_dim=64, max_seq_len=32,
            n_layers=2, n_heads=2, d_ff=128,
            causal=True, embedding_type="flat",
        )
        model = HCLMD(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 256)
