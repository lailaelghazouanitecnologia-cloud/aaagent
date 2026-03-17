"""Tests for composite embedding shapes and forward pass."""

import torch
import pytest

from model.embedding.composite import CompositeEmbedding, FlatEmbedding
from model.embedding.local import LocalEmbedding
from model.embedding.positional import LearnedPositionalEncoding


class TestLocalEmbedding:
    def test_forward_shape(self):
        emb = LocalEmbedding(vocab_size=8192, embed_dim=384)
        x = torch.randint(0, 8192, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 384)

    def test_weight_property(self):
        emb = LocalEmbedding(vocab_size=8192, embed_dim=384)
        assert emb.weight.shape == (8192, 384)


class TestFlatEmbedding:
    def test_forward_shape(self):
        emb = FlatEmbedding(vocab_size=8192, embed_dim=384, max_seq_len=512)
        x = torch.randint(0, 8192, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 384)


class TestCompositeEmbedding:
    def test_forward_shape(self):
        emb = CompositeEmbedding(
            vocab_size=8192, embed_dim=384, max_seq_len=512,
            fine_clusters=64, coarse_clusters=8,
        )
        x = torch.randint(0, 8192, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 384)

    def test_gate_scale(self):
        emb = CompositeEmbedding(
            vocab_size=8192, embed_dim=384, max_seq_len=512,
        )
        x = torch.randint(0, 8192, (2, 16))
        out_zero = emb(x, gate_scale=torch.tensor(0.0))
        out_full = emb(x, gate_scale=torch.tensor(1.0))
        # With gate_scale=0, structural component is zeroed out
        # With gate_scale=1, full learned gate is applied
        assert out_zero.shape == out_full.shape
        assert not torch.allclose(out_zero, out_full)

    def test_no_hierarchy(self):
        emb = CompositeEmbedding(
            vocab_size=8192, embed_dim=384, max_seq_len=512,
            fine_clusters=64, coarse_clusters=0, use_hierarchy=False,
        )
        x = torch.randint(0, 8192, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 384)

    def test_no_gate(self):
        emb = CompositeEmbedding(
            vocab_size=8192, embed_dim=384, max_seq_len=512,
            use_gate=False,
        )
        x = torch.randint(0, 8192, (2, 16))
        out = emb(x)
        assert out.shape == (2, 16, 384)


class TestPositionalEncoding:
    def test_shape(self):
        pos = LearnedPositionalEncoding(max_seq_len=512, embed_dim=384)
        out = pos(16)
        assert out.shape == (1, 16, 384)
