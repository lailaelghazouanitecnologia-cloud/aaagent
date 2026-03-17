"""Tests for Za v6 Block IR."""

import torch
import pytest

from blocks.block import Block, BlockType
from blocks.registry import BlockRegistry
from blocks.composer import BlockComposer, CompositionNetwork
from blocks.meta_embedding import MetaEmbeddingGenerator
from blocks.pattern_detector import PatternDetector
from blocks.dynamic import DynamicBlockManager, HyperNetDelta


# ── Block ──

def _make_block(name="test", block_type=BlockType.TOKEN, embed_dim=384):
    return Block(
        block_type=block_type,
        embedding=torch.randn(embed_dim),
        name=name,
    )


def _make_template(name="tmpl", n_filled=2, n_empty=1, embed_dim=384):
    filled = [_make_block(f"slot_{i}") for i in range(n_filled)]
    slots = filled + [None] * n_empty
    return Block(
        block_type=BlockType.TEMPLATE,
        embedding=torch.randn(embed_dim),
        slots=slots,
        name=name,
        depth=1,
    )


class TestBlock:
    def test_token_block(self):
        b = _make_block("hello", BlockType.TOKEN)
        assert b.is_leaf
        assert b.n_slots == 0
        assert b.is_complete

    def test_template_block(self):
        t = _make_template("tmpl", n_filled=2, n_empty=1)
        assert not t.is_leaf
        assert t.n_slots == 3
        assert t.n_filled == 2
        assert t.n_empty == 1
        assert not t.is_complete

    def test_mask(self):
        t = _make_template("tmpl", n_filled=2, n_empty=1)
        assert t.mask == [True, True, False]

    def test_semantic_hash_deterministic(self):
        b1 = Block(block_type=BlockType.TOKEN, embedding=torch.ones(384), name="a")
        b2 = Block(block_type=BlockType.TOKEN, embedding=torch.ones(384), name="a")
        assert b1.semantic_hash() == b2.semantic_hash()

    def test_semantic_hash_different_embedding(self):
        b1 = Block(block_type=BlockType.TOKEN, embedding=torch.ones(384))
        b2 = Block(block_type=BlockType.TOKEN, embedding=torch.zeros(384))
        assert b1.semantic_hash() != b2.semantic_hash()

    def test_hex_hash(self):
        b = _make_block()
        h = b.hex_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_instance_hash(self):
        b = _make_block()
        h1 = b.instance_hash(parent_hash=b"parent", position=0)
        h2 = b.instance_hash(parent_hash=b"parent", position=0)
        assert h1 == h2

    def test_all_blocks(self):
        child1 = _make_block("c1")
        child2 = _make_block("c2")
        parent = Block(
            block_type=BlockType.TEMPLATE,
            embedding=torch.randn(384),
            slots=[child1, child2],
        )
        all_b = parent.all_blocks()
        assert len(all_b) == 3  # parent + 2 children

    def test_max_depth(self):
        leaf = _make_block()
        mid = Block(block_type=BlockType.TEMPLATE, embedding=torch.randn(384), slots=[leaf])
        root = Block(block_type=BlockType.TEMPLATE, embedding=torch.randn(384), slots=[mid])
        assert root.max_depth() == 2

    def test_flatten_tokens(self):
        t1 = Block(block_type=BlockType.TOKEN, embedding=torch.randn(384), token_id=10)
        t2 = Block(block_type=BlockType.TOKEN, embedding=torch.randn(384), token_id=20)
        parent = Block(block_type=BlockType.PLAN, embedding=torch.randn(384), slots=[t1, t2])
        assert parent.flatten_tokens() == [10, 20]


# ── Registry ──

class TestBlockRegistry:
    def test_register_and_lookup(self):
        reg = BlockRegistry(max_size=100)
        b = _make_block()
        reg.register(b)
        found = reg.lookup(b.semantic_hash())
        assert found is not None
        assert found.name == b.name

    def test_deduplication(self):
        reg = BlockRegistry()
        b = Block(block_type=BlockType.TOKEN, embedding=torch.ones(384))
        reg.register(b)
        reg.register(b)
        assert len(reg) == 1
        assert reg.get_use_count(b.semantic_hash()) == 2

    def test_lookup_by_name(self):
        reg = BlockRegistry()
        b = _make_block("findme")
        reg.register(b)
        found = reg.lookup_by_name("findme")
        assert found is not None

    def test_cache_result(self):
        reg = BlockRegistry()
        b = _make_block()
        reg.register(b)
        reg.cache_result(b.semantic_hash(), torch.tensor(42.0))
        cached = reg.get_cached_result(b.semantic_hash())
        assert cached is not None
        assert cached.item() == 42.0

    def test_eviction(self):
        reg = BlockRegistry(max_size=3)
        blocks = [_make_block(f"b{i}") for i in range(5)]
        for b in blocks:
            reg.register(b)
        assert len(reg) <= 3

    def test_most_used(self):
        reg = BlockRegistry()
        b1 = _make_block("popular")
        reg.register(b1)
        for _ in range(10):
            reg.lookup(b1.semantic_hash())
        top = reg.most_used(1)
        assert len(top) == 1
        assert top[0][1] >= 10

    def test_clear(self):
        reg = BlockRegistry()
        reg.register(_make_block())
        reg.clear()
        assert len(reg) == 0


# ── Composer ──

class TestBlockComposer:
    def test_compose_two_blocks(self):
        composer = BlockComposer(embed_dim=384, meta_dim=128)
        b1 = _make_block("a")
        b2 = _make_block("b")
        b1.meta_embedding = torch.randn(128)
        b2.meta_embedding = torch.randn(128)

        composed = composer.compose([b1, b2], name="ab")
        assert composed.name == "ab"
        assert composed.embedding.shape == (384,)
        assert composed.meta_embedding.shape == (128,)
        assert len(composed.slots) == 2

    def test_compose_with_open_slots(self):
        composer = BlockComposer(embed_dim=384, meta_dim=128)
        b1 = _make_block("a")
        b1.meta_embedding = torch.randn(128)

        composed = composer.compose([b1], name="tmpl", n_open_slots=2)
        assert composed.block_type == BlockType.TEMPLATE
        assert composed.n_filled == 1
        assert composed.n_empty == 2

    def test_composition_network_forward(self):
        net = CompositionNetwork(meta_dim=128, max_components=8)
        metas = [torch.randn(128) for _ in range(3)]
        result = net(metas)
        assert result.shape == (128,)


# ── MetaEmbeddingGenerator ──

class TestMetaEmbeddingGenerator:
    def test_generate(self):
        gen = MetaEmbeddingGenerator(embed_dim=384, meta_dim=128)
        b = _make_block()
        meta = gen.generate_for_block(b)
        assert meta.shape == (128,)

    def test_get_section(self):
        gen = MetaEmbeddingGenerator(embed_dim=384, meta_dim=128)
        meta = torch.randn(128)
        section = gen.get_section(meta, "type")
        assert section.shape == (16,)

    def test_extract_struct_features(self):
        gen = MetaEmbeddingGenerator(embed_dim=384, meta_dim=128)
        b = _make_template()
        features = gen.extract_struct_features(b)
        assert features.shape == (8,)  # 5 one-hot + 3 scalars


# ── PatternDetector ──

class TestPatternDetector:
    def test_observe_and_detect(self):
        reg = BlockRegistry()
        composer = BlockComposer(embed_dim=384, meta_dim=128)
        detector = PatternDetector(reg, composer, min_frequency=3)

        b1 = _make_block("read")
        b1.meta_embedding = torch.randn(128)
        b2 = _make_block("filter")
        b2.meta_embedding = torch.randn(128)
        reg.register(b1)
        reg.register(b2)

        # Observe same pattern multiple times
        for _ in range(5):
            detector.observe([b1, b2])

        templates = detector.detect_and_create()
        assert len(templates) >= 1


# ── HyperNetDelta ──

class TestHyperNetDelta:
    def test_forward(self):
        net = HyperNetDelta(embed_dim=384, context_dim=128)
        emb = torch.randn(2, 10, 384)
        ctx = torch.randn(2, 10, 128)
        result = net(emb, ctx)
        assert result.shape == (2, 10, 384)


# ── DynamicBlockManager ──

class TestDynamicBlockManager:
    def test_create_token_block(self):
        reg = BlockRegistry()
        composer = BlockComposer(embed_dim=384, meta_dim=128)
        meta_gen = MetaEmbeddingGenerator(embed_dim=384, meta_dim=128)
        mgr = DynamicBlockManager(reg, composer, meta_gen)

        block = mgr.create_token_block(42, torch.randn(384))
        assert block.block_type == BlockType.TOKEN
        assert block.token_id == 42
        assert block.meta_embedding is not None

    def test_create_template(self):
        reg = BlockRegistry()
        composer = BlockComposer(embed_dim=384, meta_dim=128)
        meta_gen = MetaEmbeddingGenerator(embed_dim=384, meta_dim=128)
        mgr = DynamicBlockManager(reg, composer, meta_gen)

        filled = [_make_block("read"), _make_block("filter")]
        template = mgr.create_template("pipeline", filled, n_empty_slots=1)
        assert template.block_type == BlockType.TEMPLATE
        assert template.n_filled == 2
        assert template.n_empty == 1
        assert mgr.n_dynamic == 1
