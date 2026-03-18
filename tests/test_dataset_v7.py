"""Tests for Za v7 dataset with structure annotation."""

import torch
import pytest

from data.dataset import ZaDataset, HCLMDataset, collate_fn
from data.tokens import (
    PAD_TOKEN_ID,
    SLOT_START_ID,
    SLOT_END_ID,
    BLOCK_START_ID,
    BLOCK_END_ID,
)


def _make_tokens(seq_len: int = 64, n_seqs: int = 4) -> torch.Tensor:
    """Create fake token data with structure delimiters."""
    tokens = torch.randint(100, 1000, (n_seqs * seq_len,))

    # Insert structure into each sequence
    for s in range(n_seqs):
        base = s * seq_len
        # Block at positions 5-15
        tokens[base + 5] = BLOCK_START_ID
        tokens[base + 15] = BLOCK_END_ID
        # Block at positions 30-45
        tokens[base + 30] = BLOCK_START_ID
        tokens[base + 45] = BLOCK_END_ID
        # Slot at position 8 (inside first block)
        tokens[base + 8] = SLOT_START_ID
        tokens[base + 12] = SLOT_END_ID
        # Slot at position 50 (outside blocks)
        tokens[base + 50] = SLOT_START_ID
        tokens[base + 55] = SLOT_END_ID

    return tokens


class TestZaDataset:
    def test_basic_shape(self):
        tokens = torch.randint(100, 1000, (256,))
        ds = ZaDataset(tokens, seq_len=64)
        assert len(ds) == 4  # 256 / 64

        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["slot_positions"].shape[0] == 32  # MAX_SLOTS
        assert item["block_boundaries"].shape == (16, 2)  # MAX_BLOCKS

    def test_structure_detection(self):
        tokens = _make_tokens(seq_len=64, n_seqs=4)
        ds = ZaDataset(tokens, seq_len=64)
        item = ds[0]

        # Should find 2 slots
        n_slots = item["n_slots"].item()
        assert n_slots == 2
        assert item["slot_positions"][0].item() == 8  # First SLOT_START
        assert item["slot_positions"][1].item() == 50  # Second SLOT_START

        # Should find 2 blocks
        n_blocks = item["n_blocks"].item()
        assert n_blocks == 2
        assert item["block_boundaries"][0, 0].item() == 5  # First block start
        assert item["block_boundaries"][0, 1].item() == 16  # First block end (exclusive)
        assert item["block_boundaries"][1, 0].item() == 30  # Second block start

    def test_no_structure(self):
        tokens = torch.randint(100, 1000, (128,))
        ds = ZaDataset(tokens, seq_len=64)
        item = ds[0]
        assert item["n_slots"].item() == 0
        assert item["n_blocks"].item() == 0
        assert (item["slot_positions"] == -1).all()
        assert (item["block_boundaries"] == -1).all()

    def test_padding_mask(self):
        tokens = torch.randint(100, 1000, (64,))
        tokens[50:] = PAD_TOKEN_ID
        ds = ZaDataset(tokens, seq_len=64)
        item = ds[0]
        assert item["attention_mask"][:50].sum() == 50
        assert item["attention_mask"][50:].sum() == 0

    def test_collate(self):
        tokens = _make_tokens(seq_len=64, n_seqs=4)
        ds = ZaDataset(tokens, seq_len=64)
        batch = collate_fn([ds[0], ds[1], ds[2]])
        assert batch["input_ids"].shape == (3, 64)
        assert batch["attention_mask"].shape == (3, 64)
        assert batch["slot_positions"].shape == (3, 32)
        assert batch["block_boundaries"].shape == (3, 16, 2)
        assert batch["n_slots"].shape == (3,)
        assert batch["n_blocks"].shape == (3,)

    def test_backward_compat_alias(self):
        assert HCLMDataset is ZaDataset


class TestStructureAnnotation:
    def test_annotate_narrative(self):
        from data.prep import annotate_structure
        text = 'First paragraph.\n\n"Hello," said Tom.\n\nEnd.'
        result = annotate_structure(text, domain="story")
        assert "[BLOCK_START]" in result
        assert "[BLOCK_END]" in result
        assert "[SLOT_START]" in result
        assert "[SLOT_END]" in result

    def test_annotate_code(self):
        from data.prep import annotate_structure
        code = 'def foo():\n    """Docstring."""\n    pass\n\nx = 1'
        result = annotate_structure(code, domain="code")
        assert "[BLOCK_START]" in result
        assert "[BLOCK_END]" in result

    def test_annotate_qa(self):
        from data.prep import annotate_structure
        qa = "Question: What is 2+2? Answer: 4."
        result = annotate_structure(qa, domain="qa")
        assert "[SLOT_START]" in result
        assert "[BLOCK_START]" in result

    def test_no_annotation(self):
        from data.prep import annotate_structure
        text = "Plain text with no structure markers."
        result = annotate_structure(text, domain="unknown_domain")
        assert result == text
