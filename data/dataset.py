"""Dataset for Za v7 training with structure-aware masking.

Each item returns:
  - input_ids: Clean token IDs [seq_len]
  - attention_mask: Padding mask [seq_len] (1=real, 0=pad)
  - slot_positions: Positions of slot starts [max_slots] (-1=empty)
  - block_boundaries: Block (start, end) pairs [max_blocks, 2] (-1=empty)
  - n_slots: Number of real slots (scalar)
  - n_blocks: Number of real blocks (scalar)

Masking is NOT applied in the dataset — the trainer handles it
(DiffusionMasker in Phase 0-1, MultiLevelMasker in Phase 2+).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from data.tokens import (
    PAD_TOKEN_ID,
    SLOT_START_ID,
    SLOT_END_ID,
    BLOCK_START_ID,
    BLOCK_END_ID,
)


# Limits for collation (fixed-size tensors)
MAX_SLOTS_PER_SEQ = 32
MAX_BLOCKS_PER_SEQ = 16


class ZaDataset(Dataset):
    """Dataset for Za v7 with structure annotation.

    Scans each sequence for [SLOT_START]/[SLOT_END] and
    [BLOCK_START]/[BLOCK_END] delimiters and extracts
    slot positions and block boundaries for multi-level masking.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 2048,
        max_slots: int = MAX_SLOTS_PER_SEQ,
        max_blocks: int = MAX_BLOCKS_PER_SEQ,
    ):
        self.seq_len = seq_len
        self.max_slots = max_slots
        self.max_blocks = max_blocks

        # Chunk the flat token tensor into sequences
        n_tokens = len(token_ids)
        n_seqs = n_tokens // seq_len
        self.sequences = token_ids[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.sequences[idx]
        attention_mask = (input_ids != PAD_TOKEN_ID).long()

        # Extract structure from delimiter tokens
        slot_positions, n_slots = self._find_slots(input_ids)
        block_boundaries, n_blocks = self._find_blocks(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_positions": slot_positions,
            "block_boundaries": block_boundaries,
            "n_slots": torch.tensor(n_slots, dtype=torch.long),
            "n_blocks": torch.tensor(n_blocks, dtype=torch.long),
        }

    def _find_slots(self, ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Find positions of [SLOT_START] tokens in the sequence.

        Returns:
            (positions [max_slots], count)
        """
        positions = torch.full((self.max_slots,), -1, dtype=torch.long)
        slot_idx = 0

        for i in range(len(ids)):
            if ids[i].item() == SLOT_START_ID and slot_idx < self.max_slots:
                positions[slot_idx] = i
                slot_idx += 1

        return positions, slot_idx

    def _find_blocks(self, ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Find (start, end) boundaries of blocks.

        A block is delimited by [BLOCK_START] ... [BLOCK_END].
        Nested blocks are NOT supported (first match wins).

        Returns:
            (boundaries [max_blocks, 2], count)
        """
        boundaries = torch.full((self.max_blocks, 2), -1, dtype=torch.long)
        block_idx = 0
        in_block = False
        start = 0

        for i in range(len(ids)):
            tok = ids[i].item()
            if tok == BLOCK_START_ID and not in_block:
                in_block = True
                start = i
            elif tok == BLOCK_END_ID and in_block:
                if block_idx < self.max_blocks:
                    boundaries[block_idx, 0] = start
                    boundaries[block_idx, 1] = i + 1  # exclusive end
                    block_idx += 1
                in_block = False

        return boundaries, block_idx


# Backward compat alias
HCLMDataset = ZaDataset


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a batch of dataset items into batched tensors.

    All items have fixed-size tensors (padded to MAX_SLOTS/MAX_BLOCKS),
    so torch.stack should always work. Includes a defensive check.
    """
    result = {}
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]
        # Verify shapes match (they should, but be safe)
        shape0 = tensors[0].shape
        if not all(t.shape == shape0 for t in tensors):
            # Pad to max shape along each dim
            max_shape = list(shape0)
            for t in tensors[1:]:
                for d in range(len(max_shape)):
                    max_shape[d] = max(max_shape[d], t.shape[d])
            padded = []
            for t in tensors:
                if t.shape == tuple(max_shape):
                    padded.append(t)
                else:
                    p = torch.full(max_shape, -1, dtype=t.dtype)
                    slices = tuple(slice(0, s) for s in t.shape)
                    p[slices] = t
                    padded.append(p)
            result[key] = torch.stack(padded)
        else:
            result[key] = torch.stack(tensors)
    return result
