"""Multi-level masking for Za v6: token, span, slot, and block masking.

Extends LLaDA masking to operate at multiple structural levels:

Level 0: Token masking     — standard LLaDA [MASK] per token
Level 1: Span masking      — mask contiguous token sequences
Level 2: Slot masking      — mask slots within templates
Level 3: Block masking     — mask entire blocks (subtrees)

The diffusion process can mask at any level. Unmasking follows
a coarse-to-fine schedule: blocks → slots → spans → tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class MaskLevel(IntEnum):
    """Levels of masking granularity."""
    TOKEN = 0
    SPAN = 1
    SLOT = 2
    BLOCK = 3


@dataclass
class MultiLevelMask:
    """Multi-level mask for a sequence.

    Each level has its own mask tensor. Higher levels override lower levels:
    if a block is masked, all its tokens/slots are also masked.
    """
    token_mask: torch.Tensor      # [batch, seq_len] bool
    span_mask: torch.Tensor       # [batch, seq_len] bool
    slot_mask: torch.Tensor       # [batch, n_slots] bool
    block_mask: torch.Tensor      # [batch, n_blocks] bool

    # Block boundaries: [n_blocks, 2] with (start, end) positions
    block_boundaries: torch.Tensor

    @property
    def combined_mask(self) -> torch.Tensor:
        """Unified mask: True = masked at any level. [batch, seq_len]"""
        return self.token_mask | self.span_mask

    def mask_level_at(self, position: int) -> MaskLevel:
        """What level of masking applies at a given position."""
        if self.block_mask.any():
            # Check if position falls in a masked block
            for i in range(self.block_boundaries.size(0)):
                start, end = self.block_boundaries[i]
                if start <= position < end and self.block_mask[0, i]:
                    return MaskLevel.BLOCK

        if self.slot_mask.any():
            return MaskLevel.SLOT

        if self.span_mask[0, position]:
            return MaskLevel.SPAN

        if self.token_mask[0, position]:
            return MaskLevel.TOKEN

        return MaskLevel.TOKEN  # Not masked


class MultiLevelMasker:
    """Applies multi-level masking for training.

    At each training step, randomly selects masking levels:
    - With probability p_token: mask individual tokens (LLaDA)
    - With probability p_span: mask contiguous spans
    - With probability p_slot: mask template slots
    - With probability p_block: mask entire blocks

    Probabilities are independent and can overlap.
    """

    def __init__(
        self,
        mask_token_id: int = 0,
        # Masking probabilities per level
        p_token: float = 0.6,
        p_span: float = 0.2,
        p_slot: float = 0.1,
        p_block: float = 0.1,
        # Span parameters
        mean_span_len: int = 4,
        max_span_len: int = 16,
        # Block parameters
        max_block_mask_ratio: float = 0.3,
    ):
        self.mask_token_id = mask_token_id
        self.p_token = p_token
        self.p_span = p_span
        self.p_slot = p_slot
        self.p_block = p_block
        self.mean_span_len = mean_span_len
        self.max_span_len = max_span_len
        self.max_block_mask_ratio = max_block_mask_ratio

    def mask_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        block_boundaries: torch.Tensor | None = None,
        slot_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MultiLevelMask]:
        """Apply multi-level masking to a batch.

        Args:
            input_ids: [batch, seq_len] clean token IDs
            attention_mask: [batch, seq_len] padding mask
            block_boundaries: [n_blocks, 2] start/end positions of blocks
            slot_positions: [n_slots] positions of template slots

        Returns:
            (masked_ids, MultiLevelMask)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize masks
        token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        span_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        n_blocks = block_boundaries.size(0) if block_boundaries is not None else 0
        block_mask_tensor = torch.zeros(batch_size, max(n_blocks, 1), dtype=torch.bool, device=device)

        n_slots = slot_positions.size(0) if slot_positions is not None else 0
        slot_mask_tensor = torch.zeros(batch_size, max(n_slots, 1), dtype=torch.bool, device=device)

        if block_boundaries is None:
            block_boundaries = torch.zeros(0, 2, dtype=torch.long, device=device)

        # Sample which masking levels to use per example
        level_dice = torch.rand(batch_size, 4, device=device)

        for b in range(batch_size):
            # Per-example masking ratio
            t = torch.empty(1, device=device).uniform_(0.0, 1.0).item()

            # Level 0: Token masking (standard LLaDA)
            if level_dice[b, 0] < self.p_token:
                token_rand = torch.rand(seq_len, device=device)
                token_mask[b] = token_rand < t

            # Level 1: Span masking
            if level_dice[b, 1] < self.p_span:
                span_mask[b] = self._sample_spans(seq_len, t, device)

            # Level 2: Slot masking
            if level_dice[b, 2] < self.p_slot and n_slots > 0:
                slot_rand = torch.rand(n_slots, device=device)
                slot_mask_tensor[b, :n_slots] = slot_rand < t
                # Propagate slot masks to token positions
                for s in range(n_slots):
                    if slot_mask_tensor[b, s]:
                        pos = slot_positions[s].item()
                        token_mask[b, pos] = True

            # Level 3: Block masking
            if level_dice[b, 3] < self.p_block and n_blocks > 0:
                block_rand = torch.rand(n_blocks, device=device)
                n_to_mask = max(1, int(n_blocks * self.max_block_mask_ratio))
                _, top_idx = block_rand.topk(min(n_to_mask, n_blocks))
                block_mask_tensor[b, top_idx] = True
                # Propagate block masks to token positions
                for i in top_idx:
                    start, end = block_boundaries[i]
                    token_mask[b, start:end] = True
                    span_mask[b, start:end] = True

        # Don't mask padding
        if attention_mask is not None:
            valid = attention_mask.bool()
            token_mask = token_mask & valid
            span_mask = span_mask & valid

        # Apply masking
        combined = token_mask | span_mask
        masked_ids = input_ids.clone()
        masked_ids[combined] = self.mask_token_id

        ml_mask = MultiLevelMask(
            token_mask=token_mask,
            span_mask=span_mask,
            slot_mask=slot_mask_tensor,
            block_mask=block_mask_tensor,
            block_boundaries=block_boundaries,
        )

        return masked_ids, ml_mask

    def _sample_spans(self, seq_len: int, t: float, device: torch.device) -> torch.Tensor:
        """Sample random spans to mask.

        Uses geometric distribution for span lengths.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        pos = 0
        while pos < seq_len:
            # Skip with probability (1-t)
            if torch.rand(1, device=device).item() > t:
                pos += 1
                continue

            # Sample span length
            span_len = min(
                int(torch.geometric(torch.tensor(1.0 / self.mean_span_len)).item()) + 1,
                self.max_span_len,
                seq_len - pos,
            )
            mask[pos:pos + span_len] = True
            pos += span_len

        return mask

    def mask_batch_token_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback: token-level only masking (LLaDA compatible).

        Returns (masked_ids, bool_mask) like the original DiffusionMasker.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        t = torch.empty(batch_size, 1, device=device).uniform_(0.0, 1.0)
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < t

        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        masked_ids = input_ids.clone()
        masked_ids[mask] = self.mask_token_id

        return masked_ids, mask
