"""Streaming dataset for HCLM-D training with diffusion masking."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from data.masking import DiffusionMasker
from data.tokenizer import PAD_TOKEN_ID


class HCLMDataset(Dataset):
    """Dataset that returns tokenized sequences ready for masked diffusion training.

    Each item returns:
        - input_ids: Original token IDs (clean sequence), shape [seq_len]
        - masked_ids: Token IDs with diffusion masking applied, shape [seq_len]
        - mask: Boolean mask indicating which positions are masked, shape [seq_len]
        - attention_mask: Padding mask (1 = real token, 0 = padding), shape [seq_len]
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 512,
        masker: Optional[DiffusionMasker] = None,
        mask_token_id: int = 0,
    ):
        """
        Args:
            token_ids: Flat tensor of all token IDs in the corpus.
            seq_len: Fixed sequence length (will chunk the corpus).
            masker: DiffusionMasker instance. If None, creates default.
            mask_token_id: Token ID used for [MASK].
        """
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.masker = masker or DiffusionMasker(mask_token_id=mask_token_id)

        # Chunk the flat token tensor into sequences
        n_tokens = len(token_ids)
        n_seqs = n_tokens // seq_len
        self.sequences = token_ids[: n_seqs * seq_len].reshape(n_seqs, seq_len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.sequences[idx]
        attention_mask = (input_ids != PAD_TOKEN_ID).long()

        # Apply diffusion masking
        masked_ids, mask = self.masker.mask(input_ids, attention_mask)

        return {
            "input_ids": input_ids,
            "masked_ids": masked_ids,
            "mask": mask,
            "attention_mask": attention_mask,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a batch of dataset items into batched tensors."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
