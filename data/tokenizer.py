"""BPE tokenizer (vocab_size=32,868) using HuggingFace tokenizers.

Vocabulary layout:
  IDs 0-99:    Special tokens (from data.tokens registry)
  IDs 100+:    BPE-learned subwords (32,768 tokens)
  Total:       32,868 tokens

The tokenizer is trained with BPE on the corpus, then special tokens
are injected at their fixed IDs. This ensures consistent IDs across
tokenizer retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

from data.tokens import (
    SPECIAL_TOKEN_LIST,
    NUM_SPECIAL_TOKENS,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID,
    BOS_TOKEN,
    EOS_TOKEN,
)

# Re-export for backward compatibility
SPECIAL_TOKENS = SPECIAL_TOKEN_LIST[:5]  # Legacy: first 5 only
MASK_TOKEN_ID = MASK_TOKEN_ID
PAD_TOKEN_ID = PAD_TOKEN_ID
BOS_TOKEN_ID = BOS_TOKEN_ID
EOS_TOKEN_ID = EOS_TOKEN_ID
UNK_TOKEN_ID = UNK_TOKEN_ID


def build_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 32768,
    save_path: str = "data/tokenizer.json",
) -> Tokenizer:
    """Train a BPE tokenizer from corpus files.

    The BPE vocabulary is trained to `vocab_size` subwords. Then 100 special
    tokens are registered at IDs 0-99, giving a total vocabulary of
    vocab_size + 100.

    Args:
        corpus_files: List of paths to text files for training.
        vocab_size: Number of BPE-learned tokens (default 32,768).
        save_path: Where to save the trained tokenizer.

    Returns:
        Trained tokenizer instance.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size + NUM_SPECIAL_TOKENS,
        special_tokens=SPECIAL_TOKEN_LIST,
        show_progress=True,
        min_frequency=2,
    )

    tokenizer.train(corpus_files, trainer)

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        special_tokens=[
            (BOS_TOKEN, BOS_TOKEN_ID),
            (EOS_TOKEN, EOS_TOKEN_ID),
        ],
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str = "data/tokenizer.json") -> Tokenizer:
    """Load a pre-trained tokenizer from disk."""
    return Tokenizer.from_file(path)


def get_vocab_size(tokenizer: Tokenizer) -> int:
    """Get the total vocabulary size of a loaded tokenizer."""
    return tokenizer.get_vocab_size()
