"""BPE tokenizer (vocab_size=8192) using HuggingFace tokenizers."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[BOS]", "[EOS]", "[UNK]"]
MASK_TOKEN_ID = 0
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
UNK_TOKEN_ID = 4


def build_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 8192,
    save_path: str = "data/tokenizer.json",
) -> Tokenizer:
    """Train a BPE tokenizer from corpus files.

    Args:
        corpus_files: List of paths to text files for training.
        vocab_size: Target vocabulary size.
        save_path: Where to save the trained tokenizer.

    Returns:
        Trained tokenizer instance.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    tokenizer.train(corpus_files, trainer)

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        special_tokens=[
            ("[BOS]", BOS_TOKEN_ID),
            ("[EOS]", EOS_TOKEN_ID),
        ],
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str = "data/tokenizer.json") -> Tokenizer:
    """Load a pre-trained tokenizer from disk."""
    return Tokenizer.from_file(path)
