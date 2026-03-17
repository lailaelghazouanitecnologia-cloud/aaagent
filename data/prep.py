"""Download and preprocess TinyStories dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def download_tinystories(data_dir: str = "data/tinystories") -> Path:
    """Download TinyStories dataset using HuggingFace datasets.

    Returns:
        Path to the data directory containing processed files.
    """
    from datasets import load_dataset

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_file = data_path / "train.txt"
    val_file = data_path / "val.txt"

    if train_file.exists() and val_file.exists():
        logger.info("TinyStories already downloaded at %s", data_path)
        return data_path

    logger.info("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories")

    # Write train split
    logger.info("Writing train split...")
    with open(train_file, "w", encoding="utf-8") as f:
        for example in ds["train"]:
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")

    # Write validation split
    logger.info("Writing validation split...")
    with open(val_file, "w", encoding="utf-8") as f:
        for example in ds["validation"]:
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")

    logger.info("TinyStories saved to %s", data_path)
    return data_path


def tokenize_corpus(
    text_file: str,
    tokenizer_path: str,
    output_path: str,
    max_length: int = 512,
) -> torch.Tensor:
    """Tokenize a text file and save as a flat tensor of token IDs.

    Args:
        text_file: Path to input text file.
        tokenizer_path: Path to trained tokenizer JSON.
        output_path: Where to save the token tensor (.pt file).
        max_length: Max tokens per encoding batch (for chunking).

    Returns:
        Flat tensor of all token IDs.
    """
    from data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)

    logger.info("Tokenizing %s...", text_file)
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize in chunks to avoid memory issues
    chunk_size = 1_000_000  # characters
    all_ids: list[int] = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        encoding = tokenizer.encode(chunk)
        all_ids.extend(encoding.ids)

    token_tensor = torch.tensor(all_ids, dtype=torch.long)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_tensor, output_path)
    logger.info("Saved %d tokens to %s", len(token_tensor), output_path)

    return token_tensor


def prepare_data(config: dict) -> None:
    """Full data preparation pipeline: download, tokenize, and build tokenizer."""
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "data/tinystories")
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer.json")
    vocab_size = config.get("model", {}).get("vocab_size", 8192)

    # Download
    data_path = download_tinystories(data_dir)

    # Build tokenizer if not exists
    if not Path(tokenizer_path).exists():
        from data.tokenizer import build_tokenizer

        logger.info("Training tokenizer with vocab_size=%d...", vocab_size)
        build_tokenizer(
            corpus_files=[str(data_path / "train.txt")],
            vocab_size=vocab_size,
            save_path=tokenizer_path,
        )

    # Tokenize train and val
    for split in ["train", "val"]:
        text_file = data_path / f"{split}.txt"
        output_file = data_path / f"{split}_tokens.pt"
        if not output_file.exists():
            tokenize_corpus(
                str(text_file),
                tokenizer_path,
                str(output_file),
            )
