"""Download and preprocess datasets for Za v7.

Supports:
  - TinyStories (default, English narratives)
  - Multilingual (EN + ES + Python mixed corpus)

Structure annotation:
  Inserts [BLOCK_START]/[BLOCK_END] around paragraphs and
  [SLOT_START]/[SLOT_END] around dialogue/template slots.
  This gives the multi-level masker real boundaries to work with.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def download_tinystories(data_dir: str = "data/tinystories") -> Path:
    """Download TinyStories dataset using HuggingFace datasets."""
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

    logger.info("Writing train split...")
    with open(train_file, "w", encoding="utf-8") as f:
        for example in ds["train"]:
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")

    logger.info("Writing validation split...")
    with open(val_file, "w", encoding="utf-8") as f:
        for example in ds["validation"]:
            text = example["text"].strip()
            if text:
                f.write(text + "\n\n")

    logger.info("TinyStories saved to %s", data_path)
    return data_path


def annotate_structure(text: str, domain: str = "story") -> str:
    """Insert structure tokens into raw text.

    Annotation rules by domain:
      story/wiki:
        - Paragraphs (double newline separated) → [BLOCK_START]...[BLOCK_END]
        - Dialogue (quoted speech) → [SLOT_START]...[SLOT_END]
      code:
        - Functions/classes → [BLOCK_START]...[BLOCK_END]
        - Docstrings → [SLOT_START]...[SLOT_END]
      qa:
        - Questions → [SLOT_START]...[SLOT_END]
        - Answers → [BLOCK_START]...[BLOCK_END]

    Args:
        text: Raw text to annotate.
        domain: One of "story", "wiki", "code", "qa".

    Returns:
        Text with structure tokens inserted.
    """
    if domain in ("story", "wiki"):
        return _annotate_narrative(text)
    elif domain == "code":
        return _annotate_code(text)
    elif domain == "qa":
        return _annotate_qa(text)
    return text


def _annotate_narrative(text: str) -> str:
    """Annotate narrative text (stories, Wikipedia).

    - Paragraphs → blocks
    - Quoted dialogue → slots
    """
    paragraphs = re.split(r'\n\s*\n', text)
    parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Wrap each paragraph as a block
        # Mark quoted dialogue as slots within the block
        annotated = _mark_dialogue_slots(para)
        parts.append(f"[BLOCK_START] {annotated} [BLOCK_END]")

    return "\n".join(parts)


def _mark_dialogue_slots(text: str) -> str:
    """Mark quoted dialogue as slots.

    "Hello," said the girl. → [SLOT_START] "Hello," [SLOT_END] said the girl.
    """
    # Match quoted strings (double or single quotes)
    def _replace_quote(m):
        return f"[SLOT_START] {m.group(0)} [SLOT_END]"

    return re.sub(r'"[^"]{2,}"', _replace_quote, text)


def _annotate_code(text: str) -> str:
    """Annotate Python code.

    - Function/class definitions → blocks
    - Docstrings → slots
    """
    lines = text.split('\n')
    parts = []
    in_block = False
    block_lines = []

    for line in lines:
        stripped = line.strip()

        # Detect function/class start
        if re.match(r'^(def |class |async def )', stripped):
            # Close previous block
            if in_block and block_lines:
                block_text = '\n'.join(block_lines)
                parts.append(f"[BLOCK_START] {block_text} [BLOCK_END]")
                block_lines = []
            in_block = True
            block_lines.append(line)
        elif in_block:
            block_lines.append(line)
            # Detect end of block (empty line at indent level 0)
            if stripped == '' and len(block_lines) > 2:
                block_text = '\n'.join(block_lines)
                # Mark docstrings as slots
                block_text = re.sub(
                    r'"""[\s\S]*?"""',
                    lambda m: f'[SLOT_START] {m.group(0)} [SLOT_END]',
                    block_text,
                )
                parts.append(f"[BLOCK_START] {block_text} [BLOCK_END]")
                block_lines = []
                in_block = False
        else:
            parts.append(line)

    # Flush remaining block
    if in_block and block_lines:
        block_text = '\n'.join(block_lines)
        parts.append(f"[BLOCK_START] {block_text} [BLOCK_END]")

    return '\n'.join(parts)


def _annotate_qa(text: str) -> str:
    """Annotate Q&A text.

    Question: ... → [SLOT_START] Question: ... [SLOT_END]
    Answer: ... → [BLOCK_START] Answer: ... [BLOCK_END]
    """
    text = re.sub(
        r'(Question:\s*.+?)(?=Answer:|$)',
        lambda m: f'[SLOT_START] {m.group(1).strip()} [SLOT_END] ',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'(Answer:\s*.+?)(?=Question:|$)',
        lambda m: f'[BLOCK_START] {m.group(1).strip()} [BLOCK_END] ',
        text,
        flags=re.DOTALL,
    )
    return text


def tokenize_corpus(
    text_file: str,
    tokenizer_path: str,
    output_path: str,
    max_length: int = 2048,
    annotate: bool = True,
    domain: str = "story",
) -> torch.Tensor:
    """Tokenize a text file and save as a flat tensor of token IDs.

    If annotate=True, inserts structure tokens before tokenizing.

    Args:
        text_file: Path to input text file.
        tokenizer_path: Path to trained tokenizer JSON.
        output_path: Where to save the token tensor (.pt file).
        max_length: Max tokens per encoding batch.
        annotate: Whether to insert structure annotations.
        domain: Domain hint for annotation ("story", "code", "qa", "wiki").

    Returns:
        Flat tensor of all token IDs.
    """
    from data.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)

    logger.info("Tokenizing %s (annotate=%s, domain=%s)...", text_file, annotate, domain)
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize in chunks to avoid memory issues
    chunk_size = 1_000_000  # characters
    total_chunks = (len(text) + chunk_size - 1) // chunk_size
    all_ids: list[int] = []

    for i, start in enumerate(range(0, len(text), chunk_size)):
        chunk = text[start: start + chunk_size]

        # Annotate structure before tokenizing
        if annotate:
            chunk = annotate_structure(chunk, domain=domain)

        encoding = tokenizer.encode(chunk)
        all_ids.extend(encoding.ids)
        pct = (i + 1) / total_chunks * 100
        logger.info(
            "  Tokenizing: %d/%d chunks (%.0f%%) — %d tokens so far",
            i + 1, total_chunks, pct, len(all_ids),
        )

    token_tensor = torch.tensor(all_ids, dtype=torch.long)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_tensor, output_path)
    logger.info("Saved %d tokens to %s", len(token_tensor), output_path)

    return token_tensor


def prepare_data(config: dict) -> None:
    """Full data preparation pipeline.

    Detects dataset type from config:
    - "tinystories": English-only TinyStories
    - "multilingual": EN + ES + Python mixed corpus
    """
    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("dataset", "tinystories")

    if dataset_type == "multilingual":
        from data.prep_multilingual import prepare_multilingual
        prepare_multilingual(config)
        return

    # Default: TinyStories pipeline
    data_dir = data_cfg.get("data_dir", "data/tinystories")
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer.json")
    vocab_size = config.get("model", {}).get("vocab_size", 32868)

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

    # Tokenize train and val (with structure annotation)
    for split in ["train", "val"]:
        text_file = data_path / f"{split}.txt"
        output_file = data_path / f"{split}_tokens.pt"
        if not output_file.exists():
            tokenize_corpus(
                str(text_file),
                tokenizer_path,
                str(output_file),
                annotate=True,
                domain="story",
            )
