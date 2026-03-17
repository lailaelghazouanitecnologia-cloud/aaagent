"""Multi-source data preparation — English + Spanish + Python code.

Downloads, filters, and tokenizes data from multiple sources into
a single training corpus with configurable domain ratios.

Datasets:
    EN: roneneldan/TinyStories (simple english, good for small models)
    ES: datificate/SomosNLP-ultrachat_200k-es (conversational spanish)
    Python: bigcode/the-stack-dedup (language=python, filtered)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Default domain ratios and token budgets
DEFAULT_RATIOS = {"en": 0.25, "es": 0.25, "python": 0.50}
DEFAULT_TOTAL_TOKENS = 50_000_000  # 50M


# ── Downloaders ──

def _download_english(data_dir: Path, max_chars: int) -> Path:
    """Download English text from TinyStories."""
    from datasets import load_dataset

    out = data_dir / "en_raw.txt"
    if out.exists():
        logger.info("English data already exists at %s", out)
        return out

    logger.info("Downloading English (TinyStories)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)

    chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            chars += len(text)
            if chars >= max_chars:
                break

    logger.info("English: %d chars written to %s", chars, out)
    return out


def _download_spanish(data_dir: Path, max_chars: int) -> Path:
    """Download Spanish text."""
    from datasets import load_dataset

    out = data_dir / "es_raw.txt"
    if out.exists():
        logger.info("Spanish data already exists at %s", out)
        return out

    logger.info("Downloading Spanish (ultrachat_200k-es)...")
    try:
        ds = load_dataset(
            "datificate/SomosNLP-ultrachat_200k-es",
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: use wikipedia spanish
        logger.info("Fallback: downloading Spanish Wikipedia...")
        ds = load_dataset("wikipedia", "20220301.es", split="train", trust_remote_code=True)

    chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            # Try common field names
            text = example.get("text", example.get("content", "")).strip()
            if not text or len(text) < 50:
                continue
            f.write(text + "\n\n")
            chars += len(text)
            if chars >= max_chars:
                break

    logger.info("Spanish: %d chars written to %s", chars, out)
    return out


def _download_python(data_dir: Path, max_chars: int) -> Path:
    """Download Python code from The Stack."""
    from datasets import load_dataset

    out = data_dir / "python_raw.txt"
    if out.exists():
        logger.info("Python data already exists at %s", out)
        return out

    logger.info("Downloading Python code (the-stack-dedup)...")
    try:
        ds = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: codeparrot/github-code
        logger.info("Fallback: downloading from codeparrot/github-code...")
        ds = load_dataset(
            "codeparrot/github-code",
            languages=["Python"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

    chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            code = example.get("content", example.get("code", "")).strip()
            if not code or len(code) < 100 or len(code) > 10000:
                continue
            # Basic quality filter
            if "def " not in code and "class " not in code and "import " not in code:
                continue
            f.write(code + "\n\n")
            chars += len(code)
            if chars >= max_chars:
                break

    logger.info("Python: %d chars written to %s", chars, out)
    return out


# ── Mixing ──

def _mix_corpus(data_dir: Path, ratios: dict[str, float], total_tokens: int) -> Path:
    """Read raw files and write a shuffled mixed corpus."""
    out = data_dir / "mixed_corpus.txt"
    if out.exists():
        logger.info("Mixed corpus already exists at %s", out)
        return out

    logger.info("Mixing corpus with ratios: %s", ratios)

    # Read all domain texts as chunks (paragraph-level)
    chunks: list[tuple[str, str]] = []  # (domain, text_chunk)

    for domain in ["en", "es", "python"]:
        raw_file = data_dir / f"{domain}_raw.txt"
        if not raw_file.exists():
            logger.warning("Missing %s, skipping", raw_file)
            continue

        with open(raw_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into paragraphs/documents
        docs = [d.strip() for d in text.split("\n\n") if d.strip()]
        for doc in docs:
            chunks.append((domain, doc))

    # Shuffle
    random.shuffle(chunks)

    # Write mixed — we'll let the tokenizer handle the final token count
    with open(out, "w", encoding="utf-8") as f:
        for domain, text in chunks:
            f.write(text + "\n\n")

    logger.info("Mixed corpus: %d documents written to %s", len(chunks), out)
    return out


# ── Main pipeline ──

def download_multilingual(
    data_dir: str = "data/multilingual",
    total_tokens: int = DEFAULT_TOTAL_TOKENS,
    ratios: dict[str, float] | None = None,
) -> Path:
    """Download all 3 domain datasets.

    Estimates ~4 chars per token for text, ~3 chars per token for code.
    """
    ratios = ratios or DEFAULT_RATIOS
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Estimate chars needed per domain
    en_tokens = int(total_tokens * ratios.get("en", 0.25))
    es_tokens = int(total_tokens * ratios.get("es", 0.25))
    py_tokens = int(total_tokens * ratios.get("python", 0.50))

    # ~4 chars/token for text, ~3 for code (conservative)
    en_chars = en_tokens * 4
    es_chars = es_tokens * 5  # Spanish words are longer
    py_chars = py_tokens * 3

    _download_english(data_path, en_chars)
    _download_spanish(data_path, es_chars)
    _download_python(data_path, py_chars)
    _mix_corpus(data_path, ratios, total_tokens)

    return data_path


def prepare_multilingual(config: dict) -> None:
    """Full multilingual data preparation: download, tokenize, split."""
    from data.tokenizer import build_tokenizer, load_tokenizer

    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "data/multilingual")
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer_multilingual.json")
    vocab_size = config.get("model", {}).get("vocab_size", 32768)
    total_tokens = data_cfg.get("total_tokens", DEFAULT_TOTAL_TOKENS)
    ratios = data_cfg.get("domain_ratios", DEFAULT_RATIOS)

    # 1. Download
    data_path = download_multilingual(data_dir, total_tokens, ratios)
    mixed_corpus = data_path / "mixed_corpus.txt"

    # 2. Build tokenizer on mixed corpus (covers all 3 domains)
    if not Path(tokenizer_path).exists():
        logger.info("Training multilingual tokenizer (vocab=%d)...", vocab_size)
        corpus_files = []
        for domain in ["en_raw.txt", "es_raw.txt", "python_raw.txt"]:
            p = data_path / domain
            if p.exists():
                corpus_files.append(str(p))
        build_tokenizer(
            corpus_files=corpus_files,
            vocab_size=vocab_size,
            save_path=tokenizer_path,
        )
    else:
        logger.info("Tokenizer already exists at %s", tokenizer_path)

    # 3. Tokenize mixed corpus
    tokenizer = load_tokenizer(tokenizer_path)

    for split_name, split_ratio in [("train", 0.95), ("val", 0.05)]:
        output_file = data_path / f"{split_name}_tokens.pt"
        if output_file.exists():
            logger.info("%s tokens already exist at %s", split_name, output_file)
            continue

        logger.info("Tokenizing %s split...", split_name)
        with open(mixed_corpus, "r", encoding="utf-8") as f:
            text = f.read()

        # Split text
        split_point = int(len(text) * 0.95)
        if split_name == "train":
            chunk = text[:split_point]
        else:
            chunk = text[split_point:]

        # Tokenize in chunks
        chunk_size = 1_000_000
        all_ids: list[int] = []
        for start in range(0, len(chunk), chunk_size):
            part = chunk[start : start + chunk_size]
            encoding = tokenizer.encode(part)
            all_ids.extend(encoding.ids)

        token_tensor = torch.tensor(all_ids, dtype=torch.long)
        torch.save(token_tensor, output_file)
        logger.info("%s: %d tokens saved to %s", split_name, len(token_tensor), output_file)
