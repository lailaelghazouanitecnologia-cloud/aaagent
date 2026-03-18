"""Multi-source data preparation — English + Spanish + Python code.

Downloads, filters, annotates structure, and tokenizes data from multiple
sources into a single training corpus with configurable domain ratios.

Default ratios: 70% English / 20% Spanish / 10% Python

Datasets:
    EN:
        - HuggingFaceFW/fineweb-edu-score-2 (high-quality educational web text)
        - roneneldan/TinyStories (simple stories, narrative structure)
    ES:
        - oscar-corpus/OSCAR-2301 (language=es, web corpus)
        - wikipedia/20220301.es (fallback)
    Python:
        - bigcode/the-stack-v2-train-smol-ids (filtered to Python)
        - bigcode/the-stack-dedup (fallback)

Each document is:
    1. Filtered by quality (min length, dedup, structure for code)
    2. Annotated with structure tokens ([BLOCK_START], [SLOT_START], etc.)
    3. Prefixed with language/domain tags ([LANG_EN], [DOMAIN_CODE], etc.)
    4. Mixed respecting target ratios via weighted sampling
    5. Split train/val at document boundaries (no leakage)
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Default domain ratios and token budgets
DEFAULT_RATIOS = {"en": 0.70, "es": 0.20, "python": 0.10}
DEFAULT_TOTAL_TOKENS = 50_000_000  # 50M

# Chars-per-token estimates by domain (conservative, slightly over-download)
CHARS_PER_TOKEN = {
    "en": 4.5,       # FineWeb-Edu: technical/educational English
    "es": 5.0,       # Spanish words are longer on average
    "python": 3.0,   # Code: shorter tokens, more symbols
}

# English sub-sources with proportions (within the EN budget)
EN_SOURCES = [
    {
        "name": "FineWeb-Edu",
        "dataset": "HuggingFaceFW/fineweb-edu-score-2",
        "split": "train",
        "field": "text",
        "ratio": 0.93,        # 93% — high-quality educational web text
        "streaming": True,
        "min_len": 200,
        "domain": "wiki",     # annotation domain hint
    },
    {
        "name": "TinyStories",
        "dataset": "roneneldan/TinyStories",
        "split": "train",
        "field": "text",
        "ratio": 0.07,        # 7% — simple narrative structure
        "streaming": False,
        "min_len": 50,
        "domain": "story",
    },
]


# ── Downloaders ──

def _download_en_source(source: dict, data_dir: Path, max_chars: int) -> int:
    """Download a single English source. Returns chars written."""
    from datasets import load_dataset

    name = source["name"]
    out = data_dir / f"en_{name.lower().replace(' ', '_').replace('-', '_')}.txt"

    if out.exists():
        size = out.stat().st_size
        logger.info("  %s already exists (%d chars)", name, size)
        return size

    logger.info("  Downloading %s (budget: %d chars)...", name, max_chars)

    kwargs = {
        "path": source["dataset"],
        "split": source["split"],
        "trust_remote_code": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    if source.get("streaming"):
        kwargs["streaming"] = True

    try:
        ds = load_dataset(**kwargs)
    except Exception as e:
        logger.warning("  Failed to load %s: %s — skipping", name, e)
        return 0

    field = source["field"]
    min_len = source.get("min_len", 50)
    chars = 0

    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get(field, "").strip()
            if not text or len(text) < min_len:
                continue
            # Cap individual documents at 8000 chars for diversity
            if len(text) > 8000:
                text = text[:8000]
            f.write(text + "\n\n")
            chars += len(text)
            if chars >= max_chars:
                break

    logger.info("  %s: %d chars written", name, chars)
    return chars


def _download_english(data_dir: Path, max_chars: int) -> Path:
    """Download English text from multiple sources for diversity."""
    out = data_dir / "en_raw.txt"
    if out.exists():
        logger.info("English data already exists at %s", out)
        return out

    logger.info("Downloading English (multi-source, budget: %d chars)...", max_chars)

    # Download each source with its proportion of the budget
    for source in EN_SOURCES:
        source_budget = int(max_chars * source["ratio"])
        _download_en_source(source, data_dir, source_budget)

    # Merge all EN source files into en_raw.txt, interleaved for diversity
    logger.info("Merging English sources...")
    source_chunks: list[list[str]] = []

    for source in EN_SOURCES:
        name = source["name"].lower().replace(" ", "_").replace("-", "_")
        src_file = data_dir / f"en_{name}.txt"
        if src_file.exists():
            with open(src_file, "r", encoding="utf-8") as f:
                docs = [d.strip() for d in f.read().split("\n\n") if d.strip()]
            source_chunks.append(docs)

    if not source_chunks:
        logger.error("No English sources downloaded!")
        # Write empty file to avoid downstream errors
        out.touch()
        return out

    # Interleave: weighted round-robin respecting source ratios
    merged: list[str] = []
    indices = [0] * len(source_chunks)
    total_docs = sum(len(c) for c in source_chunks)

    while len(merged) < total_docs:
        added = False
        for i, chunks in enumerate(source_chunks):
            if indices[i] < len(chunks):
                merged.append(chunks[indices[i]])
                indices[i] += 1
                added = True
        if not added:
            break

    # Final shuffle for randomness
    random.shuffle(merged)

    with open(out, "w", encoding="utf-8") as f:
        for doc in merged:
            f.write(doc + "\n\n")

    total_chars = sum(len(d) for d in merged)
    logger.info("English merged: %d docs, %d chars → %s", len(merged), total_chars, out)
    return out


def _download_spanish(data_dir: Path, max_chars: int) -> Path:
    """Download Spanish text from OSCAR 2301."""
    from datasets import load_dataset

    out = data_dir / "es_raw.txt"
    if out.exists():
        logger.info("Spanish data already exists at %s", out)
        return out

    # Try OSCAR 2301 first (large, diverse web corpus)
    logger.info("Downloading Spanish (OSCAR 2301, budget: %d chars)...", max_chars)
    ds = None
    for attempt_name, attempt_kwargs in [
        ("OSCAR-2301", {
            "path": "oscar-corpus/OSCAR-2301",
            "name": "es",
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }),
        ("Wikipedia ES", {
            "path": "wikipedia",
            "name": "20220301.es",
            "split": "train",
            "trust_remote_code": True,
        }),
    ]:
        try:
            logger.info("  Trying %s...", attempt_name)
            ds = load_dataset(**attempt_kwargs)
            logger.info("  %s loaded successfully", attempt_name)
            break
        except Exception as e:
            logger.warning("  Failed to load %s: %s", attempt_name, e)
            continue

    if ds is None:
        logger.error("All Spanish sources failed!")
        out.touch()
        return out

    chars = 0
    seen_hashes: set[str] = set()  # Basic dedup

    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", example.get("content", "")).strip()
            if not text or len(text) < 100:
                continue
            # Cap at 8000 chars
            if len(text) > 8000:
                text = text[:8000]
            # Basic dedup: skip if first 200 chars already seen
            doc_hash = hashlib.md5(text[:200].encode()).hexdigest()
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)

            f.write(text + "\n\n")
            chars += len(text)
            if chars >= max_chars:
                break

    logger.info("Spanish: %d chars, %d unique docs → %s", chars, len(seen_hashes), out)
    return out


def _download_python(data_dir: Path, max_chars: int) -> Path:
    """Download Python code from The Stack v2."""
    from datasets import load_dataset

    out = data_dir / "python_raw.txt"
    if out.exists():
        logger.info("Python data already exists at %s", out)
        return out

    logger.info("Downloading Python code (budget: %d chars)...", max_chars)

    # Try sources in order of preference
    ds = None
    for attempt_name, attempt_kwargs in [
        ("The Stack v2 (smol)", {
            "path": "bigcode/the-stack-v2-train-smol-ids",
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }),
        ("The Stack dedup", {
            "path": "bigcode/the-stack-dedup",
            "data_dir": "data/python",
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }),
        ("codeparrot/github-code", {
            "path": "codeparrot/github-code",
            "languages": ["Python"],
            "split": "train",
            "streaming": True,
            "trust_remote_code": True,
        }),
    ]:
        try:
            logger.info("  Trying %s...", attempt_name)
            ds = load_dataset(**attempt_kwargs)
            logger.info("  %s loaded successfully", attempt_name)
            break
        except Exception as e:
            logger.warning("  Failed to load %s: %s", attempt_name, e)
            continue

    if ds is None:
        logger.error("All Python sources failed!")
        out.touch()
        return out

    chars = 0
    with open(out, "w", encoding="utf-8") as f:
        for example in ds:
            # Handle different field names across datasets
            code = ""
            for field in ("content", "code", "text"):
                code = example.get(field, "")
                if code:
                    break
            code = code.strip()
            if not code or len(code) < 100 or len(code) > 10000:
                continue

            # Filter: only files with Python language tag or real Python content
            lang = example.get("language", example.get("lang", "Python"))
            if isinstance(lang, str) and lang.lower() != "python":
                continue

            # Quality filter: must have at least one def/class/import
            if "def " not in code and "class " not in code and "import " not in code:
                continue

            # Skip auto-generated files
            first_line = code.split("\n")[0].lower()
            if "auto-generated" in first_line or "do not edit" in first_line:
                continue

            f.write(code + "\n\n")
            chars += len(code)
            if chars >= max_chars:
                break

    logger.info("Python: %d chars written → %s", chars, out)
    return out


# ── Structure annotation ──

def _annotate_document(text: str, domain: str) -> str:
    """Annotate a single document with structure tokens.

    Uses the annotation functions from data.prep for domain-specific
    structure detection (blocks, slots).
    """
    from data.prep import annotate_structure
    return annotate_structure(text, domain=domain)


def _get_lang_tag(domain: str) -> str:
    """Get the language tag prefix for a domain."""
    tags = {
        "en": "[LANG_EN]",
        "es": "[LANG_ES]",
        "python": "[LANG_PY]",
    }
    return tags.get(domain, "[LANG_EN]")


def _get_domain_tag(domain: str, annotation_domain: str) -> str:
    """Get the domain tag prefix."""
    tags = {
        "story": "[DOMAIN_STORY]",
        "wiki": "[DOMAIN_WIKI]",
        "qa": "[DOMAIN_QA]",
        "code": "[DOMAIN_CODE]",
    }
    return tags.get(annotation_domain, "")


# ── Mixing ──

def _mix_corpus(
    data_dir: Path,
    ratios: dict[str, float],
    total_tokens: int,
) -> tuple[Path, Path]:
    """Read raw files, annotate, tag, mix by ratio, and split train/val.

    Returns (train_corpus_path, val_corpus_path).
    """
    train_out = data_dir / "mixed_train.txt"
    val_out = data_dir / "mixed_val.txt"

    if train_out.exists() and val_out.exists():
        logger.info("Mixed corpus already exists at %s", data_dir)
        return train_out, val_out

    logger.info("Mixing corpus with ratios: %s", ratios)

    # Annotation domain mapping
    annotation_domains = {
        "en": "wiki",       # FineWeb-Edu is mostly encyclopedic/educational
        "es": "wiki",       # OSCAR is web text, treat like wiki
        "python": "code",   # Python code
    }

    # Read and annotate documents per domain
    domain_docs: dict[str, list[str]] = {}

    for domain in ["en", "es", "python"]:
        raw_file = data_dir / f"{domain}_raw.txt"
        if not raw_file.exists() or raw_file.stat().st_size == 0:
            logger.warning("Missing or empty %s, skipping", raw_file)
            domain_docs[domain] = []
            continue

        logger.info("Processing %s...", domain)
        ann_domain = annotation_domains[domain]
        lang_tag = _get_lang_tag(domain)
        domain_tag = _get_domain_tag(domain, ann_domain)
        prefix = f"{lang_tag} {domain_tag}".strip()

        docs: list[str] = []
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        raw_docs = [d.strip() for d in raw_text.split("\n\n") if d.strip()]
        del raw_text  # Free memory

        for doc in raw_docs:
            # Annotate structure (blocks, slots)
            annotated = _annotate_document(doc, domain=ann_domain)
            # Prepend language + domain tags
            tagged = f"{prefix} {annotated}"
            docs.append(tagged)

        domain_docs[domain] = docs
        logger.info("  %s: %d documents annotated and tagged", domain, len(docs))

    # Weighted sampling to respect ratios
    # Compute how many docs we need from each domain
    total_docs = sum(len(d) for d in domain_docs.values())
    if total_docs == 0:
        logger.error("No documents collected from any domain!")
        train_out.touch()
        val_out.touch()
        return train_out, val_out

    # Sample docs weighted by ratio
    all_docs: list[str] = []
    for domain, docs in domain_docs.items():
        target_ratio = ratios.get(domain, 0.0)
        if not docs or target_ratio <= 0:
            continue
        # How many docs to sample (with replacement if needed)
        target_count = int(total_docs * target_ratio)
        if len(docs) >= target_count:
            sampled = random.sample(docs, target_count)
        else:
            # Not enough docs: use all + sample extra with replacement
            sampled = docs.copy()
            remaining = target_count - len(docs)
            sampled.extend(random.choices(docs, k=remaining))
        all_docs.extend(sampled)
        logger.info("  %s: sampled %d docs (target ratio %.0f%%)",
                     domain, len(sampled), target_ratio * 100)

    # Shuffle
    random.shuffle(all_docs)

    # Split train/val at document boundaries (no leakage!)
    val_count = max(1, int(len(all_docs) * 0.05))
    train_docs = all_docs[:-val_count]
    val_docs = all_docs[-val_count:]

    # Write
    with open(train_out, "w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(doc + "\n\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(doc + "\n\n")

    logger.info(
        "Mixed corpus: %d train docs, %d val docs → %s",
        len(train_docs), len(val_docs), data_dir,
    )
    return train_out, val_out


# ── Main pipeline ──

def download_multilingual(
    data_dir: str = "data/multilingual",
    total_tokens: int = DEFAULT_TOTAL_TOKENS,
    ratios: dict[str, float] | None = None,
) -> Path:
    """Download all 3 domain datasets.

    Validates ratios sum to 1.0 and estimates char budgets per domain
    using calibrated chars-per-token estimates.
    """
    ratios = ratios or DEFAULT_RATIOS

    # Validate ratios
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 0.01:
        logger.warning(
            "Domain ratios sum to %.3f (expected 1.0). Normalizing.", ratio_sum
        )
        ratios = {k: v / ratio_sum for k, v in ratios.items()}

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Estimate chars needed per domain using calibrated estimates
    en_tokens = int(total_tokens * ratios.get("en", 0.70))
    es_tokens = int(total_tokens * ratios.get("es", 0.20))
    py_tokens = int(total_tokens * ratios.get("python", 0.10))

    en_chars = int(en_tokens * CHARS_PER_TOKEN["en"])
    es_chars = int(es_tokens * CHARS_PER_TOKEN["es"])
    py_chars = int(py_tokens * CHARS_PER_TOKEN["python"])

    logger.info(
        "Token budget: EN=%dM (%dM chars), ES=%dM (%dM chars), PY=%dM (%dM chars)",
        en_tokens // 1_000_000, en_chars // 1_000_000,
        es_tokens // 1_000_000, es_chars // 1_000_000,
        py_tokens // 1_000_000, py_chars // 1_000_000,
    )

    _download_english(data_path, en_chars)
    _download_spanish(data_path, es_chars)
    _download_python(data_path, py_chars)

    return data_path


def prepare_multilingual(config: dict) -> None:
    """Full multilingual data preparation: download, tokenize, split.

    The vocab_size in config should be the BPE vocab size (excluding the
    100 special tokens which are added automatically by build_tokenizer).
    """
    from data.tokenizer import build_tokenizer, load_tokenizer
    from data.tokens import NUM_SPECIAL_TOKENS

    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "data/multilingual")
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer_multilingual.json")
    total_tokens = data_cfg.get("total_tokens", DEFAULT_TOTAL_TOKENS)
    ratios = data_cfg.get("domain_ratios", DEFAULT_RATIOS)

    # vocab_size in config is total (BPE + special). build_tokenizer expects BPE-only.
    config_vocab = config.get("model", {}).get("vocab_size", 32868)
    bpe_vocab_size = config_vocab - NUM_SPECIAL_TOKENS
    if bpe_vocab_size < 1000:
        logger.warning(
            "vocab_size=%d seems too small (BPE would be %d). "
            "Using default BPE vocab of 32768.",
            config_vocab, bpe_vocab_size,
        )
        bpe_vocab_size = 32768

    # 1. Download raw data
    data_path = download_multilingual(data_dir, total_tokens, ratios)

    # 2. Build tokenizer on raw corpus (covers all 3 domains)
    tokenizer_file = Path(tokenizer_path)
    if not tokenizer_file.exists():
        logger.info("Training multilingual tokenizer (BPE vocab=%d + %d special)...",
                     bpe_vocab_size, NUM_SPECIAL_TOKENS)
        corpus_files = []
        for domain_file in ["en_raw.txt", "es_raw.txt", "python_raw.txt"]:
            p = data_path / domain_file
            if p.exists() and p.stat().st_size > 0:
                corpus_files.append(str(p))
        if not corpus_files:
            raise RuntimeError("No corpus files found for tokenizer training!")
        build_tokenizer(
            corpus_files=corpus_files,
            vocab_size=bpe_vocab_size,
            save_path=tokenizer_path,
        )
    else:
        # Validate existing tokenizer matches expected vocab
        existing = load_tokenizer(tokenizer_path)
        expected_total = bpe_vocab_size + NUM_SPECIAL_TOKENS
        actual_total = existing.get_vocab_size()
        if actual_total != expected_total:
            logger.warning(
                "Existing tokenizer vocab=%d doesn't match expected=%d. "
                "Delete %s to retrain.",
                actual_total, expected_total, tokenizer_path,
            )
        logger.info("Tokenizer already exists at %s (vocab=%d)", tokenizer_path, actual_total)

    # 3. Mix, annotate, tag, and split into train/val
    train_corpus, val_corpus = _mix_corpus(data_path, ratios, total_tokens)

    # 4. Tokenize each split
    tokenizer = load_tokenizer(tokenizer_path)

    for split_name, corpus_file in [("train", train_corpus), ("val", val_corpus)]:
        output_file = data_path / f"{split_name}_tokens.pt"
        if output_file.exists():
            logger.info("%s tokens already exist at %s", split_name, output_file)
            continue

        if not corpus_file.exists() or corpus_file.stat().st_size == 0:
            logger.warning("Corpus %s is empty, skipping", corpus_file)
            continue

        logger.info("Tokenizing %s split from %s...", split_name, corpus_file)

        # Tokenize in chunks to avoid memory issues
        chunk_size = 500_000  # chars per chunk (smaller for safety)
        all_ids: list[int] = []

        with open(corpus_file, "r", encoding="utf-8") as f:
            text = f.read()

        total_chunks = (len(text) + chunk_size - 1) // chunk_size
        for i, start in enumerate(range(0, len(text), chunk_size)):
            part = text[start: start + chunk_size]
            encoding = tokenizer.encode(part)
            all_ids.extend(encoding.ids)
            if (i + 1) % 10 == 0 or i == total_chunks - 1:
                logger.info(
                    "  Tokenizing %s: %d/%d chunks — %d tokens so far",
                    split_name, i + 1, total_chunks, len(all_ids),
                )

        token_tensor = torch.tensor(all_ids, dtype=torch.long)
        torch.save(token_tensor, output_file)
        logger.info("%s: %d tokens saved to %s", split_name, len(token_tensor), output_file)
