#!/usr/bin/env python3
"""Entry point: python scripts/train.py --config configs/base.yaml"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.lm import HCLMD
from data.dataset import HCLMDataset, collate_fn
from data.prep import prepare_data
from training.trainer import Trainer


def _create_loader_with_fallback(
    dataset: HCLMDataset,
    batch_size: int,
    num_workers: int,
    seq_len: int,
    shuffle: bool = True,
    embed_dim: int = 384,
    n_layers: int = 8,
) -> DataLoader:
    """Try to create a DataLoader; reduce batch size on OOM.

    Tests a single forward pass to detect OOM before training starts.
    Tries batch_size → batch_size * 3/4 → batch_size // 2 → min(batch_size, 16).
    """
    min_batch = min(batch_size, 16)
    candidates = sorted(set([batch_size, batch_size * 3 // 4, batch_size // 2, min_batch]), reverse=True)
    candidates = [b for b in candidates if b >= 1]

    for bs in candidates:
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        if not torch.cuda.is_available() or bs == candidates[-1]:
            logging.info("Using batch_size=%d", bs)
            return loader

        # Test if batch fits in GPU memory
        try:
            test_batch = next(iter(loader))
            test_input = test_batch["input_ids"].to("cuda")
            # Rough memory estimate: batch × seq × embed × 4 (activations) × layers
            mem_needed = bs * seq_len * embed_dim * 4 * n_layers * 4  # conservative bytes estimate
            mem_available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            del test_input, test_batch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if mem_needed < mem_available * 0.85:  # 85% safety margin
                logging.info("Using batch_size=%d (fits in GPU memory)", bs)
                return loader
            else:
                logging.info("batch_size=%d may not fit (need ~%.1fGB, available %.1fGB), trying smaller",
                           bs, mem_needed / 1e9, mem_available / 1e9)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning("batch_size=%d caused OOM, trying smaller", bs)
                torch.cuda.empty_cache()
            else:
                raise

    # Fallback to smallest candidate
    logging.info("Falling back to batch_size=%d", min_batch)
    return DataLoader(
        dataset, batch_size=min_batch, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True,
    )


def load_config(path: str) -> dict:
    """Load config, resolving _base_ inheritance."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Handle _base_ inheritance
    if "_base_" in cfg:
        base_path = Path(path).parent / cfg.pop("_base_")
        base_cfg = load_config(str(base_path))
        # Deep merge: child overrides base
        merged = deep_merge(base_cfg, cfg)
        return merged

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Train HCLM-D")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--phase", type=str, default="train", choices=["prep", "train"])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    # Data preparation only
    if args.phase == "prep":
        prepare_data(config)
        return

    # Build model
    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)

    param_counts = model.count_parameters()
    logging.info("Model parameters: %s", param_counts)

    # Load data
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/tinystories"))
    seq_len = config.get("model", {}).get("max_seq_len", 512)
    batch_size = config.get("training", {}).get("batch_size", 64)

    # Auto num_workers
    num_workers_cfg = data_cfg.get("num_workers", 4)
    if num_workers_cfg == "auto":
        import os
        cpu_count = os.cpu_count() or 4
        num_workers = min(8, cpu_count // 2)
        num_workers = max(2, num_workers)
        logging.info("Auto num_workers: %d (detected %d CPUs)", num_workers, cpu_count)
    else:
        num_workers = int(num_workers_cfg)

    train_tokens_path = data_dir / "train_tokens.pt"
    val_tokens_path = data_dir / "val_tokens.pt"

    if not train_tokens_path.exists():
        logging.error("Training data not found. Run with --phase prep first.")
        sys.exit(1)

    train_tokens = torch.load(train_tokens_path, weights_only=True)
    train_dataset = HCLMDataset(train_tokens, seq_len=seq_len)

    # Auto batch size: try configured, fall back on OOM
    embed_dim = config.get("model", {}).get("embed_dim", 384)
    n_layers = config.get("model", {}).get("transformer", {}).get("n_layers", 8)
    train_loader = _create_loader_with_fallback(
        train_dataset, batch_size, num_workers, seq_len, shuffle=True,
        embed_dim=embed_dim, n_layers=n_layers,
    )
    actual_batch_size = train_loader.batch_size

    # Scale LR if batch size differs from config
    configured_batch = config.get("training", {}).get("batch_size", 64)
    if actual_batch_size != configured_batch:
        base_lr = config.get("training", {}).get("lr", 3e-4)
        scaled_lr = base_lr * math.sqrt(actual_batch_size / configured_batch)
        config.setdefault("training", {})["lr"] = scaled_lr
        config["training"]["batch_size"] = actual_batch_size
        logging.info(
            "Batch size adjusted: %d → %d | LR scaled: %.2e → %.2e",
            configured_batch, actual_batch_size, base_lr, scaled_lr,
        )

    val_loader = None
    if val_tokens_path.exists():
        val_tokens = torch.load(val_tokens_path, weights_only=True)
        val_dataset = HCLMDataset(val_tokens, seq_len=seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=max(2, num_workers // 2),
            pin_memory=True,
        )

    # Load tokenizer for sample generation
    tokenizer = None
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer.json")
    if Path(tokenizer_path).exists():
        try:
            from data.tokenizer import load_tokenizer
            tokenizer = load_tokenizer(tokenizer_path)
            logging.info("Tokenizer loaded for sample generation")
        except Exception as e:
            logging.warning("Could not load tokenizer: %s", e)

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    if tokenizer:
        trainer.set_tokenizer(tokenizer)

    if args.resume:
        from training.checkpointing import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, trainer.optimizer)
        trainer.global_step = ckpt.get("step", 0)

    trainer.train()


if __name__ == "__main__":
    main()
