#!/usr/bin/env python3
"""Entry point: python scripts/train.py --config configs/base.yaml"""

from __future__ import annotations

import argparse
import logging
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

    train_tokens_path = data_dir / "train_tokens.pt"
    val_tokens_path = data_dir / "val_tokens.pt"

    if not train_tokens_path.exists():
        logging.error("Training data not found. Run with --phase prep first.")
        sys.exit(1)

    train_tokens = torch.load(train_tokens_path, weights_only=True)
    train_dataset = HCLMDataset(train_tokens, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = None
    if val_tokens_path.exists():
        val_tokens = torch.load(val_tokens_path, weights_only=True)
        val_dataset = HCLMDataset(val_tokens, seq_len=seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)

    if args.resume:
        from training.checkpointing import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, trainer.optimizer)
        trainer.global_step = ckpt.get("step", 0)

    trainer.train()


if __name__ == "__main__":
    main()
