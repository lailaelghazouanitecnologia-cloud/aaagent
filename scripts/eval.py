#!/usr/bin/env python3
"""Entry point: python scripts/eval.py --checkpoint checkpoints/best.pt"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.lm import HCLMD
from data.dataset import HCLMDataset, collate_fn
from training.checkpointing import load_checkpoint
from eval.perplexity import compute_nll


def main():
    parser = argparse.ArgumentParser(description="Evaluate HCLM-D")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    from scripts.train import load_config
    config = load_config(args.config)

    # Build and load model
    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    # Load validation data
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/tinystories"))
    seq_len = config.get("model", {}).get("max_seq_len", 512)
    batch_size = config.get("training", {}).get("batch_size", 64)

    val_tokens = torch.load(data_dir / "val_tokens.pt", weights_only=True)
    val_dataset = HCLMDataset(val_tokens, seq_len=seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Evaluate
    results = compute_nll(model, val_loader, device=device)
    logging.info("Results: %s", results)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
