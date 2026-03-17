#!/usr/bin/env python3
"""Entry point: python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" """

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.lm import HCLMD
from data.tokenizer import load_tokenizer
from training.checkpointing import load_checkpoint
from eval.generation import generate_samples


def main():
    parser = argparse.ArgumentParser(description="Generate text with HCLM-D")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config and model
    from scripts.train import load_config
    config = load_config(args.config)

    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = config.get("data", {}).get("tokenizer_path", "data/tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path)

    # Generate
    prompts = [args.prompt] * args.n_samples
    results = generate_samples(
        model, tokenizer, prompts,
        seq_len=args.seq_len,
        sampling_steps=args.steps,
        temperature=args.temperature,
        device=device,
    )

    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated:\n{result['generated']}")


if __name__ == "__main__":
    main()
