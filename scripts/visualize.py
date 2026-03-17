#!/usr/bin/env python3
"""Embedding + cluster visualizations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.lm import HCLMD
from training.checkpointing import load_checkpoint
from eval.embedding_viz import visualize_embeddings


def main():
    parser = argparse.ArgumentParser(description="Visualize HCLM-D embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--output", type=str, default="figures/embeddings.png")
    parser.add_argument("--n_tokens", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cpu"  # Visualization runs on CPU

    from scripts.train import load_config
    config = load_config(args.config)

    model_config = ModelConfig.from_dict(config)
    model = HCLMD(model_config)
    load_checkpoint(args.checkpoint, model, device=device)

    visualize_embeddings(
        model,
        method=args.method,
        output_path=args.output,
        n_tokens=args.n_tokens,
    )
    logging.info("Visualization saved to %s", args.output)


if __name__ == "__main__":
    main()
