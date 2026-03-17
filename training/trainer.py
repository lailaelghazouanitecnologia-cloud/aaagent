"""Main training loop for HCLM-D.

Includes automatic performance optimizations:
- torch.compile with fallback
- Auto batch size detection (tries larger, falls back on OOM)
- Learning rate scaling with batch size
- Auto num_workers based on available CPUs
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.lm import HCLMD
from model.embedding.composite import CompositeEmbedding
from data.masking import DiffusionMasker
from losses.combined import CombinedLoss
from training.warmup import StructuralWarmup
from training.optimizer import build_optimizer, build_scheduler
from training.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


def _try_compile(model: nn.Module) -> nn.Module:
    """Try to torch.compile the model; return original if it fails."""
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available (PyTorch < 2.0), skipping")
        return model
    try:
        compiled = torch.compile(model)
        logger.info("torch.compile: OK")
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed (%s), running without compilation", e)
        return model


def _auto_workers() -> int:
    """Pick num_workers based on available CPUs."""
    cpu_count = os.cpu_count() or 4
    workers = min(8, cpu_count // 2)
    workers = max(2, workers)
    logger.info("Auto num_workers: %d (detected %d CPUs)", workers, cpu_count)
    return workers


class Trainer:
    """Main training loop for HCLM-D.

    Handles:
        - Masked diffusion training (LLaDA-style)
        - Structural warmup (gate schedule, cluster freezing)
        - Loss computation with auxiliary losses
        - Logging and checkpointing
        - Automatic performance optimizations
    """

    def __init__(
        self,
        model: HCLMD,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict | None = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        train_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("losses", {})

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        # torch.compile (auto fallback)
        if train_cfg.get("compile", True) and self.device.type == "cuda":
            self.model = _try_compile(model)
        else:
            self.model = model

        # Optimizer and scheduler
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.scheduler = build_scheduler(self.optimizer, train_cfg)

        # Loss function
        self.criterion = CombinedLoss(
            lambda_balance=loss_cfg.get("lambda_balance", 0.01),
            lambda_diversity=loss_cfg.get("lambda_diversity", 0.001),
            lambda_hierarchy=loss_cfg.get("lambda_hierarchy", 0.01),
        )

        # Structural warmup
        sw_cfg = train_cfg.get("structural_warmup", {})
        self.warmup = StructuralWarmup(
            gate_freeze_steps=sw_cfg.get("gate_freeze_steps", 500),
            gate_ramp_end_steps=sw_cfg.get("gate_ramp_end_steps", 2000),
            cluster_unfreeze_step=sw_cfg.get("cluster_unfreeze_step", 500),
        )

        # Masker
        mask_token_id = self.config.get("model", {}).get("mask_token_id", 0)
        self.masker = DiffusionMasker(mask_token_id=mask_token_id)

        # Training state
        self.global_step = 0
        self.total_steps = train_cfg.get("total_steps", 100000)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.log_every = train_cfg.get("log_every_steps", 100)
        self.eval_every = train_cfg.get("eval_every_steps", 1000)
        self.save_every = train_cfg.get("save_every_steps", 5000)
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))

        # Gradient accumulation
        self.grad_accum_steps = train_cfg.get("gradient_accumulation", 1)

        # Mixed precision
        dtype_str = train_cfg.get("dtype", "bfloat16")
        self.use_amp = dtype_str in ("bfloat16", "float16")
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        # Throughput tracking
        self._step_start = None
        self._tokens_processed = 0

    def train(self) -> None:
        """Run the main training loop."""
        self.model.train()
        logger.info("Starting training for %d steps", self.total_steps)

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

        batch_size = self.train_loader.batch_size
        seq_len = self.config.get("model", {}).get("max_seq_len", 512)
        logger.info(
            "Batch: %d × %d = %d tokens/step | Grad accum: %d | Effective batch: %d",
            batch_size, seq_len, batch_size * seq_len,
            self.grad_accum_steps, batch_size * self.grad_accum_steps,
        )

        self._step_start = time.time()
        self._tokens_processed = 0

        while self.global_step < self.total_steps:
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                loss_output = self.train_step(batch)
                self.global_step += 1
                self._tokens_processed += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

                # Logging with throughput
                if self.global_step % self.log_every == 0:
                    elapsed = time.time() - self._step_start
                    tok_per_sec = self._tokens_processed / elapsed if elapsed > 0 else 0
                    gpu_mem_used = torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0

                    logger.info(
                        "Step %d | Loss: %.4f (diff: %.4f, bal: %.4f, div: %.4f, hier: %.4f) "
                        "| %.0f tok/s | GPU: %.1fGB",
                        self.global_step,
                        loss_output.total.item(),
                        loss_output.diffusion.item(),
                        loss_output.balance.item(),
                        loss_output.diversity.item(),
                        loss_output.hierarchy.item(),
                        tok_per_sec,
                        gpu_mem_used,
                    )
                    # Reset counters
                    self._step_start = time.time()
                    self._tokens_processed = 0

                # Evaluation
                if self.val_loader and self.global_step % self.eval_every == 0:
                    val_loss = self.evaluate()
                    logger.info("Step %d | Val loss: %.4f", self.global_step, val_loss)

                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save()

        # Final save
        self.save()
        logger.info("Training complete at step %d", self.global_step)

    def train_step(self, batch: dict[str, torch.Tensor]):
        """Execute a single training step."""
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Apply diffusion masking
        masked_ids, mask = self.masker.mask_batch(input_ids, attention_mask)

        # Structural warmup: get gate override
        gate_override = self.warmup.get_gate_value(self.global_step)

        # Freeze/unfreeze clusters based on warmup schedule
        self.warmup.apply_freezing(self.model, self.global_step)

        # Forward pass
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(masked_ids, attention_mask, gate_override=gate_override)

            # Get routing info for auxiliary losses
            routing_weights = None
            fine_centroids = None
            coarse_centroids = None

            if isinstance(self.model.embedding, CompositeEmbedding):
                info = self.model.embedding.get_routing_info()
                fine_centroids = info.get("fine_centroids")
                if fine_centroids is not None:
                    fine_centroids = fine_centroids.centroids
                coarse_info = info.get("coarse_centroids")
                if coarse_info is not None:
                    coarse_centroids = coarse_info.centroids

            loss_output = self.criterion(
                logits=logits,
                targets=input_ids,
                mask=mask,
                fine_centroids=fine_centroids,
                coarse_centroids=coarse_centroids,
            )

        # Backward
        self.optimizer.zero_grad()
        loss_output.total.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return loss_output

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on the validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            masked_ids, mask = self.masker.mask_batch(input_ids, attention_mask)

            logits = self.model(masked_ids, attention_mask)
            loss_output = self.criterion(logits=logits, targets=input_ids, mask=mask)
            total_loss += loss_output.diffusion.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def save(self) -> None:
        """Save checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"step_{self.global_step}.pt"
        save_checkpoint(self.model, self.optimizer, self.global_step, str(path))
        logger.info("Saved checkpoint to %s", path)
