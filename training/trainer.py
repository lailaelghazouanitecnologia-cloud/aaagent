"""Main training loop for HCLM-D (v0.2).

Includes:
- torch.compile with fallback
- Meta-optimizer integration (Phase 5)
- Dashboard reporter (real-time metrics)
- Sample generation during training
- Auto eval (perplexity) at checkpoints
- Throughput logging
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


class Trainer:
    """Main training loop for HCLM-D.

    Handles:
        - Masked diffusion training (LLaDA-style)
        - Structural warmup (gate schedule, cluster freezing)
        - Loss computation with auxiliary losses
        - Meta-optimizer (optional, Phase 5)
        - Dashboard reporting (optional)
        - Sample generation during training
        - Logging and checkpointing
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

        # Keep reference to unwrapped model for generation/meta
        self._raw_model = model

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
            coarse_unfreeze_step=sw_cfg.get("coarse_unfreeze_step", 3000),
            temp_anneal_start=sw_cfg.get("temp_anneal_start", 2000),
            temp_anneal_end=sw_cfg.get("temp_anneal_end", 20000),
            temp_start=sw_cfg.get("temp_start", 1.0),
            temp_end=sw_cfg.get("temp_end", 0.3),
            loss_ramp_start=sw_cfg.get("loss_ramp_start", 2000),
            loss_ramp_end=sw_cfg.get("loss_ramp_end", 20000),
            loss_multiplier_min=sw_cfg.get("loss_multiplier_min", 0.1),
            loss_multiplier_hierarchy_max=sw_cfg.get("loss_multiplier_hierarchy_max", 10.0),
            loss_multiplier_balance_max=sw_cfg.get("loss_multiplier_balance_max", 5.0),
            loss_multiplier_diversity_max=sw_cfg.get("loss_multiplier_diversity_max", 10.0),
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
        self.generate_every = train_cfg.get("generate_every_steps", 5000)

        # Gradient accumulation
        self.grad_accum_steps = train_cfg.get("gradient_accumulation", 1)

        # Mixed precision
        dtype_str = train_cfg.get("dtype", "bfloat16")
        self.use_amp = dtype_str in ("bfloat16", "float16")
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

        # Throughput tracking
        self._step_start = None
        self._tokens_processed = 0
        self._best_loss = float("inf")

        # ── Meta-optimizer (optional) ──
        meta_cfg = train_cfg.get("meta", {})
        self.meta_enabled = meta_cfg.get("enabled", False)
        self.meta_optimizer = None
        self.meta_scheduler = None
        self.grad_stats = None

        if self.meta_enabled:
            self._init_meta_optimizer(meta_cfg)

        # ── Dashboard reporter (optional) ──
        self.reporter = None
        self._init_dashboard_reporter(train_cfg)

        # ── Generation config ──
        self.sample_prompts = train_cfg.get("sample_prompts", [
            "Once upon a time",
            "The little girl",
            "There was a big",
        ])
        self.tokenizer = None  # Set externally via set_tokenizer()

    def _init_meta_optimizer(self, meta_cfg: dict) -> None:
        """Initialize meta-optimizer if enabled."""
        try:
            from meta.meta_model import MetaOptimizer
            from meta.groups import get_parameter_groups
            from meta.scheduler import MetaScheduler
            from meta.stats import GradientStats

            param_groups = get_parameter_groups(self._raw_model)
            self.meta_optimizer = MetaOptimizer(
                param_groups=param_groups,
                meta_lr=meta_cfg.get("meta_lr", 1e-4),
                hidden_dim=meta_cfg.get("hidden_dim", 128),
                update_every=meta_cfg.get("update_every", 100),
            )
            self.meta_scheduler = MetaScheduler(
                enable_after_step=meta_cfg.get("enable_after_step", 100000),
                update_every=meta_cfg.get("update_every", 100),
            )
            self.grad_stats = GradientStats()
            # Move meta model to device
            self.meta_optimizer.meta_model = self.meta_optimizer.meta_model.to(self.device)
            logger.info(
                "Meta-optimizer: ON (activates at step %d, updates every %d steps)",
                meta_cfg.get("enable_after_step", 100000),
                meta_cfg.get("update_every", 100),
            )
        except Exception as e:
            logger.warning("Meta-optimizer init failed (%s), disabling", e)
            self.meta_enabled = False

    def _init_dashboard_reporter(self, train_cfg: dict) -> None:
        """Initialize dashboard reporter if reachable."""
        try:
            from training.dashboard_reporter import DashboardReporter

            run_name = self.config.get("wandb", {}).get("run_name", "default")
            self.reporter = DashboardReporter(
                run=run_name,
                batch_size=train_cfg.get("log_every_steps", 100),
            )
            if not self.reporter._enabled:
                self.reporter = None
        except Exception as e:
            logger.debug("Dashboard reporter not available: %s", e)
            self.reporter = None

    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer for sample generation during training."""
        self.tokenizer = tokenizer

    def train(self) -> None:
        """Run the main training loop."""
        self.model.train()
        logger.info("Starting training for %d steps", self.total_steps)

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

        batch_size = self.train_loader.batch_size
        seq_len = self.config.get("model", {}).get("max_seq_len", 512)
        logger.info(
            "Batch: %d × %d = %d tokens/step | Grad accum: %d | Effective batch: %d",
            batch_size, seq_len, batch_size * seq_len,
            self.grad_accum_steps, batch_size * self.grad_accum_steps,
        )

        # Log feature status
        features = []
        if self.meta_enabled:
            features.append("meta-optimizer")
        if self.reporter:
            features.append("dashboard")
        if self.tokenizer:
            features.append("generation")
        if features:
            logger.info("Active features: %s", ", ".join(features))

        self._step_start = time.time()
        self._tokens_processed = 0

        while self.global_step < self.total_steps:
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                loss_output = self.train_step(batch)
                self.global_step += 1
                self._tokens_processed += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

                loss_val = loss_output.total.item()

                # Track best loss
                if loss_val < self._best_loss:
                    self._best_loss = loss_val

                # ── Meta-optimizer update ──
                if self.meta_enabled and self.meta_scheduler.should_update(self.global_step):
                    self._meta_step(loss_val)

                # ── Logging with throughput ──
                if self.global_step % self.log_every == 0:
                    self._log_step(loss_output)

                # ── Dashboard report ──
                if self.reporter and self.global_step % self.log_every == 0:
                    self._report_to_dashboard(loss_output)

                # ── Evaluation ──
                if self.val_loader and self.global_step % self.eval_every == 0:
                    val_loss = self.evaluate()
                    logger.info("Step %d | Val loss: %.4f", self.global_step, val_loss)
                    if self.reporter:
                        self.reporter.report(
                            step=self.global_step,
                            extra={"val_loss": val_loss},
                        )

                # ── Checkpointing ──
                if self.global_step % self.save_every == 0:
                    self.save()

                # ── Sample generation ──
                if self.global_step % self.generate_every == 0:
                    self._generate_samples()

        # Final save and generation
        self.save()
        self._generate_samples()
        if self.reporter:
            self.reporter.close()
        logger.info("Training complete at step %d", self.global_step)

    def train_step(self, batch: dict[str, torch.Tensor]):
        """Execute a single training step."""
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Apply diffusion masking
        masked_ids, mask = self.masker.mask_batch(input_ids, attention_mask)

        # Structural warmup: get gate override, temperature, and loss multipliers
        gate_override = self.warmup.get_gate_value(self.global_step)
        router_temp = self.warmup.get_router_temperature(self.global_step)
        loss_multipliers = self.warmup.get_loss_multipliers(self.global_step)

        # Freeze/unfreeze clusters based on warmup schedule
        self.warmup.apply_freezing(self.model, self.global_step)

        # Forward pass
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(
                masked_ids, attention_mask,
                gate_override=gate_override,
                router_temperature=router_temp,
            )

            # Get routing info for auxiliary losses
            fine_centroids = None
            coarse_centroids = None
            routing_weights = None
            fine_to_coarse_weights = None

            if isinstance(self.model.embedding, CompositeEmbedding):
                info = self.model.embedding.get_routing_info()
                fine_centroids_mod = info.get("fine_centroids")
                if fine_centroids_mod is not None:
                    fine_centroids = fine_centroids_mod.centroids
                coarse_centroids_mod = info.get("coarse_centroids")
                if coarse_centroids_mod is not None:
                    coarse_centroids = coarse_centroids_mod.centroids
                # Cached routing weights from forward pass
                routing_weights = info.get("fine_weights")
                fine_to_coarse_weights = info.get("coarse_weights")
                # For hierarchy loss: we need [K, M] mapping (average coarse
                # assignment per fine cluster). Approximate from batch data.
                if (
                    routing_weights is not None
                    and fine_to_coarse_weights is not None
                ):
                    # routing_weights: [B, S, K], fine_to_coarse_weights: [B, S, M]
                    # Compute: for each fine cluster k, average coarse assignment
                    # fine_to_coarse: [K, M] = (routing^T @ coarse) / routing.sum
                    rw_flat = routing_weights.reshape(-1, routing_weights.shape[-1])  # [N, K]
                    cw_flat = fine_to_coarse_weights.reshape(-1, fine_to_coarse_weights.shape[-1])  # [N, M]
                    fine_to_coarse_weights = torch.matmul(rw_flat.T, cw_flat)  # [K, M]
                    fine_to_coarse_weights = fine_to_coarse_weights / (rw_flat.sum(dim=0, keepdim=True).T + 1e-8)

            loss_output = self.criterion(
                logits=logits,
                targets=input_ids,
                mask=mask,
                routing_weights=routing_weights,
                fine_centroids=fine_centroids,
                coarse_centroids=coarse_centroids,
                fine_to_coarse_weights=fine_to_coarse_weights,
                loss_multipliers=loss_multipliers,
            )

        # Backward
        self.optimizer.zero_grad()
        loss_output.total.backward()

        # ── Meta-optimizer: collect gradient stats ──
        if self.meta_enabled and self.grad_stats is not None and self.meta_scheduler.is_active(self.global_step):
            from meta.groups import get_parameter_groups
            param_groups = get_parameter_groups(self._raw_model)
            for group_name, params in param_groups.items():
                for p in params:
                    if p.grad is not None:
                        self.grad_stats.update(group_name, p)
                        break  # One representative param per group

        # ── Meta-optimizer: apply LR multipliers ──
        if (self.meta_enabled and self.meta_optimizer is not None
                and self.meta_scheduler.is_active(self.global_step)
                and self.grad_stats is not None):
            multipliers = self.meta_optimizer.get_multipliers(self.grad_stats)
            self._apply_meta_multipliers(multipliers)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return loss_output

    def _meta_step(self, current_loss: float) -> None:
        """Update the meta-optimizer."""
        if self.meta_optimizer is None or self.grad_stats is None:
            return
        try:
            self.meta_optimizer.update(current_loss, self.grad_stats)
            multipliers = self.meta_optimizer.get_multipliers(self.grad_stats)
            logger.info(
                "Step %d | Meta multipliers: %s",
                self.global_step,
                {k: f"{v:.3f}" for k, v in multipliers.items()},
            )
            self.grad_stats.reset()
        except Exception as e:
            logger.warning("Meta-optimizer update failed: %s", e)

    def _apply_meta_multipliers(self, multipliers: dict[str, float]) -> None:
        """Scale gradients by meta-optimizer multipliers."""
        from meta.groups import get_parameter_groups
        param_groups = get_parameter_groups(self._raw_model)
        for group_name, params in param_groups.items():
            mult = multipliers.get(group_name, 1.0)
            if mult != 1.0:
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(mult)

    def _log_step(self, loss_output) -> None:
        """Log training metrics with throughput."""
        elapsed = time.time() - self._step_start
        tok_per_sec = self._tokens_processed / elapsed if elapsed > 0 else 0
        gpu_mem_used = torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0

        # Cluster health info
        cluster_info = ""
        if isinstance(self._raw_model.embedding, CompositeEmbedding):
            try:
                info = self._raw_model.embedding.get_routing_info()
                parts = []

                # Gate statistics
                gate_values = info.get("gate_values")
                if gate_values is not None:
                    parts.append(f"gate: {gate_values.mean().item():.3f}±{gate_values.std().item():.3f}")

                # Alpha and beta (learned mixing weights)
                alpha = info.get("alpha")
                beta = info.get("beta")
                if alpha is not None:
                    parts.append(f"α: {alpha.item():.4f}")
                if beta is not None:
                    parts.append(f"β: {beta.item():.4f}")

                # Router entropy (measures cluster specialization)
                fine_weights = info.get("fine_weights")
                if fine_weights is not None:
                    # H = -sum(p * log(p)), max = log2(K)
                    eps = 1e-8
                    with torch.no_grad():
                        entropy = -(fine_weights * (fine_weights + eps).log2()).sum(dim=-1).mean()
                    max_entropy = math.log2(fine_weights.shape[-1])
                    parts.append(f"H_router: {entropy.item():.2f}/{max_entropy:.2f}")

                # Temperature and phase info
                temp = self.warmup.get_router_temperature(self.global_step)
                phase = self.warmup.get_phase(self.global_step)
                parts.append(f"τ: {temp:.2f}")
                parts.append(f"phase: {phase}")

                if parts:
                    cluster_info = " | " + " | ".join(parts)
            except Exception:
                pass

        logger.info(
            "Step %d | Loss: %.4f (diff: %.4f, bal: %.4f, div: %.4f, hier: %.4f)"
            " | %.0f tok/s | GPU: %.1fGB%s",
            self.global_step,
            loss_output.total.item(),
            loss_output.diffusion.item(),
            loss_output.balance.item(),
            loss_output.diversity.item(),
            loss_output.hierarchy.item(),
            tok_per_sec,
            gpu_mem_used,
            cluster_info,
        )

        # Reset counters
        self._step_start = time.time()
        self._tokens_processed = 0

    def _report_to_dashboard(self, loss_output) -> None:
        """Send metrics to dashboard."""
        if not self.reporter:
            return

        gpu_mem = torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0
        gpu_util = 0.0
        if self.device.type == "cuda":
            try:
                gpu_util = torch.cuda.utilization(0) / 100.0
            except Exception:
                pass

        # Extract all structural metrics from cached routing info
        cluster_health = {}
        gate_metrics = {}
        hierarchy_metrics = {}
        extra = {}

        if isinstance(self._raw_model.embedding, CompositeEmbedding):
            try:
                from eval.cluster_health import full_cluster_health
                from eval.hierarchy_metrics import coarse_fine_alignment

                info = self._raw_model.embedding.get_routing_info()

                # Gate stats (from cached forward pass)
                gate_values = info.get("gate_values")
                if gate_values is not None:
                    gate_metrics = {
                        "mean": gate_values.mean().item(),
                        "std": gate_values.std().item(),
                    }

                # Cluster health (from cached routing weights)
                fine_weights = info.get("fine_weights")
                fine_centroids_mod = info.get("fine_centroids")
                if fine_weights is not None and fine_centroids_mod is not None:
                    health = full_cluster_health(fine_weights, fine_centroids_mod.centroids)
                    cluster_health = {
                        "entropy_ratio": health["entropy_ratio"],
                        "dead_clusters": health["dead_clusters"],
                        "centroid_similarity": health["mean_cosine"],
                    }

                # Hierarchy metrics
                coarse_centroids_mod = info.get("coarse_centroids")
                if fine_centroids_mod is not None and coarse_centroids_mod is not None:
                    alignment = coarse_fine_alignment(
                        fine_centroids_mod.centroids.detach(),
                        coarse_centroids_mod.centroids.detach(),
                    )
                    hierarchy_metrics = {
                        "coarse_fine_alignment": alignment["coherence"],
                        "balance": alignment["balance_score"],
                    }

                # Extra: alpha, beta for tracking
                alpha = info.get("alpha")
                beta = info.get("beta")
                if alpha is not None:
                    extra["alpha"] = alpha.item()
                if beta is not None:
                    extra["beta"] = beta.item()

            except Exception:
                pass

        elapsed = time.time() - self._step_start if self._step_start else 1
        tok_per_sec = self._tokens_processed / elapsed if elapsed > 0 else 0

        self.reporter.report(
            step=self.global_step,
            losses={
                "total": loss_output.total.item(),
                "diffusion": loss_output.diffusion.item(),
                "balance": loss_output.balance.item(),
                "diversity": loss_output.diversity.item(),
                "hierarchy": loss_output.hierarchy.item(),
            },
            cluster_health=cluster_health if cluster_health else None,
            gate=gate_metrics if gate_metrics else None,
            hierarchy=hierarchy_metrics if hierarchy_metrics else None,
            throughput={
                "tokens_per_sec": tok_per_sec,
                "gpu_memory_gb": gpu_mem,
                "gpu_utilization": gpu_util,
            },
            extra=extra if extra else None,
        )

    def _generate_samples(self) -> None:
        """Generate text samples to show training progress."""
        if self.tokenizer is None:
            return

        try:
            from eval.generation import generate_samples

            logger.info("Step %d | Generating samples...", self.global_step)
            results = generate_samples(
                self._raw_model,
                self.tokenizer,
                prompts=self.sample_prompts,
                seq_len=128,
                sampling_steps=32,
                temperature=0.8,
                device=str(self.device),
            )

            for r in results:
                prompt = r["prompt"]
                generated = r["generated"]
                # Truncate for logging
                gen_short = generated[:200] + "..." if len(generated) > 200 else generated
                logger.info("  Prompt: %s", prompt)
                logger.info("  Output: %s", gen_short)

                # Report to dashboard
                if self.reporter:
                    self.reporter.report_generation(
                        step=self.global_step,
                        text=generated,
                        prompt=prompt,
                    )

            self.model.train()  # Back to training mode
        except Exception as e:
            logger.warning("Sample generation failed: %s", e)

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
