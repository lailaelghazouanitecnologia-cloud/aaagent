"""Structural warmup: gate schedule, cluster freezing, temperature annealing,
loss curriculum, and staggered coarse unfreeze.

Phase 0 (0 - gate_freeze_steps): Gate=0, all clusters frozen. Only E_local trains.
Phase 1 (gate_freeze_steps - gate_ramp_end_steps): Gate ramps 0→learned.
    Fine clusters unfreeze at cluster_unfreeze_step.
Phase 2 (gate_ramp_end_steps - coarse_unfreeze_step): Gate learned. Fine active.
    Coarse clusters unfreeze at coarse_unfreeze_step (Proposal D).
Phase 3 (coarse_unfreeze_step+): Full training, all components active.

Temperature annealing (Proposal A): τ decays from τ_start → τ_end between
    temp_anneal_start and temp_anneal_end steps.

Loss curriculum (Proposal B): lambda multipliers ramp from λ_min → λ_max between
    loss_ramp_start and loss_ramp_end steps, then scale by final multiplier.
"""

from __future__ import annotations

import torch.nn as nn


class StructuralWarmup:
    """Manages the structural warmup schedule for HCLM-D.

    Controls gate values, cluster parameter freezing, router temperature,
    and loss weight scheduling during training.
    """

    def __init__(
        self,
        gate_freeze_steps: int = 500,
        gate_ramp_end_steps: int = 2000,
        cluster_unfreeze_step: int = 500,
        # Proposal D: staggered coarse unfreeze
        coarse_unfreeze_step: int = 3000,
        # Proposal A: temperature annealing
        temp_anneal_start: int = 2000,
        temp_anneal_end: int = 20000,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        # Proposal B: loss curriculum
        loss_ramp_start: int = 2000,
        loss_ramp_end: int = 20000,
        loss_multiplier_min: float = 0.1,
        loss_multiplier_hierarchy_max: float = 10.0,
        loss_multiplier_balance_max: float = 5.0,
        loss_multiplier_diversity_max: float = 10.0,
    ):
        self.gate_freeze_steps = gate_freeze_steps
        self.gate_ramp_end_steps = gate_ramp_end_steps
        self.cluster_unfreeze_step = cluster_unfreeze_step
        self.coarse_unfreeze_step = coarse_unfreeze_step

        # Temperature annealing
        self.temp_anneal_start = temp_anneal_start
        self.temp_anneal_end = temp_anneal_end
        self.temp_start = temp_start
        self.temp_end = temp_end

        # Loss curriculum
        self.loss_ramp_start = loss_ramp_start
        self.loss_ramp_end = loss_ramp_end
        self.loss_multiplier_min = loss_multiplier_min
        self.loss_multiplier_hierarchy_max = loss_multiplier_hierarchy_max
        self.loss_multiplier_balance_max = loss_multiplier_balance_max
        self.loss_multiplier_diversity_max = loss_multiplier_diversity_max

        self._fine_frozen = True
        self._coarse_frozen = True

    def get_gate_value(self, step: int) -> float | None:
        """Get the gate override value for the current step.

        Returns:
            Gate override value (0.0 to None). None means use learned value.
        """
        if step < self.gate_freeze_steps:
            return 0.0
        elif step < self.gate_ramp_end_steps:
            progress = (step - self.gate_freeze_steps) / max(
                self.gate_ramp_end_steps - self.gate_freeze_steps, 1
            )
            return None if progress >= 1.0 else progress
        else:
            return None

    def get_router_temperature(self, step: int) -> float:
        """Get the router softmax temperature for the current step.

        Anneals linearly from temp_start → temp_end between
        temp_anneal_start and temp_anneal_end.
        """
        if step < self.temp_anneal_start:
            return self.temp_start
        elif step >= self.temp_anneal_end:
            return self.temp_end
        else:
            progress = (step - self.temp_anneal_start) / max(
                self.temp_anneal_end - self.temp_anneal_start, 1
            )
            return self.temp_start + progress * (self.temp_end - self.temp_start)

    def get_loss_multipliers(self, step: int) -> dict[str, float]:
        """Get dynamic loss weight multipliers for the current step.

        Returns dict with keys: 'hierarchy', 'balance', 'diversity'.
        Each value is a multiplier applied to the base lambda.
        """
        if step < self.loss_ramp_start:
            t = self.loss_multiplier_min
            return {"hierarchy": t, "balance": t, "diversity": t}
        elif step >= self.loss_ramp_end:
            return {
                "hierarchy": self.loss_multiplier_hierarchy_max,
                "balance": self.loss_multiplier_balance_max,
                "diversity": self.loss_multiplier_diversity_max,
            }
        else:
            progress = (step - self.loss_ramp_start) / max(
                self.loss_ramp_end - self.loss_ramp_start, 1
            )
            return {
                "hierarchy": self.loss_multiplier_min + progress * (
                    self.loss_multiplier_hierarchy_max - self.loss_multiplier_min
                ),
                "balance": self.loss_multiplier_min + progress * (
                    self.loss_multiplier_balance_max - self.loss_multiplier_min
                ),
                "diversity": self.loss_multiplier_min + progress * (
                    self.loss_multiplier_diversity_max - self.loss_multiplier_min
                ),
            }

    def apply_freezing(self, model: nn.Module, step: int) -> None:
        """Freeze/unfreeze cluster parameters based on warmup schedule.

        Fine components (fine_router, fine_centroids, alpha) unfreeze at
        cluster_unfreeze_step. Coarse components (coarse_router,
        coarse_centroids, beta) unfreeze later at coarse_unfreeze_step.
        """
        should_freeze_fine = step < self.cluster_unfreeze_step
        should_freeze_coarse = step < self.coarse_unfreeze_step

        fine_changed = should_freeze_fine != self._fine_frozen
        coarse_changed = should_freeze_coarse != self._coarse_frozen

        if not fine_changed and not coarse_changed:
            return

        for name, param in model.named_parameters():
            # Fine components
            if "fine_router" in name or "fine_centroids" in name or "alpha" in name:
                if fine_changed:
                    param.requires_grad = not should_freeze_fine
            # Coarse components
            elif "coarse_router" in name or "coarse_centroids" in name or "beta" in name:
                if coarse_changed:
                    param.requires_grad = not should_freeze_coarse

        self._fine_frozen = should_freeze_fine
        self._coarse_frozen = should_freeze_coarse

    def get_phase(self, step: int) -> int:
        """Get the current training phase."""
        if step < self.gate_freeze_steps:
            return 0
        elif step < self.gate_ramp_end_steps:
            return 1
        elif step < self.coarse_unfreeze_step:
            return 2
        else:
            return 3
