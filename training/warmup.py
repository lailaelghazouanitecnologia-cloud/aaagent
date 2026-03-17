"""Structural warmup: gate schedule and cluster freezing.

Phase 0 (0 - gate_freeze_steps): Gate=0, clusters frozen. Only E_local trains.
Phase 1 (gate_freeze_steps - gate_ramp_end_steps): Gate ramps 0→learned. Clusters unfreeze.
Phase 2 (gate_ramp_end_steps+): Full training, all components active.
"""

from __future__ import annotations

import torch.nn as nn


class StructuralWarmup:
    """Manages the structural warmup schedule for HCLM-D.

    Controls gate values and cluster parameter freezing during early training.
    """

    def __init__(
        self,
        gate_freeze_steps: int = 500,
        gate_ramp_end_steps: int = 2000,
        cluster_unfreeze_step: int = 500,
    ):
        self.gate_freeze_steps = gate_freeze_steps
        self.gate_ramp_end_steps = gate_ramp_end_steps
        self.cluster_unfreeze_step = cluster_unfreeze_step
        self._clusters_frozen = True

    def get_gate_value(self, step: int) -> float | None:
        """Get the gate override value for the current step.

        Args:
            step: Current training step.

        Returns:
            Gate override value (0.0 to None). None means use learned value.
        """
        if step < self.gate_freeze_steps:
            # Phase 0: Gate completely off
            return 0.0
        elif step < self.gate_ramp_end_steps:
            # Phase 1: Linear ramp from 0 to learned
            progress = (step - self.gate_freeze_steps) / max(
                self.gate_ramp_end_steps - self.gate_freeze_steps, 1
            )
            # Ramp from 0 → 1 linearly; at 1.0 switch to learned gate
            return None if progress >= 1.0 else progress
        else:
            # Phase 2+: Use learned gate
            return None

    def apply_freezing(self, model: nn.Module, step: int) -> None:
        """Freeze/unfreeze cluster parameters based on warmup schedule.

        Args:
            model: The HCLM-D model.
            step: Current training step.
        """
        should_freeze = step < self.cluster_unfreeze_step

        if should_freeze == self._clusters_frozen:
            return  # No change needed

        for name, param in model.named_parameters():
            if "router" in name or "centroids" in name or "alpha" in name or "beta" in name:
                param.requires_grad = not should_freeze

        self._clusters_frozen = should_freeze

    def get_phase(self, step: int) -> int:
        """Get the current training phase."""
        if step < self.gate_freeze_steps:
            return 0
        elif step < self.gate_ramp_end_steps:
            return 1
        else:
            return 2
