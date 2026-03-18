"""6-phase structural warmup for Za v7.

Phase 0 (0-1K):        RWKV base
  Only RWKV forward. No backward. No diffusion bidirectional.
  Gate=0, clusters frozen.

Phase 1 (1K-4K):       RWKV bidi + token diffusion
  Activate backward + merge. M2T over tokens.
  Gate ramps. Fine clusters unfreeze.

Phase 2 (4K-12K):      Structure emerges
  Gate fully open. Coarse clusters unfreeze.
  M2T + T2T. Multi-level masking (token + slot + block).

Phase 3 (12K-25K):     VM integration
  Activate L_plan. Execute real plans.
  Model learns to generate functional plans.

Phase 4 (25K-35K):     Ranking + retrieval
  Activate ranking opcodes.
  Model learns to search and compose in embedding space.

Phase 5 (35K-50K):     Dynamic blocks + SSD
  Activate block creation.
  System starts persisting templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch.nn as nn

from training.warmup import StructuralWarmup


class V7Phase(IntEnum):
    """Training phases for Za v7."""
    RWKV_BASE = 0       # Forward only, no bidi
    RWKV_BIDI = 1       # Forward + backward + merge
    STRUCTURE = 2        # Clusters emerge, multi-level masking
    VM = 3               # VM integration, plan execution
    RANKING = 4          # Ranking + retrieval opcodes
    DYNAMIC = 5          # Dynamic blocks + SSD persistence


@dataclass
class V7WarmupState:
    """Snapshot of v7 warmup state at a given step."""
    phase: V7Phase
    gate_scale: float
    fine_temperature: float
    coarse_temperature: float
    bidi_active: bool
    multilevel_active: bool
    vm_active: bool
    ranking_active: bool
    dynamic_active: bool


class V7Warmup(StructuralWarmup):
    """Extended warmup with 6 phases for Za v7.

    Inherits gate/temperature/loss curriculum from StructuralWarmup.
    Adds RWKV-specific phase boundaries and component activation.
    """

    def __init__(
        self,
        # Phase boundaries
        phase0_end: int = 1000,      # RWKV base
        phase1_end: int = 4000,      # RWKV bidi + token diffusion
        phase2_end: int = 12000,     # Structure emerges
        phase3_end: int = 25000,     # VM integration
        phase4_end: int = 35000,     # Ranking + retrieval
        # phase5 = 35K-50K (dynamic blocks)
        # v4 warmup params
        gate_freeze_steps: int = 1000,
        gate_ramp_end_steps: int = 4000,
        cluster_unfreeze_step: int = 1000,
        coarse_unfreeze_step: int = 4000,
        temp_anneal_start: int = 4000,
        temp_anneal_end: int = 40000,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        coarse_temp_ratio: float = 0.6,
        loss_ramp_start: int = 4000,
        loss_ramp_end: int = 40000,
        loss_multiplier_min: float = 0.1,
        loss_multiplier_hierarchy_max: float = 10.0,
        loss_multiplier_balance_max: float = 5.0,
        loss_multiplier_diversity_max: float = 10.0,
    ):
        super().__init__(
            gate_freeze_steps=gate_freeze_steps,
            gate_ramp_end_steps=gate_ramp_end_steps,
            cluster_unfreeze_step=cluster_unfreeze_step,
            coarse_unfreeze_step=coarse_unfreeze_step,
            temp_anneal_start=temp_anneal_start,
            temp_anneal_end=temp_anneal_end,
            temp_start=temp_start,
            temp_end=temp_end,
            coarse_temp_ratio=coarse_temp_ratio,
            loss_ramp_start=loss_ramp_start,
            loss_ramp_end=loss_ramp_end,
            loss_multiplier_min=loss_multiplier_min,
            loss_multiplier_hierarchy_max=loss_multiplier_hierarchy_max,
            loss_multiplier_balance_max=loss_multiplier_balance_max,
            loss_multiplier_diversity_max=loss_multiplier_diversity_max,
        )

        # Phase boundaries
        self.phase0_end = phase0_end
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.phase3_end = phase3_end
        self.phase4_end = phase4_end

        # Track activation state
        self._bidi_activated = False
        self._backward_frozen = True

    def get_phase(self, step: int) -> int:
        """Get the current v7 training phase (0-5)."""
        if step < self.phase0_end:
            return 0
        elif step < self.phase1_end:
            return 1
        elif step < self.phase2_end:
            return 2
        elif step < self.phase3_end:
            return 3
        elif step < self.phase4_end:
            return 4
        else:
            return 5

    def is_bidi_active(self, step: int) -> bool:
        """Whether bidirectional RWKV (backward + merge) is active."""
        return step >= self.phase0_end

    def is_multilevel_active(self, step: int) -> bool:
        """Whether multi-level masking (slot + block) is active."""
        return step >= self.phase1_end

    def is_vm_active(self, step: int) -> bool:
        """Whether VM execution and plan feedback are active."""
        return step >= self.phase2_end

    def is_ranking_active(self, step: int) -> bool:
        """Whether ranking opcodes are active."""
        return step >= self.phase3_end

    def is_dynamic_active(self, step: int) -> bool:
        """Whether dynamic block creation is active."""
        return step >= self.phase4_end

    def apply_v7_freezing(self, model: nn.Module, step: int) -> None:
        """Apply phase-specific freezing for v7 RWKV components.

        Phase 0: backward_layers and merge are frozen.
        Phase 1+: backward_layers and merge are unfrozen.
        """
        # Base freezing (gate, clusters, coarse)
        self.apply_freezing(model, step)

        # RWKV-specific: freeze backward stack in phase 0
        should_freeze_backward = step < self.phase0_end

        if should_freeze_backward != self._backward_frozen:
            for name, param in model.named_parameters():
                if "backward_layers" in name or "merge" in name:
                    param.requires_grad = not should_freeze_backward
            self._backward_frozen = should_freeze_backward

    def get_loss_multipliers(self, step: int) -> dict[str, float]:
        """Extended multipliers including v7 phase-specific losses."""
        base = super().get_loss_multipliers(step)
        phase = self.get_phase(step)

        # Slot/block losses ramp during phase 2
        if phase >= 2:
            progress = min(1.0, (step - self.phase1_end) / max(self.phase2_end - self.phase1_end, 1))
            base["slot"] = progress
            base["block"] = progress
        else:
            base["slot"] = 0.0
            base["block"] = 0.0

        # Plan loss ramps during phase 3
        if phase >= 3:
            progress = min(1.0, (step - self.phase2_end) / max(self.phase3_end - self.phase2_end, 1))
            base["plan"] = progress
        else:
            base["plan"] = 0.0

        # Ranking loss ramps during phase 4
        if phase >= 4:
            progress = min(1.0, (step - self.phase3_end) / max(self.phase4_end - self.phase3_end, 1))
            base["rank"] = progress
            base["compose"] = progress
        else:
            base["rank"] = 0.0
            base["compose"] = 0.0

        # Storage loss ramps during phase 5
        if phase >= 5:
            base["storage"] = 1.0
        else:
            base["storage"] = 0.0

        return base

    def get_state(self, step: int) -> V7WarmupState:
        """Get a complete warmup state snapshot."""
        phase = V7Phase(self.get_phase(step))
        gate_val = self.get_gate_value(step)

        return V7WarmupState(
            phase=phase,
            gate_scale=gate_val if gate_val is not None else 1.0,
            fine_temperature=self.get_router_temperature(step),
            coarse_temperature=self.get_coarse_temperature(step),
            bidi_active=self.is_bidi_active(step),
            multilevel_active=self.is_multilevel_active(step),
            vm_active=self.is_vm_active(step),
            ranking_active=self.is_ranking_active(step),
            dynamic_active=self.is_dynamic_active(step),
        )
