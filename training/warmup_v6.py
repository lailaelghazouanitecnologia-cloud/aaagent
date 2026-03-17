"""6-phase structural warmup for Za v6.

Phase 0 (0-2K):       Embedding warmup
  Only E_local. Gate=0. Clusters frozen. Block features off.

Phase 1 (2K-10K):     Structure emerges
  Gate opens. Clusters unfreeze. Token-level diffusion only.

Phase 2 (10K-30K):    Multi-level diffusion
  Activate L_slot, L_block. Span/slot/block masking active.

Phase 3 (30K-50K):    VM integration
  Activate L_plan. Execute real plans, feedback to model.

Phase 4 (50K-70K):    Dynamic embeddings
  Activate block creation. L_compose, L_template, L_hash.
  System starts self-extending.

Phase 5 (70K+):       Meta-learning
  Meta-optimizer. Meta-embeddings active. Online learning.
"""

from __future__ import annotations

import torch.nn as nn

from training.warmup import StructuralWarmup


class V6Warmup(StructuralWarmup):
    """Extended warmup with 6 phases for Za v6.

    Inherits phases 0-3 from StructuralWarmup and adds
    phases 4-5 for dynamic blocks and meta-learning.
    """

    def __init__(
        self,
        # v4 warmup params
        gate_freeze_steps: int = 500,
        gate_ramp_end_steps: int = 2000,
        cluster_unfreeze_step: int = 500,
        coarse_unfreeze_step: int = 3000,
        temp_anneal_start: int = 2000,
        temp_anneal_end: int = 20000,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        loss_ramp_start: int = 2000,
        loss_ramp_end: int = 20000,
        loss_multiplier_min: float = 0.1,
        loss_multiplier_hierarchy_max: float = 10.0,
        loss_multiplier_balance_max: float = 5.0,
        loss_multiplier_diversity_max: float = 10.0,
        # v6 phase boundaries
        multilevel_start: int = 10000,
        vm_start: int = 30000,
        dynamic_start: int = 50000,
        meta_start: int = 70000,
        # v6 block feature ramp
        block_gate_ramp_start: int = 10000,
        block_gate_ramp_end: int = 20000,
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
            loss_ramp_start=loss_ramp_start,
            loss_ramp_end=loss_ramp_end,
            loss_multiplier_min=loss_multiplier_min,
            loss_multiplier_hierarchy_max=loss_multiplier_hierarchy_max,
            loss_multiplier_balance_max=loss_multiplier_balance_max,
            loss_multiplier_diversity_max=loss_multiplier_diversity_max,
        )

        # v6 phase boundaries
        self.multilevel_start = multilevel_start
        self.vm_start = vm_start
        self.dynamic_start = dynamic_start
        self.meta_start = meta_start

        # Block feature gate ramp
        self.block_gate_ramp_start = block_gate_ramp_start
        self.block_gate_ramp_end = block_gate_ramp_end

        # Track what's been unfrozen
        self._block_unfrozen = False
        self._vm_activated = False
        self._dynamic_activated = False
        self._meta_activated = False

    def get_phase(self, step: int) -> int:
        """Get the current v6 training phase (0-5)."""
        if step < self.gate_freeze_steps:
            return 0
        elif step < self.multilevel_start:
            return 1
        elif step < self.vm_start:
            return 2
        elif step < self.dynamic_start:
            return 3
        elif step < self.meta_start:
            return 4
        else:
            return 5

    def get_block_gate_scale(self, step: int) -> float:
        """Get the block feature gate scale (0.0 → 1.0).

        Block features (E_block, E_template, E_meta, ΔE_dyn) are
        ramped in during phase 2.
        """
        if step < self.block_gate_ramp_start:
            return 0.0
        elif step >= self.block_gate_ramp_end:
            return 1.0
        else:
            progress = (step - self.block_gate_ramp_start) / max(
                self.block_gate_ramp_end - self.block_gate_ramp_start, 1
            )
            return progress

    def is_multilevel_active(self, step: int) -> bool:
        """Whether multi-level masking (span/slot/block) is active."""
        return step >= self.multilevel_start

    def is_vm_active(self, step: int) -> bool:
        """Whether VM execution and plan feedback are active."""
        return step >= self.vm_start

    def is_dynamic_active(self, step: int) -> bool:
        """Whether dynamic block creation is active."""
        return step >= self.dynamic_start

    def is_meta_active(self, step: int) -> bool:
        """Whether meta-learning features are active."""
        return step >= self.meta_start

    def apply_v6_freezing(self, model: nn.Module, step: int) -> None:
        """Apply phase-specific freezing for v6 components.

        Extends apply_freezing with block/VM/meta component management.
        """
        # Base v4 freezing
        self.apply_freezing(model, step)

        # Unfreeze block features at phase 2
        if step >= self.multilevel_start and not self._block_unfrozen:
            for name, param in model.named_parameters():
                if "block_embedding" in name or "block_encoder" in name:
                    param.requires_grad = True
                if "template_encoder" in name:
                    param.requires_grad = True
            self._block_unfrozen = True

        # Unfreeze meta features at phase 5
        if step >= self.meta_start and not self._meta_activated:
            for name, param in model.named_parameters():
                if "meta_encoder" in name or "hyper_gate" in name or "hyper_delta" in name:
                    param.requires_grad = True
            self._meta_activated = True

    def get_loss_multipliers(self, step: int) -> dict[str, float]:
        """Extended multipliers including v6 losses."""
        base = super().get_loss_multipliers(step)

        # v6 loss multipliers: ramp in at their phase start
        phase = self.get_phase(step)

        # Slot/block losses ramp during phase 2
        if phase >= 2:
            slot_progress = min(1.0, (step - self.multilevel_start) / max(self.vm_start - self.multilevel_start, 1))
            base["slot"] = slot_progress
            base["block"] = slot_progress
        else:
            base["slot"] = 0.0
            base["block"] = 0.0

        # Plan loss ramps during phase 3
        if phase >= 3:
            plan_progress = min(1.0, (step - self.vm_start) / max(self.dynamic_start - self.vm_start, 1))
            base["plan"] = plan_progress
        else:
            base["plan"] = 0.0

        # Compose/template/hash losses ramp during phase 4
        if phase >= 4:
            dyn_progress = min(1.0, (step - self.dynamic_start) / max(self.meta_start - self.dynamic_start, 1))
            base["compose"] = dyn_progress
            base["template"] = dyn_progress
            base["hash"] = dyn_progress
        else:
            base["compose"] = 0.0
            base["template"] = 0.0
            base["hash"] = 0.0

        return base
