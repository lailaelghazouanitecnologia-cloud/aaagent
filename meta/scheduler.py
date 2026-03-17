"""Meta-model invocation schedule."""

from __future__ import annotations


class MetaScheduler:
    """Controls when the meta-optimizer is invoked during training."""

    def __init__(
        self,
        enable_after_step: int = 100000,
        update_every: int = 100,
    ):
        self.enable_after_step = enable_after_step
        self.update_every = update_every

    def is_active(self, step: int) -> bool:
        """Check if meta-optimizer should be active at this step."""
        return step >= self.enable_after_step

    def should_update(self, step: int) -> bool:
        """Check if meta-optimizer should update at this step."""
        return self.is_active(step) and step % self.update_every == 0
