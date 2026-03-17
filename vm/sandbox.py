"""VM Sandbox — security and resource limits for VM execution.

Enforces:
- Maximum execution time
- Maximum memory usage
- Maximum recursion depth
- Opcode allowlists/blocklists
- No filesystem access by default
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class SandboxViolation(RuntimeError):
    """Raised when sandbox security limits are exceeded."""
    pass


@dataclass
class SandboxPolicy:
    """Security policy for VM execution."""

    max_steps: int = 100
    timeout_ms: float = 5000.0
    max_recursion_depth: int = 10
    max_output_size: int = 10000  # chars
    max_list_size: int = 10000
    max_memory_bytes: int = 0  # 0 = unlimited

    # Opcode restrictions
    allowed_opcodes: Optional[set[str]] = None  # None = allow all
    blocked_opcodes: set[str] = field(default_factory=lambda: {"http_get", "write", "read"})

    # Resource limits
    allow_io: bool = False
    allow_network: bool = False


# Backward compatibility alias
SandboxConfig = SandboxPolicy


class Sandbox:
    """Enforces execution limits on the VM."""

    def __init__(self, policy: Optional[SandboxPolicy] = None):
        self.policy = policy or SandboxPolicy()
        # Also accept as 'config' for backward compat
        self.config = self.policy
        self._step_count = 0
        self._recursion_depth = 0

    def is_allowed(self, name: str) -> bool:
        """Check if an opcode is allowed."""
        if self.policy.allowed_opcodes is not None:
            return name in self.policy.allowed_opcodes
        return name not in self.policy.blocked_opcodes

    # Alias for backward compat
    def check_opcode(self, name: str) -> bool:
        return self.is_allowed(name)

    def check_step(self) -> None:
        """Check and increment step count. Raises SandboxViolation if exceeded."""
        self._step_count += 1
        if self._step_count > self.policy.max_steps:
            raise SandboxViolation(
                f"step limit exceeded: {self._step_count} > {self.policy.max_steps}"
            )

    def check_step_limit(self) -> bool:
        """Legacy API: returns bool instead of raising."""
        self._step_count += 1
        return self._step_count <= self.policy.max_steps

    def check_memory(self, bytes_used: int) -> None:
        """Check memory usage. Raises SandboxViolation if exceeded."""
        if self.policy.max_memory_bytes > 0 and bytes_used > self.policy.max_memory_bytes:
            raise SandboxViolation(
                f"memory limit exceeded: {bytes_used} > {self.policy.max_memory_bytes}"
            )

    def enter_recursion(self) -> bool:
        self._recursion_depth += 1
        return self._recursion_depth <= self.policy.max_recursion_depth

    def exit_recursion(self) -> None:
        self._recursion_depth = max(0, self._recursion_depth - 1)

    def check_output_size(self, output: str) -> bool:
        return len(output) <= self.policy.max_output_size

    def check_list_size(self, lst: list) -> bool:
        return len(lst) <= self.policy.max_list_size

    def reset(self) -> None:
        self._step_count = 0
        self._recursion_depth = 0
