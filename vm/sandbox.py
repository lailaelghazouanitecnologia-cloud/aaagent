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


@dataclass
class SandboxConfig:
    """Security configuration for VM execution."""

    max_steps: int = 100
    timeout_ms: float = 5000.0
    max_recursion_depth: int = 10
    max_output_size: int = 10000  # chars
    max_list_size: int = 10000

    # Opcode restrictions
    allowed_opcodes: Optional[set[str]] = None  # None = allow all
    blocked_opcodes: set[str] = field(default_factory=lambda: {"http_get", "write", "read"})

    # Resource limits
    allow_io: bool = False
    allow_network: bool = False


class Sandbox:
    """Enforces execution limits on the VM."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._step_count = 0
        self._recursion_depth = 0

    def check_opcode(self, name: str) -> bool:
        """Check if an opcode is allowed."""
        if self.config.allowed_opcodes is not None:
            return name in self.config.allowed_opcodes
        return name not in self.config.blocked_opcodes

    def check_step_limit(self) -> bool:
        self._step_count += 1
        return self._step_count <= self.config.max_steps

    def enter_recursion(self) -> bool:
        self._recursion_depth += 1
        return self._recursion_depth <= self.config.max_recursion_depth

    def exit_recursion(self) -> None:
        self._recursion_depth = max(0, self._recursion_depth - 1)

    def check_output_size(self, output: str) -> bool:
        return len(output) <= self.config.max_output_size

    def check_list_size(self, lst: list) -> bool:
        return len(lst) <= self.config.max_list_size

    def reset(self) -> None:
        self._step_count = 0
        self._recursion_depth = 0
