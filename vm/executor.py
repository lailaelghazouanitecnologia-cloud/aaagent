"""VM Executor — runs plans as block trees.

The executor traverses a plan (tree of blocks), executes opcodes
at the leaves, and propagates results upward.

For v6, the executor supports:
- Hierarchical block execution (trees, not just sequences)
- Template expansion (fill slots → execute)
- Cached results by hash (memoization)
- Sandboxed execution with timeouts
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from blocks.block import Block, BlockType
from blocks.registry import BlockRegistry
from vm.context import VMContext
from vm.opcodes import OPCODE_REGISTRY

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of executing a block/plan."""

    def __init__(
        self,
        value: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        steps_executed: int = 0,
        time_ms: float = 0.0,
    ):
        self.value = value
        self.success = success
        self.error = error
        self.steps_executed = steps_executed
        self.time_ms = time_ms


class VMExecutor:
    """Executes block plans with hierarchical traversal.

    Supports:
    - Sequential plan execution (flat opcode list)
    - Hierarchical plan execution (block tree)
    - Template expansion
    - Hash-based memoization
    - Timeout enforcement
    """

    def __init__(
        self,
        registry: Optional[BlockRegistry] = None,
        max_steps: int = 100,
        timeout_ms: float = 5000.0,
    ):
        self.registry = registry or BlockRegistry()
        self.max_steps = max_steps
        self.timeout_ms = timeout_ms

    def execute_plan(
        self,
        steps: list[tuple[str, list[Any]]],
        context: Optional[VMContext] = None,
    ) -> ExecutionResult:
        """Execute a flat plan (list of (opcode, args) tuples).

        This is the v4/v5-compatible interface.
        """
        ctx = context or VMContext(self.max_steps)
        start = time.monotonic()
        n_executed = 0

        for i, (opcode_name, args) in enumerate(steps):
            if n_executed >= self.max_steps:
                return ExecutionResult(
                    error=f"Max steps ({self.max_steps}) exceeded",
                    success=False,
                    steps_executed=n_executed,
                )

            elapsed_ms = (time.monotonic() - start) * 1000
            if elapsed_ms > self.timeout_ms:
                return ExecutionResult(
                    error=f"Timeout ({self.timeout_ms}ms) exceeded",
                    success=False,
                    steps_executed=n_executed,
                )

            result = self._execute_opcode(opcode_name, args, ctx)
            ctx.set_step_result(i, result)
            n_executed += 1

            if ctx.should_abort:
                return ExecutionResult(
                    value=result,
                    success=False,
                    error="Aborted",
                    steps_executed=n_executed,
                    time_ms=(time.monotonic() - start) * 1000,
                )

            if ctx.should_replan:
                return ExecutionResult(
                    value=result,
                    success=True,
                    error="replan_requested",
                    steps_executed=n_executed,
                    time_ms=(time.monotonic() - start) * 1000,
                )

        elapsed_ms = (time.monotonic() - start) * 1000
        last_result = ctx.get_step_result(len(steps) - 1) if steps else None

        return ExecutionResult(
            value=last_result,
            success=True,
            steps_executed=n_executed,
            time_ms=elapsed_ms,
        )

    def execute_block(
        self,
        block: Block,
        context: Optional[VMContext] = None,
    ) -> ExecutionResult:
        """Execute a block tree (v6 hierarchical execution).

        Recursively traverses the block tree:
        1. Check cache by hash
        2. Execute children (slots) first
        3. Execute this block with children results as args
        4. Cache result
        """
        ctx = context or VMContext(self.max_steps)
        start = time.monotonic()

        # Check cache
        h = block.semantic_hash()
        cached = self.registry.get_cached_result(h)
        if cached is not None:
            return ExecutionResult(value=cached, success=True, time_ms=0.0)

        # Execute children first
        child_results = []
        for slot in block.slots:
            if slot is None:
                child_results.append(None)
                continue

            child_result = self.execute_block(slot, ctx)
            if not child_result.success:
                return child_result
            child_results.append(child_result.value)

        # Execute this block
        if block.block_type == BlockType.TOKEN:
            result_value = block.token_id
        elif block.name and block.name in OPCODE_REGISTRY:
            result_value = self._execute_opcode(block.name, child_results, ctx)
        else:
            # Compound block: return child results as tuple
            result_value = tuple(r for r in child_results if r is not None)

        # Cache
        if result_value is not None:
            import torch
            if isinstance(result_value, (int, float)):
                self.registry.cache_result(h, torch.tensor(result_value))

        elapsed_ms = (time.monotonic() - start) * 1000
        return ExecutionResult(
            value=result_value,
            success=True,
            steps_executed=len(block.all_blocks()),
            time_ms=elapsed_ms,
        )

    def _execute_opcode(
        self,
        name: str,
        args: list[Any],
        ctx: VMContext,
    ) -> Any:
        """Execute a single opcode."""
        fn = OPCODE_REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown opcode: %s", name)
            return None

        try:
            ctx_dict = ctx.to_dict()

            # Resolve context references in args
            resolved_args = []
            for arg in args:
                if isinstance(arg, str) and arg.startswith("@"):
                    resolved_args.append(ctx.resolve_ref(arg))
                else:
                    resolved_args.append(arg)

            result = fn(resolved_args, ctx_dict)
            ctx.from_dict(ctx_dict)
            return result

        except Exception as e:
            logger.warning("Opcode %s failed: %s", name, e)
            return None
