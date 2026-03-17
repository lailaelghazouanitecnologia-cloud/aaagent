"""VM Context — execution state for plan execution.

Provides variable storage with special namespaces:
  @0, @1, ...          Step results
  @template:<hash>     Registered templates
  @meta:<name>         Meta-embeddings
  @block:latest        Most recent block created

Supports stack frames for nested execution with local/global scoping.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch


class VMContext:
    """Execution context for the VM.

    Stores step results, registered templates, and meta state.
    Supports frame-based scoping for nested plan execution.
    """

    def __init__(self, max_steps: int = 100, max_depth: int = 10):
        self.max_steps = max_steps
        self.max_depth = max_depth
        self._step_results: dict[int, Any] = {}
        self._variables: dict[str, Any] = {}
        self._output: list[str] = []

        # Control flags
        self._abort = False
        self._replan = False
        self._think_budget = 0
        self._current_step = 0

        # Frame stack for nested execution
        self._frames: list[dict[str, Any]] = []  # Stack of local scopes
        self._globals: dict[str, Any] = {}
        self._result_stack: list[Any] = []

    # ── Frame-based API ──

    def push_frame(self, name: str) -> None:
        """Push a new execution frame (local scope)."""
        if len(self._frames) >= self.max_depth:
            raise RuntimeError(f"max recursion depth exceeded: {self.max_depth}")
        self._frames.append({"_name": name})

    def pop_frame(self) -> None:
        """Pop the current execution frame."""
        if self._frames:
            self._frames.pop()

    def set_local(self, key: str, value: Any) -> None:
        """Set a variable in the current frame's local scope."""
        if self._frames:
            self._frames[-1][key] = value

    def get_local(self, key: str) -> Any:
        """Get a variable from the current frame's local scope only."""
        if self._frames:
            frame = self._frames[-1]
            return frame.get(key) if key != "_name" else None
        return None

    def set_global(self, key: str, value: Any) -> None:
        """Set a global variable (visible across all frames)."""
        self._globals[key] = value

    def get_global(self, key: str) -> Any:
        """Get a global variable."""
        return self._globals.get(key)

    def push_result(self, value: Any) -> None:
        """Push a value onto the result stack."""
        self._result_stack.append(value)

    def pop_result(self) -> Any:
        """Pop a value from the result stack."""
        if self._result_stack:
            return self._result_stack.pop()
        return None

    def snapshot(self) -> dict[str, Any]:
        """Take a snapshot of the entire context state."""
        return {
            "frames": copy.deepcopy(self._frames),
            "globals": copy.deepcopy(self._globals),
            "result_stack": copy.deepcopy(self._result_stack),
            "step_results": copy.deepcopy(self._step_results),
            "variables": copy.deepcopy(self._variables),
            "abort": self._abort,
            "replan": self._replan,
            "think_budget": self._think_budget,
            "current_step": self._current_step,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore context from a snapshot."""
        self._frames = snap["frames"]
        self._globals = snap["globals"]
        self._result_stack = snap["result_stack"]
        self._step_results = snap["step_results"]
        self._variables = snap["variables"]
        self._abort = snap["abort"]
        self._replan = snap["replan"]
        self._think_budget = snap["think_budget"]
        self._current_step = snap["current_step"]

    # ── Step results API ──

    def set_step_result(self, step: int, result: Any) -> None:
        """Store result from a plan step."""
        self._step_results[step] = result
        self._current_step = step

    def get_step_result(self, step: int) -> Any:
        """Get result from a previous step. @0, @1, etc."""
        return self._step_results.get(step)

    def set(self, key: str, value: Any) -> None:
        self._variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if key.isdigit():
            return self._step_results.get(int(key), default)
        return self._variables.get(key, default)

    def resolve_ref(self, ref: str) -> Any:
        """Resolve a reference like @0, @template:hash, @meta:name."""
        if ref.startswith("@"):
            ref = ref[1:]

        if ref.isdigit():
            return self.get_step_result(int(ref))

        if ref.startswith("template:"):
            return self._variables.get(ref)

        if ref.startswith("meta:"):
            return self._variables.get(ref)

        if ref == "block:latest":
            return self._variables.get("_latest_block")

        return self._variables.get(ref)

    def to_dict(self) -> dict[str, Any]:
        """Export context as a flat dict for opcode execution."""
        d = dict(self._variables)
        d["_output"] = self._output
        d["_abort"] = self._abort
        d["_replan"] = self._replan
        d["_think_budget"] = self._think_budget
        for step, result in self._step_results.items():
            d[str(step)] = result
        return d

    def from_dict(self, d: dict[str, Any]) -> None:
        """Import control flags back from opcode execution."""
        self._abort = d.get("_abort", False)
        self._replan = d.get("_replan", False)
        self._think_budget = d.get("_think_budget", 0)
        if "_output" in d:
            self._output = d["_output"]

    @property
    def should_abort(self) -> bool:
        return self._abort

    @property
    def should_replan(self) -> bool:
        return self._replan

    @property
    def output(self) -> list[str]:
        return self._output

    def reset(self) -> None:
        self._step_results.clear()
        self._variables.clear()
        self._output.clear()
        self._frames.clear()
        self._globals.clear()
        self._result_stack.clear()
        self._abort = False
        self._replan = False
        self._think_budget = 0
        self._current_step = 0
