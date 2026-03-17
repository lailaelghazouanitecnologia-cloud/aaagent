"""VM Context — execution state for plan execution.

Provides variable storage with special namespaces:
  @0, @1, ...          Step results
  @template:<hash>     Registered templates
  @meta:<name>         Meta-embeddings
  @block:latest        Most recent block created
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class VMContext:
    """Execution context for the VM.

    Stores step results, registered templates, and meta state.
    Acts as the mutable state threaded through plan execution.
    """

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self._step_results: dict[int, Any] = {}
        self._variables: dict[str, Any] = {}
        self._output: list[str] = []

        # Control flags
        self._abort = False
        self._replan = False
        self._think_budget = 0
        self._current_step = 0

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
        # Check step results first (@0, @1, etc.)
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
        self._abort = False
        self._replan = False
        self._think_budget = 0
        self._current_step = 0
