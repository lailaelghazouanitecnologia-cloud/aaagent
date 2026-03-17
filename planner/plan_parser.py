"""Plan Parser — converts model output tokens into executable block graphs.

Takes a sequence of tokens (including plan/template/block markers)
and produces a structured Plan that the VM can execute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from blocks.block import Block, BlockType
from blocks.registry import BlockRegistry
from vm.interceptor import OutputInterceptor, PlanSpec, TemplateRef

logger = logging.getLogger(__name__)


@dataclass
class PlanNode:
    """A node in the execution graph."""
    node_id: str
    node_type: str                       # "opcode", "template", "block", "text"
    opcode: str = ""                     # Opcode name (if type=opcode)
    args: list[Any] = field(default_factory=list)
    children: list[PlanNode] = field(default_factory=list)
    template_ref: Optional[TemplateRef] = None
    block: Optional[Block] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class ExecutionPlan:
    """A parsed plan ready for VM execution."""
    root: PlanNode
    mode: str = "direct"
    expected_steps: int = 0
    expected_complexity: str = "low"
    expected_depth: int = 0

    def to_flat_steps(self) -> list[tuple[str, list[Any]]]:
        """Convert to flat (opcode, args) list for sequential VM execution."""
        steps = []
        self._flatten(self.root, steps)
        return steps

    def _flatten(self, node: PlanNode, steps: list) -> None:
        # Children first (DFS)
        for child in node.children:
            self._flatten(child, steps)
        if node.opcode:
            steps.append((node.opcode, node.args))


class PlanParser:
    """Parses model output into executable plans."""

    def __init__(self, registry: Optional[BlockRegistry] = None):
        self.registry = registry or BlockRegistry()
        self.interceptor = OutputInterceptor()
        self._node_counter = 0

    def parse(self, text: str) -> list[ExecutionPlan]:
        """Parse model output text into executable plans.

        Returns a list of plans (model may generate multiple).
        """
        scan_result = self.interceptor.scan(text)

        if not scan_result["has_plan"]:
            return []

        plans = []
        for plan_spec in scan_result["plans"]:
            plan = self._build_plan(plan_spec)
            if plan is not None:
                plans.append(plan)

        return plans

    def _build_plan(self, spec: PlanSpec) -> Optional[ExecutionPlan]:
        """Build an ExecutionPlan from a PlanSpec."""
        if not spec.operations:
            return None

        # Build node tree from operations
        children = []
        for op in spec.operations:
            node = self._build_node(op)
            if node is not None:
                children.append(node)

        if not children:
            return None

        root = PlanNode(
            node_id=self._next_id("root"),
            node_type="plan",
            children=children,
        )

        return ExecutionPlan(
            root=root,
            mode=spec.mode,
            expected_steps=spec.steps,
            expected_complexity=spec.complexity,
            expected_depth=spec.depth,
        )

    def _build_node(self, op: dict[str, Any]) -> Optional[PlanNode]:
        """Build a PlanNode from a parsed operation."""
        op_type = op.get("type", "text")

        if op_type == "template":
            ref = op.get("ref")
            if ref is None:
                return None
            return PlanNode(
                node_id=self._next_id("tmpl"),
                node_type="template",
                template_ref=ref,
                opcode="expand_template",
                args=[ref.hash, ref.slot_bindings],
            )

        elif op_type == "block":
            return PlanNode(
                node_id=self._next_id("block"),
                node_type="block",
                args=[op.get("raw", "")],
            )

        elif op_type == "text":
            content = op.get("content", "")
            # Try to parse as opcode
            parts = content.split(None, 1)
            if parts and parts[0].lower() in _KNOWN_OPCODES:
                opcode = parts[0].lower()
                args = parts[1].split(",") if len(parts) > 1 else []
                args = [a.strip() for a in args]
                return PlanNode(
                    node_id=self._next_id("op"),
                    node_type="opcode",
                    opcode=opcode,
                    args=args,
                )

            return PlanNode(
                node_id=self._next_id("text"),
                node_type="text",
                args=[content],
            )

        return None

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"


# Known opcodes for parsing
_KNOWN_OPCODES = {
    "add", "sub", "mul", "div", "mod", "pow", "sqrt", "abs", "round",
    "sum", "mean", "max", "min", "count", "is_prime", "gcd",
    "concat", "split", "join", "upper", "lower", "replace", "trim",
    "contains", "length", "slice", "format", "regex",
    "map", "filter", "reduce", "sort", "reverse", "unique",
    "flatten", "zip", "take", "skip", "enumerate", "range",
    "get", "set", "keys", "values", "merge", "has_key", "delete",
    "from_pairs", "group_by",
    "parse_json", "dump_json", "print",
    "if", "pipe", "sequence",
    "think", "retrieve", "plan", "replan", "abort",
    "create_template", "register_block", "lookup_hash",
    "expand_template", "bind_slot", "merge_block", "hash_block",
    "rewrite_block", "freeze_block", "abstract_block", "inline_block",
}
