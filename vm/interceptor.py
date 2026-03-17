"""VM Interceptor — detects plan/template/block tokens in model output.

Scans generated token sequences for structural markers:
  [PLAN:...]       → Parse and execute as a plan
  [TMPL:hash ...]  → Expand a template
  [CREATE:...]     → Create a new block
  [/PLAN]          → End of plan

Returns structured representations that the VM can execute.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlanSpec:
    """Parsed plan specification from model output."""
    mode: str = "direct"           # direct, compute, pipeline, compose
    steps: int = 0
    complexity: str = "low"        # low, mid, high
    depth: int = 0                 # Template nesting depth
    operations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TemplateRef:
    """Reference to a template in model output."""
    hash: str = ""
    name: str = ""
    slot_bindings: dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateSpec:
    """Specification for creating a new block."""
    name: str = ""
    n_slots: int = 0
    from_hashes: list[str] = field(default_factory=list)


class OutputInterceptor:
    """Intercepts structural tokens in model output.

    Parses sequences of tokens looking for plan, template, and
    create markers. Returns structured specs for the VM.
    """

    # Token patterns (these map to special token IDs)
    PLAN_START = "[PLAN:"
    PLAN_END = "[/PLAN]"
    TMPL_START = "[TMPL:"
    TMPL_END = "[/TMPL]"
    CREATE_START = "[CREATE:"
    BLOCK_START = "[BLOCK"
    BLOCK_END = "[/BLOCK]"
    ARG_START = "[ARG"
    MASK_TOKEN = "[MASK_TOKEN]"
    MASK_SPAN = "[MASK_SPAN]"
    MASK_BLOCK = "[MASK_BLOCK]"
    MASK_TEMPLATE = "[MASK_TEMPLATE"

    def scan(self, text: str) -> dict[str, Any]:
        """Scan text for structural markers.

        Returns:
            {
                "has_plan": bool,
                "plans": [PlanSpec, ...],
                "templates": [TemplateRef, ...],
                "creates": [CreateSpec, ...],
                "text_segments": [str, ...],  # Non-structural text
            }
        """
        result = {
            "has_plan": False,
            "plans": [],
            "templates": [],
            "creates": [],
            "text_segments": [],
        }

        # Find plans
        plans = self._extract_plans(text)
        if plans:
            result["has_plan"] = True
            result["plans"] = plans

        # Find template references
        result["templates"] = self._extract_templates(text)

        # Find create directives
        result["creates"] = self._extract_creates(text)

        # Extract plain text segments
        result["text_segments"] = self._extract_text(text)

        return result

    def _extract_plans(self, text: str) -> list[PlanSpec]:
        """Extract [PLAN:...][/PLAN] blocks."""
        plans = []
        pattern = r'\[PLAN:([^\]]*)\](.*?)\[/PLAN\]'
        for match in re.finditer(pattern, text, re.DOTALL):
            header = match.group(1)
            body = match.group(2)
            plan = self._parse_plan_header(header)
            plan.operations = self._parse_plan_body(body)
            plans.append(plan)
        return plans

    def _parse_plan_header(self, header: str) -> PlanSpec:
        """Parse plan header: mode|steps=N|complexity=X|depth=D"""
        spec = PlanSpec()
        parts = [p.strip() for p in header.split("|")]

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "steps":
                    spec.steps = int(value)
                elif key == "complexity":
                    spec.complexity = value
                elif key == "depth":
                    spec.depth = int(value)
            else:
                spec.mode = part

        return spec

    def _parse_plan_body(self, body: str) -> list[dict[str, Any]]:
        """Parse plan body into operations."""
        ops = []
        # Simple parsing: look for opcode-like patterns
        lines = body.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to parse as template ref
            if line.startswith("[TMPL:"):
                tmpl = self._parse_template_ref(line)
                if tmpl:
                    ops.append({"type": "template", "ref": tmpl})
            elif line.startswith("[BLOCK"):
                ops.append({"type": "block", "raw": line})
            else:
                ops.append({"type": "text", "content": line})
        return ops

    def _extract_templates(self, text: str) -> list[TemplateRef]:
        """Extract [TMPL:hash slot=value ...] references."""
        templates = []
        pattern = r'\[TMPL:([^\s\]]+)([^\]]*)\]'
        for match in re.finditer(pattern, text):
            ref = TemplateRef(hash=match.group(1))
            bindings_str = match.group(2).strip()
            if bindings_str:
                for binding in re.finditer(r'(\w+)=["\']?([^"\'\s\]]+)["\']?', bindings_str):
                    ref.slot_bindings[binding.group(1)] = binding.group(2)
            templates.append(ref)
        return templates

    def _extract_creates(self, text: str) -> list[CreateSpec]:
        """Extract [CREATE:name|slots=N|from=hash1,hash2] directives."""
        creates = []
        pattern = r'\[CREATE:([^\]]*)\]'
        for match in re.finditer(pattern, text):
            spec = CreateSpec()
            parts = [p.strip() for p in match.group(1).split("|")]
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    if key.strip() == "slots":
                        spec.n_slots = int(value.strip())
                    elif key.strip() == "from":
                        spec.from_hashes = [h.strip() for h in value.split(",")]
                else:
                    spec.name = part
            creates.append(spec)
        return creates

    def _extract_text(self, text: str) -> list[str]:
        """Extract non-structural text segments."""
        # Remove all structural markers
        clean = re.sub(r'\[PLAN:[^\]]*\].*?\[/PLAN\]', '', text, flags=re.DOTALL)
        clean = re.sub(r'\[TMPL:[^\]]*\]', '', clean)
        clean = re.sub(r'\[CREATE:[^\]]*\]', '', clean)
        clean = re.sub(r'\[BLOCK[^\]]*\]', '', clean)
        clean = re.sub(r'\[/BLOCK\]', '', clean)
        clean = re.sub(r'\[ARG[^\]]*\]', '', clean)

        segments = [s.strip() for s in clean.split("\n") if s.strip()]
        return segments
