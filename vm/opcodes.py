"""VM Opcodes — ~80 base operations + META opcodes.

Each opcode is a function: (*args) → OpResult.
Opcodes are stateless; all state flows through the context.

Categories:
  MATH:    Arithmetic and aggregation
  STRING:  Text manipulation
  LIST:    Sequence operations
  DICT:    Key-value operations
  IO:      Read/write/parse
  CONTROL: Flow control
  META:    System self-modification (create templates, register blocks)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ── Result type ──

@dataclass
class OpResult:
    """Result of executing an opcode."""
    value: Any = None
    success: bool = True
    error: Optional[str] = None


# ── Category enum ──

class OpCategory(Enum):
    MATH = "math"
    STRING = "string"
    LIST = "list"
    DICT = "dict"
    IO = "io"
    CONTROL = "control"
    META = "meta"
    BLOCK = "block"


# Type alias for opcode functions
OpcodeFunc = Callable[..., OpResult]


def _ok(value: Any) -> OpResult:
    return OpResult(value=value, success=True)


def _err(msg: str) -> OpResult:
    return OpResult(value=None, success=False, error=msg)


# ── MATH ──

def op_add(a, b):
    return _ok(a + b)

def op_sub(a, b):
    return _ok(a - b)

def op_mul(a, b):
    return _ok(a * b)

def op_div(a, b):
    if b == 0:
        return _err("division by zero")
    return _ok(a / b)

def op_mod(a, b):
    return _ok(a % b)

def op_pow(a, b):
    return _ok(a ** b)

def op_sqrt(a):
    return _ok(math.sqrt(abs(a)))

def op_abs(a):
    return _ok(abs(a))

def op_round(a, decimals=0):
    return _ok(round(a, int(decimals)))

def op_sum(lst):
    return _ok(sum(lst))

def op_mean(lst):
    return _ok(sum(lst) / len(lst) if lst else 0)

def op_max(lst):
    return _ok(max(lst))

def op_min(lst):
    return _ok(min(lst))

def op_count(lst):
    return _ok(len(lst))

def op_is_prime(n):
    n = int(n)
    if n < 2:
        return _ok(False)
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return _ok(False)
    return _ok(True)

def op_gcd(a, b):
    return _ok(math.gcd(int(a), int(b)))


# ── STRING ──

def op_concat(*args):
    return _ok("".join(str(a) for a in args))

def op_split(s, sep=None):
    return _ok(s.split(sep if sep is not None else " "))

def op_join(lst, sep=" "):
    return _ok(sep.join(str(x) for x in lst))

def op_upper(s):
    return _ok(str(s).upper())

def op_lower(s):
    return _ok(str(s).lower())

def op_replace(s, old, new):
    return _ok(str(s).replace(str(old), str(new)))

def op_trim(s):
    return _ok(str(s).strip())

def op_contains(s, sub):
    return _ok(str(sub) in str(s))

def op_length(s):
    return _ok(len(s))

def op_slice(s, start=0, end=None):
    return _ok(s[int(start):end if end is None else int(end)])

def op_format(template, *args):
    return _ok(str(template).format(*args))

def op_regex(s, pattern):
    return _ok(re.findall(str(pattern), str(s)))


# ── LIST ──

def op_map(lst, fn_name, fn_args=None):
    """Map an opcode over a list. fn_args are extra args passed after each element."""
    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None:
        return _err(f"unknown opcode: {fn_name}")
    extra = fn_args or []
    results = []
    for x in lst:
        r = fn(x, *extra)
        if isinstance(r, OpResult):
            results.append(r.value)
        else:
            results.append(r)
    return _ok(results)

def op_filter(lst, fn_name, fn_args=None):
    """Filter list by a comparator. Supports 'gt', 'lt', 'eq', etc."""
    extra = fn_args or []
    # Built-in comparators
    comparators = {
        "gt": lambda x, t: x > t,
        "lt": lambda x, t: x < t,
        "eq": lambda x, t: x == t,
        "gte": lambda x, t: x >= t,
        "lte": lambda x, t: x <= t,
        "ne": lambda x, t: x != t,
    }
    if fn_name in comparators:
        cmp = comparators[fn_name]
        return _ok([x for x in lst if cmp(x, *extra)])

    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None:
        return _err(f"unknown opcode: {fn_name}")
    result = []
    for x in lst:
        r = fn(x, *extra)
        val = r.value if isinstance(r, OpResult) else r
        if val:
            result.append(x)
    return _ok(result)

def op_reduce(lst, fn_name):
    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None or not lst:
        return _ok(lst)
    result = lst[0]
    for x in lst[1:]:
        r = fn(result, x)
        result = r.value if isinstance(r, OpResult) else r
    return _ok(result)

def op_sort(lst, reverse=False):
    return _ok(sorted(lst, reverse=bool(reverse)))

def op_reverse(lst):
    return _ok(list(reversed(lst)))

def op_unique(lst):
    seen = set()
    result = []
    for x in lst:
        key = str(x)
        if key not in seen:
            seen.add(key)
            result.append(x)
    return _ok(result)

def op_flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return _ok(result)

def op_zip(*lists):
    return _ok(list(zip(*lists)))

def op_take(lst, n):
    return _ok(lst[:int(n)])

def op_skip(lst, n):
    return _ok(lst[int(n):])

def op_enumerate(lst):
    return _ok(list(enumerate(lst)))

def op_range(*args):
    return _ok(list(range(*(int(a) for a in args))))


# ── DICT ──

def op_get(d, key, default=None):
    return _ok(d.get(key, default) if isinstance(d, dict) else default)

def op_set(d, key, value):
    result = dict(d) if isinstance(d, dict) else {}
    result[key] = value
    return _ok(result)

def op_keys(d):
    return _ok(list(d.keys()) if isinstance(d, dict) else [])

def op_values(d):
    return _ok(list(d.values()) if isinstance(d, dict) else [])

def op_merge(*dicts):
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return _ok(result)

def op_has_key(d, key):
    return _ok(key in d if isinstance(d, dict) else False)

def op_delete(d, key):
    result = dict(d) if isinstance(d, dict) else {}
    result.pop(key, None)
    return _ok(result)

def op_from_pairs(pairs):
    return _ok(dict(pairs))

def op_group_by(lst, key):
    result = {}
    for item in lst:
        k = item.get(key) if isinstance(item, dict) else str(item)
        result.setdefault(k, []).append(item)
    return _ok(result)


# ── IO ──

def op_parse_json(s):
    return _ok(json.loads(str(s)))

def op_dump_json(obj):
    return _ok(json.dumps(obj, indent=2, default=str))

def op_print(*args):
    result = " ".join(str(a) for a in args)
    return _ok(result)


# ── CONTROL ──

def op_if(condition, then_val, else_val=None):
    return _ok(then_val if condition else else_val)

def op_pipe(value, steps):
    """Pipe: result of each step feeds into next.

    Args:
        value: Initial value
        steps: List of (opcode_name, extra_args) tuples
    """
    for step in steps:
        fn_name, extra_args = step
        fn = OPCODE_REGISTRY.get(fn_name)
        if fn is None:
            return _err(f"unknown opcode in pipe: {fn_name}")
        r = fn(value, *extra_args)
        value = r.value if isinstance(r, OpResult) else r
    return _ok(value)

def op_sequence(*items):
    """Execute opcodes in sequence, return last result."""
    result = None
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            fn_name, fn_args = item
            fn = OPCODE_REGISTRY.get(fn_name)
            if fn:
                r = fn(*fn_args)
                result = r.value if isinstance(r, OpResult) else r
    return _ok(result)


# ── META ──

def op_think(steps=1):
    return _ok(int(steps))

def op_retrieve(key):
    return _ok(None)  # Placeholder — resolved by executor via context

def op_plan(*steps):
    return _ok({"type": "plan", "steps": list(steps)})

def op_replan(reason=""):
    return _ok({"type": "replan", "reason": str(reason)})

def op_abort(reason=""):
    return _ok({"type": "abort", "reason": str(reason)})

def op_create_template(name, slots=0):
    return _ok({"type": "create_template", "name": str(name), "slots": int(slots)})

def op_register_block(block):
    return _ok({"type": "register_block", "block": block})

def op_lookup_hash(h):
    return _ok({"type": "lookup_hash", "hash": str(h)})


# ── BLOCK OPERATIONS (v6) ──

def op_expand_template(template, *bindings):
    return _ok({"type": "expand_template", "template": template, "bindings": list(bindings)})

def op_bind_slot(slot_idx, value):
    return _ok({"type": "bind_slot", "slot_idx": int(slot_idx), "value": value})

def op_merge_block(*blocks):
    return _ok({"type": "merge_block", "blocks": list(blocks)})

def op_hash_block(block):
    if hasattr(block, "semantic_hash"):
        return _ok(block.semantic_hash().hex())
    return _ok(str(hash(str(block))))

def op_rewrite_block(original, replacement):
    return _ok({"type": "rewrite_block", "original": original, "replacement": replacement})

def op_freeze_block(block):
    return _ok({"type": "freeze_block", "block": block})

def op_abstract_block(block):
    return _ok({"type": "abstract_block", "block": block})

def op_inline_block(template):
    return _ok({"type": "inline_block", "template": template})


# ── REGISTRY ──

# Category mapping for each opcode
_OPCODE_CATEGORIES: dict[str, OpCategory] = {}

def _register_category(cat: OpCategory, names: list[str]) -> None:
    for name in names:
        _OPCODE_CATEGORIES[name] = cat

_register_category(OpCategory.MATH, [
    "add", "sub", "mul", "div", "mod", "pow", "sqrt", "abs",
    "round", "sum", "mean", "max", "min", "count", "is_prime", "gcd",
])
_register_category(OpCategory.STRING, [
    "concat", "split", "join", "upper", "lower", "replace",
    "trim", "contains", "length", "slice", "format", "regex",
])
_register_category(OpCategory.LIST, [
    "map", "filter", "reduce", "sort", "reverse", "unique",
    "flatten", "zip", "take", "skip", "enumerate", "range",
])
_register_category(OpCategory.DICT, [
    "get", "set", "keys", "values", "merge", "has_key",
    "delete", "from_pairs", "group_by",
])
_register_category(OpCategory.IO, ["parse_json", "dump_json", "print"])
_register_category(OpCategory.CONTROL, ["if", "pipe", "sequence"])
_register_category(OpCategory.META, [
    "think", "retrieve", "plan", "replan", "abort",
    "create_template", "register_block", "lookup_hash",
])
_register_category(OpCategory.BLOCK, [
    "expand_template", "bind_slot", "merge_block", "hash_block",
    "rewrite_block", "freeze_block", "abstract_block", "inline_block",
])


OPCODE_REGISTRY: dict[str, OpcodeFunc] = {
    # Math
    "add": op_add, "sub": op_sub, "mul": op_mul, "div": op_div,
    "mod": op_mod, "pow": op_pow, "sqrt": op_sqrt, "abs": op_abs,
    "round": op_round, "sum": op_sum, "mean": op_mean, "max": op_max,
    "min": op_min, "count": op_count, "is_prime": op_is_prime, "gcd": op_gcd,
    # String
    "concat": op_concat, "split": op_split, "join": op_join,
    "upper": op_upper, "lower": op_lower, "replace": op_replace,
    "trim": op_trim, "contains": op_contains, "length": op_length,
    "slice": op_slice, "format": op_format, "regex": op_regex,
    # List
    "map": op_map, "filter": op_filter, "reduce": op_reduce,
    "sort": op_sort, "reverse": op_reverse, "unique": op_unique,
    "flatten": op_flatten, "zip": op_zip, "take": op_take,
    "skip": op_skip, "enumerate": op_enumerate, "range": op_range,
    # Dict
    "get": op_get, "set": op_set, "keys": op_keys, "values": op_values,
    "merge": op_merge, "has_key": op_has_key, "delete": op_delete,
    "from_pairs": op_from_pairs, "group_by": op_group_by,
    # IO
    "parse_json": op_parse_json, "dump_json": op_dump_json, "print": op_print,
    # Control
    "if": op_if, "pipe": op_pipe, "sequence": op_sequence,
    # Meta
    "think": op_think, "retrieve": op_retrieve, "plan": op_plan,
    "replan": op_replan, "abort": op_abort,
    "create_template": op_create_template, "register_block": op_register_block,
    "lookup_hash": op_lookup_hash,
    # Block operations
    "expand_template": op_expand_template, "bind_slot": op_bind_slot,
    "merge_block": op_merge_block, "hash_block": op_hash_block,
    "rewrite_block": op_rewrite_block, "freeze_block": op_freeze_block,
    "abstract_block": op_abstract_block, "inline_block": op_inline_block,
}


class OpcodeRegistry:
    """Registry for looking up opcodes by name or category."""

    def __init__(self) -> None:
        self._ops = dict(OPCODE_REGISTRY)
        self._categories = dict(_OPCODE_CATEGORIES)

    def lookup(self, name: str) -> Optional[OpcodeFunc]:
        """Look up an opcode by name."""
        return self._ops.get(name)

    def by_category(self, category: OpCategory) -> dict[str, OpcodeFunc]:
        """Get all opcodes in a category."""
        return {
            name: self._ops[name]
            for name, cat in self._categories.items()
            if cat == category and name in self._ops
        }

    def register(self, name: str, fn: OpcodeFunc, category: OpCategory) -> None:
        """Register a new opcode."""
        self._ops[name] = fn
        self._categories[name] = category

    def names(self) -> list[str]:
        return list(self._ops.keys())
