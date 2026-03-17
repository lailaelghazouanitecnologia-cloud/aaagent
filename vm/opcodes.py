"""VM Opcodes — ~80 base operations + META opcodes.

Each opcode is a function: (args, context) → result.
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
from typing import Any, Callable


# Type alias for opcode functions
OpcodeFunc = Callable[[list[Any], dict[str, Any]], Any]


# ── MATH ──

def op_add(args, ctx):
    return args[0] + args[1]

def op_sub(args, ctx):
    return args[0] - args[1]

def op_mul(args, ctx):
    return args[0] * args[1]

def op_div(args, ctx):
    if args[1] == 0:
        return float("inf")
    return args[0] / args[1]

def op_mod(args, ctx):
    return args[0] % args[1]

def op_pow(args, ctx):
    return args[0] ** args[1]

def op_sqrt(args, ctx):
    return math.sqrt(abs(args[0]))

def op_abs(args, ctx):
    return abs(args[0])

def op_round(args, ctx):
    decimals = args[1] if len(args) > 1 else 0
    return round(args[0], int(decimals))

def op_sum(args, ctx):
    return sum(args[0])

def op_mean(args, ctx):
    lst = args[0]
    return sum(lst) / len(lst) if lst else 0

def op_max(args, ctx):
    return max(args[0])

def op_min(args, ctx):
    return min(args[0])

def op_count(args, ctx):
    return len(args[0])

def op_is_prime(args, ctx):
    n = int(args[0])
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def op_gcd(args, ctx):
    return math.gcd(int(args[0]), int(args[1]))


# ── STRING ──

def op_concat(args, ctx):
    return "".join(str(a) for a in args)

def op_split(args, ctx):
    return args[0].split(args[1] if len(args) > 1 else " ")

def op_join(args, ctx):
    sep = args[1] if len(args) > 1 else " "
    return sep.join(str(x) for x in args[0])

def op_upper(args, ctx):
    return str(args[0]).upper()

def op_lower(args, ctx):
    return str(args[0]).lower()

def op_replace(args, ctx):
    return str(args[0]).replace(str(args[1]), str(args[2]))

def op_trim(args, ctx):
    return str(args[0]).strip()

def op_contains(args, ctx):
    return str(args[1]) in str(args[0])

def op_length(args, ctx):
    return len(args[0])

def op_slice(args, ctx):
    start = int(args[1]) if len(args) > 1 else 0
    end = int(args[2]) if len(args) > 2 else None
    return args[0][start:end]

def op_format(args, ctx):
    template = str(args[0])
    return template.format(*args[1:])

def op_regex(args, ctx):
    pattern = str(args[1])
    return re.findall(pattern, str(args[0]))


# ── LIST ──

def op_map(args, ctx):
    fn_name = args[1]
    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None:
        return args[0]
    return [fn([x], ctx) for x in args[0]]

def op_filter(args, ctx):
    fn_name = args[1]
    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None:
        return args[0]
    return [x for x in args[0] if fn([x], ctx)]

def op_reduce(args, ctx):
    fn_name = args[1]
    fn = OPCODE_REGISTRY.get(fn_name)
    if fn is None or not args[0]:
        return args[0]
    result = args[0][0]
    for x in args[0][1:]:
        result = fn([result, x], ctx)
    return result

def op_sort(args, ctx):
    reverse = bool(args[1]) if len(args) > 1 else False
    return sorted(args[0], reverse=reverse)

def op_reverse(args, ctx):
    return list(reversed(args[0]))

def op_unique(args, ctx):
    seen = set()
    result = []
    for x in args[0]:
        key = str(x)
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result

def op_flatten(args, ctx):
    result = []
    for item in args[0]:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

def op_zip(args, ctx):
    return list(zip(*args))

def op_take(args, ctx):
    return args[0][:int(args[1])]

def op_skip(args, ctx):
    return args[0][int(args[1]):]

def op_enumerate(args, ctx):
    return list(enumerate(args[0]))

def op_range(args, ctx):
    if len(args) == 1:
        return list(range(int(args[0])))
    elif len(args) == 2:
        return list(range(int(args[0]), int(args[1])))
    else:
        return list(range(int(args[0]), int(args[1]), int(args[2])))


# ── DICT ──

def op_get(args, ctx):
    d = args[0]
    key = args[1]
    default = args[2] if len(args) > 2 else None
    return d.get(key, default) if isinstance(d, dict) else default

def op_set(args, ctx):
    d = dict(args[0]) if isinstance(args[0], dict) else {}
    d[args[1]] = args[2]
    return d

def op_keys(args, ctx):
    return list(args[0].keys()) if isinstance(args[0], dict) else []

def op_values(args, ctx):
    return list(args[0].values()) if isinstance(args[0], dict) else []

def op_merge(args, ctx):
    result = {}
    for d in args:
        if isinstance(d, dict):
            result.update(d)
    return result

def op_has_key(args, ctx):
    return args[1] in args[0] if isinstance(args[0], dict) else False

def op_delete(args, ctx):
    d = dict(args[0]) if isinstance(args[0], dict) else {}
    d.pop(args[1], None)
    return d

def op_from_pairs(args, ctx):
    return dict(args[0])

def op_group_by(args, ctx):
    lst = args[0]
    key = args[1]
    result = {}
    for item in lst:
        k = item.get(key) if isinstance(item, dict) else str(item)
        result.setdefault(k, []).append(item)
    return result


# ── IO ──

def op_parse_json(args, ctx):
    return json.loads(str(args[0]))

def op_dump_json(args, ctx):
    return json.dumps(args[0], indent=2, default=str)

def op_print(args, ctx):
    result = " ".join(str(a) for a in args)
    ctx.setdefault("_output", []).append(result)
    return result


# ── CONTROL ──

def op_if(args, ctx):
    condition, then_val = args[0], args[1]
    else_val = args[2] if len(args) > 2 else None
    return then_val if condition else else_val

def op_pipe(args, ctx):
    """Pipe: result of each step feeds into next."""
    value = args[0]
    for fn_name in args[1:]:
        fn = OPCODE_REGISTRY.get(fn_name)
        if fn:
            value = fn([value], ctx)
    return value

def op_sequence(args, ctx):
    """Execute opcodes in sequence, return last result."""
    result = None
    for item in args:
        if isinstance(item, tuple) and len(item) == 2:
            fn_name, fn_args = item
            fn = OPCODE_REGISTRY.get(fn_name)
            if fn:
                result = fn(fn_args, ctx)
    return result


# ── META ──
# These opcodes allow the system to self-modify its block registry.

def op_think(args, ctx):
    """Allocate compute budget (no-op in VM, signal for planner)."""
    steps = int(args[0]) if args else 1
    ctx["_think_budget"] = steps
    return steps

def op_retrieve(args, ctx):
    """Retrieve a value from context."""
    key = str(args[0])
    return ctx.get(key)

def op_plan(args, ctx):
    """Create a plan structure (parsed by planner, not VM)."""
    return {"type": "plan", "steps": args}

def op_replan(args, ctx):
    """Signal to re-plan from current state."""
    ctx["_replan"] = True
    return {"type": "replan", "reason": str(args[0]) if args else ""}

def op_abort(args, ctx):
    """Abort current execution."""
    ctx["_abort"] = True
    return {"type": "abort", "reason": str(args[0]) if args else ""}

def op_create_template(args, ctx):
    """Create a new template block at runtime."""
    return {"type": "create_template", "name": str(args[0]), "slots": int(args[1]) if len(args) > 1 else 0}

def op_register_block(args, ctx):
    """Register a block in the global registry."""
    return {"type": "register_block", "block": args[0]}

def op_lookup_hash(args, ctx):
    """Look up a block by hash."""
    return {"type": "lookup_hash", "hash": str(args[0])}


# ── BLOCK OPERATIONS (new for v6) ──

def op_expand_template(args, ctx):
    """Expand a template by filling its slots."""
    return {"type": "expand_template", "template": args[0], "bindings": args[1:]}

def op_bind_slot(args, ctx):
    """Bind a value to a template slot."""
    return {"type": "bind_slot", "slot_idx": int(args[0]), "value": args[1]}

def op_merge_block(args, ctx):
    """Merge two blocks into one."""
    return {"type": "merge_block", "blocks": args}

def op_hash_block(args, ctx):
    """Compute hash of a block."""
    if hasattr(args[0], "semantic_hash"):
        return args[0].semantic_hash().hex()
    return str(hash(str(args[0])))

def op_rewrite_block(args, ctx):
    """Rewrite a block (T2T correction)."""
    return {"type": "rewrite_block", "original": args[0], "replacement": args[1]}

def op_freeze_block(args, ctx):
    """Mark a block as immutable."""
    return {"type": "freeze_block", "block": args[0]}

def op_abstract_block(args, ctx):
    """Abstract a block into a template (extract common structure)."""
    return {"type": "abstract_block", "block": args[0]}

def op_inline_block(args, ctx):
    """Inline a template block (expand in place)."""
    return {"type": "inline_block", "template": args[0]}


# ── REGISTRY ──

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
