"""Tests for Za v6 VM."""

import torch
import pytest

from vm.opcodes import (
    OpcodeRegistry, OpCategory, OpResult,
    op_add, op_sub, op_mul, op_div, op_concat, op_split,
    op_map, op_filter, op_sort, op_reverse, op_unique,
    op_get, op_set, op_keys, op_values, op_merge,
    op_if, op_pipe, op_sequence,
)
from vm.context import VMContext
from vm.executor import VMExecutor, ExecutionResult
from vm.sandbox import Sandbox, SandboxPolicy, SandboxViolation


# ── Opcodes ──

class TestOpcodes:
    def test_math_ops(self):
        assert op_add(3, 4).value == 7
        assert op_sub(10, 3).value == 7
        assert op_mul(3, 4).value == 12
        assert op_div(10, 2).value == 5.0

    def test_div_zero(self):
        r = op_div(10, 0)
        assert not r.success

    def test_string_ops(self):
        assert op_concat("hello", " world").value == "hello world"
        assert op_split("a,b,c", ",").value == ["a", "b", "c"]

    def test_list_ops(self):
        assert op_sort([3, 1, 2]).value == [1, 2, 3]
        assert op_reverse([1, 2, 3]).value == [3, 2, 1]
        assert op_unique([1, 2, 2, 3]).value == [1, 2, 3]

    def test_dict_ops(self):
        d = {"a": 1, "b": 2}
        assert op_get(d, "a").value == 1
        r = op_set(d, "c", 3)
        assert r.value == {"a": 1, "b": 2, "c": 3}
        assert op_keys(d).value == ["a", "b"]
        assert op_values(d).value == [1, 2]

    def test_control_flow(self):
        assert op_if(True, "yes", "no").value == "yes"
        assert op_if(False, "yes", "no").value == "no"

    def test_registry_lookup(self):
        reg = OpcodeRegistry()
        add_fn = reg.lookup("add")
        assert add_fn is not None
        r = add_fn(2, 3)
        assert r.value == 5

    def test_registry_category(self):
        reg = OpcodeRegistry()
        math_ops = reg.by_category(OpCategory.MATH)
        assert "add" in math_ops
        assert "sub" in math_ops

    def test_pipe(self):
        r = op_pipe(5, [("mul", [2]), ("add", [3])])
        assert r.value == 13  # (5*2) + 3

    def test_map(self):
        r = op_map([1, 2, 3], "mul", [2])
        assert r.value == [2, 4, 6]

    def test_filter(self):
        r = op_filter([1, 2, 3, 4, 5], "gt", [3])
        assert r.value == [4, 5]


# ── VMContext ──

class TestVMContext:
    def test_push_pop_frame(self):
        ctx = VMContext()
        ctx.push_frame("main")
        ctx.set_local("x", 42)
        assert ctx.get_local("x") == 42

        ctx.push_frame("inner")
        assert ctx.get_local("x") is None  # Not visible in inner frame
        ctx.set_local("x", 99)
        assert ctx.get_local("x") == 99

        ctx.pop_frame()
        assert ctx.get_local("x") == 42  # Restored

    def test_globals(self):
        ctx = VMContext()
        ctx.push_frame("main")
        ctx.set_global("g", "global_val")
        assert ctx.get_global("g") == "global_val"
        ctx.push_frame("inner")
        assert ctx.get_global("g") == "global_val"

    def test_result_stack(self):
        ctx = VMContext()
        ctx.push_frame("main")
        ctx.push_result("a")
        ctx.push_result("b")
        assert ctx.pop_result() == "b"
        assert ctx.pop_result() == "a"

    def test_max_depth(self):
        ctx = VMContext(max_depth=3)
        ctx.push_frame("f1")
        ctx.push_frame("f2")
        ctx.push_frame("f3")
        with pytest.raises(RuntimeError, match="max recursion"):
            ctx.push_frame("f4")

    def test_snapshot_restore(self):
        ctx = VMContext()
        ctx.push_frame("main")
        ctx.set_local("x", 1)
        snap = ctx.snapshot()
        ctx.set_local("x", 999)
        ctx.restore(snap)
        assert ctx.get_local("x") == 1


# ── Sandbox ──

class TestSandbox:
    def test_default_policy(self):
        sb = Sandbox()
        # Default allows all opcodes
        assert sb.is_allowed("add")
        assert sb.is_allowed("concat")

    def test_allowlist(self):
        policy = SandboxPolicy(allowed_opcodes={"add", "sub"})
        sb = Sandbox(policy=policy)
        assert sb.is_allowed("add")
        assert not sb.is_allowed("mul")

    def test_blocklist(self):
        policy = SandboxPolicy(blocked_opcodes={"exec", "eval"})
        sb = Sandbox(policy=policy)
        assert sb.is_allowed("add")
        assert not sb.is_allowed("exec")

    def test_resource_limits(self):
        policy = SandboxPolicy(max_steps=5)
        sb = Sandbox(policy=policy)
        for _ in range(5):
            sb.check_step()
        with pytest.raises(SandboxViolation, match="step limit"):
            sb.check_step()

    def test_memory_limit(self):
        policy = SandboxPolicy(max_memory_bytes=100)
        sb = Sandbox(policy=policy)
        with pytest.raises(SandboxViolation, match="memory"):
            sb.check_memory(200)


# ── VMExecutor ──

class TestVMExecutor:
    def test_execute_simple(self):
        executor = VMExecutor(max_steps=50)
        steps = [("add", [2, 3]), ("mul", [None, 10])]
        result = executor.execute(steps)
        assert result.success
        assert result.final_value == 50  # (2+3) * 10

    def test_execute_string_ops(self):
        executor = VMExecutor(max_steps=50)
        steps = [("concat", ["hello", " world"]), ("upper", [None])]
        result = executor.execute(steps)
        assert result.success
        assert result.final_value == "HELLO WORLD"

    def test_step_limit(self):
        executor = VMExecutor(max_steps=2)
        steps = [("add", [1, 1])] * 10
        result = executor.execute(steps)
        assert not result.success
        assert "step limit" in result.error.lower() or result.steps_executed <= 2

    def test_execute_empty(self):
        executor = VMExecutor()
        result = executor.execute([])
        assert result.success
        assert result.final_value is None

    def test_unknown_opcode(self):
        executor = VMExecutor()
        result = executor.execute([("nonexistent_op_xyz", [1, 2])])
        assert not result.success
