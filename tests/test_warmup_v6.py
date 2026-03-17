"""Tests for Za v6 6-phase warmup."""

import pytest

from training.warmup_v6 import WarmupSchedulerV6, WarmupPhase


class TestWarmupSchedulerV6:
    def _make_scheduler(self, **overrides):
        defaults = dict(
            gate_freeze_steps=500,
            gate_ramp_end=2000,
            cluster_unfreeze=500,
            coarse_unfreeze=3000,
            temp_anneal_start=2000,
            temp_anneal_end=20000,
            temp_start=1.0,
            temp_end=0.3,
            multilevel_start=10000,
            vm_start=30000,
            dynamic_start=50000,
            meta_start=70000,
            block_gate_ramp_start=10000,
            block_gate_ramp_end=20000,
        )
        defaults.update(overrides)
        return WarmupSchedulerV6(**defaults)

    def test_phase_0(self):
        sched = self._make_scheduler()
        state = sched.get_state(0)
        assert state.phase == WarmupPhase.EMBED_FREEZE
        assert state.gate_alpha == 0.0
        assert not state.multilevel_active
        assert not state.vm_active
        assert not state.dynamic_active
        assert not state.meta_active

    def test_phase_1_ramp(self):
        sched = self._make_scheduler()
        state = sched.get_state(1000)
        assert state.phase == WarmupPhase.EMBED_RAMP
        assert 0.0 < state.gate_alpha < 1.0

    def test_phase_2_multilevel(self):
        sched = self._make_scheduler()
        state = sched.get_state(15000)
        assert state.phase == WarmupPhase.MULTILEVEL
        assert state.multilevel_active
        assert not state.vm_active
        assert state.block_gate > 0.0

    def test_phase_3_vm(self):
        sched = self._make_scheduler()
        state = sched.get_state(35000)
        assert state.phase == WarmupPhase.VM
        assert state.multilevel_active
        assert state.vm_active
        assert not state.dynamic_active

    def test_phase_4_dynamic(self):
        sched = self._make_scheduler()
        state = sched.get_state(55000)
        assert state.phase == WarmupPhase.DYNAMIC
        assert state.dynamic_active
        assert not state.meta_active

    def test_phase_5_meta(self):
        sched = self._make_scheduler()
        state = sched.get_state(75000)
        assert state.phase == WarmupPhase.META
        assert state.meta_active
        assert state.dynamic_active
        assert state.vm_active
        assert state.multilevel_active

    def test_temperature_annealing(self):
        sched = self._make_scheduler()
        state_early = sched.get_state(2000)
        state_late = sched.get_state(20000)
        assert state_early.temperature >= state_late.temperature
        assert state_late.temperature == pytest.approx(0.3, abs=0.01)

    def test_gate_ramp(self):
        sched = self._make_scheduler()
        s0 = sched.get_state(0)
        s_mid = sched.get_state(1250)
        s_end = sched.get_state(2000)
        assert s0.gate_alpha == 0.0
        assert 0.0 < s_mid.gate_alpha < 1.0
        assert s_end.gate_alpha == pytest.approx(1.0, abs=0.01)

    def test_block_gate_ramp(self):
        sched = self._make_scheduler()
        s_before = sched.get_state(9000)
        s_mid = sched.get_state(15000)
        s_after = sched.get_state(20000)
        assert s_before.block_gate == 0.0
        assert 0.0 < s_mid.block_gate < 1.0
        assert s_after.block_gate == pytest.approx(1.0, abs=0.01)
