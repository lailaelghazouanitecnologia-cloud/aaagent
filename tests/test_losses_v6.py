"""Tests for Za v6 losses."""

import torch
import pytest

from losses.slot_loss import SlotLoss
from losses.block_loss import BlockLoss
from losses.template_loss import TemplateLoss
from losses.plan_loss import PlanLoss
from losses.combined_v6 import CombinedLossV6


B, T, V = 2, 32, 1000  # batch, seq, vocab


def _logits():
    return torch.randn(B, T, V)


def _targets():
    return torch.randint(0, V, (B, T))


def _mask():
    return torch.ones(B, T, dtype=torch.bool)


class TestSlotLoss:
    def test_forward(self):
        loss_fn = SlotLoss(vocab_size=V)
        logits = _logits()
        targets = _targets()
        mask = _mask()

        # Slot positions: mark some positions as slots
        slot_mask = torch.zeros(B, T, dtype=torch.bool)
        slot_mask[:, 5:10] = True

        loss = loss_fn(logits, targets, mask, slot_mask)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_no_slots(self):
        loss_fn = SlotLoss(vocab_size=V)
        slot_mask = torch.zeros(B, T, dtype=torch.bool)
        loss = loss_fn(_logits(), _targets(), _mask(), slot_mask)
        assert loss.item() == 0.0


class TestBlockLoss:
    def test_forward(self):
        loss_fn = BlockLoss(embed_dim=384)
        # Block boundary predictions vs ground truth
        boundary_logits = torch.randn(B, T)
        boundary_labels = torch.zeros(B, T)
        boundary_labels[:, [0, 8, 16, 24]] = 1.0  # Block boundaries

        loss = loss_fn(boundary_logits, boundary_labels)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestTemplateLoss:
    def test_forward(self):
        loss_fn = TemplateLoss(embed_dim=384)
        # Template hash prediction
        pred_hash = torch.randn(B, 384)
        true_hash = torch.randn(B, 384)
        true_hash = true_hash / true_hash.norm(dim=-1, keepdim=True)

        loss = loss_fn(pred_hash, true_hash)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestPlanLoss:
    def test_forward(self):
        loss_fn = PlanLoss(n_complexity_classes=3)

        # Complexity prediction
        complexity_logits = torch.randn(B, 3)
        complexity_labels = torch.randint(0, 3, (B,))

        # Step count prediction
        pred_steps = torch.tensor([5.0, 3.0])
        true_steps = torch.tensor([4.0, 3.0])

        loss = loss_fn(complexity_logits, complexity_labels, pred_steps, true_steps)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestCombinedLossV6:
    def test_forward(self):
        loss_fn = CombinedLossV6(
            vocab_size=V,
            embed_dim=384,
            lambda_slot=0.5,
            lambda_block=0.3,
            lambda_template=0.1,
            lambda_plan=0.1,
        )

        logits = _logits()
        targets = _targets()
        mask = _mask()

        loss, breakdown = loss_fn(
            logits=logits,
            targets=targets,
            mask=mask,
        )
        assert loss.shape == ()
        assert loss.item() >= 0
        assert "ce" in breakdown

    def test_with_all_components(self):
        loss_fn = CombinedLossV6(
            vocab_size=V,
            embed_dim=384,
            lambda_slot=0.5,
            lambda_block=0.3,
            lambda_template=0.1,
            lambda_plan=0.1,
        )

        slot_mask = torch.zeros(B, T, dtype=torch.bool)
        slot_mask[:, 5:10] = True

        loss, breakdown = loss_fn(
            logits=_logits(),
            targets=_targets(),
            mask=_mask(),
            slot_mask=slot_mask,
            boundary_logits=torch.randn(B, T),
            boundary_labels=torch.zeros(B, T),
        )
        assert loss.item() >= 0
        assert "slot" in breakdown or "block" in breakdown
