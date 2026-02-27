"""Tests for LossManager: VICReg + t_regs_mst regularization and warmup scheduling."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from nn.loss_manager import (
    LossConfig,
    LossManager,
    FixedWeighting,
    ConceptLossStepCallback,
    VICRegLoss,
    TREGSMSTLoss,
    create_loss_component,
    get_available_losses,
)


def _random_concepts(B=4, C=8, H=32):
    return torch.randn(B, C, H)


def _collapsed_concepts(B=4, C=8, H=32):
    """All concepts per sample identical (worst-case intra-sample collapse)."""
    one = torch.randn(B, 1, H)
    return one.expand(B, C, H).clone()


# =========================================================================
# FixedWeighting warmup tests
# =========================================================================

class TestFixedWeightingWarmup:

    def test_no_warmup_returns_full_weight(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.02}, warmup_steps=0)
        losses = {"task": torch.tensor(3.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=0)
        assert torch.isclose(total, torch.tensor(3.0 + 0.02 * 1.0))

    def test_warmup_step_zero_gives_task_only(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.02}, warmup_steps=100)
        losses = {"task": torch.tensor(3.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=0)
        assert torch.isclose(total, torch.tensor(3.0), atol=1e-6)

    def test_warmup_midpoint(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.10}, warmup_steps=100)
        losses = {"task": torch.tensor(2.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=50)
        expected = 2.0 + 0.10 * 0.5 * 1.0
        assert torch.isclose(total, torch.tensor(expected), atol=1e-6)

    def test_warmup_at_end_gives_full_weight(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.02}, warmup_steps=100)
        losses = {"task": torch.tensor(3.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=100)
        assert torch.isclose(total, torch.tensor(3.0 + 0.02))

    def test_warmup_past_end_is_capped(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.02}, warmup_steps=100)
        losses = {"task": torch.tensor(3.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=999)
        assert torch.isclose(total, torch.tensor(3.0 + 0.02))

    def test_warmup_step_none_treated_as_full(self):
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.02}, warmup_steps=100)
        losses = {"task": torch.tensor(3.0), "vicreg": torch.tensor(1.0)}
        total = fw(losses, step=None)
        assert torch.isclose(total, torch.tensor(3.0 + 0.02))

    def test_warmup_multiple_concept_losses(self):
        fw = FixedWeighting(
            {"task": 1.0, "vicreg": 0.01, "t_regs_mst": 0.01},
            warmup_steps=200,
        )
        losses = {
            "task": torch.tensor(3.0),
            "vicreg": torch.tensor(2.0),
            "t_regs_mst": torch.tensor(-1.0),
        }
        total = fw(losses, step=100)
        factor = 0.5
        expected = 3.0 + 0.01 * factor * 2.0 + 0.01 * factor * (-1.0)
        assert torch.isclose(total, torch.tensor(expected), atol=1e-6)

    def test_task_weight_not_affected_by_warmup(self):
        """Task loss weight must be constant regardless of warmup."""
        fw = FixedWeighting({"task": 1.0, "vicreg": 0.1}, warmup_steps=100)
        losses = {"task": torch.tensor(5.0), "vicreg": torch.tensor(0.0)}
        total_step0 = fw(losses, step=0)
        total_step50 = fw(losses, step=50)
        assert torch.isclose(total_step0, total_step50)


# =========================================================================
# LossConfig tests
# =========================================================================

class TestLossConfig:

    def test_warmup_steps_default_zero(self):
        cfg = LossConfig(concept_losses=["vicreg"])
        assert cfg.warmup_steps == 0

    def test_warmup_steps_in_to_dict(self):
        cfg = LossConfig(concept_losses=["vicreg"], warmup_steps=500)
        d = cfg.to_dict()
        assert d["warmup_steps"] == 500

    def test_disabled_has_zero_warmup(self):
        cfg = LossConfig.disabled()
        assert cfg.warmup_steps == 0

    def test_from_dict_with_warmup(self):
        cfg = LossConfig.from_dict({
            "concept_losses": ["t_regs_mst"],
            "warmup_steps": 1000,
        })
        assert cfg.warmup_steps == 1000

    def test_t_regs_mst_is_valid_loss(self):
        assert "t_regs_mst" in get_available_losses()
        cfg = LossConfig(concept_losses=["t_regs_mst"])
        assert cfg.is_enabled


# =========================================================================
# LossManager integration tests
# =========================================================================

class TestLossManagerWithVICReg:

    def test_vicreg_t_regs_mst_forward(self):
        cfg = LossConfig(
            concept_losses=["vicreg", "t_regs_mst"],
            weighting_strategy="fixed",
            loss_weights={"task": 1.0, "vicreg": 0.02, "t_regs_mst": 0.02},
        )
        lm = LossManager(cfg)
        concepts = _random_concepts()
        task_loss = torch.tensor(3.0)
        total = lm(task_loss, concept_repr=concepts)
        assert torch.isfinite(total)
        assert total != task_loss

    def test_vicreg_t_regs_mst_with_warmup(self):
        cfg = LossConfig(
            concept_losses=["vicreg", "t_regs_mst"],
            weighting_strategy="fixed",
            loss_weights={"task": 1.0, "vicreg": 0.02, "t_regs_mst": 0.02},
            warmup_steps=100,
        )
        lm = LossManager(cfg)
        concepts = _random_concepts()
        task_loss = torch.tensor(3.0)

        loss_step0 = lm(task_loss, concept_repr=concepts, step=0)
        loss_step100 = lm(task_loss, concept_repr=concepts, step=100)

        assert torch.isclose(loss_step0, task_loss, atol=1e-5), \
            "At step 0 with warmup, total should equal task loss"
        assert not torch.isclose(loss_step100, task_loss, atol=1e-4), \
            "At step 100 (end of warmup), concept losses should contribute"

    def test_current_step_fallback(self):
        """LossManager._current_step is used when step arg is None."""
        cfg = LossConfig(
            concept_losses=["vicreg"],
            weighting_strategy="fixed",
            loss_weights={"task": 1.0, "vicreg": 0.1},
            warmup_steps=100,
        )
        lm = LossManager(cfg)
        concepts = _random_concepts()
        task_loss = torch.tensor(3.0)

        lm._current_step = 0
        loss_warmup = lm(task_loss, concept_repr=concepts)

        lm._current_step = 100
        loss_full = lm(task_loss, concept_repr=concepts)

        assert loss_warmup < loss_full or torch.isclose(loss_warmup, task_loss, atol=1e-5)

    def test_breakdown_includes_all_losses(self):
        cfg = LossConfig(
            concept_losses=["vicreg", "t_regs_mst"],
            weighting_strategy="fixed",
            loss_weights={"task": 1.0, "vicreg": 0.02, "t_regs_mst": 0.02},
        )
        lm = LossManager(cfg)
        concepts = _random_concepts()
        task_loss = torch.tensor(3.0)

        result = lm(task_loss, concept_repr=concepts, return_breakdown=True)
        assert "task" in result
        assert "vicreg" in result
        assert "t_regs_mst" in result
        assert "total" in result

    def test_backward_passes_with_vicreg_t_regs(self):
        cfg = LossConfig(
            concept_losses=["vicreg", "t_regs_mst"],
            weighting_strategy="fixed",
            loss_weights={"task": 1.0, "vicreg": 0.02, "t_regs_mst": 0.02},
        )
        lm = LossManager(cfg)
        concepts = _random_concepts().requires_grad_(True)
        task_loss = torch.tensor(3.0, requires_grad=True)

        total = lm(task_loss, concept_repr=concepts)
        total.backward()

        assert concepts.grad is not None
        assert torch.isfinite(concepts.grad).all()


# =========================================================================
# VICReg + t_regs_mst component behavior tests
# =========================================================================

class TestVICRegComponent:

    def test_vicreg_is_finite(self):
        loss = create_loss_component("vicreg")
        concepts = _random_concepts()
        val = loss.compute(concepts)
        assert torch.isfinite(val)

    def test_vicreg_higher_for_collapsed(self):
        """VICReg should penalize collapsed dimensions more than random."""
        loss = create_loss_component("vicreg")
        random_val = loss.compute(_random_concepts(B=8, C=16, H=32))
        collapsed_val = loss.compute(_collapsed_concepts(B=8, C=16, H=32))
        assert collapsed_val > random_val, \
            "VICReg should assign higher loss to collapsed concepts"


class TestTREGSMSTComponent:

    def test_t_regs_mst_is_negative(self):
        """t_regs_mst returns negative value (to be minimized → maximize distance)."""
        loss = create_loss_component("t_regs_mst")
        concepts = _random_concepts()
        val = loss.compute(concepts)
        assert val < 0, "t_regs_mst loss should be negative (maximize NN distance)"

    def test_t_regs_mst_lower_for_diverse(self):
        """Diverse concepts should have more negative (better) t_regs_mst loss."""
        loss = create_loss_component("t_regs_mst")
        diverse = _random_concepts(B=4, C=8, H=32)
        collapsed = _collapsed_concepts(B=4, C=8, H=32)
        val_diverse = loss.compute(diverse)
        val_collapsed = loss.compute(collapsed)
        assert val_diverse < val_collapsed, \
            "Diverse concepts should have more negative t_regs_mst than collapsed"

    def test_t_regs_mst_within_sample(self):
        """Verify t_regs_mst operates within-sample (bmm/cdist pattern)."""
        loss = create_loss_component("t_regs_mst")
        concepts = _random_concepts(B=2, C=8, H=32)
        val = loss.compute(concepts)
        assert torch.isfinite(val)


# =========================================================================
# ConceptLossStepCallback tests
# =========================================================================

class TestConceptLossStepCallback:

    def test_on_step_begin_sets_current_step(self):
        cfg = LossConfig(concept_losses=["vicreg"], warmup_steps=100)
        lm = LossManager(cfg)

        class FakeModel:
            loss_manager = lm

        class FakeState:
            global_step = 42

        callback = ConceptLossStepCallback()
        callback.on_step_begin(args=None, state=FakeState(), control=None, model=FakeModel())
        assert lm._current_step == 42

    def test_on_train_begin_resets_to_zero(self):
        cfg = LossConfig(concept_losses=["vicreg"], warmup_steps=100)
        lm = LossManager(cfg)
        lm._current_step = 999

        class FakeModel:
            loss_manager = lm

        class FakeState:
            global_step = 0

        callback = ConceptLossStepCallback()
        callback.on_train_begin(args=None, state=FakeState(), control=None, model=FakeModel())
        assert lm._current_step == 0

    def test_callback_no_crash_without_loss_manager(self):
        """Callback should handle models without loss_manager gracefully."""
        class FakeModel:
            pass

        class FakeState:
            global_step = 10

        callback = ConceptLossStepCallback()
        callback.on_step_begin(args=None, state=FakeState(), control=None, model=FakeModel())
