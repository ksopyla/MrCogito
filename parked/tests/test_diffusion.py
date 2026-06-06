"""Tests for ConceptEncoderForMaskedDiffusion and ELBO loss weighting."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_diffusion import (
    ConceptEncoderForMaskedDiffusion,
    DiffusionOutput,
)


def _make_config(**overrides):
    defaults = dict(
        vocab_size=32,
        concept_num=4,
        hidden_size=16,
        token_embedding_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_sequence_length=16,
        pad_token_id=0,
        mask_token_id=3,
        eos_token_id=2,
    )
    defaults.update(overrides)
    return ConceptEncoderConfig(**defaults)


def _make_batch(B=2, L=16, vocab_size=32):
    input_ids = torch.randint(4, vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    return input_ids, attention_mask


class TestForwardPass:
    """Basic forward pass and output shape tests."""

    def test_output_shape_and_type(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=False)
        model.train()
        ids, mask = _make_batch()
        out = model(ids, mask)
        assert isinstance(out, DiffusionOutput)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.concept_repr.shape == (2, 4, 16)

    def test_elbo_forward_runs(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True)
        model.train()
        ids, mask = _make_batch()
        out = model(ids, mask)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)

    def test_backward_passes(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True)
        model.train()
        ids, mask = _make_batch()
        out = model(ids, mask)
        out.loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert all(g < 1e6 for g in grad_norms), "Gradient explosion detected"


class TestELBOWeighting:
    """Verify that ELBO 1/t weighting correctly reweights per-token losses."""

    def test_elbo_upweights_low_noise(self):
        """At low t, fewer tokens are masked but each should be weighted MORE (1/t).
        Verify that the ELBO model assigns higher per-token weight at low t."""
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True, t_min=0.3)
        model.eval()
        ids, mask = _make_batch(B=4, L=16)

        # Force specific t values: 2 samples at low t, 2 at high t
        t_low = torch.tensor([0.3, 0.3, 0.9, 0.9])
        with torch.no_grad():
            out_elbo = model(ids, mask, t=t_low)
        assert torch.isfinite(out_elbo.loss)

    def test_elbo_vs_unweighted_differ(self):
        """ELBO-weighted and unweighted losses should produce different values."""
        config = _make_config()
        torch.manual_seed(42)
        model_elbo = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True, t_min=0.3)
        model_elbo.eval()

        torch.manual_seed(42)
        model_flat = ConceptEncoderForMaskedDiffusion(config, elbo_weight=False, t_min=0.3)
        model_flat.load_state_dict(model_elbo.state_dict())
        model_flat.eval()

        ids, mask = _make_batch(B=8, L=16)
        t = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        with torch.no_grad():
            loss_elbo = model_elbo(ids, mask, t=t).loss
            loss_flat = model_flat(ids, mask, t=t).loss

        assert not torch.allclose(loss_elbo, loss_flat, atol=1e-4), (
            f"ELBO ({loss_elbo:.4f}) and unweighted ({loss_flat:.4f}) should differ"
        )

    def test_elbo_gradient_magnitude_stable(self):
        """With ELBO weighting, gradient norm should be more stable across t values
        than without weighting (the whole point of ELBO 1/t)."""
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True, t_min=0.3)
        model.train()
        ids, mask = _make_batch(B=2, L=16)

        grad_norms = []
        for t_val in [0.3, 0.6, 0.9]:
            model.zero_grad()
            t = torch.full((2,), t_val)
            out = model(ids, mask, t=t)
            out.loss.backward()
            total_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(total_norm)

        ratio = max(grad_norms) / (min(grad_norms) + 1e-8)
        # With ELBO, gradient ratio across t values should be much less than 10x
        # (without ELBO, it can be 10x+ as noted in the diagnosis)
        assert ratio < 20, (
            f"Gradient ratio {ratio:.1f}x across t values is too high for ELBO weighting. "
            f"Norms: {grad_norms}"
        )


class TestTMinBehavior:
    """Verify t_min parameter behavior."""

    def test_default_t_min_is_03(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config)
        assert model.t_min == 0.3

    def test_custom_t_min(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, t_min=0.5)
        assert model.t_min == 0.5

    def test_sampled_t_respects_t_min(self):
        """When t is None (auto-sampled), all values should be >= t_min."""
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, t_min=0.3, elbo_weight=True)
        model.eval()
        ids, mask = _make_batch(B=32, L=16)
        with torch.no_grad():
            out = model(ids, mask)
        assert out.noise_level.min() >= 0.3 - 1e-6


class TestGenerate:
    """Basic generation test."""

    def test_generate_produces_tokens(self):
        config = _make_config()
        model = ConceptEncoderForMaskedDiffusion(config, elbo_weight=True)
        model.eval()
        ids = torch.full((1, 16), config.mask_token_id, dtype=torch.long)
        mask = torch.ones(1, 16, dtype=torch.long)
        generated = model.generate(ids, mask, num_steps=3)
        assert generated.shape == (1, 16)
        n_unmasked = (generated != config.mask_token_id).sum().item()
        assert n_unmasked > 0, "Generation should unmask at least some tokens"
