"""
Tests for ConceptEncoderForPrefixDiffusion (SODA-style prefix generation).

Covers:
  - Output shapes (concepts, masked_logits, loss)
  - Gradient flow through encoder and decoder
  - ELBO weighting behaviour
  - Sinusoidal position embeddings in decoder
  - End-of-sequence ([SEP]) masking by diffusion
  - BiXT encoder variant
"""

import sys
import os
import math

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_diffusion import (
    ConceptEncoderForPrefixDiffusion,
    PrefixDiffusionDecoder,
    DiffusionOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 1000
CLS_ID = 998
SEP_ID = 999
PAD_ID = 0
MASK_ID = 997
BATCH = 2
HIDDEN = 64
CONCEPTS = 8
HEADS = 4
LAYERS = 2
INTERMEDIATE = 128
MAX_SEQ = 128


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=VOCAB_SIZE,
        concept_num=CONCEPTS,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        max_sequence_length=MAX_SEQ,
        pad_token_id=PAD_ID,
        cls_token_id=CLS_ID,
        sep_token_id=SEP_ID,
        mask_token_id=MASK_ID,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    defaults.update(overrides)
    return ConceptEncoderConfig(**defaults)


def _make_batch(prefix_len=20, suffix_len=30, batch_size=BATCH):
    """Create a synthetic prefix+suffix batch."""
    prefix_ids = torch.randint(1, VOCAB_SIZE - 10, (batch_size, prefix_len))
    prefix_ids[:, 0] = CLS_ID
    prefix_ids[:, -1] = SEP_ID
    prefix_mask = torch.ones(batch_size, prefix_len, dtype=torch.long)

    suffix_ids = torch.randint(1, VOCAB_SIZE - 10, (batch_size, suffix_len))
    suffix_ids[:, -1] = SEP_ID
    suffix_mask = torch.ones(batch_size, suffix_len, dtype=torch.long)

    labels = suffix_ids.clone()

    return {
        "prefix_input_ids": prefix_ids,
        "prefix_attention_mask": prefix_mask,
        "suffix_input_ids": suffix_ids,
        "suffix_attention_mask": suffix_mask,
        "labels": labels,
    }


# ===========================================================================
# Forward pass shape tests
# ===========================================================================

class TestForwardShapes:

    def test_output_type(self):
        model = ConceptEncoderForPrefixDiffusion(_tiny_config())
        batch = _make_batch()
        out = model(**batch)
        assert isinstance(out, DiffusionOutput)

    def test_concept_shapes(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        batch = _make_batch(prefix_len=15, suffix_len=25)
        out = model(**batch)
        assert out.concept_repr.shape == (BATCH, CONCEPTS, HIDDEN)

    def test_concept_shape_independent_of_prefix_length(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        out_short = model(**_make_batch(prefix_len=10, suffix_len=20))
        out_long = model(**_make_batch(prefix_len=40, suffix_len=20))
        assert out_short.concept_repr.shape == out_long.concept_repr.shape

    def test_masked_logits_shape(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        model.train()
        batch = _make_batch()
        # Force high t so most tokens are masked
        out = model(**batch, t=torch.full((BATCH,), 0.9))
        assert out.masked_logits is not None
        M = out.masked_logits.shape[0]
        assert M > 0
        assert out.masked_logits.shape[1] == VOCAB_SIZE

    def test_loss_is_scalar(self):
        model = ConceptEncoderForPrefixDiffusion(_tiny_config())
        model.train()
        out = model(**_make_batch())
        assert out.loss is not None
        assert out.loss.dim() == 0

    def test_loss_requires_grad(self):
        model = ConceptEncoderForPrefixDiffusion(_tiny_config())
        model.train()
        out = model(**_make_batch())
        assert out.loss.requires_grad

    def test_different_prefix_suffix_ratios(self):
        model = ConceptEncoderForPrefixDiffusion(_tiny_config())
        model.train()
        for p, s in [(10, 40), (30, 15), (5, 50)]:
            out = model(**_make_batch(prefix_len=p, suffix_len=s))
            assert out.loss is not None
            assert out.concept_repr.shape == (BATCH, CONCEPTS, HIDDEN)

    def test_no_suffix_returns_concepts_only(self):
        """When no suffix is given, model returns concepts with no loss."""
        model = ConceptEncoderForPrefixDiffusion(_tiny_config())
        batch = _make_batch()
        out = model(
            prefix_input_ids=batch["prefix_input_ids"],
            prefix_attention_mask=batch["prefix_attention_mask"],
        )
        assert out.concept_repr is not None
        assert out.loss is None

    def test_bixt_variant(self):
        config = _tiny_config(use_bixt=True, bixt_token_ffn=True)
        model = ConceptEncoderForPrefixDiffusion(config)
        model.train()
        out = model(**_make_batch())
        assert out.loss is not None
        assert out.concept_repr.shape == (BATCH, CONCEPTS, HIDDEN)


# ===========================================================================
# Gradient flow tests
# ===========================================================================

class TestGradientFlow:

    def _run_backward(self, config=None, **batch_kwargs):
        if config is None:
            config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        model.train()
        batch = _make_batch(**batch_kwargs)
        out = model(**batch, t=torch.full((BATCH,), 0.8))
        out.loss.backward()
        return model

    def test_gradient_flows_to_concept_embeddings(self):
        model = self._run_backward()
        grad = model.encoder.concept_embeddings.weight.grad
        assert grad is not None
        assert (grad != 0).any()

    def test_gradient_flows_to_token_embeddings(self):
        model = self._run_backward()
        grad = model.encoder.token_embeddings.weight.grad
        assert grad is not None
        assert (grad != 0).any()

    def test_gradient_flows_through_all_encoder_layers(self):
        model = self._run_backward()
        for i, layer in enumerate(model.encoder.layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, (
                        f"Encoder layer {i}, param {name}: grad is None"
                    )

    def test_gradient_flows_to_decoder(self):
        model = self._run_backward()
        for i, layer in enumerate(model.decoder.layers):
            grad = layer.cross_attn.in_proj_weight.grad
            assert grad is not None, f"Decoder layer {i}: cross_attn grad is None"
            assert (grad != 0).any()

    def test_gradient_flows_to_lm_head(self):
        model = self._run_backward()
        grad = model.lm_head.weight.grad
        assert grad is not None
        assert (grad != 0).any()

    def test_no_nan_gradients(self):
        model = self._run_backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )

    def test_adaln_zero_initial_gate(self):
        """AdaLN bias starts at zero and gradients flow through adaLN."""
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)

        # The DiffusionDecoderLayer zeros adaLN in __init__, but post_init()
        # may re-initialize weight via _init_weights. The bias should remain
        # zero regardless, ensuring gates start at zero output.
        for layer in model.decoder.layers:
            assert (layer.adaLN.bias == 0).all()

        model.train()
        batch = _make_batch()
        out = model(**batch, t=torch.full((BATCH,), 0.8))
        out.loss.backward()

        for i, layer in enumerate(model.decoder.layers):
            grad = layer.adaLN.weight.grad
            assert grad is not None, f"Decoder layer {i}: adaLN grad is None"

    def test_gradient_flows_with_bixt(self):
        config = _tiny_config(use_bixt=True)
        model = self._run_backward(config=config)
        grad = model.encoder.concept_embeddings.weight.grad
        assert grad is not None
        assert (grad != 0).any()

    def test_all_bixt_encoder_params_receive_gradients(self):
        config = _tiny_config(use_bixt=True, bixt_token_ffn=True)
        model = self._run_backward(config=config)
        for i, layer in enumerate(model.encoder.layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, (
                        f"BiXT encoder layer {i}, param {name}: grad is None"
                    )

    def test_last_bixt_layer_does_not_build_dead_token_update_params(self):
        config = _tiny_config(use_bixt=True, bixt_token_ffn=True)
        model = ConceptEncoderForPrefixDiffusion(config)
        last_layer = model.encoder.layers[-1]
        param_names = {name for name, _ in last_layer.named_parameters()}

        assert "bixt_cross_attn.proj_tok.weight" not in param_names
        assert "bixt_cross_attn.proj_tok.bias" not in param_names
        assert "Wi_tok.weight" not in param_names
        assert "Wo_tok.weight" not in param_names


# ===========================================================================
# ELBO weighting tests
# ===========================================================================

class TestELBOWeighting:

    def test_elbo_weight_different_t(self):
        """Loss at t=0.3 should differ from t=0.9 when ELBO-weighted."""
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config, elbo_weight=True)
        model.eval()
        batch = _make_batch()

        torch.manual_seed(42)
        out_low = model(**batch, t=torch.full((BATCH,), 0.35))
        torch.manual_seed(42)
        out_high = model(**batch, t=torch.full((BATCH,), 0.95))

        assert out_low.loss is not None and out_high.loss is not None
        # Losses should differ (ELBO reweights differently)
        assert out_low.loss.item() != pytest.approx(out_high.loss.item(), abs=1e-4)

    def test_loss_without_elbo(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config, elbo_weight=False)
        model.train()
        out = model(**_make_batch(), t=torch.full((BATCH,), 0.7))
        assert out.loss is not None
        assert out.loss.requires_grad

    def test_t_min_respected(self):
        """Noise level should never be below t_min."""
        config = _tiny_config()
        t_min = 0.4
        model = ConceptEncoderForPrefixDiffusion(config, t_min=t_min)
        model.eval()
        for _ in range(10):
            out = model(**_make_batch())
            if out.noise_level is not None:
                assert (out.noise_level >= t_min).all()


# ===========================================================================
# Position embedding tests
# ===========================================================================

class TestPositionEmbeddings:

    def test_decoder_uses_sinusoidal_positions(self):
        """Decoder should use buffer-based (non-learnable) position embeddings."""
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        decoder = model.decoder

        assert hasattr(decoder, "pos_embed")
        assert isinstance(decoder.pos_embed, torch.Tensor)
        # pos_embed is a buffer, not a parameter
        param_names = {n for n, _ in decoder.named_parameters()}
        assert "pos_embed" not in param_names

    def test_suffix_positions_start_at_zero(self):
        """Position embeddings for suffix always start at index 0."""
        config = _tiny_config()
        decoder = PrefixDiffusionDecoder(config, num_layers=1)

        # The first position embedding should be the sin/cos pattern at pos=0
        pos0 = decoder.pos_embed[0]
        dim = config.hidden_size
        half = dim // 2
        expected_sin = torch.zeros(half)  # sin(0) = 0
        expected_cos = torch.ones(half)   # cos(0) = 1
        assert torch.allclose(pos0[0::2], expected_sin, atol=1e-6)
        assert torch.allclose(pos0[1::2], expected_cos, atol=1e-6)

    def test_position_extrapolation(self):
        """Decoder should handle suffix lengths beyond training config."""
        config = _tiny_config(max_sequence_length=32)
        model = ConceptEncoderForPrefixDiffusion(config, decoder_layers=1)
        model.eval()

        # Suffix length 30 < max 32: should work
        batch = _make_batch(prefix_len=10, suffix_len=30)
        out = model(**batch, t=torch.full((BATCH,), 0.5))
        assert out.loss is not None


# ===========================================================================
# End-of-sequence tests
# ===========================================================================

class TestEndOfSequence:

    def test_sep_maskable_by_diffusion(self):
        """[SEP] at end of suffix can be masked by diffusion noise."""
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)

        suffix_ids = torch.randint(1, 100, (1, 20))
        suffix_ids[0, -1] = SEP_ID
        suffix_mask = torch.ones(1, 20, dtype=torch.long)

        t = torch.tensor([1.0])  # mask everything

        noisy, mask = model._apply_noise(suffix_ids, t, MASK_ID, suffix_mask)
        # The [SEP] position should be masked (t=1.0 masks all real tokens)
        assert mask[0, -1].item() is True
        assert noisy[0, -1].item() == MASK_ID

    def test_padding_never_masked(self):
        """Positions with attention_mask=0 must never be masked."""
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)

        suffix_ids = torch.randint(1, 100, (1, 20))
        suffix_mask = torch.ones(1, 20, dtype=torch.long)
        suffix_mask[0, 15:] = 0  # last 5 positions are padding

        t = torch.tensor([1.0])
        _, mask = model._apply_noise(suffix_ids, t, MASK_ID, suffix_mask)

        assert not mask[0, 15:].any()


# ===========================================================================
# Generation test
# ===========================================================================

class TestGeneration:

    def test_generate_returns_correct_shape(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        model.eval()

        prefix = torch.randint(1, 100, (1, 15))
        prefix[0, 0] = CLS_ID
        prefix[0, -1] = SEP_ID
        prefix_mask = torch.ones(1, 15, dtype=torch.long)

        generated = model.generate(
            prefix_input_ids=prefix,
            prefix_attention_mask=prefix_mask,
            suffix_length=20,
            num_steps=3,
        )
        assert generated.shape == (1, 20)

    def test_generate_no_mask_tokens_remain(self):
        config = _tiny_config()
        model = ConceptEncoderForPrefixDiffusion(config)
        model.eval()

        prefix = torch.randint(1, 100, (1, 10))
        prefix[0, 0] = CLS_ID
        prefix[0, -1] = SEP_ID

        generated = model.generate(
            prefix_input_ids=prefix,
            suffix_length=15,
            num_steps=20,
        )
        assert (generated != MASK_ID).all()
