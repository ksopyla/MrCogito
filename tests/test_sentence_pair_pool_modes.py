"""Unit tests for the sentence-pair concept pooling modes (mean vs attention).

The attention-pool is the frozen-encoder probe readout that makes information
*distributed across the C concepts* visible (mean-pool is permutation-invariant
and can hide it). These tests check the wiring: shapes for both the cosine and
classifier paths, that ``pool_mode='mean'`` is unchanged, and that the attention
query is trainable. Tiny random tensors only — runs on CPU.
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    AttentionPool,
    ConceptEncoderForSentencePairClassification,
)


def _tiny_config(**overrides) -> ConceptEncoderConfig:
    base = dict(
        vocab_size=40,
        hidden_size=32,
        token_embedding_dim=16,
        concept_num=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        hidden_act="silu",
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        num_labels=1,
    )
    base.update(overrides)
    return ConceptEncoderConfig(**base)


def _pair_inputs(B=4, L=10, vocab=40):
    ids_a = torch.randint(3, vocab, (B, L))
    ids_b = torch.randint(3, vocab, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    return ids_a, mask, ids_b, mask


def test_attention_pool_module_shapes():
    pool = AttentionPool(hidden_size=32)
    out = pool(torch.randn(4, 8, 32))
    assert out.shape == (4, 32)


def test_mean_mode_is_default_and_has_no_attn_pool():
    model = ConceptEncoderForSentencePairClassification(_tiny_config())
    assert model.pool_mode == "mean"
    assert not hasattr(model, "attn_pool")


def test_attention_mode_builds_pool_and_is_trainable():
    model = ConceptEncoderForSentencePairClassification(_tiny_config(pool_mode="attention"))
    assert model.pool_mode == "attention"
    assert hasattr(model, "attn_pool")
    assert model.attn_pool.query.requires_grad


def test_cosine_path_shapes_both_modes():
    ids_a, m_a, ids_b, m_b = _pair_inputs()
    for mode in ("mean", "attention"):
        model = ConceptEncoderForSentencePairClassification(_tiny_config(pool_mode=mode)).eval()
        with torch.no_grad():
            out = model(
                input_ids_a=ids_a, attention_mask_a=m_a,
                input_ids_b=ids_b, attention_mask_b=m_b,
                cosine_only=True, return_dict=True,
            )
        assert out.logits.shape == (4, 1)
        assert torch.isfinite(out.logits).all()


def test_classifier_path_shapes_both_modes():
    ids_a, m_a, ids_b, m_b = _pair_inputs()
    labels = torch.randint(0, 2, (4,))
    for mode in ("mean", "attention"):
        cfg = _tiny_config(pool_mode=mode, num_labels=2)
        model = ConceptEncoderForSentencePairClassification(cfg)
        out = model(
            input_ids_a=ids_a, attention_mask_a=m_a,
            input_ids_b=ids_b, attention_mask_b=m_b,
            labels=labels, return_dict=True,
        )
        assert out.logits.shape == (4, 2)
        assert out.loss is not None and torch.isfinite(out.loss)


def test_mean_mode_unchanged_by_pool_mode_field():
    # pool_mode='mean' must reproduce the original mean-pool path exactly.
    ids_a, m_a, ids_b, m_b = _pair_inputs()
    torch.manual_seed(0)
    model = ConceptEncoderForSentencePairClassification(_tiny_config(pool_mode="mean")).eval()
    with torch.no_grad():
        concepts = model.encoder(input_ids=ids_a, attention_mask=m_a, return_dict=True).last_hidden_state
        expected = model.pool_norm(concepts.mean(dim=1))
        got = model._pool_concepts(concepts)
    assert torch.allclose(expected, got, atol=1e-6)
