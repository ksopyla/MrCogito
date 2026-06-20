"""Unit tests for L1/L3 generation & compression faithfulness metrics.

Tiny random concept_ar model on CPU — checks wiring, shapes, value ranges, and the
order-independent token_f1 helper. We do not assert recovery quality (the model is
untrained); we assert the metrics are finite and in valid ranges, that specificity
returns a finite drop, and that the compression curve buckets the lengths present.
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from analysis.concept_generation_eval import (
    token_f1,
    compute_roundtrip_recovery,
    compute_latent_specificity,
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
        max_sequence_length=32,
        decoder_num_layers=2,
        decoder_type="causal_ar",
        decoder_pos_type="rope",
        decoder_word_dropout=0.0,
        hidden_act="silu",
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.update(overrides)
    return ConceptEncoderConfig(**base)


def _batches(B=4, T=12, vocab=40, n_batches=2):
    out = []
    for _ in range(n_batches):
        ids = torch.randint(3, vocab, (B, T))
        mask = torch.ones_like(ids)
        out.append((ids, mask))
    return out


# --- token_f1 ---

def test_token_f1_identical_is_one():
    assert token_f1([5, 6, 7], [5, 6, 7]) == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1([1, 2], [3, 4]) == 0.0


def test_token_f1_order_independent():
    assert token_f1([5, 6, 7], [7, 6, 5]) == 1.0


def test_token_f1_partial_between_zero_and_one():
    f1 = token_f1([1, 2, 3, 4], [1, 2, 9, 9])
    assert 0.0 < f1 < 1.0


def test_token_f1_empty_is_zero():
    assert token_f1([], [1]) == 0.0


# --- round-trip recovery + compression curve ---

def test_roundtrip_recovery_shapes_and_ranges():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    out = compute_roundtrip_recovery(
        model, _batches(), device="cpu", concept_num=8, free_running_examples=3
    )
    for k in ("teacher_forced_token_acc", "free_running_exact_match", "free_running_token_f1"):
        assert 0.0 <= out[k] <= 1.0
    assert out["free_running_n"] == 3
    assert isinstance(out["compression_curve"], dict) and out["compression_curve"]
    for bucket in out["compression_curve"].values():
        assert bucket["compression_ratio"] >= 1
        assert 0.0 <= bucket["teacher_forced_token_acc"] <= 1.0


def test_compression_curve_buckets_by_ratio():
    # T=24 with C=8 -> ceil(24/8)=3; T=8 -> ceil(8/8)=1. Expect two distinct buckets.
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    long_ids = torch.randint(3, 40, (2, 24))
    short_ids = torch.randint(3, 40, (2, 8))
    batches = [
        (long_ids, torch.ones_like(long_ids)),
        (short_ids, torch.ones_like(short_ids)),
    ]
    out = compute_roundtrip_recovery(model, batches, device="cpu", concept_num=8, free_running_examples=0)
    ratios = {b["compression_ratio"] for b in out["compression_curve"].values()}
    assert 1 in ratios and 3 in ratios


def test_roundtrip_handles_padding():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    ids = torch.randint(3, 40, (3, 10))
    mask = torch.ones_like(ids)
    mask[:, 6:] = 0  # pad the tail
    out = compute_roundtrip_recovery(model, [(ids, mask)], device="cpu", concept_num=8, free_running_examples=2)
    assert 0.0 <= out["teacher_forced_token_acc"] <= 1.0


# --- latent specificity ---

def test_latent_specificity_returns_finite_drop():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    out = compute_latent_specificity(model, _batches(B=4), device="cpu")
    for k in ("specificity_acc_matched", "specificity_acc_shuffled", "specificity_acc_drop"):
        assert out[k] == out[k]  # not NaN
    assert out["specificity_symmetric_kl"] >= 0.0
    # matched/shuffled accuracies are valid probabilities
    assert 0.0 <= out["specificity_acc_matched"] <= 1.0
    assert 0.0 <= out["specificity_acc_shuffled"] <= 1.0


def test_latent_specificity_needs_batch_ge_2():
    model = ConceptEncoderForConditionalLM(_tiny_config()).eval()
    ids = torch.randint(3, 40, (1, 10))
    out = compute_latent_specificity(model, [(ids, torch.ones_like(ids))], device="cpu")
    assert out["specificity_acc_drop"] != out["specificity_acc_drop"]  # NaN, skipped
