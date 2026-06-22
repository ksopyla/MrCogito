"""Unit tests for E05 — windowed causal decoder + concepts as cross-window memory.

Covers the experiment's single change (sliding-window causal mask on the AR decoder),
the beyond-window concept-ablation metric, and the multi-dataset mix loader spec.
Tiny random tensors / no network — runs on CPU.
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForConditionalLM,
    build_sliding_window_causal_mask,
)


def _tiny_config(**overrides) -> ConceptEncoderConfig:
    base = dict(
        vocab_size=40,
        hidden_size=32,
        token_embedding_dim=16,
        concept_num=8,
        num_hidden_layers=3,
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


def test_window_mask_pattern():
    """Boolean mask is causal AND restricted to the last K tokens."""
    mask = build_sliding_window_causal_mask(6, window=3, device="cpu")
    assert mask.shape == (6, 6)
    assert mask.dtype == torch.bool
    # Row i may attend j iff i-3 < j <= i.
    for i in range(6):
        for j in range(6):
            expected = (j <= i) and (i - j < 3)
            assert bool(mask[i, j]) == expected, (i, j)
    # Diagonal always visible; nothing in the future visible.
    assert mask.diagonal().all()
    assert not torch.triu(mask, diagonal=1).any()


def test_window_default_is_full_causal():
    """decoder_context_window=None keeps the full-causal (is_causal) path: no mask built."""
    config = _tiny_config(decoder_context_window=None)
    model = ConceptEncoderForConditionalLM(config).eval()
    assert model.decoder._sliding_window_mask(16, torch.device("cpu")) is None
    # Window >= seq_len also falls back to the cheap path.
    config2 = _tiny_config(decoder_context_window=64)
    model2 = ConceptEncoderForConditionalLM(config2).eval()
    assert model2.decoder._sliding_window_mask(16, torch.device("cpu")) is None


def test_windowed_decoder_forward_shapes_and_finite_loss():
    config = _tiny_config(decoder_context_window=4)
    model = ConceptEncoderForConditionalLM(config)
    B, T = 2, 24
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.logits.shape == (B, T, config.vocab_size)
    assert torch.isfinite(out.loss)


def test_single_layer_window_blocks_far_back_context():
    """With ONE decoder layer the receptive field is exactly K: a perturbation at p cannot
    reach positions p+K and beyond, so out-of-window context must travel through concepts.

    (With L stacked window layers the effective field grows to ~L*(K-1)+1 — that depth
    interaction is checked separately in test_window_receptive_field_grows_with_depth.)
    """
    K = 4
    config = _tiny_config(decoder_context_window=K, decoder_num_layers=1)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 1, 20
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        concepts = model.encode_concepts(input_ids, attention_mask, return_dict=True).last_hidden_state
        dec_in = model._shift_right(input_ids)
        p = 5
        dec_in_pert = dec_in.clone()
        dec_in_pert[:, p] = (dec_in_pert[:, p] + 7) % config.vocab_size
        la = model.decode_logits(concepts, dec_in)
        lb = model.decode_logits(concepts, dec_in_pert)

    diff = (la - lb).abs().amax(dim=-1)[0]  # [T]
    # Far-back position (p + K and beyond) is unaffected by the perturbation at p.
    assert torch.allclose(la[:, p + K:], lb[:, p + K:], atol=1e-5)
    # At least one within-window position from p does change (the perturbation matters).
    assert diff[p:p + K].max() > 1e-5


def test_window_receptive_field_grows_with_depth():
    """L stacked window-K layers reach ~L*(K-1) back, but still finitely — positions far
    beyond L*(K-1) stay blind to a perturbation, so deep stacks must use concepts too.

    Design note: the effective local field is ~decoder_num_layers*(K-1); pick K (and depth)
    so it stays well below the sequence length, else few positions are forced through concepts.
    """
    K, L = 3, 2
    config = _tiny_config(decoder_context_window=K, decoder_num_layers=L)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 1, 24
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        concepts = model.encode_concepts(input_ids, attention_mask, return_dict=True).last_hidden_state
        dec_in = model._shift_right(input_ids)
        p = 4
        dec_in_pert = dec_in.clone()
        dec_in_pert[:, p] = (dec_in_pert[:, p] + 7) % config.vocab_size
        la = model.decode_logits(concepts, dec_in)
        lb = model.decode_logits(concepts, dec_in_pert)

    reach = L * (K - 1)  # = 4: positions p+reach+1 and beyond must be blind
    assert torch.allclose(la[:, p + reach + 1:], lb[:, p + reach + 1:], atol=1e-5)


def test_beyond_window_ablation_metrics_present():
    """concept_ablation_ce reports beyond/within-window deltas when window_k is set."""
    config = _tiny_config(decoder_context_window=4)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 3, 20
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    m = model.concept_ablation_ce(input_ids, attention_mask, labels, window_k=4)
    for key in (
        "window_k",
        "ce_intact_beyond_window",
        "ce_intact_within_window",
        "delta_zero_beyond_window",
        "delta_shuffle_beyond_window",
    ):
        assert key in m
    assert m["window_k"] == 4
    # Without window_k the beyond-window keys are absent (back-compat).
    m2 = model.concept_ablation_ce(input_ids, attention_mask, labels)
    assert "delta_zero_beyond_window" not in m2


def test_long_context_mix_is_registered_and_normalisable():
    """The long-context base mix exists, sums sensibly, and references real loadable sources."""
    from data.dataset_preprocess import DATASET_MIXES

    mix = DATASET_MIXES["long_2k_base_v1"]
    assert len(mix) >= 2
    total_w = sum(s["weight"] for s in mix)
    assert abs(total_w - 1.0) < 1e-6
    for spec in mix:
        assert spec.get("hf_id") or spec.get("data_files")
        assert spec.get("text_columns")
        assert spec.get("max_samples", 0) > 0
    # FinePDFs (the long-doc backbone) carries the most weight.
    backbone = max(mix, key=lambda s: s["weight"])
    assert "finepdfs" in backbone["name"].lower()
