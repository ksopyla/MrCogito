"""Unit tests for the E01 autoregressive concept-conditioned decoder.

Covers the spec's success-critical wiring: output shapes, decoder causality,
that the concepts actually feed the decoder (zero/shuffle change the loss), and
that decoder-input word-dropout routes through the learned dropout embedding.
Tiny random tensors only — runs on CPU.
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import ConceptEncoderForConditionalLM
from training.train_perceiver_denoise import (
    DataTrainingArguments,
    ModelArguments,
    build_perceiver_denoise_config,
)


def _tiny_config(**overrides) -> ConceptEncoderConfig:
    base = dict(
        vocab_size=40,
        hidden_size=32,
        token_embedding_dim=16,        # asymmetry kept (Ht < H), like E01
        concept_num=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        decoder_num_layers=2,          # lean decoder < encoder
        decoder_type="causal_ar",
        decoder_pos_type="rope",
        decoder_word_dropout=0.0,
        hidden_act="silu",             # SwiGLU
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.update(overrides)
    return ConceptEncoderConfig(**base)


def test_forward_shapes_and_finite_loss():
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config)
    B, T = 2, 16
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.logits.shape == (B, T, config.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_prefix_suffix_forward_shapes_and_finite_loss():
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config)
    B, P, S = 2, 7, 10
    prefix_input_ids = torch.randint(3, config.vocab_size, (B, P))
    prefix_attention_mask = torch.ones_like(prefix_input_ids)
    suffix_input_ids = torch.randint(3, config.vocab_size, (B, S))
    labels = suffix_input_ids.clone()

    out = model(
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        suffix_input_ids=suffix_input_ids,
        labels=labels,
    )
    assert out.logits.shape == (B, S, config.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_decoder_self_attention_is_causal():
    """Changing a future target token must not change earlier-position logits."""
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 1, 12
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits_a = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Flip the LAST decoder-input position: decoder_input is shift-right of input_ids,
        # so changing input_ids[:, -1] only affects the decoder input at the last position.
        perturbed = input_ids.clone()
        perturbed[:, -1] = (perturbed[:, -1] + 5) % config.vocab_size
        logits_b = model(input_ids=perturbed, attention_mask=attention_mask).logits

    # Concepts are recomputed from the encoder (bidirectional), so only compare the
    # decoder causal behavior by holding concepts fixed:
    with torch.no_grad():
        concepts = model.encode_concepts(input_ids, attention_mask, return_dict=True).last_hidden_state
        dec_in = model._shift_right(input_ids)
        dec_in_perturbed = dec_in.clone()
        dec_in_perturbed[:, -1] = (dec_in_perturbed[:, -1] + 5) % config.vocab_size
        la = model.decode_logits(concepts, dec_in)
        lb = model.decode_logits(concepts, dec_in_perturbed)

    # All positions except the last must be identical (causal mask).
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1], atol=1e-5)
    del logits_a, logits_b


def test_prefix_suffix_decoder_self_attention_is_causal():
    """Suffix AR decoding must not let future suffix tokens affect earlier logits."""
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config).eval()
    B, P, S = 1, 8, 12
    prefix_input_ids = torch.randint(3, config.vocab_size, (B, P))
    prefix_attention_mask = torch.ones_like(prefix_input_ids)
    suffix_input_ids = torch.randint(3, config.vocab_size, (B, S))

    with torch.no_grad():
        concepts = model.encode_concepts(
            prefix_input_ids,
            prefix_attention_mask,
            return_dict=True,
        ).last_hidden_state
        dec_in = model._shift_right(suffix_input_ids)
        dec_in_perturbed = dec_in.clone()
        dec_in_perturbed[:, -1] = (dec_in_perturbed[:, -1] + 5) % config.vocab_size
        la = model.decode_logits(concepts, dec_in)
        lb = model.decode_logits(concepts, dec_in_perturbed)

    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1], atol=1e-5)


def test_loss_is_single_shift_teacher_forcing():
    """Regression guard for the E01-warmup double-shift bug.

    Decoder inputs are already shift-right-ed ([bos, x0..x_{N-2}]), so logits[t]
    is conditioned on [bos..x_{t-1}] and must be scored against labels[t] = x_t
    directly. The buggy version shifted again (logits[:-1] vs labels[1:]), pairing
    logits[t] with x_{t+1} — a skip-one objective where the decoder never sees the
    immediately preceding token of its target.
    """
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 2, 12
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, -2] = -100  # mixed ignore positions must be honored too

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        expected = torch.nn.functional.cross_entropy(
            out.logits.view(-1, config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
    assert torch.allclose(out.loss, expected, atol=1e-6)

    # And the same single-shift contract on the prefix->suffix path.
    P, S = 6, 10
    prefix_ids = torch.randint(3, config.vocab_size, (B, P))
    suffix_ids = torch.randint(3, config.vocab_size, (B, S))
    suffix_labels = suffix_ids.clone()
    with torch.no_grad():
        out_ps = model(
            prefix_input_ids=prefix_ids,
            prefix_attention_mask=torch.ones_like(prefix_ids),
            suffix_input_ids=suffix_ids,
            labels=suffix_labels,
        )
        expected_ps = torch.nn.functional.cross_entropy(
            out_ps.logits.view(-1, config.vocab_size),
            suffix_labels.view(-1),
            ignore_index=-100,
        )
    assert torch.allclose(out_ps.loss, expected_ps, atol=1e-6)


def test_concepts_are_used_zero_and_shuffle_change_loss():
    # Seeded: an unlucky identity randperm in the shuffle ablation makes delta_shuffle
    # exactly 0 and the test flaky (observed in full-suite runs).
    torch.manual_seed(0)
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 4, 16
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    m = model.concept_ablation_ce(input_ids, attention_mask, labels)
    # Zeroing / shuffling concepts must change next-token CE (concepts wired into path).
    assert abs(m["delta_zero"]) > 1e-4
    assert abs(m["delta_shuffle"]) > 1e-4


def test_prefix_suffix_concepts_are_used_zero_and_shuffle_change_loss():
    torch.manual_seed(0)
    config = _tiny_config()
    model = ConceptEncoderForConditionalLM(config).eval()
    B, P, S = 4, 8, 12
    prefix_input_ids = torch.randint(3, config.vocab_size, (B, P))
    prefix_attention_mask = torch.ones_like(prefix_input_ids)
    suffix_input_ids = torch.randint(3, config.vocab_size, (B, S))
    labels = suffix_input_ids.clone()

    m = model.concept_ablation_ce(
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        suffix_input_ids=suffix_input_ids,
        labels=labels,
    )
    assert abs(m["delta_zero"]) > 1e-4
    assert abs(m["delta_shuffle"]) > 1e-4


def test_word_dropout_routes_through_learned_embedding():
    # hidden_dropout_prob=0 makes embed_dropout an identity. With p=1.0 and the RoPE path
    # (no abs-pos add), every decoder-input embedding should equal the learned dropout
    # embedding exactly. Callers gate on training mode; embed() honors the explicit rate.
    config = _tiny_config(decoder_word_dropout=1.0, hidden_dropout_prob=0.0)
    model = ConceptEncoderForConditionalLM(config).train()
    B, T = 2, 16
    dec_in = torch.randint(3, config.vocab_size, (B, T))

    emb = model.decoder.embed(dec_in, word_dropout_p=1.0)
    expected = model.decoder.dropout_embedding.expand(B, T, config.hidden_size)
    assert torch.allclose(emb, expected, atol=1e-6)


def test_word_dropout_applies_in_eval_when_explicitly_requested():
    """Eval-mode diagnostics (ce_intact_wd) must be able to force the train-matched rate."""
    config = _tiny_config(hidden_dropout_prob=0.0)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 2, 16
    dec_in = torch.randint(3, config.vocab_size, (B, T))

    emb = model.decoder.embed(dec_in, word_dropout_p=1.0)
    expected = model.decoder.dropout_embedding.expand(B, T, config.hidden_size)
    assert torch.allclose(emb, expected, atol=1e-6)

    # And forward() in eval mode must still use CLEAN inputs (word-dropout off).
    clean = model.decoder.embed(dec_in, word_dropout_p=0.0)
    assert not torch.allclose(clean, expected, atol=1e-6)


def test_concept_ablation_reports_train_matched_word_dropout_ce():
    config = _tiny_config(decoder_word_dropout=0.4)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 4, 16
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    m = model.concept_ablation_ce(input_ids, attention_mask, labels)
    assert "ce_intact_wd" in m and "gap_clean_vs_wd" in m
    assert torch.isfinite(torch.tensor(m["ce_intact_wd"]))
    assert abs(m["gap_clean_vs_wd"] - (m["ce_intact"] - m["ce_intact_wd"])) < 1e-5

    # Without word-dropout in the config, the matched-condition keys must be absent.
    config0 = _tiny_config(decoder_word_dropout=0.0)
    model0 = ConceptEncoderForConditionalLM(config0).eval()
    m0 = model0.concept_ablation_ce(input_ids, attention_mask, labels)
    assert "ce_intact_wd" not in m0


def test_build_config_causal_ar_eval_contract():
    class _Tok:
        pad_token_id, mask_token_id, cls_token_id = 0, None, None
        sep_token_id, bos_token_id, eos_token_id, unk_token_id = None, 1, 2, None

        def __len__(self):
            return 40

    config = build_perceiver_denoise_config(
        _Tok(),
        ModelArguments(
            hidden_size=32,
            token_embedding_dim=16,
            num_hidden_layers=3,
            concept_num=8,
            intermediate_size=64,
            decoder_num_layers=2,
            decoder_type="causal_ar",
            decoder_pos_type="rope",
            decoder_word_dropout=0.4,
            hidden_act="silu",
            norm_type="rmsnorm",
            use_bixt=True,
        ),
        DataTrainingArguments(max_seq_length=16, tokenizer_name="dummy"),
    )
    assert config.checkpoint_family == "concept_ar"
    assert config.canonical_single_eval_mode == "weighted_pool"
    assert config.canonical_pair_eval_mode == "sentence_pair"
    assert config.decoder_type == "causal_ar"
    assert config.decoder_posonly is False
    assert config.norm_type == "rmsnorm"
    assert config.hidden_act == "silu"


def test_build_config_causal_ar_prefix_suffix_objective():
    class _Tok:
        pad_token_id, mask_token_id, cls_token_id = 0, None, None
        sep_token_id, bos_token_id, eos_token_id, unk_token_id = None, 1, 2, None

        def __len__(self):
            return 40

    config = build_perceiver_denoise_config(
        _Tok(),
        ModelArguments(
            hidden_size=32,
            token_embedding_dim=16,
            num_hidden_layers=3,
            concept_num=8,
            intermediate_size=64,
            decoder_num_layers=2,
            decoder_type="causal_ar",
            decoder_pos_type="rope",
            hidden_act="silu",
            norm_type="rmsnorm",
            use_bixt=True,
            objective_variant="prefix_suffix",
        ),
        DataTrainingArguments(max_seq_length=16, tokenizer_name="dummy"),
    )
    assert config.checkpoint_family == "concept_ar"
    assert config.pretraining_objective == "ar_prefix_suffix_generation"
