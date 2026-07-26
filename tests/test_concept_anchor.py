"""Unit tests for the E03 frozen-encoder hidden-state anchor (de-collapse).

Covers the wiring that must hold regardless of the (network-downloaded) teacher:
backward-compatibility when disabled, the anchor head's output shape, that the
anchor gradient reaches BOTH the head and the encoder/concepts (the de-collapse
pressure), and the disabled-path guard. Tiny random tensors only — runs on CPU,
no teacher download (the teacher-driven loss is exercised by the MPS smoke run).
"""

import torch

from nn.concept_encoder import ConceptEncoderConfig
from nn.concept_encoder_perceiver import (
    ConceptEncoderForConditionalLM,
    masked_standardized_mse,
)
from training.train_concept_pretraining import (
    DataTrainingArguments,
    ModelArguments,
    build_perceiver_denoise_config,
)

TEACHER_HIDDEN = 20  # pretend teacher hidden size (≠ model hidden, like SmolLM2's 576 vs 768)


def _tiny_config(**overrides) -> ConceptEncoderConfig:
    base = dict(
        vocab_size=40,
        hidden_size=32,
        token_embedding_dim=16,
        concept_num=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        decoder_num_layers=2,
        decoder_type="causal_ar",
        decoder_pos_type="rope",
        decoder_word_dropout=0.2,
        hidden_act="silu",
        norm_type="rmsnorm",
        use_bixt=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    base.update(overrides)
    return ConceptEncoderConfig(**base)


def test_anchor_disabled_is_backward_compatible():
    """Default (anchor_loss=False) builds NO anchor submodules → identical state_dict to E01."""
    config = _tiny_config()
    assert config.anchor_loss is False
    model = ConceptEncoderForConditionalLM(config)
    assert model.anchor_head is None
    assert not any("anchor_head" in k for k in model.state_dict())


def test_anchor_enabled_builds_head_and_predicts_shape():
    config = _tiny_config(anchor_loss=True, anchor_teacher_hidden=TEACHER_HIDDEN, anchor_head_layers=2)
    model = ConceptEncoderForConditionalLM(config).eval()
    assert model.anchor_head is not None
    assert any("anchor_head" in k for k in model.state_dict())

    B, C, H, N = 2, config.concept_num, config.hidden_size, 12
    concepts = torch.randn(B, C, H)
    pred = model.anchor_predict(concepts, N)
    assert pred.shape == (B, N, TEACHER_HIDDEN)
    assert torch.isfinite(pred).all()


def test_anchor_predict_raises_when_disabled():
    model = ConceptEncoderForConditionalLM(_tiny_config())  # anchor_loss=False
    try:
        model.anchor_predict(torch.randn(1, 8, 32), 5)
    except RuntimeError as exc:
        assert "anchor_loss" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("anchor_predict should raise when the head is not built")


def test_anchor_gradient_reaches_encoder_and_head():
    """The de-collapse pressure must flow through the concepts into the encoder, not only the head."""
    torch.manual_seed(0)
    config = _tiny_config(anchor_loss=True, anchor_teacher_hidden=TEACHER_HIDDEN)
    model = ConceptEncoderForConditionalLM(config).train()
    B, T, N = 2, 16, 16
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    attention_mask = torch.ones_like(input_ids)

    concepts = model.encode_concepts(input_ids, attention_mask, return_dict=True).last_hidden_state
    # Exercise the PRODUCTION loss path (compute_anchor_loss → masked_standardized_mse) with a
    # random stand-in for the frozen teacher's hidden states (no teacher download needed).
    teacher_hidden = torch.randn(B, N, TEACHER_HIDDEN)
    target_mask = torch.ones(B, N)
    loss = model.compute_anchor_loss(concepts, teacher_hidden, target_mask, standardize=True)
    loss.backward()

    assert model.anchor_head.proj.weight.grad is not None
    # concept_embeddings feed concepts → anchor head; a non-None grad proves the path is end-to-end.
    assert model.encoder.concept_embeddings.weight.grad is not None
    assert torch.isfinite(model.encoder.concept_embeddings.weight.grad).all()


def test_anchor_head_is_lean_by_config():
    config = _tiny_config(anchor_loss=True, anchor_teacher_hidden=TEACHER_HIDDEN, anchor_head_layers=1)
    model = ConceptEncoderForConditionalLM(config)
    assert len(model.anchor_head.layers) == 1


def test_forward_unchanged_by_anchor_flag_no_teacher_needed():
    """model.forward (the eval/AR path) never touches the teacher and stays well-formed with anchor on."""
    config = _tiny_config(anchor_loss=True, anchor_teacher_hidden=TEACHER_HIDDEN)
    model = ConceptEncoderForConditionalLM(config).eval()
    B, T = 2, 16
    input_ids = torch.randint(3, config.vocab_size, (B, T))
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids.clone())
    assert out.logits.shape == (B, T, config.vocab_size)
    assert torch.isfinite(out.loss)


def test_build_config_leaves_anchor_off_by_default():
    """build_perceiver_denoise_config without anchor_loss must not set anchor fields (no teacher fetch)."""

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
        ),
        DataTrainingArguments(max_seq_length=16, tokenizer_name="dummy"),
    )
    assert config.anchor_loss is False
    assert config.anchor_teacher_hidden is None


def test_masked_standardized_mse_ignores_padding():
    """The masked MSE must be invariant to whatever sits at padded positions."""
    torch.manual_seed(0)
    B, N, D = 2, 6, 8
    pred = torch.randn(B, N, D)
    target = torch.randn(B, N, D)
    mask = torch.ones(B, N)
    mask[:, -2:] = 0  # last two positions are padding

    loss = masked_standardized_mse(pred, target, mask, standardize=True)
    assert torch.isfinite(loss) and loss.item() > 0

    perturbed = pred.clone()
    perturbed[:, -2:] += 100.0  # garbage in the masked region must not move the loss
    loss_perturbed = masked_standardized_mse(perturbed, target, mask, standardize=True)
    assert torch.allclose(loss, loss_perturbed, atol=1e-5)


def test_masked_standardized_mse_zero_for_matched_prediction():
    """A prediction equal to the (standardized) target gives ~0 loss; standardize flag is honored."""
    torch.manual_seed(0)
    B, N, D = 2, 5, 8
    target = torch.randn(B, N, D)
    mask = torch.ones(B, N)

    std_target = torch.nn.functional.layer_norm(target.float(), (D,))
    assert masked_standardized_mse(std_target, target, mask, standardize=True).item() < 1e-6
    assert masked_standardized_mse(target, target, mask, standardize=False).item() < 1e-6
