"""Unit tests for the E10 backbone-concept graft (nn/backbone_concept_lm.py).

All tests run on a tiny RANDOM Gemma3TextConfig (no hub access): 6 layers with the native
5-sliding:1-global pattern, sliding_window = concept_block = 8, H=64, V=256.
"""

import pytest
import torch
import torch.nn as nn
from transformers import GenerationConfig

from data.data_collators import DataCollatorForCausalLM
from data.dataset_preprocess import configure_text_tokenizer_for_model_vocab
from analysis.run_e10_comparison import evaluate_length
from nn.backbone_concept_lm import (
    BackboneConceptConfig,
    BackboneConceptLM,
    ConceptWriteHead,
    GlobalLayerWithConceptRead,
    _AttnWithConceptResidual,
)
from training.train_concept_pretraining import align_special_tokens_for_training

VOCAB = 256
K = 8   # sliding window == concept block
H = 64


def tiny_backbone_dict():
    return dict(
        vocab_size=VOCAB,
        hidden_size=H,
        intermediate_size=128,
        num_hidden_layers=6,           # native pattern → layers 0-4 sliding, layer 5 global
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        query_pre_attn_scalar=32,
        sliding_window=K,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        attention_dropout=0.0,
    )


def two_global_backbone_dict():
    config = tiny_backbone_dict()
    config["num_hidden_layers"] = 12
    return config


def make_model(concept_num=4, lora_r=2, seed=0, **config_overrides):
    torch.manual_seed(seed)
    config_kwargs = dict(
        backbone_model="tiny-random-gemma3",
        backbone_config=tiny_backbone_dict(),
        concept_num=concept_num,
        concept_block=K,
        write_num_heads=2,
        lora_r=lora_r,
        lora_dropout=0.0,
    )
    config_kwargs.update(config_overrides)
    cfg = BackboneConceptConfig(**config_kwargs)
    model = BackboneConceptLM(cfg)
    model.eval()
    return model


def make_batch(B=2, S=24, seed=1, pad_row_from=None):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(3, VOCAB, (B, S), generator=g)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    if pad_row_from is not None:
        input_ids[1, pad_row_from:] = 0
        attention_mask[1, pad_row_from:] = 0
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def _assert_close_where_valid(a, b, atol=1e-4):
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    valid = ~torch.isnan(a)
    assert torch.allclose(a[valid], b[valid], atol=atol), (
        f"max diff {(a[valid] - b[valid]).abs().max().item()}"
    )


def test_global_layer_wrapped_and_control_not():
    model = make_model(concept_num=4)
    types = [type(l).__name__ for l in model.backbone.model.layers]
    assert types[5] == "GlobalLayerWithConceptRead"
    assert all(t == "Gemma3DecoderLayer" for t in types[:5])
    control = make_model(concept_num=0)
    assert all(type(l).__name__ == "Gemma3DecoderLayer" for l in control.backbone.model.layers)


def test_shared_depth_mode_discovers_global_layers_and_builds_depth_gates():
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        write_gate_init=0.01,
    )
    assert model.global_layer_indices == (5, 11)
    assert model.write_head.alpha is None
    assert model.write_head.depth_alphas.shape == (2,)
    assert model.write_head.depth_alphas.tolist() == pytest.approx([0.01, 0.01])

    legacy = make_model(backbone_config=two_global_backbone_dict())
    assert legacy.write_head.alpha.ndim == 0
    assert legacy.write_head.depth_alphas is None


def test_shared_depth_writes_once_after_each_global_layer_and_chains_state():
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.2,
        write_gate_init=0.3,
    )
    reads = []
    writes = []
    depth_indices = []
    for layer_index in model.global_layer_indices:
        branch = model.backbone.model.layers[layer_index].read_branch
        original_read = branch.forward

        def tracked_read(x, z, attn, *, _original=original_read):
            reads.append(z.detach().clone())
            return _original(x, z, attn)

        branch.forward = tracked_read

    original_write = model.write_head.forward

    def tracked_write(*args, **kwargs):
        depth_indices.append(kwargs["depth_index"])
        output = original_write(*args, **kwargs)
        writes.append(output.detach().clone())
        return output

    model.write_head.forward = tracked_write
    input_ids, attention_mask, _ = make_batch(B=2, S=K)
    final_state = model.encode_concepts(input_ids, attention_mask).last_hidden_state

    assert depth_indices == [0, 1]
    assert len(reads) == len(writes) == 2
    assert torch.equal(reads[1], writes[0])
    assert torch.equal(final_state, writes[1])


def test_shared_depth_explicit_loop_matches_native_gemma_when_concepts_disabled():
    shared = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        seed=7,
    )
    legacy = make_model(backbone_config=two_global_backbone_dict(), seed=7)
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    shared_ce = shared.per_position_ce(
        input_ids, attention_mask, labels, concept_mode="zero"
    )
    legacy_ce = legacy.per_position_ce(
        input_ids, attention_mask, labels, concept_mode="zero"
    )
    _assert_close_where_valid(shared_ce, legacy_ce, atol=1e-6)


def test_read_concept_norm_is_optional_and_normalizes_low_rms_state():
    default = make_model(concept_num=4)
    enabled = make_model(concept_num=4, read_concept_norm=True)
    assert isinstance(default.backbone.model.layers[5].read_branch.concept_norm, nn.Identity)
    assert isinstance(enabled.backbone.model.layers[5].read_branch.concept_norm, nn.RMSNorm)
    assert not any("read_branch.concept_norm.weight" in key for key in default.state_dict())
    assert any("read_branch.concept_norm.weight" in key for key in enabled.state_dict())

    z = torch.randn(2, 4, H) * 0.03
    normalized = enabled.backbone.model.layers[5].read_branch.concept_norm(z)
    assert normalized.square().mean().sqrt().item() == pytest.approx(1.0, abs=2e-3)


def test_read_concept_norm_forward_and_roundtrip(tmp_path):
    model = make_model(concept_num=4, read_concept_norm=True)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    assert torch.isfinite(model(input_ids, attention_mask, labels=labels).loss)
    model.save_pretrained(tmp_path / "norm_ckpt")
    reloaded = BackboneConceptLM.from_pretrained(tmp_path / "norm_ckpt")
    assert isinstance(reloaded.backbone.model.layers[5].read_branch.concept_norm, nn.RMSNorm)
    assert torch.equal(
        model.backbone.model.layers[5].read_branch.concept_norm.weight,
        reloaded.backbone.model.layers[5].read_branch.concept_norm.weight,
    )


def test_nonzero_gate_init_opens_full_memory_gradient_path():
    model = make_model(
        concept_num=4,
        read_concept_norm=True,
        read_gate_init=0.01,
        write_gate_init=0.01,
    )
    model.train()
    layer = model.backbone.model.layers[5]
    assert layer.gate.item() == pytest.approx(0.01)
    assert model.write_head.alpha.item() == pytest.approx(0.01)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    model(input_ids, attention_mask, labels=labels).loss.backward()

    assert layer.read_branch.concept_norm.weight.grad is not None
    assert layer.read_branch.concept_norm.weight.grad.abs().sum() > 0
    assert model.concept_init.grad is not None and model.concept_init.grad.abs().sum() > 0
    bixt_grads = [p.grad for p in model.write_head.bixt.parameters() if p.grad is not None]
    assert bixt_grads and any(grad.abs().sum() > 0 for grad in bixt_grads)
    lora_b_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.grad is not None
    ]
    assert lora_b_grads and any(grad.abs().sum() > 0 for grad in lora_b_grads)


def test_zero_init_equivalence_blockloop_vs_single_windowed_first_two_blocks():
    """Load-bearing mask/RoPE test: with gates at 0, the block loop matches one forward with
    every layer window-masked EXACTLY for the first two blocks (whose dec_in covers the full
    history) — validating the mask surgery, the per-block position reset, and graft inertness.

    From block 2 on, the two deliberately diverge: stacked windowed layers grow the receptive
    field by (W-1) per layer in the single forward, while the block loop's one-block carry
    truncates history — that truncated context is exactly what the concepts must supply, and
    the trained control arm shares the identical block protocol, so A/B attribution is clean.
    """
    model = make_model(concept_num=4)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    ce_block = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    ce_single = model.per_position_ce(input_ids, attention_mask, labels, mode="single_windowed")
    _assert_close_where_valid(ce_block[:, : 2 * K], ce_single[:, : 2 * K])
    # And the divergence beyond IS there (if it weren't, the recurrence would be pointless).
    valid = ~torch.isnan(ce_block[:, 2 * K :])
    assert (ce_block[:, 2 * K :][valid] - ce_single[:, 2 * K :][valid]).abs().max() > 0


def test_zero_init_equivalence_with_padding_and_ragged_last_block():
    model = make_model(concept_num=4)
    input_ids, attention_mask, labels = make_batch(B=2, S=21, pad_row_from=10)
    ce_block = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    ce_single = model.per_position_ce(input_ids, attention_mask, labels, mode="single_windowed")
    _assert_close_where_valid(ce_block[:, : 2 * K], ce_single[:, : 2 * K])


def test_control_arm_matches_concept_arm_at_zero_gates():
    """concept_num=0 must take the same code path result as gates-at-zero (the two arms are
    identical at step 0)."""
    model = make_model(concept_num=4)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    ce_real = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    ce_zero = model.per_position_ce(
        input_ids, attention_mask, labels, mode="blockwise", concept_mode="zero"
    )
    _assert_close_where_valid(ce_real, ce_zero, atol=1e-6)


def test_forward_loss_finite_and_matches_per_position_mean():
    model = make_model(concept_num=4)
    input_ids, attention_mask, labels = make_batch(B=2, S=21, pad_row_from=10)
    out = model(input_ids, attention_mask, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    pos_ce = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    assert torch.isclose(out.loss.float(), pos_ce.nanmean(), atol=1e-3)


def test_gradients_reach_gate_and_lora_at_init():
    model = make_model(concept_num=4)
    model.train()
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    loss = model(input_ids, attention_mask, labels=labels).loss
    loss.backward()
    gate = model.backbone.model.layers[5].gate
    assert gate.grad is not None and gate.grad.abs().item() > 0
    lora_b_grads = [
        p.grad for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None
    ]
    assert lora_b_grads and any(g.abs().sum() > 0 for g in lora_b_grads)
    # At read-gate=0 the concept state is disconnected from the loss (by design): z0/write
    # grads are exactly zero but must EXIST (DDP find_unused_parameters=False contract).
    assert model.concept_init.grad is not None
    assert model.write_head.alpha.grad is not None


def test_gradients_reach_concept_state_with_open_gates():
    model = make_model(concept_num=4)
    model.train()
    model.backbone.model.layers[5].gate.data.fill_(0.5)     # read gate open
    input_ids, attention_mask, labels = make_batch(B=2, S=24)   # 3 blocks → writes are read
    loss = model(input_ids, attention_mask, labels=labels).loss
    loss.backward()
    assert model.concept_init.grad is not None and model.concept_init.grad.abs().sum() > 0
    # At write-gate α=0 the BiXT write params are still multiplied by tanh(0)=0 → only α
    # itself gets gradient (it opens first). With α open, the write params get signal too.
    assert model.write_head.alpha.grad is not None and model.write_head.alpha.grad.abs() > 0
    model.zero_grad()
    model.write_head.alpha.data.fill_(0.3)                  # write gate open
    loss = model(input_ids, attention_mask, labels=labels).loss
    loss.backward()
    bixt_grads = [p.grad for p in model.write_head.bixt.parameters() if p.grad is not None]
    assert bixt_grads and any(g.abs().sum() > 0 for g in bixt_grads)


def test_gradients_reach_concepts_with_gradient_checkpointing():
    """The production launcher enables checkpointing; the wrapped global layer and its
    mutable per-block concept state must still preserve read/write gradients."""
    model = make_model(concept_num=4)
    model.train()
    model.gradient_checkpointing_enable()
    checkpoint_func = model.backbone.model.layers[0]._gradient_checkpointing_func
    assert checkpoint_func.keywords["use_reentrant"] is False
    model.backbone.model.layers[5].gate.data.fill_(0.5)
    model.write_head.alpha.data.fill_(0.3)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    model(input_ids, attention_mask, labels=labels).loss.backward()
    assert model.concept_init.grad is not None and model.concept_init.grad.abs().sum() > 0
    assert model.write_head.alpha.grad is not None and model.write_head.alpha.grad.abs() > 0
    bixt_grads = [p.grad for p in model.write_head.bixt.parameters() if p.grad is not None]
    assert bixt_grads and any(g.abs().sum() > 0 for g in bixt_grads)


def test_shared_depth_gates_and_writer_receive_finite_gradients():
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_concept_norm=True,
        read_gate_init=0.01,
        write_gate_init=0.01,
    )
    model.train()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    loss = model(input_ids, attention_mask, labels=labels).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert model.write_head.depth_alphas.grad is not None
    assert torch.isfinite(model.write_head.depth_alphas.grad).all()
    writer_grads = [
        parameter.grad
        for parameter in model.write_head.parameters()
        if parameter is not model.write_head.depth_alphas
    ]
    assert writer_grads
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in writer_grads)


def test_shared_depth_non_reentrant_checkpointing_matches_plain_gradients():
    plain = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.1,
        write_gate_init=0.1,
        seed=9,
    )
    checkpointed = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.1,
        write_gate_init=0.1,
        seed=9,
    )
    plain.train()
    checkpointed.train()
    checkpointed.gradient_checkpointing_enable()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K, seed=4)

    plain_loss = plain(input_ids, attention_mask, labels=labels).loss
    checkpointed_loss = checkpointed(input_ids, attention_mask, labels=labels).loss
    plain_loss.backward()
    checkpointed_loss.backward()

    assert torch.allclose(plain_loss, checkpointed_loss, atol=1e-6)
    assert torch.allclose(
        plain.write_head.depth_alphas.grad,
        checkpointed.write_head.depth_alphas.grad,
        atol=1e-6,
    )
    assert torch.isfinite(checkpointed.write_head.depth_alphas.grad).all()


def test_no_leak_from_future_blocks():
    """Perturbing block 2's tokens must not change CE in blocks 0-1 (block causality of both
    the windowed attention and the write recurrence), even with open gates."""
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.5)
    model.write_head.alpha.data.fill_(0.3)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    ce_a = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    perturbed = input_ids.clone()
    perturbed[:, 16:] = torch.randint(3, VOCAB, (2, 8), generator=torch.Generator().manual_seed(9))
    labels_p = perturbed.masked_fill(attention_mask == 0, -100)
    ce_b = model.per_position_ce(perturbed, attention_mask, labels_p, mode="blockwise")
    _assert_close_where_valid(ce_a[:, :16], ce_b[:, :16], atol=1e-6)


def test_open_read_gate_makes_concepts_matter():
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(1.0)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    ce_real = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    ce_zero = model.per_position_ce(
        input_ids, attention_mask, labels, mode="blockwise", concept_mode="zero"
    )
    valid = ~torch.isnan(ce_real)
    assert (ce_real[valid] - ce_zero[valid]).abs().max() > 1e-6


def test_concept_ablation_ce_contract():
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.5)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    metrics = model.concept_ablation_ce(input_ids, attention_mask, labels)
    for key in ("ce_real", "ce_shuffle", "ce_zero", "delta_shuffle", "delta_zero",
                "ce_static", "ce_one_block", "delta_static_beyond",
                "delta_one_block_beyond", "ce_real_carry", "ce_real_beyond",
                "delta_zero_beyond"):
        assert key in metrics, f"missing {key}"
        assert isinstance(metrics[key], float) and metrics[key] == metrics[key]  # not NaN


@pytest.mark.parametrize(
    "concept_mode",
    ["real", "static", "zero", "shuffle", "one_block", "permutation"],
)
def test_shared_depth_preserves_all_ablation_modes(concept_mode):
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.2,
        write_gate_init=0.2,
    )
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    kwargs = {}
    if concept_mode == "permutation":
        kwargs["concept_permutation"] = torch.tensor([1, 0])
    metrics = model.per_position_metrics(
        input_ids,
        attention_mask,
        labels,
        concept_mode=concept_mode,
        **kwargs,
    )
    assert set(metrics) == {"ce", "predictions"}
    assert torch.isfinite(metrics["ce"][~torch.isnan(metrics["ce"])]).all()


def test_recurrence_ablation_modes_isolate_static_and_one_block_memory():
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.8)
    model.write_head.alpha.data.fill_(0.5)
    input_ids, attention_mask, labels = make_batch(B=2, S=32)
    real = model.per_position_ce(
        input_ids, attention_mask, labels, concept_mode="real"
    )
    static = model.per_position_ce(
        input_ids, attention_mask, labels, concept_mode="static"
    )
    one_block = model.per_position_ce(
        input_ids, attention_mask, labels, concept_mode="one_block"
    )
    beyond = slice(2 * K, None)
    assert not torch.allclose(real[:, beyond], static[:, beyond], equal_nan=True)
    # Recurrent and previous-block-only states first diverge after two writes.
    assert not torch.allclose(real[:, 3 * K :], one_block[:, 3 * K :], equal_nan=True)


def test_sparse_per_position_metrics_and_explicit_pair_permutation():
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.8)
    model.write_head.alpha.data.fill_(0.5)
    input_ids, attention_mask, _ = make_batch(B=2, S=32)
    labels = torch.full_like(input_ids, -100)
    labels[:, 3 * K + 1] = input_ids[:, 3 * K + 1]

    real = model.per_position_metrics(input_ids, attention_mask, labels)
    assert torch.isfinite(real["ce"]).sum() == 2
    assert (real["predictions"] != -100).sum() == 2

    shuffled = model.per_position_metrics(
        input_ids,
        attention_mask,
        labels,
        concept_mode="shuffle",
    )
    permuted = model.per_position_metrics(
        input_ids,
        attention_mask,
        labels,
        concept_mode="permutation",
        concept_permutation=torch.tensor([1, 0]),
    )
    _assert_close_where_valid(shuffled["ce"], permuted["ce"], atol=1e-6)
    assert torch.equal(shuffled["predictions"], permuted["predictions"])


def test_explicit_concept_permutation_rejects_invalid_maps():
    model = make_model(concept_num=4)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    with pytest.raises(ValueError, match="requires concept_permutation"):
        model.per_position_metrics(
            input_ids,
            attention_mask,
            labels,
            concept_mode="permutation",
        )
    with pytest.raises(ValueError, match="bijection"):
        model.per_position_metrics(
            input_ids,
            attention_mask,
            labels,
            concept_mode="permutation",
            concept_permutation=torch.tensor([0, 0]),
        )


def test_e10_paired_comparison_contract():
    concept = make_model(concept_num=4)
    control = make_model(concept_num=0)
    concept.backbone.model.layers[5].gate.data.fill_(0.8)
    concept.write_head.alpha.data.fill_(0.5)
    rows, _, _ = make_batch(B=3, S=32)
    report = evaluate_length(
        concept, control, rows, seq_len=32, gap=0.2,
        batch_size=2, device="cpu", seed=42,
    )
    assert report["beyond_local_start"] == 2 * K
    assert "control_minus_concept_beyond_1024" in report
    assert "static_minus_recurrent_beyond_1024" in report
    assert "one_block_minus_recurrent_beyond_1024" in report
    assert report["concept_rank"]["within_sample_rankme_mean"] > 0


def test_concept_gate_metrics_contract():
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.5)
    model.write_head.alpha.data.fill_(0.3)
    metrics = model.concept_gate_metrics()
    assert metrics["concept_gates/read_0"] == pytest.approx(torch.tanh(torch.tensor(0.5)).item())
    assert metrics["concept_gates/read_layer_5"] == metrics["concept_gates/read_0"]
    assert metrics["concept_gates/write"] == pytest.approx(torch.tanh(torch.tensor(0.3)).item())
    assert make_model(concept_num=0).concept_gate_metrics() == {}


def test_shared_depth_gate_metrics_expose_depth_and_layer_keys():
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
    )
    model.write_head.depth_alphas.data.copy_(torch.tensor([0.2, -0.3]))
    metrics = model.concept_gate_metrics()
    for depth_index, layer_index in enumerate((5, 11)):
        expected = torch.tanh(model.write_head.depth_alphas[depth_index]).item()
        assert metrics[f"concept_gates/write_{depth_index}"] == pytest.approx(expected)
        assert metrics[f"concept_gates/write_layer_{layer_index}"] == pytest.approx(expected)
    assert "concept_gates/write" not in metrics


def test_encode_concepts_shape():
    model = make_model(concept_num=4)
    input_ids, attention_mask, _ = make_batch(B=3, S=20)
    z = model.encode_concepts(input_ids, attention_mask).last_hidden_state
    assert z.shape == (3, 4, H)
    assert torch.isfinite(z).all()


def test_control_arm_forward_and_encode_guard():
    model = make_model(concept_num=0)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    out = model(input_ids, attention_mask, labels=labels)
    assert torch.isfinite(out.loss)
    assert model.concept_ablation_ce(input_ids, attention_mask, labels) == {}
    with pytest.raises(RuntimeError):
        model.encode_concepts(input_ids, attention_mask)


def test_save_load_roundtrip(tmp_path):
    model = make_model(concept_num=4)
    model.backbone.model.layers[5].gate.data.fill_(0.3)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    ce_before = model.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    model.save_pretrained(tmp_path / "ckpt")
    reloaded = BackboneConceptLM.from_pretrained(tmp_path / "ckpt")
    reloaded.eval()
    ce_after = reloaded.per_position_ce(input_ids, attention_mask, labels, mode="blockwise")
    _assert_close_where_valid(ce_before, ce_after, atol=1e-5)


def test_shared_depth_save_load_roundtrip_preserves_mode_gates_loss_and_state(tmp_path):
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.2,
        write_gate_init=0.3,
    )
    model.write_head.depth_alphas.data.copy_(torch.tensor([0.25, -0.15]))
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    loss_before = model(input_ids, attention_mask, labels=labels).loss
    state_before = model.encode_concepts(input_ids, attention_mask).last_hidden_state

    model.save_pretrained(tmp_path / "shared_depth_ckpt")
    reloaded = BackboneConceptLM.from_pretrained(tmp_path / "shared_depth_ckpt")
    reloaded.eval()
    loss_after = reloaded(input_ids, attention_mask, labels=labels).loss
    state_after = reloaded.encode_concepts(input_ids, attention_mask).last_hidden_state

    assert reloaded.config.concept_io_mode == "shared_depth_recurrent"
    assert torch.equal(
        model.write_head.depth_alphas, reloaded.write_head.depth_alphas
    )
    assert torch.allclose(loss_before, loss_after, atol=1e-6)
    assert torch.allclose(state_before, state_after, atol=1e-6)


def test_shared_depth_concept_zero_keeps_legacy_control_path():
    shared = make_model(
        concept_num=0,
        concept_io_mode="shared_depth_recurrent",
        seed=13,
    )
    legacy = make_model(concept_num=0, concept_io_mode="global_kv", seed=13)
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    shared_ce = shared.per_position_ce(input_ids, attention_mask, labels)
    legacy_ce = legacy.per_position_ce(input_ids, attention_mask, labels)
    _assert_close_where_valid(shared_ce, legacy_ce, atol=1e-6)


def test_causal_lm_collator():
    class TokStub:
        pad_token_id = 0

        def __len__(self):
            return VOCAB

    collator = DataCollatorForCausalLM(TokStub(), max_length=16)
    batch = collator([{"input_ids": [5, 6, 7]}, {"input_ids": list(range(3, 25))}])
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"][0].sum() == 3
    assert (batch["labels"][0, 3:] == -100).all()
    assert (batch["labels"][1] != -100).all()

    class TokStubWithUnk(TokStub):
        unk_token_id = 1

    model_bounded = DataCollatorForCausalLM(
        TokStubWithUnk(), max_length=16, model_vocab_size=VOCAB - 1
    )
    clamped = model_bounded([{"input_ids": [5, VOCAB - 1]}])
    assert clamped["input_ids"][0, 1].item() == 1  # unk replacement
    assert clamped["labels"][0, 1].item() == -100


def test_causal_lm_collator_preserves_sparse_precomputed_labels():
    class TokStub:
        pad_token_id = 0

        def __len__(self):
            return VOCAB

    collator = DataCollatorForCausalLM(
        TokStub(),
        max_length=5,
        preserve_precomputed_labels=True,
    )
    batch = collator(
        [
            {"input_ids": [5, 6, 7], "labels": [-100, -100, 7]},
            {
                "input_ids": [8, 9, 10, 11, 12, 13],
                "labels": [-100, 9, -100, -100, -100, 13],
            },
        ]
    )
    assert batch["labels"].tolist() == [
        [-100, -100, 7, -100, -100],
        [-100, 9, -100, -100, -100],
    ]
    with pytest.raises(ValueError, match="requires labels on every feature"):
        collator(
            [
                {"input_ids": [5], "labels": [-100]},
                {"input_ids": [6]},
            ]
        )


def test_text_tokenizer_splits_out_of_model_special_tokens():
    class TokStub:
        split_special_tokens = False

        def __len__(self):
            return 101

    tokenizer = TokStub()
    assert configure_text_tokenizer_for_model_vocab(tokenizer, 100)
    assert tokenizer.split_special_tokens is True
    assert not configure_text_tokenizer_for_model_vocab(tokenizer, 101)


def test_tokenize_fn_rejects_ids_outside_model_vocab():
    from data.dataset_preprocess import _make_tokenize_fn

    class TokStub:
        split_special_tokens = False

        def __len__(self):
            return 101

        def __call__(self, texts, **kwargs):
            assert kwargs.get("split_special_tokens") is True
            # Pretend the tokenizer still emitted a tokenizer-only id.
            return {"input_ids": [[0, 100]], "attention_mask": [[1, 1]]}

    tokenize = _make_tokenize_fn(
        TokStub(), max_seq_length=8, append_eos_token_id=None, model_vocab_size=100
    )
    with pytest.raises(ValueError, match="outside model vocabulary range"):
        tokenize({"text": ["hello"]})


def test_filter_rows_outside_model_vocab():
    from datasets import Dataset

    from data.dataset_preprocess import filter_rows_outside_model_vocab

    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [1, 99], [4, 5]],
            "attention_mask": [[1, 1, 1], [1, 1], [1, 1]],
        }
    )
    filtered, dropped = filter_rows_outside_model_vocab(ds, model_vocab_size=10)
    assert dropped == 1
    assert len(filtered) == 2
    assert filtered["input_ids"] == [[1, 2, 3], [4, 5]]


def test_training_special_tokens_align_generation_config_without_warning_path():
    class TokStub:
        pad_token_id = 0
        bos_token_id = 2
        eos_token_id = 1

    model = make_model(concept_num=4)
    model.generation_config = GenerationConfig(
        pad_token_id=0, bos_token_id=2, eos_token_id=[1, 106]
    )
    changes = align_special_tokens_for_training(model, TokStub())

    assert model.config.pad_token_id == 0
    assert model.config.bos_token_id == 2
    assert model.config.eos_token_id == 1
    assert model.generation_config.eos_token_id == 1
    assert changes["generation_config.eos_token_id"] == {"from": [1, 106], "to": 1}


def test_config_facade_attributes_for_shared_plumbing():
    """init_wandb / the FA probe / the encoder-log line read config.hidden_size,
    num_attention_heads, etc. unconditionally. BackboneConceptConfig must mirror the
    backbone's headline dims so that plumbing gets real values, not AttributeError."""
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    bb = Gemma3TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, sliding_window=8,
        max_position_embeddings=128,
    )
    cfg = BackboneConceptConfig(
        backbone_model="tiny", backbone_config=bb.to_dict(),
        concept_num=4, concept_block=8,
    )
    for attr, expected in [
        ("hidden_size", 64), ("num_hidden_layers", 6), ("num_attention_heads", 2),
        ("intermediate_size", 128), ("vocab_size", 256), ("token_embedding_dim", 64),
        ("max_sequence_length", 128), ("head_dim", 32), ("sliding_window", 8),
        ("concept_num", 4), ("checkpoint_family", "backbone_concept"),
    ]:
        assert getattr(cfg, attr) == expected, f"{attr}={getattr(cfg, attr, 'MISSING')}"


def test_wandb_identity_both_arms_share_group_differ_on_arm_tag():
    """The concept/control A/B must share ONE W&B group (so they filter together, like
    E05's optimizer A/B) and differ on the arm tag."""
    from training.utils_training import WandbRunIdentity

    def build(concept_num):
        backbone_model = "google/gemma-3-1b-pt"
        backbone_short = backbone_model.split("/")[-1].replace("-", "_")
        arch = f"backbone_concept_{backbone_short}_K512"
        resolved = "E10"
        arm = "concept-arm" if concept_num > 0 else "control-arm"
        return WandbRunIdentity(
            experiment_id=resolved, model_family="backbone_concept",
            objective_family="causal_lm", architecture_id=arch,
            group=f"{resolved}_{arch}", job_type="train_backbone_causal_lm",
            tags=["train", arm, "lora_r16", resolved],
        )

    concept = build(128)
    control = build(0)
    assert concept.group == control.group              # SAME group (the A/B invariant)
    assert concept.job_type == control.job_type
    assert "concept-arm" in concept.tags
    assert "control-arm" in control.tags
    assert concept.architecture_id == control.architecture_id


def test_next_token_logits_and_generate_shapes():
    model = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        read_gate_init=0.2,
        write_gate_init=0.2,
    )
    input_ids, attention_mask, _ = make_batch(B=2, S=K + 3)
    logits = model.next_token_logits(input_ids, attention_mask)
    assert logits.shape == (2, VOCAB)
    assert torch.isfinite(logits).all()

    greedy = model.generate(
        input_ids[:1], attention_mask[:1], max_new_tokens=4, do_sample=False
    )
    assert greedy.shape == (1, input_ids.shape[1] + 4)

    sampled = model.generate(
        input_ids[:1],
        attention_mask[:1],
        max_new_tokens=3,
        do_sample=True,
        temperature=0.9,
        top_k=20,
        top_p=0.9,
    )
    assert sampled.shape == (1, input_ids.shape[1] + 3)


# ------------------------------------------------------------------ E17 per_layer_banks

def _perlayer_model(**overrides):
    """Two-global-bank BackboneConceptLM (G=2 banks at global layers 5, 11) for E17 tests."""
    defaults = dict(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="per_layer_banks",
        concept_num=4,
        read_gate_init=0.2,
        write_gate_init=0.2,
    )
    defaults.update(overrides)
    return make_model(**defaults)


def _e17c_model(**overrides):
    """Tiny two-bank E17c with the registered architecture and pressure defaults."""
    defaults = dict(
        concept_read_mode="dedicated",
        tie_concept_writer=False,
        concept_write_mode="gated_replace",
        write_update_gate_init=0.25,
        read_concept_norm=True,
        read_gate_init=0.1,
        memory_carry_dropout=0.5,
        memory_pressure_tokens=2,
        memory_pressure_weight=4.0,
    )
    defaults.update(overrides)
    return _perlayer_model(**defaults)


def _trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_per_layer_banks_builds_and_param_count():
    # two_global_backbone_dict -> global layers at indices 5, 11 -> G = 2 banks.
    G = 2
    shared = make_model(
        backbone_config=two_global_backbone_dict(),
        concept_io_mode="shared_depth_recurrent",
        concept_num=4,
    )
    perlayer = _perlayer_model()
    assert tuple(perlayer.concept_init.shape) == (G, 4, H)          # one init per bank
    assert tuple(perlayer.write_head.depth_alphas.shape) == (G,)    # one write gate per bank
    # Machinery (tied writer + per-layer gates) is identical; only bank inits differ.
    assert abs(_trainable_params(perlayer) - _trainable_params(shared)) == (G - 1) * 4 * H


@pytest.mark.parametrize("concept_mode", ["real", "static", "zero", "shuffle", "one_block"])
def test_per_layer_banks_ablation_modes_finite(concept_mode):
    model = _perlayer_model()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    metrics = model.per_position_metrics(
        input_ids, attention_mask, labels, concept_mode=concept_mode
    )
    assert set(metrics) == {"ce", "predictions"}
    assert torch.isfinite(metrics["ce"][~torch.isnan(metrics["ce"])]).all()


def test_per_layer_banks_write_is_per_bank():
    # Each bank is written by exactly one layer; batch-shuffling the banks changes
    # beyond-window CE (the bank carries per-sequence content).
    model = _perlayer_model()
    model.backbone.model.layers[5].gate.data.fill_(0.8)
    model.write_head.depth_alphas.data.fill_(0.5)
    input_ids, attention_mask, labels = make_batch(B=2, S=4 * K)
    real = model.per_position_ce(input_ids, attention_mask, labels, concept_mode="real")
    shuf = model.per_position_ce(input_ids, attention_mask, labels, concept_mode="shuffle")
    beyond = slice(2 * K, None)
    assert not torch.allclose(real[:, beyond], shuf[:, beyond], equal_nan=True)


def test_per_layer_banks_encode_concepts_exposes_last_bank():
    # encode_concepts must return [B, C, H] (last bank) for probe compatibility, not [B, G, C, H].
    model = _perlayer_model()
    input_ids, attention_mask, _ = make_batch(B=2, S=2 * K)
    out = model.encode_concepts(input_ids, attention_mask)
    assert tuple(out.last_hidden_state.shape) == (2, 4, H)


def test_per_layer_banks_checkpoint_roundtrip(tmp_path):
    model = _perlayer_model()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    ce_a = model.per_position_ce(input_ids, attention_mask, labels, concept_mode="real")
    p = tmp_path / "e17_ckpt"
    model.save_pretrained(p)
    reloaded = BackboneConceptLM.from_pretrained(p)
    ce_b = reloaded.per_position_ce(input_ids, attention_mask, labels, concept_mode="real")
    assert torch.allclose(ce_a, ce_b, equal_nan=True)


def test_per_layer_banks_generate_frozen_finite():
    # The frozen decode path must work banked: encode prompt -> all banks -> read-only decode.
    model = _perlayer_model()
    ids = torch.randint(3, VOCAB, (1, 2 * K))
    mask = torch.ones_like(ids)
    out = model.generate(ids, mask, max_new_tokens=6, concept_mode="frozen", eos_token_id=9999)
    assert out.shape[0] == 1 and out.shape[1] >= 2 * K
    assert torch.isfinite(out.float()).all()


# ------------------------------------------------------------------ E17c


def test_e17c_legacy_defaults_preserve_e17b_module_contract():
    legacy = _perlayer_model()
    assert legacy.config.concept_read_mode == "backbone_qkv"
    assert legacy.config.concept_read_placement == "post_layer"
    assert legacy.config.inference_carry_policy == "normal"
    assert legacy.config.tie_concept_writer is True
    assert legacy.config.concept_write_mode == "additive"
    assert legacy.write_head is not None and legacy.write_heads is None
    keys = set(legacy.state_dict())
    assert "write_head.depth_alphas" in keys
    assert not any(key.startswith("write_heads.") for key in keys)
    assert not any(".read_branch.q_proj." in key for key in keys)


def test_e17c_builds_depth_private_readers_and_writers():
    model = _e17c_model()
    assert model.write_head is None
    assert len(model.write_heads) == 2
    assert model.write_heads[0] is not model.write_heads[1]
    assert (
        model.write_heads[0].bixt.rv_lat.weight
        is not model.write_heads[1].bixt.rv_lat.weight
    )
    readers = [
        model.backbone.model.layers[index].read_branch
        for index in model.global_layer_indices
    ]
    assert all(reader.mode == "dedicated" for reader in readers)
    assert readers[0].q_proj.weight is not readers[1].q_proj.weight


def test_e17c_gated_replace_equation_and_padded_identity():
    torch.manual_seed(11)
    writer = ConceptWriteHead(
        H,
        num_heads=2,
        update_mode="gated_replace",
        update_gate_init=0.25,
    ).eval()
    z = torch.randn(2, 4, H)
    h = torch.randn(2, K, H)
    pad = torch.zeros(2, K, dtype=torch.bool)
    pad[1] = True
    safe_pad = pad.clone()
    safe_pad[1, 0] = False
    with torch.no_grad():
        lat, _ = writer.bixt(
            writer.norm_lat(z), writer.norm_tok(h), key_padding_mask=safe_pad
        )
        candidate = writer.sandwich(lat)
        expected = torch.lerp(z, candidate, torch.full((2, 4, 1), 0.25))
        expected[1] = z[1]
        actual = writer(z, h, pad)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert writer._last_update_gate_mean == pytest.approx(0.25, abs=1e-6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed_precision = writer(z, h, pad)
    assert mixed_precision.dtype == z.dtype
    assert torch.isfinite(mixed_precision).all()


def test_e17c_three_block_gradients_reach_every_private_cell():
    model = _e17c_model(memory_carry_dropout=1.0)
    model.train()
    input_ids, attention_mask, labels = make_batch(B=2, S=3 * K)
    loss = model(input_ids, attention_mask, labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.concept_init.grad is not None
    assert model.concept_init.grad.abs().sum() > 0
    for layer_index, writer in zip(model.global_layer_indices, model.write_heads):
        reader = model.backbone.model.layers[layer_index]
        assert reader.gate.grad is not None and reader.gate.grad.abs().sum() > 0
        assert (
            reader.read_branch.q_proj.weight.grad is not None
            and reader.read_branch.q_proj.weight.grad.abs().sum() > 0
        )
        assert (
            writer.update_gate.weight.grad is not None
            and writer.update_gate.weight.grad.abs().sum() > 0
        )
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in writer.bixt.parameters()
        )
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    assert lora_grads and any(
        grad is not None and grad.abs().sum() > 0 for grad in lora_grads
    )


def test_e17c_pressure_masks_only_prior_carry_and_keeps_bos_sentinel():
    model = _e17c_model(memory_carry_dropout=1.0)
    model.train()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    seen_ids = []

    def capture_ids(_module, args):
        seen_ids.append(args[0].detach().clone())

    handle = model.backbone.model.embed_tokens.register_forward_pre_hook(capture_ids)
    try:
        model(input_ids, attention_mask, labels)
    finally:
        handle.remove()
    assert len(seen_ids) == 2
    carry = seen_ids[1][:, :K]
    assert torch.equal(carry[:, :-1], torch.zeros_like(carry[:, :-1]))
    assert torch.equal(carry[:, -1], torch.full_like(carry[:, -1], 2))
    assert torch.equal(seen_ids[1][:, K:], input_ids[:, K:])
    assert model._last_pressure_fraction == pytest.approx(1.0)
    model.eval()
    model.per_position_ce(input_ids, attention_mask, labels)
    assert model._last_pressure_fraction == pytest.approx(1.0)


def test_e17c_weighted_pressure_ce_matches_per_position_reference():
    model = _e17c_model(memory_carry_dropout=1.0)
    input_ids, attention_mask, labels = make_batch(B=2, S=3 * K, pad_row_from=2 * K + 3)
    model.eval()
    pos_ce = model.per_position_ce(
        input_ids,
        attention_mask,
        labels,
        carry_policy="drop_after_first",
    )
    pressure_mask = torch.zeros_like(pos_ce, dtype=torch.bool)
    for block_index in range(1, 3):
        start = block_index * K
        pressure_mask[:, start : start + 2] = True
    valid = ~torch.isnan(pos_ce)
    selected = valid & pressure_mask
    reference = (
        pos_ce[valid].sum() + 3.0 * pos_ce[selected].sum()
    ) / (valid.sum() + 3.0 * selected.sum())
    model.train()
    actual = model(input_ids, attention_mask, labels).loss
    assert torch.allclose(actual, reference, atol=2e-4)


@pytest.mark.parametrize("forced_pressure", [False, True])
def test_e17c_intra_block_causality(forced_pressure):
    model = _e17c_model()
    model.eval()
    input_ids, attention_mask, labels = make_batch(B=2, S=3 * K)
    changed_ids = input_ids.clone()
    changed_labels = labels.clone()
    changed_ids[:, 2 * K + 5 :] = torch.flip(changed_ids[:, 2 * K + 5 :], dims=[1])
    changed_labels[:, 2 * K + 5 :] = changed_ids[:, 2 * K + 5 :]
    carry_policy = "drop_after_first" if forced_pressure else "normal"
    before = model.per_position_ce(
        input_ids, attention_mask, labels, carry_policy=carry_policy
    )
    after = model.per_position_ce(
        changed_ids, attention_mask, changed_labels, carry_policy=carry_policy
    )
    _assert_close_where_valid(before[:, : 2 * K + 5], after[:, : 2 * K + 5])


def test_e17c_bank_api_ablation_and_checkpoint_roundtrip(tmp_path):
    model = _e17c_model(memory_carry_dropout=0.0, memory_pressure_tokens=0, memory_pressure_weight=1.0)
    input_ids, attention_mask, labels = make_batch(B=2, S=3 * K)
    banks = model.encode_concept_banks(input_ids, attention_mask)
    exposed = model.encode_concepts(input_ids, attention_mask).last_hidden_state
    assert banks.shape == (2, 2, 4, H)
    assert torch.allclose(exposed, banks[:, -1])
    permutation = torch.tensor([1, 0])
    for bank_index in range(2):
        ce = model.per_position_ce(
            input_ids,
            attention_mask,
            labels,
            concept_mode="permutation",
            concept_permutation=permutation,
            concept_bank_index=bank_index,
        )
        assert torch.isfinite(ce[~torch.isnan(ce)]).all()
    metrics = model.concept_ablation_ce(input_ids, attention_mask, labels)
    assert "pressure_delta_permutation_first64" in metrics
    assert "delta_permutation_bank_0_beyond" in metrics
    path = tmp_path / "e17c"
    model.save_pretrained(path)
    reloaded = BackboneConceptLM.from_pretrained(path)
    assert reloaded.config.concept_read_mode == "dedicated"
    assert reloaded.config.tie_concept_writer is False
    assert reloaded.config.concept_read_placement == "post_layer"
    restored = reloaded.encode_concept_banks(input_ids, attention_mask)
    assert torch.allclose(banks, restored)


# ------------------------------------------------------------------ E17d


def _e17d_model(**overrides):
    """Tiny two-bank E17d: attn-residual mix, additive writes, no token carry."""
    defaults = dict(
        concept_read_mode="dedicated",
        concept_read_placement="attn_residual",
        tie_concept_writer=False,
        concept_write_mode="additive",
        write_gate_init=0.1,
        read_concept_norm=True,
        read_gate_init=0.1,
        memory_carry_dropout=1.0,
        inference_carry_policy="drop_after_first",
        memory_pressure_tokens=0,
        memory_pressure_weight=1.0,
    )
    defaults.update(overrides)
    return _perlayer_model(**defaults)


def test_e17d_defaults_keep_e17c_sidecar_path():
    cfg = BackboneConceptConfig(backbone_config=tiny_backbone_dict())
    assert cfg.concept_read_placement == "post_layer"
    assert cfg.inference_carry_policy == "normal"
    sidecar = _e17c_model()
    wrapped = sidecar.backbone.model.layers[5]
    assert wrapped.read_placement == "post_layer"
    assert not isinstance(wrapped.layer.self_attn, _AttnWithConceptResidual)


def test_e17d_wraps_attention_and_permutation_hits_attn_residual():
    model = _e17d_model()
    wrapped = model.backbone.model.layers[5]
    assert wrapped.read_placement == "attn_residual"
    assert isinstance(wrapped.layer.self_attn, _AttnWithConceptResidual)
    for index in model.global_layer_indices:
        model.backbone.model.layers[index].gate.data.fill_(1.0)
    for writer in model.write_heads:
        writer.alpha.data.fill_(0.5)

    inner = []
    mixed = []

    def inner_hook(_module, _inp, out):
        tok = out[0] if isinstance(out, tuple) else out
        inner.append(tok.detach().clone())

    def mixed_hook(_module, _inp, out):
        tok = out[0] if isinstance(out, tuple) else out
        mixed.append(tok.detach().clone())

    attn_wrap = wrapped.layer.self_attn
    h_inner = attn_wrap.original_attn.register_forward_hook(inner_hook)
    h_mixed = attn_wrap.register_forward_hook(mixed_hook)
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    permutation = torch.tensor([1, 0])
    try:
        model.per_position_ce(
            input_ids, attention_mask, labels, concept_mode="real",
            carry_policy="drop_after_first",
        )
        n_real = len(mixed)
        model.per_position_ce(
            input_ids,
            attention_mask,
            labels,
            concept_mode="permutation",
            concept_permutation=permutation,
            carry_policy="drop_after_first",
        )
    finally:
        h_inner.remove()
        h_mixed.remove()
    assert n_real > 0 and len(mixed) == 2 * n_real and len(inner) == 2 * n_real
    for index in range(n_real):
        assert inner[index].shape == inner[n_real + index].shape
        assert torch.allclose(inner[index], inner[n_real + index], atol=1e-5)
    # Window 0 reads the shared init, so permutation is a no-op until after the first write.
    later = [
        index for index in range(n_real)
        if mixed[index].shape[1] == 2 * K
    ]
    assert later
    assert any(
        not torch.allclose(mixed[index], mixed[n_real + index], atol=1e-5)
        for index in later
    )


def test_e17d_untied_additive_writers_honor_write_gate_init():
    model = _e17d_model()
    assert model.write_head is None
    assert len(model.write_heads) == 2
    for writer in model.write_heads:
        assert writer.update_mode == "additive"
        assert float(writer.alpha.detach()) == pytest.approx(0.1)
    metrics = model.concept_gate_metrics()
    assert "concept_gates/write_0" in metrics
    assert "concept_gates/write_1" in metrics


def test_e17d_no_same_window_bank_leak():
    model = _e17d_model()
    for index in model.global_layer_indices:
        model.backbone.model.layers[index].gate.data.fill_(0.8)
    for writer in model.write_heads:
        writer.alpha.data.fill_(0.5)
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    captured = []

    def pre_hook(_module, _args, kwargs):
        state = kwargs.get("concept_state")
        if torch.is_tensor(state):
            captured.append(state.detach().clone())
        return None

    handles = [
        model.backbone.model.layers[index].register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )
        for index in model.global_layer_indices
    ]
    try:
        model.eval()
        model(input_ids, attention_mask, labels)
    finally:
        for handle in handles:
            handle.remove()
    # 2 windows × 2 depths.
    assert len(captured) == 4
    after_first = model.encode_concept_banks(input_ids[:, :K], attention_mask[:, :K])
    assert torch.allclose(captured[2], after_first[:, 0], atol=1e-5)
    assert torch.allclose(captured[3], after_first[:, 1], atol=1e-5)
    assert not torch.allclose(captured[2], captured[3], atol=1e-4)


def test_e17d_generate_and_eval_forward_drop_carry():
    model = _e17d_model()
    model.eval()
    input_ids, attention_mask, labels = make_batch(B=2, S=2 * K)
    seen_ids = []

    def capture_ids(_module, args):
        seen_ids.append(args[0].detach().clone())

    handle = model.backbone.model.embed_tokens.register_forward_pre_hook(capture_ids)
    try:
        model(input_ids, attention_mask, labels)
        model.generate(
            input_ids[:1],
            attention_mask[:1],
            max_new_tokens=2,
            eos_token_id=9999,
        )
    finally:
        handle.remove()
    # First two embed calls are the eval forward's two windows; later calls are generate.
    assert len(seen_ids) >= 3
    carry = seen_ids[1][:, :K]
    assert torch.equal(carry[:, :-1], torch.zeros_like(carry[:, :-1]))
    assert torch.equal(carry[:, -1], torch.full_like(carry[:, -1], 2))
    gen_second_block = None
    for ids in seen_ids[2:]:
        if ids.shape[1] >= 2 * K:
            gen_second_block = ids
            break
    assert gen_second_block is not None
    gen_carry = gen_second_block[:, :K]
    assert torch.equal(gen_carry[:, :-1], torch.zeros_like(gen_carry[:, :-1]))
    assert torch.equal(gen_carry[:, -1], torch.full_like(gen_carry[:, -1], 2))


def test_e17d_late_bin_ablation_keys_and_finite_loss():
    model = _e17d_model()
    for index in model.global_layer_indices:
        model.backbone.model.layers[index].gate.data.fill_(0.5)
    input_ids, attention_mask, labels = make_batch(B=2, S=3 * K)
    metrics = model.concept_ablation_ce(input_ids, attention_mask, labels)
    assert "delta_permutation_block_256_512" in metrics
    assert "delta_permutation_bank_0_block_256_512" in metrics
    assert "delta_permutation_bank_1_block_256_512" in metrics
    assert metrics["delta_permutation_block_256_512"] == metrics[
        "delta_permutation_block_256_512"
    ]
    model.train()
    loss = None
    for _ in range(3):
        loss = model(input_ids, attention_mask, labels).loss
        loss.backward()
        model.zero_grad()
    assert torch.isfinite(loss)

