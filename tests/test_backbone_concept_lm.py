"""Unit tests for the E10 backbone-concept graft (nn/backbone_concept_lm.py).

All tests run on a tiny RANDOM Gemma3TextConfig (no hub access): 6 layers with the native
5-sliding:1-global pattern, sliding_window = concept_block = 8, H=64, V=256.
"""

import pytest
import torch

from data.data_collators import DataCollatorForCausalLM
from data.dataset_preprocess import configure_text_tokenizer_for_model_vocab
from analysis.run_e10_comparison import evaluate_length
from nn.backbone_concept_lm import BackboneConceptConfig, BackboneConceptLM

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


def make_model(concept_num=4, lora_r=2, seed=0):
    torch.manual_seed(seed)
    cfg = BackboneConceptConfig(
        backbone_model="tiny-random-gemma3",
        backbone_config=tiny_backbone_dict(),
        concept_num=concept_num,
        concept_block=K,
        write_num_heads=2,
        lora_r=lora_r,
        lora_dropout=0.0,
    )
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
    model.backbone.model.layers[5].gate.data.fill_(0.5)
    model.write_head.alpha.data.fill_(0.3)
    input_ids, attention_mask, labels = make_batch(B=2, S=24)
    model(input_ids, attention_mask, labels=labels).loss.backward()
    assert model.concept_init.grad is not None and model.concept_init.grad.abs().sum() > 0
    assert model.write_head.alpha.grad is not None and model.write_head.alpha.grad.abs() > 0
    bixt_grads = [p.grad for p in model.write_head.bixt.parameters() if p.grad is not None]
    assert bixt_grads and any(g.abs().sum() > 0 for g in bixt_grads)


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

    model_bounded = DataCollatorForCausalLM(
        TokStub(), max_length=16, model_vocab_size=VOCAB - 1
    )
    with pytest.raises(ValueError, match="outside model vocabulary range"):
        model_bounded([{"input_ids": [5, VOCAB - 1]}])


def test_text_tokenizer_splits_out_of_model_special_tokens():
    class TokStub:
        split_special_tokens = False

        def __len__(self):
            return 101

    tokenizer = TokStub()
    assert configure_text_tokenizer_for_model_vocab(tokenizer, 100)
    assert tokenizer.split_special_tokens is True
    assert not configure_text_tokenizer_for_model_vocab(tokenizer, 101)


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
