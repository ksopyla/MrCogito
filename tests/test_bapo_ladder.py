"""BAPO ladder configs, info reports, and tiny architecture smoke."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from data.bapo_ladder import SCALES, SOLVABLE_ACC, TINY_PROOF_TASKS, config_for, rung_card
from data.symbolic_tasks import TASKS, chance_accuracy, prize_bits
from evaluation.bapo_metrics import info_report
from evaluation.bapo_models import ArchSpec, build_model, n_params
from nn.encdec_lm import EncDecConfig, EncoderDecoderLM


@pytest.mark.parametrize("scale", list(SCALES))
@pytest.mark.parametrize("task", TASKS)
def test_every_rung_fits(scale, task):
    cfg = config_for(scale, task)
    card = rung_card(scale, task)
    assert cfg.seq_len == SCALES[scale].seq_len
    assert card["prize_bits"] > 0
    assert card["solvable_acc"] == SOLVABLE_ACC
    assert card["chance_acc"] == pytest.approx(chance_accuracy(cfg))


def test_tiny_core_is_what_the_cpu_probe_runs():
    assert TINY_PROOF_TASKS == ("far_copy", "recall", "select", "chain_ordered", "chain")
    for task in TINY_PROOF_TASKS:
        cfg = config_for("tiny", task)
        assert prize_bits(cfg) == pytest.approx(cfg.answer_len * math.log(cfg.n_symbols) / math.log(2))


def test_tiny_core_answers_are_packed():
    """Packed CE is the difference between chance and a solvable rung (Arm-A span=8 vs 32)."""
    from data.bapo_ladder import pack_overrides, target_answer_len

    assert target_answer_len("tiny") == 16
    for task in TINY_PROOF_TASKS:
        packed = config_for("tiny", task)
        unpacked = config_for("tiny", task, pack=False)
        if task in {"chain", "chain_ordered"}:
            assert packed.answer_len >= 12, (task, packed.answer_len)
        else:
            assert packed.answer_len >= 16, (task, packed.answer_len)
        over = pack_overrides("tiny", task)
        assert over  # every user-core task has a pack field
        assert packed.seq_len == 128
        assert packed.answer_len >= 8, (task, packed.answer_len)


def test_medium_retrieval_packs_to_32_supervised_tokens():
    for task in ("far_copy", "recall", "select"):
        cfg = config_for("medium", task)
        assert cfg.answer_len == 32, (task, cfg.answer_len)
    for task in ("chain", "chain_ordered"):
        cfg = config_for("medium", task)
        assert cfg.answer_len == 32, (task, cfg.answer_len, cfg.key_len)


def test_pack_keeps_min_gap_contract():
    from data.symbolic_tasks import generate_row

    rng = np.random.default_rng(0)
    for task in TINY_PROOF_TASKS:
        cfg = config_for("tiny", task)
        row = generate_row(cfg, rng)
        assert row.gap >= cfg.min_gap + 1, (task, row.gap, cfg.min_gap)
        assert row.answer_len == cfg.answer_len


def test_info_report_perfect_recovery():
    cfg = config_for("tiny", "far_copy")
    floor = math.log(cfg.n_symbols)
    rep = info_report(
        ce_nats=0.0,
        acc=1.0,
        n_supervised=64,
        cfg=cfg,
        window=16,
        nominal_b_tokens=128,
        nominal_a_bytes=128.0,
    )
    assert rep.information_flow == pytest.approx(1.0)
    assert rep.recovered_bits == pytest.approx(prize_bits(cfg), abs=1e-6)
    assert rep.bytes_per_input_token == pytest.approx(rep.recovered_bits / 8.0 / cfg.seq_len)
    at_floor = info_report(
        ce_nats=floor, acc=0.25, n_supervised=64, cfg=cfg, window=16,
        nominal_b_tokens=16, nominal_a_bytes=0.0,
    )
    assert at_floor.information_flow == pytest.approx(0.0)
    assert at_floor.recovered_bits == pytest.approx(0.0)


def test_info_report_perfect_recovery_on_glyph_reverse():
    from data.glyph_tasks import GlyphTaskConfig, prize_bits as glyph_prize

    cfg = config_for("tiny", "reverse")
    assert isinstance(cfg, GlyphTaskConfig)
    floor = math.log(cfg.n_symbols)
    rep = info_report(
        ce_nats=0.0,
        acc=1.0,
        n_supervised=cfg.answer_len,
        cfg=cfg,
        window=16,
        nominal_b_tokens=128,
        nominal_a_bytes=128.0,
    )
    assert rep.information_flow == pytest.approx(1.0)
    assert rep.recovered_bits == pytest.approx(glyph_prize(cfg), abs=1e-6)
    at_floor = info_report(
        ce_nats=floor,
        acc=1.0 / cfg.n_symbols,
        n_supervised=cfg.answer_len,
        cfg=cfg,
        window=16,
        nominal_b_tokens=16,
        nominal_a_bytes=0.0,
    )
    assert at_floor.information_flow == pytest.approx(0.0)
    assert at_floor.recovered_bits == pytest.approx(0.0)


def test_encdec_sinusoidal_positions_differ():
    pe = EncoderDecoderLM._sinusoidal(8, 32, torch.device("cpu"), torch.float32)
    assert pe.shape == (8, 32)
    assert not torch.allclose(pe[0], pe[1])


def test_calibrated_recipes_construct_at_every_scale():
    from data.bapo_ladder import CALIBRATED_RECIPES, UNCALIBRATED_AT_TINY, resolve_recipe

    assert resolve_recipe("recall_single").task == "recall"
    assert resolve_recipe("recall_single").overrides["n_distractors"] == 0
    assert resolve_recipe("select_1decoy").overrides == {"n_distractors": 0, "n_decoys": 1}
    for scale in SCALES:
        for rec in CALIBRATED_RECIPES.values():
            cfg = config_for(scale, rec.task, **rec.overrides)
            assert cfg.seq_len == SCALES[scale].seq_len
            assert cfg.answer_len >= 1
    # Uncalibrated hunts still construct (they are not scored until S0).
    for rec in UNCALIBRATED_AT_TINY.values():
        cfg = config_for("tiny", rec.task, **rec.overrides)
        assert cfg.task == rec.task


def test_bridge_scales_keep_e18_local_blind():
    """K2: the leak control is only valid when the stack window cannot see the evidence."""
    from data.symbolic_tasks import generate_row

    for name in ("bridge", "bridge_1k"):
        sc = SCALES[name]
        assert sc.local_window < sc.min_gap, name
        cfg = config_for(name, "far_copy", evidence_align="right")
        assert cfg.seq_len == sc.seq_len
        assert cfg.evidence_align == "right"
        row = generate_row(cfg, np.random.default_rng(0))
        assert row.gap == cfg.min_gap + 1, (name, row.gap, cfg.min_gap)


def test_recall_single_fact_still_respects_min_gap():
    from data.symbolic_tasks import generate_row

    cfg = config_for("tiny", "recall", n_distractors=0)
    row = generate_row(cfg, np.random.default_rng(0))
    assert cfg.n_distractors == 0
    assert row.gap >= cfg.min_gap + 1


def test_encdec_forward_shapes_and_finite_loss():
    from data.symbolic_tasks import generate_row

    cfg = config_for("tiny", "far_copy")
    rng = np.random.default_rng(0)
    rows = [generate_row(cfg, rng) for _ in range(4)]
    ids = torch.from_numpy(np.stack([r.input_ids for r in rows])).long()
    labels = torch.from_numpy(np.stack([r.labels for r in rows])).long()
    model = EncoderDecoderLM(
        EncDecConfig(
            vocab_size=cfg.vocab.vocab_size,
            hidden_size=32,
            intermediate_size=64,
            enc_layers=1,
            dec_layers=1,
            num_attention_heads=2,
            head_dim=16,
            answer_start=cfg.answer_start,
            pad_token_id=cfg.vocab.control("eos"),
        )
    )
    out, per, valid = model(ids, labels=labels, return_per_token_loss=True)
    assert torch.isfinite(out.loss)
    assert per.shape == (4, cfg.seq_len - 1)
    assert int(valid.sum()) == 4 * cfg.answer_len
    logits = model(ids, return_logits=True).logits
    assert logits.shape == (4, cfg.seq_len, cfg.vocab.vocab_size)


def test_right_align_puts_span_just_before_min_gap():
    from data.symbolic_tasks import generate_row

    cfg = config_for("tiny", "far_copy", seq_len=512, min_gap=8, evidence_align="right")
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap == cfg.min_gap + 1


def test_seq_len_and_min_gap_overrides_construct():
    cfg = config_for("medium", "far_copy", seq_len=512, min_gap=64)
    assert cfg.seq_len == 512
    assert cfg.min_gap == 64
    assert cfg.answer_len == 32


def test_ssmax_and_mha_factory():
    cfg = config_for("tiny", "far_copy")
    spec = ArchSpec(
        name="dense",
        hidden=64,
        head_dim=16,
        n_kv_heads=0,
        global_logit_scale="log",
        attn_backend="sdpa",
    )
    model = build_model(
        "dense",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    assert model.config.num_kv_heads == model.config.num_attention_heads
    assert model.config.global_logit_scale == "log"
    assert any(layer.attn.logit_scale is not None for layer in model.layers)


def test_warm_residuals_leave_wo_nonzero():
    cfg = config_for("tiny", "far_copy")
    cold = build_model(
        "dense",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=0, bos_id=1, eos_id=2,
        spec=ArchSpec(name="dense", hidden=32, head_dim=16, zero_init_residuals=True),
        seed=0,
    )
    warm = build_model(
        "dense",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=0, bos_id=1, eos_id=2,
        spec=ArchSpec(name="dense", hidden=32, head_dim=16, zero_init_residuals=False),
        seed=0,
    )
    assert all(float(layer.attn.wo.weight.detach().abs().sum()) == 0.0 for layer in cold.layers)
    assert any(float(layer.attn.wo.weight.detach().abs().sum()) > 0.0 for layer in warm.layers)


@pytest.mark.parametrize("arch", ["dense", "e18", "e18_local", "e21", "encdec"])
def test_factory_builds_under_100m(arch):
    cfg = config_for("tiny", "recall")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name=arch,
        hidden=32,
        head_dim=16,
        local_window=16,
        enc_layers=1,
        dec_layers=1,
        message_boundary_token_id=qid if arch == "e21" else -1,
    )
    model = build_model(
        arch,
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    n = n_params(model)
    assert 1_000 < n < 100_000_000
    ids = torch.randint(0, cfg.vocab.vocab_size, (2, cfg.seq_len))
    if arch == "e21":
        ids[:, cfg.seq_len // 2] = qid
    labels = torch.full((2, cfg.seq_len), -100)
    labels[:, cfg.answer_start : cfg.answer_start + cfg.answer_len] = ids[:, cfg.answer_start : cfg.answer_start + cfg.answer_len]
    loss = model(ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()


def test_e21_raw_override_is_wired():
    """Factory e21 + probe `_message_cm(..., 'raw')` must change receiver logits vs real slots.

    `raw` is the uncompressed prefix-KV ceiling with QUERY still a local document start.
    At r=16, real slots ≠ raw keys (r=1 would match). Dense/e18 have no message path.
    """
    from verification.bapo_capability_probe import _message_cm

    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=qid,
        message_compress_ratio=16,
        zero_init_residuals=False,
    )
    model = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    with torch.no_grad():
        real = model(ids).logits
        with _message_cm(model, "raw"):
            assert model._message_override == "raw"
            raw = model(ids).logits
        assert model._message_override == "real"
        with _message_cm(model, "real"):
            again = model(ids).logits
    assert torch.allclose(real, again, atol=1e-6)
    assert not torch.allclose(real[:, p:], raw[:, p:], atol=1e-5)
    # dense must ignore the flag (no message path)
    dense = build_model(
        "dense",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(name="dense", hidden=32, head_dim=16, zero_init_residuals=False),
        seed=0,
    )
    with torch.no_grad():
        d0 = dense(ids).logits
        with _message_cm(dense, "raw"):
            d1 = dense(ids).logits
    assert torch.allclose(d0, d1, atol=1e-6)


def test_e21_inplace_flag_is_wired_on_factory():
    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=qid,
        message_compress_ratio=1,
        message_slots_inplace=True,
        zero_init_residuals=False,
    )
    model = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    assert model.config.message_slots_inplace is True
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)


def test_e21_inplace_raw_kv_flag_is_wired_on_factory():
    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=qid,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_inplace_raw_kv=True,
        zero_init_residuals=False,
    )
    model = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    assert model.config.message_slots_inplace is True
    assert model.config.message_inplace_raw_kv is True
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)


def test_e21_identity_slots_flag_is_wired_on_factory():
    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=qid,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        zero_init_residuals=False,
    )
    model = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    assert model.config.message_slots_inplace is True
    assert model.config.message_identity_slots is True
    assert model.config.message_inplace_raw_kv is False
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)
