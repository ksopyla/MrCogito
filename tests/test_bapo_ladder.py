"""BAPO ladder configs, info reports, and tiny architecture smoke."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from data.bapo_ladder import SCALES, SOLVABLE_ACC, TINY_PROOF_TASKS, config_for, generate_row_for, rung_card
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


def test_e30_limit_recipes_construct_at_bridge_scales():
    """E30 limit exams (lookup/lookalike/4-hop chain) build at the lengths where the
    2026-09-22 limits were measured (bridge_1k = 1024) and at bridge."""
    from data.bapo_ladder import E30_LIMIT_RECIPES, resolve_recipe

    assert resolve_recipe("lookup_1key").overrides["n_distractors"] == 0
    assert resolve_recipe("lookalike").overrides["n_decoys"] == 1  # recorded exam: select_1decoy
    assert resolve_recipe("chain_4hop").overrides["hops"] == 4
    # Recorded at bridge_1k (and 2048 via --seq_len). A 4-hop chain with packed 32-letter
    # keys does not fit in 128 tokens; the old 8-letter pin that made it fit changed the exam.
    for scale in ("bridge", "bridge_1k"):
        for rec in E30_LIMIT_RECIPES.values():
            cfg = config_for(scale, rec.task, **rec.overrides)
            assert cfg.seq_len == SCALES[scale].seq_len
            assert cfg.answer_len >= 1


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


def test_bridge_1k_seq_len_2048_keeps_window_below_gap():
    """INDEX length-wall hunt: override seq only; do not invent a 2048 scale."""
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for("bridge_1k", "far_copy", seq_len=2048, evidence_align="right")
    assert cfg.seq_len == 2048
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap == cfg.min_gap + 1
    assert row.input_ids.shape == (2048,)


def test_bridge_1k_seq_len_1536_keeps_window_below_gap():
    """INDEX length-wall bracket: override seq only; do not invent a 1536 scale.

    1024 PASS / 2048 FAIL on the same r=16 identity inplace rem-off compressor.
    Window/gap stay bridge_1k 16 < 64. Packed span stays 32 / 64-bit prize.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for("bridge_1k", "far_copy", seq_len=1536, evidence_align="right")
    assert cfg.seq_len == 1536
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert sc.local_window == 16
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap == cfg.min_gap + 1
    assert row.input_ids.shape == (1536,)
    qid = cfg.vocab.control("query")
    qpos = int(np.where(row.input_ids == qid)[0][0])
    leftover = qpos % 16
    assert leftover < 16
    assert sc.local_window < row.gap


def test_bridge_1k_seq_len_768_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 768 scale.

    `bridge_1k` (not `bridge`) keeps the 1024-passing 32-token / 64-bit pack.
    Packed `select_1decoy` plants a decoy after the fact under right-align, so
    row.gap is > min_gap+1; the K2 contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=768,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 768
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    # `bridge` would pack to 24 tokens / 48 bits — prize would confound vs 1024.
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=768,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (768,)


def test_bridge_1k_seq_len_692_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 692 scale.

    Tightens (688 PASS, 696 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap. Seq=692 is 4 tokens past 688 and 4
    before 696; prize packing stays 32/64 because the scale is still
    `bridge_1k` (target_answer_len uses the scale seq, not the override).
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=692,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 692
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=692,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (692,)


def test_select_688_692_696_geometry_and_r1_remainder_noop_pack_stride():
    """688/692/696 SELECT packing: 2 complete 35-token KV packs; 693-696 add 4 left-filler
    tokens after BOS, not an incomplete evidence block. QUERY shifts +4. r=1 remainder
    is a no-op (every sender token is already an identity slot). pack_stride=32 drops
    QUERY-aligned leftover 10/14/18 from exclusive replace; both 692 and 696 then have
    640 exclusive sender slots (the 4-token leftover is dropped, not a new pack).
    """
    from nn.perceiver_ar_lm import mix_inplace_kv

    recipe_over = dict(n_distractors=0, n_decoys=1, evidence_align="right")
    rows = {}
    for seq in (688, 692, 696):
        cfg = config_for("bridge_1k", "select", seq_len=seq, **recipe_over)
        kv = 1 + cfg.key_len + cfg.value_len
        row = generate_row_for(cfg, np.random.default_rng(0))
        qid = cfg.vocab.control("query")
        kid = cfg.vocab.control("keymark")
        did = cfg.vocab.control("decoy")
        ids = row.input_ids
        q = int(np.where(ids == qid)[0][0])
        fact0 = int(np.where(ids == kid)[0][0])
        decoy0 = int(np.where(ids == did)[0][0])
        rows[seq] = dict(cfg=cfg, q=q, fact0=fact0, decoy0=decoy0, kv=kv,
                         sender=q, leftover32=q % 32, packs32=q // 32)
        assert cfg.evidence_len == 2 * kv
        assert kv == 35
        assert cfg.answer_len == 32
        assert prize_bits(cfg) == pytest.approx(64.0)
        assert decoy0 == fact0 + kv
        assert q - (decoy0 + kv) == 60  # gap filler decoy→QUERY
    assert rows[692]["q"] - rows[688]["q"] == 4
    assert rows[696]["q"] - rows[692]["q"] == 4
    assert rows[696]["fact0"] - rows[692]["fact0"] == 4  # extra tokens are left filler
    assert rows[688]["packs32"] == rows[692]["packs32"] == rows[696]["packs32"] == 20
    assert (rows[688]["leftover32"], rows[692]["leftover32"], rows[696]["leftover32"]) == (10, 14, 18)
    # r=1 remainder no-op vs pack_stride=32 leftover drop
    cfg = rows[696]["cfg"]
    qid = cfg.vocab.control("query")
    ids = torch.from_numpy(generate_row_for(cfg, np.random.default_rng(0)).input_ids[None].astype(np.int64))

    def _replace(remainder, pack_stride):
        spec = ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            message_pool_remainder=remainder,
            message_pack_stride=pack_stride,
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
        pos = model._positions(ids.shape[1], 1, None, ids.device)
        ctx = model._message_context(ids, None, None, pos)
        k = torch.zeros(1, ids.shape[1], model.config.num_kv_heads, model.config.head_dim)
        _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
        q = rows[696]["q"]
        return int(replace[0, :q].sum()), model.config.message_pack_stride

    n_off, _ = _replace(False, 0)
    n_rem, _ = _replace(True, 0)
    n_pack, stride = _replace(False, 32)
    n_pack_rem, _ = _replace(True, 32)
    assert n_off == n_rem == rows[696]["sender"]  # r=1 remainder is a no-op
    assert stride == 32
    assert n_pack == 20 * 32 == 640  # leftover 18 dropped
    assert n_pack_rem == rows[696]["sender"]  # remainder keeps leftover as identity slots
    # 692 vs 696 with pack_stride=32: same exclusive sender count (640)
    cfg692 = rows[692]["cfg"]
    ids692 = torch.from_numpy(
        generate_row_for(cfg692, np.random.default_rng(0)).input_ids[None].astype(np.int64)
    )
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=cfg692.vocab.control("query"),
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_pack_stride=32,
        zero_init_residuals=False,
    )
    m692 = build_model(
        "e21",
        vocab_size=cfg692.vocab.vocab_size,
        seq_len=cfg692.seq_len,
        answer_start=cfg692.answer_start,
        pad_id=cfg692.vocab.control("eos"),
        bos_id=cfg692.vocab.control("bos"),
        eos_id=cfg692.vocab.control("eos"),
        spec=spec,
        seed=0,
    )
    pos = m692._positions(ids692.shape[1], 1, None, ids692.device)
    ctx = m692._message_context(ids692, None, None, pos)
    k = torch.zeros(1, ids692.shape[1], m692.config.num_kv_heads, m692.config.head_dim)
    _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
    assert int(replace[0, : rows[692]["q"]].sum()) == 640


def test_bridge_1k_seq_len_696_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 696 scale.

    Tightens (688 PASS, 704 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=696,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 696
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=696,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (696,)


def test_bridge_1k_seq_len_688_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 688 scale.

    Tightens (672 PASS, 704 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=688,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 688
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=688,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (688,)


def test_bridge_1k_seq_len_672_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 672 scale.

    Tightens (640 PASS, 704 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=672,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 672
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=672,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (672,)


def test_bridge_1k_seq_len_704_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 704 scale.

    Tightens (640 PASS, 768 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=704,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 704
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=704,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (704,)


def test_bridge_1k_seq_len_640_select_keeps_window_below_gap():
    """SELECT length-wall hunt: override seq only; do not invent a 640 scale.

    Tightens (512 PASS, 768 FAIL] on the same 1024-passing 32-token / 64-bit
    pack (`bridge_1k`, not `bridge`). Packed `select_1decoy` plants a decoy
    after the fact under right-align, so row.gap is > min_gap+1; the K2
    contract is still window < gap.
    """
    from data.symbolic_tasks import generate_row

    sc = SCALES["bridge_1k"]
    cfg = config_for(
        "bridge_1k",
        "select",
        seq_len=640,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert cfg.seq_len == 640
    assert cfg.min_gap == sc.min_gap == 64
    assert sc.local_window < cfg.min_gap
    assert cfg.answer_len == 32
    assert prize_bits(cfg) == pytest.approx(64.0)
    bridge_cfg = config_for(
        "bridge",
        "select",
        seq_len=640,
        evidence_align="right",
        n_distractors=0,
        n_decoys=1,
    )
    assert bridge_cfg.answer_len == 24
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.gap >= cfg.min_gap + 1
    assert sc.local_window < row.gap
    assert row.input_ids.shape == (640,)


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


@pytest.mark.parametrize("arch", ["dense", "e18", "e18_local", "e21", "e30", "encdec"])
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
        message_boundary_token_id=qid if arch in ("e21", "e30") else -1,
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
    if arch in ("e21", "e30"):
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
    assert model.config.message_pack_stride == 0
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)


def test_e21_pack_stride_flag_is_wired_on_factory():
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
        message_pack_stride=32,
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
    assert model.config.message_pack_stride == 32
    assert model.config.message_pool_remainder is False
    assert model.config.message_keep_local_swa is False
    e18 = build_model(
        "e18",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(name="e18", hidden=32, head_dim=16, message_pack_stride=32,
                      zero_init_residuals=False),
        seed=0,
    )
    assert e18.config.message_enabled is False
    assert e18.config.message_pack_stride == 0
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)


def test_e21_keep_local_swa_flag_is_wired_on_factory():
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
        message_keep_local_swa=True,
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
    assert model.config.message_keep_local_swa is True
    assert model.config.message_inplace_raw_kv is False
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)
    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_keep_local_swa is False


def test_e21_extra_slot_attends_flag_is_wired_on_factory():
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
        message_extra_slot_attends=1,
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
    assert model.config.message_extra_slot_attends == 1
    assert model.config.message_keep_local_swa is False
    assert model.config.message_inplace_raw_kv is False
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)
    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_extra_slot_attends == 0
    e18 = build_model(
        "e18",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e18",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_extra_slot_attends=1,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert e18.config.message_boundary_token_id == -1
    assert e18.config.message_extra_slot_attends == 0
    assert e18.config.message_update_slot_kv is False


def test_e21_update_slot_kv_flag_is_wired_on_factory():
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
        message_extra_slot_attends=1,
        message_update_slot_kv=True,
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
    assert model.config.message_extra_slot_attends == 1
    assert model.config.message_update_slot_kv is True
    assert model.config.message_keep_local_swa is False
    assert model.config.message_global_anchors == "none"
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)
    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            message_extra_slot_attends=1,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_update_slot_kv is False
    e18 = build_model(
        "e18",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e18",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_extra_slot_attends=1,
            message_update_slot_kv=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert e18.config.message_boundary_token_id == -1
    assert e18.config.message_extra_slot_attends == 0
    assert e18.config.message_update_slot_kv is False


def test_e21_global_layers_two_is_wired_on_factory():
    """--global_layers 2 is two exclusive global Blocks, not extra hops and not extra SWA."""
    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        pre_layers=1,
        global_layers=2,
        stack_layers=2,
        message_boundary_token_id=qid,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        zero_init_residuals=False,
    )
    kw = dict(
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        seed=0,
    )
    e21 = build_model("e21", spec=spec, **kw)
    assert e21.config.global_layers == 2
    assert e21.config.message_extra_slot_attends == 0
    assert e21.config.message_update_slot_kv is False
    assert e21.config.message_global_anchors == "none"
    pats = [l.attn.pattern for l in e21.layers]
    assert pats == ["swa", "full", "full", "swa", "swa"]
    full = [i for i, p in enumerate(pats) if p == "full"]
    assert full == [1, 2]
    for i in full:
        assert e21.layers[i].attn.compressor is not None
    default = build_model(
        "e21",
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        **kw,
    )
    assert default.config.global_layers == 1
    assert [l.attn.pattern for l in default.layers].count("full") == 1
    e18 = build_model("e18", spec=spec, **kw)
    assert e18.config.global_layers == 2
    assert e18.config.message_boundary_token_id == -1
    assert e18.config.message_global_anchors == "none"
    assert e18.config.message_prefix_ae is False
    assert [l.attn.pattern for l in e18.layers] == ["swa", "full", "full", "swa", "swa"]
    assert all(l.attn.compressor is None for l in e18.layers)
    local = build_model("e18_local", spec=spec, **kw)
    assert local.config.global_layers == 0
    assert all(l.attn.pattern == "swa" for l in local.layers)
    assert len(local.layers) == len(e21.layers)
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = e21(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)


def test_e21_global_anchors_flag_is_wired_on_factory():
    cfg = config_for("tiny", "far_copy")
    qid = cfg.vocab.control("query")
    marks = (cfg.vocab.control("keymark"), cfg.vocab.control("decoy"))
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=qid,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_global_anchors="type_marks",
        message_anchor_token_ids=marks,
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
    assert model.config.message_global_anchors == "type_marks"
    assert model.config.message_anchor_token_ids == marks
    assert model.config.message_extra_slot_attends == 0
    assert model.config.message_keep_local_swa is False
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    loss = model(ids, labels=torch.full_like(ids, -100)).loss
    assert torch.isfinite(loss)
    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_global_anchors == "none"
    assert off.config.message_anchor_token_ids == ()
    e18 = build_model(
        "e18",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e18",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_global_anchors="type_marks",
            message_anchor_token_ids=marks,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert e18.config.message_boundary_token_id == -1
    assert e18.config.message_global_anchors == "none"
    assert e18.config.message_anchor_token_ids == ()


def test_select_1decoy_type_mark_anchors_are_sparse_vs_seq():
    """select_1decoy: 2 type marks (keymark+decoy) join exclusive K/V; count << seq.

    At r=8 remainder-off they are extra raw keys, not the full prefix. At r=1 they
    are already identity slots; the flag still marks them as anchors (2 vs 1024).
    """
    from nn.perceiver_ar_lm import exclusive_visible, mix_inplace_kv

    cfg = config_for("bridge_1k", "select", n_distractors=0, n_decoys=1, evidence_align="right")
    assert cfg.seq_len == 1024
    row = generate_row_for(cfg, np.random.default_rng(0))
    ids_np = row.input_ids
    keymark = cfg.vocab.control("keymark")
    decoy = cfg.vocab.control("decoy")
    query = cfg.vocab.control("query")
    n_marks = int((ids_np == keymark).sum() + (ids_np == decoy).sum())
    assert n_marks == 2
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=query,
        message_compress_ratio=8,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_global_anchors="type_marks",
        message_anchor_token_ids=(keymark, decoy),
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
    ids = torch.from_numpy(ids_np[None].astype(np.int64))
    pos = model._positions(ids.shape[1], 1, None, ids.device)
    with torch.no_grad():
        ctx = model._message_context(ids, None, None, pos)
    S = ids.shape[1]
    qpos = int(np.where(ids_np == query)[0][0])
    assert int(ctx.anchor[0].sum()) == 2
    assert int(ctx.anchor[0].sum()) < S
    k = torch.zeros(1, S, model.config.num_kv_heads, model.config.head_dim)
    _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
    vis = exclusive_visible(replace, ctx)
    extra = vis & ~replace
    assert int(ctx.anchor[0].sum()) == 2
    assert not bool(vis[0, :qpos].all())  # exclusive is slots+anchors, not full prefix at r=8
    assert int(extra[0, :qpos].sum()) <= 2
    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=query,
            message_compress_ratio=8,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_global_anchors == "none"


def test_select_1decoy_type_cues_land_in_r1_identity_slots():
    """SELECT keymark/decoy sit in the sender prefix (before QUERY) and are replace slots at r=1."""
    from nn.perceiver_ar_lm import dense_inplace_mask, mix_inplace_kv

    cfg = config_for("tiny", "select", n_distractors=0, n_decoys=1, evidence_align="right")
    row = generate_row_for(cfg, np.random.default_rng(0))
    ids_np = row.input_ids
    keymark = cfg.vocab.control("keymark")
    decoy = cfg.vocab.control("decoy")
    query = cfg.vocab.control("query")
    km = int(np.where(ids_np == keymark)[0][0])
    dc = int(np.where(ids_np == decoy)[0][0])
    qpos = int(np.where(ids_np == query)[0][0])
    assert km < qpos and dc < qpos
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=query,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_extra_slot_attends=1,
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
    ids = torch.from_numpy(ids_np[None].astype(np.int64))
    pos = model._positions(ids.shape[1], 1, None, ids.device)
    with torch.no_grad():
        ctx = model._message_context(ids, None, None, pos)
    S = ids.shape[1]
    k = torch.zeros(1, S, model.config.num_kv_heads, model.config.head_dim)
    v = torch.zeros_like(k)
    _, _, replace = mix_inplace_kv(k, v, k, v, ctx)
    assert bool(replace[0, km]) and bool(replace[0, dc])
    assert not bool(replace[0, qpos])
    mask = dense_inplace_mask(S, ctx, None, replace, "cpu")
    assert bool(mask[0, 0, qpos, km]) and bool(mask[0, 0, qpos, dc])


def test_select_1decoy_query_side_anchors_are_not_r1_prefix_slots():
    """SELECT type request sits at/after QUERY; query_side leaks those, not r=1 slots.

    `query_nbhd` (prefix before QUERY) is already identity slots at r=1 (extra 0),
    same no-op as type_marks. `query_side` must mark QUERY + a small window after,
    extra count << seq, default none unchanged.
    """
    from nn.perceiver_ar_lm import exclusive_visible, mix_inplace_kv

    cfg = config_for("bridge_1k", "select", n_distractors=0, n_decoys=1, evidence_align="right")
    assert cfg.seq_len == 1024
    assert cfg.query_len == cfg.key_len == 2
    row = generate_row_for(cfg, np.random.default_rng(0))
    ids_np = row.input_ids
    query = cfg.vocab.control("query")
    answer = cfg.vocab.control("answer")
    qpos = int(np.where(ids_np == query)[0][0])
    apos = int(np.where(ids_np == answer)[0][0])
    assert apos == qpos + 1 + cfg.query_len
    leaked = list(range(qpos, qpos + 4))  # QUERY, key, key, ANSWER
    assert leaked[-1] == apos
    spec = ArchSpec(
        name="e21",
        hidden=32,
        head_dim=16,
        local_window=16,
        message_boundary_token_id=query,
        message_compress_ratio=1,
        message_slots_inplace=True,
        message_identity_slots=True,
        message_global_anchors="query_side",
        message_anchor_window=4,
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
    assert model.config.message_global_anchors == "query_side"
    ids = torch.from_numpy(ids_np[None].astype(np.int64))
    pos = model._positions(ids.shape[1], 1, None, ids.device)
    with torch.no_grad():
        ctx = model._message_context(ids, None, None, pos)
    S = ids.shape[1]
    anc = ctx.anchor[0]
    for i in leaked:
        assert bool(anc[i]), i
    assert not bool(anc[qpos - 1])
    assert int(anc.sum()) == 4
    assert int(anc.sum()) < S
    k = torch.zeros(1, S, model.config.num_kv_heads, model.config.head_dim)
    _, _, replace = mix_inplace_kv(k, k, k, k, ctx)
    extra = exclusive_visible(replace, ctx) & ~replace
    assert bool(replace[0, qpos - 1]) and not bool(replace[0, qpos])
    for i in leaked:
        assert not bool(replace[0, i])
        assert bool(extra[0, i])
    assert int(extra[0].sum()) == 4
    assert int(extra[0].sum()) < cfg.seq_len
    # exact tokens: QUERY, the two query-key symbols, ANSWER — not prefix slots
    assert int(ids_np[qpos]) == query
    assert int(ids_np[apos]) == answer
    assert int(ids_np[qpos + 1]) != query and int(ids_np[qpos + 2]) != query

    nbhd = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=query,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            message_global_anchors="query_nbhd",
            message_anchor_window=4,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    with torch.no_grad():
        ctx_n = nbhd._message_context(ids, None, None, pos)
    _, _, replace_n = mix_inplace_kv(k, k, k, k, ctx_n)
    extra_n = exclusive_visible(replace_n, ctx_n) & ~replace_n
    assert int(extra_n[0].sum()) == 0
    assert not bool(ctx_n.anchor[0, qpos])

    off = build_model(
        "e21",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=query,
            message_compress_ratio=1,
            message_slots_inplace=True,
            message_identity_slots=True,
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert off.config.message_global_anchors == "none"
    e18 = build_model(
        "e18",
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        spec=ArchSpec(
            name="e18",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_global_anchors="query_side",
            zero_init_residuals=False,
        ),
        seed=0,
    )
    assert e18.config.message_boundary_token_id == -1
    assert e18.config.message_global_anchors == "none"


def test_e21_key_spans_and_prefix_ae_flags_are_wired_on_factory():
    cfg = config_for("tiny", "recall")
    qid = cfg.vocab.control("query")
    keymark = cfg.vocab.control("keymark")
    kw = dict(
        vocab_size=cfg.vocab.vocab_size,
        seq_len=cfg.seq_len,
        answer_start=cfg.answer_start,
        pad_id=cfg.vocab.control("eos"),
        bos_id=cfg.vocab.control("bos"),
        eos_id=cfg.vocab.control("eos"),
        seed=0,
    )
    hybrid = build_model(
        "e21",
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=16,
            message_slots_inplace=True,
            message_identity_slots=True,
            message_global_anchors="key_spans",
            message_anchor_token_ids=(keymark,),
            message_anchor_key_len=2,
            zero_init_residuals=False,
        ),
        **kw,
    )
    assert hybrid.config.message_global_anchors == "key_spans"
    assert hybrid.config.message_anchor_key_len == 2
    assert hybrid.config.message_anchor_token_ids == (keymark,)
    ae = build_model(
        "e21",
        spec=ArchSpec(
            name="e21",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_boundary_token_id=qid,
            message_compress_ratio=16,
            message_slots_inplace=True,
            message_identity_slots=False,
            message_prefix_ae=True,
            message_prefix_ae_weight=1.0,
            message_prefix_ae_stopgrad_answer=True,
            message_anchor_token_ids=(keymark,),
            message_anchor_key_len=2,
            zero_init_residuals=False,
        ),
        **kw,
    )
    assert ae.config.message_prefix_ae is True
    assert ae.prefix_ae_head is not None
    e18 = build_model(
        "e18",
        spec=ArchSpec(
            name="e18",
            hidden=32,
            head_dim=16,
            local_window=16,
            message_prefix_ae=True,
            message_global_anchors="key_spans",
            zero_init_residuals=False,
        ),
        **kw,
    )
    assert e18.config.message_prefix_ae is False
    assert e18.config.message_global_anchors == "none"
    torch.manual_seed(0)
    ids = torch.randint(3, cfg.vocab.vocab_size, (2, cfg.seq_len))
    p = cfg.seq_len // 2
    ids[:, p] = qid
    labels = torch.full_like(ids, -100)
    labels[:, p + 1 :] = ids[:, p + 1 :]
    loss = ae(ids, labels=labels).loss
    assert torch.isfinite(loss)
    for _ in range(3):
        loss = ae(ids, labels=labels).loss
        assert torch.isfinite(loss)
        loss.backward()
        ae.zero_grad(set_to_none=True)

