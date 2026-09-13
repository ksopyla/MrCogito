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


def test_info_report_perfect_recovery():
    cfg = config_for("tiny", "far_copy")
    floor = math.log(cfg.n_symbols)
    rep = info_report(
        ce_nats=0.0,
        acc=1.0,
        n_supervised=64,
        cfg=cfg,
        window=16,
        nominal_b_tokens=96,
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


@pytest.mark.parametrize("arch", ["dense", "e18", "e18_local", "encdec"])
def test_factory_builds_under_100m(arch):
    cfg = config_for("tiny", "recall")
    spec = ArchSpec(name=arch, hidden=32, head_dim=16, local_window=16, enc_layers=1, dec_layers=1)
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
    labels = torch.full((2, cfg.seq_len), -100)
    labels[:, cfg.answer_start : cfg.answer_start + cfg.answer_len] = ids[:, cfg.answer_start : cfg.answer_start + cfg.answer_len]
    loss = model(ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
