"""Tier A/B probe metrics: frozen E30 exams, patient criterion, train-only throughput,
pre-probe gate, per-position accuracy."""
import math

import pytest
import torch

from data.bapo_ladder import (
    E30_LIMIT_RECIPES,
    EXPECTED_PRIZE_BITS,
    SCALES,
    config_for,
    resolve_recipe,
)
from data.symbolic_tasks import prize_bits
from verification.bapo_capability_probe import (
    _examples_to_criterion,
    _preprobe_decision,
    _row_se,
    _throughput,
    evaluate,
)


def _cfg(scale, name):
    rec = resolve_recipe(name)
    return config_for(SCALES[scale], rec.task, **rec.overrides)


# --- frozen exams -------------------------------------------------------------------

@pytest.mark.parametrize("scale,name", sorted(EXPECTED_PRIZE_BITS))
def test_frozen_exams_keep_their_recorded_prize(scale, name):
    assert prize_bits(_cfg(scale, name)) == EXPECTED_PRIZE_BITS[(scale, name)]


def test_e30_limit_recipes_match_the_recorded_packs():
    """Values copied from the `pack` block of the 2026-09-22 probe JSON on Odra."""
    look = _cfg("bridge_1k", "lookup_1key")
    assert (look.key_len, look.value_len, look.n_distractors) == (2, 32, 0)
    alike = _cfg("bridge_1k", "lookalike")
    assert (alike.key_len, alike.value_len, alike.n_distractors, alike.n_decoys) == (2, 32, 0, 1)
    chain = _cfg("bridge_1k", "chain_4hop")
    assert (chain.key_len, chain.hops, chain.n_distractors) == (32, 4, 2)


def test_limit_recipes_do_not_pin_the_answer_field():
    """Overrides run after packing; pinning value_len/key_len/span_len shrinks the prize."""
    for rec in E30_LIMIT_RECIPES.values():
        assert not {"value_len", "span_len", "key_len"} & set(rec.overrides)


# --- examples to criterion ----------------------------------------------------------

def _trace(accs, every=50):
    return [{"step": (i + 1) * every, "acc": a} for i, a in enumerate(accs)]


def test_single_noisy_crossing_is_not_the_criterion():
    out = _examples_to_criterion(_trace([0.3, 0.8, 0.5, 0.78, 0.8, 0.9]), batch=8, seq_len=1024)
    assert out["first_crossing_step"] == 100
    assert out["step"] == 200 and out["confirmed"]
    assert out["examples"] == 200 * 8 and out["tokens"] == 200 * 8 * 1024


def test_crossing_at_the_last_eval_is_unconfirmed_unless_early_stop():
    late = _examples_to_criterion(_trace([0.3, 0.5, 0.8]), batch=8, seq_len=16, stop_acc=0.99)
    assert late["step"] == 150 and not late["confirmed"]
    stop = _examples_to_criterion(_trace([0.3, 0.5, 0.995]), batch=8, seq_len=16, stop_acc=0.99)
    assert stop["step"] == 150 and stop["confirmed"]


def test_never_reaching_criterion_and_patience_one_matches_old_reading():
    none = _examples_to_criterion(_trace([0.3, 0.5]), batch=8, seq_len=16)
    assert none["step"] is None and none["examples"] is None and not none["confirmed"]
    old = _examples_to_criterion(_trace([0.3, 0.8, 0.5]), batch=8, seq_len=16, patience=1)
    assert old["step"] == old["first_crossing_step"] == 100


# --- throughput -----------------------------------------------------------------------

def test_throughput_counts_training_time_only():
    t = _throughput(steps=100, batch=8, seq_len=1024, train_sec=10.0, eval_sec=30.0,
                    diag_sec=20.0, device="cpu", amp="off")
    assert t["sec_per_step"] == pytest.approx(0.1)
    assert t["tokens_per_sec"] == pytest.approx(100 * 8 * 1024 / 10.0)
    assert t["wall_sec"] == pytest.approx(60.0)


# --- pre-probe gate --------------------------------------------------------------------

def test_preprobe_statuses():
    floor = math.log(4)
    assert _preprobe_decision([{"acc": 0.6, "ce_nats": 1.3}], floor_nats=floor, min_acc=0.5)["status"] == "pass"
    learning = _preprobe_decision([{"acc": 0.3, "ce_nats": floor - 0.2}], floor_nats=floor, min_acc=0.5)
    assert learning["status"] == "learning" and learning["gain_nats"] == pytest.approx(0.2)
    assert _preprobe_decision([{"acc": 0.26, "ce_nats": floor + 0.01}], floor_nats=floor, min_acc=0.5)["status"] == "flat"
    assert _preprobe_decision([], floor_nats=floor, min_acc=0.5)["status"] == "flat"


# --- evaluate: per-position accuracy and row SE -------------------------------------------

class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Echo(torch.nn.Module):
    """Predicts the true next token on the first `k_right` supervised offsets, wrong after."""

    def __init__(self, vocab, k_right):
        super().__init__()
        self.vocab, self.k_right = vocab, k_right
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, labels=None, return_per_token_loss=False, return_logits=False):
        B, S = ids.shape
        logits = torch.zeros(B, S, self.vocab)
        nxt = torch.roll(ids, -1, dims=1)
        wrong = (nxt + 1) % self.vocab
        if self._labels is not None:
            m = self._labels[:, 1:] != -100
            off = m.long().cumsum(1) - 1
            pick = torch.where(off < self.k_right, nxt[:, :-1], wrong[:, :-1])
            logits[:, :-1].scatter_(2, pick[..., None], 5.0)
        if return_per_token_loss:
            valid = (labels[:, 1:] != -100)
            per = torch.ones_like(valid, dtype=torch.float)
            return None, per, valid
        return _Out(logits)


def test_evaluate_reports_which_answer_letters_survive():
    vocab, S, A = 12, 20, 6
    ids = torch.randint(3, vocab, (4, S))
    labels = torch.full((4, S), -100)
    labels[:, S - A:] = ids[:, S - A:]
    model = _Echo(vocab, k_right=4)
    model._labels = labels
    ev = evaluate(model, [(ids, labels)], amp="off", device=torch.device("cpu"))
    assert ev["rows"] == 4
    assert len(ev["per_position_acc"]) == A
    assert ev["per_position_acc"][:4] == [1.0] * 4 and ev["per_position_acc"][4:] == [0.0, 0.0]
    assert ev["acc"] == pytest.approx(4 / 6)
    assert ev["acc_se"] == pytest.approx(0.0)


def test_row_se():
    assert math.isnan(_row_se([0.5]))
    assert _row_se([0.0, 1.0]) == pytest.approx(0.5)
