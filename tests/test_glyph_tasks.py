"""Glyph family: typed vocab, structured noise, verifier = solvability proof.

DNA (`tests/test_symbolic_tasks.py`) stays the exact-floor A=4 control. These tests
lock the second family: every row is recoverable by `expected_answer`, the min_gap
contract holds, and a local window cannot see the evidence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from data.bapo_ladder import (
    GLYPH_CORE_RECIPES,
    GLYPH_RECIPES,
    SCALES,
    config_for,
    generate_row_for,
    resolve_recipe,
    rung_card,
)
from data.glyph_tasks import (
    GLYPH_TASKS,
    WIDTH_16,
    WIDTH_32,
    GlyphTaskConfig,
    GlyphVocab,
    _is_valid_arith,
    _is_valid_dyck,
    chance_accuracy,
    expected_answer,
    floor_nats,
    generate_row,
    iter_rows,
    prize_bits,
    verify_row,
)


def cfg_for(task: str, **over) -> GlyphTaskConfig:
    base = dict(
        task=task,
        seq_len=256,
        min_gap=32,
        width=32,
        noise="markov",
        span_len=16,
        key_len=2,
        value_len=4,
        hops=2,
        n_distractors=1,
        n_keep=8,
        k=2,
        modulus=3,
    )
    if task == "story_fact":
        base["n_distractors"] = 0
    if task == "every_k":
        base["span_len"] = 16  # divisible by k=2
    return GlyphTaskConfig(**{**base, **over})


# --------------------------------------------------------------------------------------
# Vocab
# --------------------------------------------------------------------------------------


def test_typed_layout_16_and_32_have_no_collisions():
    v16 = GlyphVocab(width=16)
    v32 = GlyphVocab(width=32)
    assert v16.vocab_size == 16 and v32.vocab_size == 32
    assert len(set(v16.names)) == 16
    assert len(set(v32.names)) == 32
    assert set(v16.digit_names).isdisjoint(v16.letter_names)
    assert "plus" in v32.names and "cat" in v32.names and "hop" in v32.names
    assert "plus" not in v16.names and "hop" not in v16.names
    assert "minus" not in v32.names and "decoy" not in v32.names
    # controls sit in the role slice, not in content
    for name in ("bos", "eos", "query", "answer", "end", "mark"):
        assert v16.control(name) >= 10
        assert v32.control(name) >= 25


def test_width_16_rejects_dyck_and_story_and_chain():
    with pytest.raises(ValueError):
        cfg_for("dyck_close", width=16)
    with pytest.raises(ValueError):
        cfg_for("story_fact", width=16)
    with pytest.raises(ValueError):
        cfg_for("chain_ordered_noise", width=16)


# --------------------------------------------------------------------------------------
# Row contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_row_shape_labels_and_vocab(task):
    cfg = cfg_for(task)
    row = generate_row(cfg, np.random.default_rng(0))
    assert row.input_ids.shape == (cfg.seq_len,)
    assert row.labels.shape == (cfg.seq_len,)
    assert row.input_ids[0] == cfg.vocab.control("bos")
    assert row.input_ids[-1] == cfg.vocab.control("eos")
    assert row.input_ids[-2] == cfg.vocab.control("end")
    supervised = np.flatnonzero(row.labels != -100)
    assert supervised.tolist() == list(range(row.answer_start, row.answer_start + cfg.answer_len))
    assert np.array_equal(row.labels[supervised], row.input_ids[supervised])
    assert row.input_ids[row.answer_start - 1] == cfg.vocab.control("answer")
    assert 0 <= int(row.input_ids.min()) and int(row.input_ids.max()) < cfg.vocab.vocab_size


@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_min_gap_is_respected(task):
    cfg = cfg_for(task)
    gaps = [row.gap for row in iter_rows(cfg, 80, seed=1)]
    assert min(gaps) >= cfg.min_gap + 1, f"{task}: min gap {min(gaps)} <= {cfg.min_gap}"


@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_generation_is_deterministic(task):
    cfg = cfg_for(task)
    a = [r.input_ids for r in iter_rows(cfg, 3, seed=7)]
    b = [r.input_ids for r in iter_rows(cfg, 3, seed=7)]
    c = [r.input_ids for r in iter_rows(cfg, 3, seed=8)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


# --------------------------------------------------------------------------------------
# Verifier = solvability proof
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_verifier_recovers_every_answer(task):
    cfg = cfg_for(task)
    for row in iter_rows(cfg, 40, seed=2):
        assert verify_row(cfg, row)
        assert expected_answer(cfg, row.input_ids) == [
            int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.answer_len]
        ]


def test_reverse_is_the_marked_span_backwards():
    cfg = cfg_for("reverse")
    v = cfg.vocab
    for row in iter_rows(cfg, 20, seed=3):
        (i,) = np.flatnonzero(row.input_ids[: row.answer_start] == v.control("mark"))
        span = [int(x) for x in row.input_ids[i + 1 : i + 1 + cfg.span_len]]
        ans = [int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.span_len]]
        assert ans == list(reversed(span))


def test_every_k_keeps_indices_0_k_2k():
    cfg = cfg_for("every_k", k=2, span_len=16)
    v = cfg.vocab
    for row in iter_rows(cfg, 20, seed=4):
        (i,) = np.flatnonzero(row.input_ids[: row.answer_start] == v.control("mark"))
        span = [int(x) for x in row.input_ids[i + 1 : i + 1 + cfg.span_len]]
        assert row.input_ids[row.answer_start - 2] == v.digit_token(2)  # k shown in the query
        assert [int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.answer_len]] == span[0::2]


def test_filter_mod_keeps_digits_divisible_by_modulus_in_order():
    cfg = cfg_for("filter_mod")
    v = cfg.vocab
    for row in iter_rows(cfg, 20, seed=5):
        (i,) = np.flatnonzero(row.input_ids[: row.answer_start] == v.control("mark"))
        span = [int(x) for x in row.input_ids[i + 1 : i + 1 + cfg.span_len]]
        kept = [t for t in span if v.digit_value(t) % cfg.modulus == 0]
        assert kept == [int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.n_keep]]
        assert len(kept) == cfg.n_keep


def test_dyck_close_is_the_stack_pop_sequence():
    cfg = cfg_for("dyck_close")
    v = cfg.vocab
    closer_of = v.closer_of
    for row in iter_rows(cfg, 20, seed=6):
        (i,) = np.flatnonzero(row.input_ids[: row.answer_start] == v.control("mark"))
        span = [int(x) for x in row.input_ids[i + 1 : i + 1 + cfg.span_len]]
        assert all(t in closer_of for t in span)
        expect = [closer_of[t] for t in reversed(span)]
        assert expect == [int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.span_len]]


# --------------------------------------------------------------------------------------
# Structured noise
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("noise", ["markov", "dyck", "arith", "mixed", "iid"])
def test_noise_runs_never_emit_controls(noise):
    cfg = cfg_for("reverse", noise=noise)
    v = cfg.vocab
    role_names = ("bos", "eos", "query", "answer", "end", "mark", "hop")
    controls = {v.control(n) for n in role_names if n in v.names}
    rng = np.random.default_rng(9)
    for _ in range(20):
        row = generate_row(cfg, rng)
        for lo, hi, _kind in row.meta["noise_runs"]:
            body = [int(x) for x in row.input_ids[lo:hi]]
            assert not (set(body) & controls)


def test_dyck_noise_fragments_are_balanced():
    cfg = cfg_for("reverse", noise="dyck")
    v = cfg.vocab
    closer_of = v.closer_of
    pad = int(v.letter_ids[0])
    for row in iter_rows(cfg, 15, seed=10):
        for lo, hi, kind in row.meta["noise_runs"]:
            assert kind == "dyck"
            seq = [int(x) for x in row.input_ids[lo:hi]]
            if seq and seq[-1] == pad and (hi - lo) % 2:
                seq = seq[:-1]
            assert _is_valid_dyck(seq, closer_of)


def test_arith_noise_is_well_formed():
    cfg = cfg_for("reverse", noise="arith")
    v = cfg.vocab
    digits = set(int(x) for x in v.digit_ids)
    ops = set(int(x) for x in v.op_ids)
    for row in iter_rows(cfg, 15, seed=11):
        for lo, hi, kind in row.meta["noise_runs"]:
            assert kind == "arith"
            assert _is_valid_arith(row.input_ids[lo:hi], digits, ops)


def test_second_chain_is_planted_when_distractors_cover_hops():
    cfg = cfg_for("chain_shuffled_noise", n_distractors=2, hops=2)
    n_second = 0
    for row in iter_rows(cfg, 30, seed=12):
        n_second += int(row.meta.get("second_chain", False))
    assert n_second == 30


# --------------------------------------------------------------------------------------
# Floor / chance
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_floor_is_log_answer_class_below_min_gap(task):
    cfg = cfg_for(task)
    assert floor_nats(cfg, cfg.min_gap) == pytest.approx(math.log(cfg.n_symbols))
    assert chance_accuracy(cfg) == pytest.approx(1.0 / cfg.n_symbols)
    assert prize_bits(cfg) == pytest.approx(cfg.answer_len * math.log(cfg.n_symbols) / math.log(2))
    assert floor_nats(cfg, cfg.min_gap + 1) == pytest.approx(0.0)


@pytest.mark.parametrize("task", ["reverse", "filter_mod", "dyck_close", "fact_markov"])
def test_answer_marginal_is_uniform_over_the_class(task):
    cfg = cfg_for(task)
    alphabet = list(int(x) for x in cfg.answer_alphabet)
    counts = {a: 0 for a in alphabet}
    for row in iter_rows(cfg, 400, seed=13):
        for tok in row.input_ids[row.answer_start : row.answer_start + cfg.answer_len]:
            counts[int(tok)] += 1
    total = sum(counts.values())
    expected = total / len(alphabet)
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    # generous: we only need "not a degenerate constant"
    assert chi2 < 40.0, (task, chi2, counts)
    assert all(c > 0 for c in counts.values())


def test_width_16_reverse_and_filter_still_verify():
    for task in ("reverse", "every_k", "filter_mod", "fact_markov", "copy_span"):
        cfg = cfg_for(task, width=16, n_distractors=0 if task == "fact_markov" else 1)
        row = generate_row(cfg, np.random.default_rng(0))
        assert cfg.vocab.vocab_size == 16
        assert verify_row(cfg, row)


# --------------------------------------------------------------------------------------
# Ladder wiring (DNA default untouched)
# --------------------------------------------------------------------------------------


def test_dna_recipes_still_resolve_and_are_not_glyph():
    rec = resolve_recipe("far_copy")
    assert rec.family == "dna"
    cfg = config_for("tiny", rec.task)
    assert type(cfg).__name__ == "SymbolicTaskConfig"
    assert cfg.n_symbols == 4


@pytest.mark.parametrize("name", list(GLYPH_RECIPES))
def test_glyph_recipes_construct_at_tiny_and_medium(name):
    rec = resolve_recipe(name)
    assert rec.family == "glyph"
    for scale in ("tiny", "tiny_wide", "bridge", "medium"):
        cfg = config_for(scale, rec.task, **rec.overrides)
        assert isinstance(cfg, GlyphTaskConfig)
        row = generate_row_for(cfg, np.random.default_rng(0))
        assert verify_row(cfg, row)
        assert row.gap >= cfg.min_gap + 1
        card = rung_card(scale, rec.task, **rec.overrides)
        assert card["family"] == "glyph"
        assert card["prize_bits"] > 0
        assert card["solvable_acc"] == 0.75


def test_glyph_core_is_the_s0_set():
    assert GLYPH_CORE_RECIPES == (
        "copy_span",
        "reverse",
        "every_k",
        "filter_mod",
        "dyck_close",
        "fact_markov_single",
    )


def test_tiny_reverse_is_packed():
    cfg = config_for("tiny", "reverse")
    assert cfg.answer_len >= 16
    assert cfg.noise == "markov"
    assert cfg.width == 32


@pytest.mark.parametrize("scale", list(SCALES))
@pytest.mark.parametrize("task", GLYPH_TASKS)
def test_every_glyph_rung_fits(scale, task):
    over = {"n_distractors": 0} if task == "story_fact" else {}
    cfg = config_for(scale, task, **over)
    card = rung_card(scale, task, **over)
    assert cfg.seq_len == SCALES[scale].seq_len
    assert card["prize_bits"] > 0
    assert cfg.answer_len >= 1
