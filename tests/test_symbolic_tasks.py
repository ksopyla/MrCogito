"""Unit tests for the symbolic long-context task suite (data/symbolic_tasks.py).

The suite is only useful if three properties actually hold, so they are tested rather than
asserted in prose:

1. **Solvable from the evidence** — the answer is recoverable by following the construction, so
   a model with a working long-range channel can in principle reach zero loss.
2. **Not solvable locally** — the supervised tokens are statistically independent of everything
   a local window can see, so a segment-confined control cannot beat the floor by cheating.
3. **The floor is right** — `floor_nats` agrees with a Monte-Carlo estimate of the entropy.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from data.symbolic_tasks import (
    CONTROL_NAMES,
    TASKS,
    SymbolicTaskConfig,
    SymbolicVocab,
    _binomial_mod_entropy,
    chance_accuracy,
    floor_nats,
    generate_row,
    iter_rows,
)

REPO = Path(__file__).resolve().parents[1]


def cfg_for(task: str, **over) -> SymbolicTaskConfig:
    base = dict(
        task=task,
        seq_len=1024,
        n_symbols=4,
        min_gap=256,
        key_len=3,
        value_len=3,
        span_len=16,
        n_distractors=5,
        hops=3,
        count_mod=4,
    )
    return SymbolicTaskConfig(**{**base, **over})


# --------------------------------------------------------------------------------------
# Vocabulary layout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sym_lo", [0, 128100])
def test_vocab_layout_has_no_collisions(sym_lo):
    v = SymbolicVocab(n_symbols=6, sym_lo=sym_lo)
    symbols = set(int(x) for x in v.symbol_ids)
    controls = {v.control(name) for name in CONTROL_NAMES}
    assert len(symbols) == 6 and len(controls) == len(CONTROL_NAMES)
    assert symbols.isdisjoint(controls)
    assert v.vocab_size == sym_lo + 6 + len(CONTROL_NAMES)
    assert max(symbols | controls) == v.vocab_size - 1


def test_vocab_rejects_degenerate_alphabet():
    with pytest.raises(ValueError):
        SymbolicVocab(n_symbols=1)
    with pytest.raises(KeyError):
        SymbolicVocab().control("nope")


def test_config_rejects_rows_that_cannot_fit():
    with pytest.raises(ValueError):
        cfg_for("far_copy", seq_len=64, min_gap=256)
    with pytest.raises(ValueError):
        cfg_for("count", count_mod=8, n_symbols=4)   # answer would not fit in one symbol
    with pytest.raises(ValueError):
        cfg_for("chain", hops=1)                     # no composition required


# --------------------------------------------------------------------------------------
# Row structure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_row_shape_and_label_placement(task):
    cfg = cfg_for(task)
    rng = np.random.default_rng(0)
    for _ in range(16):
        row = generate_row(cfg, rng)
        assert row.input_ids.shape == (cfg.seq_len,)
        assert row.labels.shape == (cfg.seq_len,)
        assert row.input_ids[0] == cfg.vocab.control("bos")
        assert row.input_ids[-1] == cfg.vocab.control("eos")
        assert row.input_ids[-2] == cfg.vocab.control("end")
        # Labels are aligned with input_ids (the model does the shift) and cover exactly the
        # answer span, which is what `labels_from_span_markers` would also produce.
        supervised = np.flatnonzero(row.labels != -100)
        assert supervised.tolist() == list(range(row.answer_start, row.answer_start + cfg.answer_len))
        assert np.array_equal(row.labels[supervised], row.input_ids[supervised])
        assert row.input_ids[row.answer_start - 1] == cfg.vocab.control("answer")
        # Ids stay inside the declared vocabulary.
        assert row.input_ids.min() >= 0 and row.input_ids.max() < cfg.vocab.vocab_size


@pytest.mark.parametrize("task", ["recall", "far_copy", "chain"])
def test_min_gap_is_respected(task):
    """The contract `floor_nats` relies on: a raw window of `min_gap` cannot reach the evidence,
    which needs `gap >= min_gap + 1` because the window includes `answer_start - 1` itself."""
    cfg = cfg_for(task)
    gaps = [row.gap for row in iter_rows(cfg, 200, seed=1)]
    assert min(gaps) >= cfg.min_gap + 1, f"{task}: min gap {min(gaps)} <= {cfg.min_gap}"


@pytest.mark.parametrize("task", TASKS)
def test_generation_is_deterministic(task):
    cfg = cfg_for(task)
    a = [r.input_ids for r in iter_rows(cfg, 4, seed=7)]
    b = [r.input_ids for r in iter_rows(cfg, 4, seed=7)]
    c = [r.input_ids for r in iter_rows(cfg, 4, seed=8)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


# --------------------------------------------------------------------------------------
# Property 1: solvable from the evidence
# --------------------------------------------------------------------------------------


def _symbols(cfg, ids):
    return [int(x) - cfg.sym_lo for x in ids]


def test_recall_answer_is_the_queried_value():
    cfg = cfg_for("recall")
    v = cfg.vocab
    for row in iter_rows(cfg, 50, seed=2):
        ids = row.input_ids
        query = _symbols(cfg, ids[row.answer_start - 1 - cfg.key_len : row.answer_start - 1])
        answer = _symbols(cfg, ids[row.answer_start : row.answer_start + cfg.value_len])
        # Scan the body for the key/value block whose key matches the query.
        found = []
        for i in np.flatnonzero(ids[: row.answer_start - 1] == v.control("keymark")):
            k = _symbols(cfg, ids[i + 1 : i + 1 + cfg.key_len])
            if k == query:
                found.append(_symbols(cfg, ids[i + 1 + cfg.key_len : i + 1 + cfg.key_len + cfg.value_len]))
        assert len(found) == 1, "the queried key must occur exactly once in the body"
        assert found[0] == answer


def test_far_copy_answer_is_the_marked_span():
    cfg = cfg_for("far_copy")
    v = cfg.vocab
    for row in iter_rows(cfg, 50, seed=3):
        ids = row.input_ids
        (i,) = np.flatnonzero(ids[: row.answer_start - 1] == v.control("spanmark"))
        span = _symbols(cfg, ids[i + 1 : i + 1 + cfg.span_len])
        assert span == _symbols(cfg, ids[row.answer_start : row.answer_start + cfg.span_len])


def test_chain_answer_requires_following_every_hop():
    cfg = cfg_for("chain")
    v = cfg.vocab
    k = cfg.key_len
    for row in iter_rows(cfg, 50, seed=4):
        ids = row.input_ids
        edges = {}
        for i in np.flatnonzero(ids[: row.answer_start - 1] == v.control("hop")):
            src = tuple(_symbols(cfg, ids[i + 1 : i + 1 + k]))
            dst = tuple(_symbols(cfg, ids[i + 1 + k : i + 1 + 2 * k]))
            assert src not in edges, "each source must have exactly one outgoing edge"
            edges[src] = dst
        node = tuple(_symbols(cfg, ids[row.answer_start - 1 - k : row.answer_start - 1]))
        for _ in range(cfg.hops):
            node = edges[node]
        assert list(node) == _symbols(cfg, ids[row.answer_start : row.answer_start + k])


def test_count_answer_is_the_residue_over_the_whole_body():
    cfg = cfg_for("count")
    tail_start = cfg.seq_len - cfg.tail_len
    for row in iter_rows(cfg, 50, seed=5):
        ids = row.input_ids
        target = int(ids[row.answer_start - 2])
        total = int(np.count_nonzero(ids[1:tail_start] == target))
        assert total == row.meta["total"]
        assert int(ids[row.answer_start]) - cfg.sym_lo == total % cfg.count_mod


def test_chain_hops_are_not_in_reading_order():
    """If the edges were laid out in chain order a left-to-right reader could shortcut it."""
    cfg = cfg_for("chain", hops=3, n_distractors=5)
    v, k = cfg.vocab, cfg.key_len
    in_order = 0
    rows = list(iter_rows(cfg, 60, seed=6))
    for row in rows:
        ids = row.input_ids
        edges = []
        for i in np.flatnonzero(ids[: row.answer_start - 1] == v.control("hop")):
            edges.append(
                (
                    tuple(_symbols(cfg, ids[i + 1 : i + 1 + k])),
                    tuple(_symbols(cfg, ids[i + 1 + k : i + 1 + 2 * k])),
                )
            )
        lookup = dict(edges)
        node = tuple(_symbols(cfg, ids[row.answer_start - 1 - k : row.answer_start - 1]))
        positions = []
        for _ in range(cfg.hops):
            positions.append(next(j for j, (s, _) in enumerate(edges) if s == node))
            node = lookup[node]
        in_order += positions == sorted(positions)
    # 3 hops among 8 edges: reading order happens by chance for ~1/6 of rows.
    assert in_order < 0.5 * len(rows), f"{in_order}/{len(rows)} rows had the chain in reading order"


# --------------------------------------------------------------------------------------
# Property 2: not solvable locally
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_answer_marginal_is_uniform(task):
    """Supervised tokens are uniform over the alphabet (mod count_mod for `count`), so guessing
    a constant cannot beat `chance_accuracy` and the entropy floor is attained, not approached."""
    cfg = cfg_for(task)
    n_out = cfg.count_mod if task == "count" else cfg.n_symbols
    counts = np.zeros(n_out)
    rows = 3000 if task == "count" else 600
    for row in iter_rows(cfg, rows, seed=11):
        ans = row.input_ids[row.answer_start : row.answer_start + cfg.answer_len] - cfg.sym_lo
        np.add.at(counts, ans.astype(int), 1.0)
    expected = counts.sum() / n_out
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # 3 or 4 degrees of freedom: the 0.1% critical value is ~16.3 / ~18.5.
    assert chi2 < 18.5, f"{task}: answer marginal not uniform (chi2={chi2:.1f}, counts={counts})"
    assert chance_accuracy(cfg) == pytest.approx(1.0 / n_out)


@pytest.mark.parametrize("task", ["recall", "chain"])
def test_answer_is_independent_of_the_query(task):
    """The query key is the only informative thing inside a local window. If the answer were
    correlated with it, a model could score below the floor without ever reaching the evidence."""
    cfg = cfg_for(task)
    A = cfg.n_symbols
    table = np.zeros((A, A))
    for row in iter_rows(cfg, 4000, seed=12):
        q = int(row.input_ids[row.answer_start - 1 - cfg.key_len]) - cfg.sym_lo
        a = int(row.input_ids[row.answer_start]) - cfg.sym_lo
        table[q, a] += 1
    row_m, col_m = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
    expected = row_m * col_m / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    # (A-1)^2 = 9 degrees of freedom; 0.1% critical value 27.9.
    assert chi2 < 27.9, f"{task}: answer depends on the query (chi2={chi2:.1f})"


def test_local_window_oracle_is_at_chance():
    """A predictor that memorises `visible window -> answer` across the training rows and is
    then asked about held-out rows must land at chance. This is the empirical version of the
    claim `floor_nats` encodes."""
    cfg = cfg_for("recall")
    window = cfg.min_gap - 1
    table: dict[bytes, list[int]] = {}
    rows = list(iter_rows(cfg, 1200, seed=13))
    for row in rows[:1000]:
        ctx = row.input_ids[row.answer_start - window : row.answer_start].tobytes()
        table.setdefault(ctx, []).append(int(row.input_ids[row.answer_start]))
    hits = 0
    for row in rows[1000:]:
        ctx = row.input_ids[row.answer_start - window : row.answer_start].tobytes()
        seen = table.get(ctx)
        pred = max(set(seen), key=seen.count) if seen else 0
        hits += pred == int(row.input_ids[row.answer_start])
    acc = hits / len(rows[1000:])
    assert acc < 0.45, f"local oracle reached {acc:.3f}, well above chance {chance_accuracy(cfg):.3f}"


# --------------------------------------------------------------------------------------
# Property 3: the floor is right
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["recall", "far_copy", "chain"])
def test_retrieval_floor_is_log_alphabet_below_min_gap(task):
    cfg = cfg_for(task)
    assert floor_nats(cfg, cfg.min_gap) == pytest.approx(math.log(cfg.n_symbols))
    assert floor_nats(cfg, cfg.min_gap + 1) == 0.0
    assert floor_nats(cfg_for(task, n_symbols=8), 1) == pytest.approx(math.log(8))
    with pytest.raises(ValueError):
        floor_nats(cfg, 0)


@pytest.mark.parametrize("n,m", [(40, 4), (17, 3), (200, 8)])
def test_binomial_mod_entropy_matches_monte_carlo(n, m):
    p = 0.25
    exact = _binomial_mod_entropy(n, p, m)
    draws = np.random.default_rng(0).binomial(n, p, size=400_000) % m
    probs = np.bincount(draws, minlength=m) / draws.size
    mc = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
    assert exact == pytest.approx(mc, abs=5e-3), f"exact {exact:.5f} vs MC {mc:.5f}"


def test_count_floor_falls_as_the_window_covers_the_body():
    """Sanity on the direction of the information: a window that sees the whole body knows the
    count exactly, a window that sees none of it is at ln(count_mod)."""
    cfg = cfg_for("count", seq_len=512, count_mod=4)
    assert floor_nats(cfg, 1) == pytest.approx(math.log(4), abs=1e-6)
    assert floor_nats(cfg, cfg.seq_len) == pytest.approx(0.0, abs=1e-9)
    windows = [1, 64, 256, 480, 505, cfg.seq_len]
    floors = [floor_nats(cfg, w) for w in windows]
    assert all(b <= a + 1e-9 for a, b in zip(floors, floors[1:])), floors
    # The statistic needs the *whole* body: the floor only moves once the window nearly covers
    # it, which is what makes `count` an aggregation task rather than a retrieval one.
    assert floors[2] == pytest.approx(math.log(4), abs=1e-6)
    assert floors[4] < math.log(4) - 0.2


def test_large_binomial_falls_back_to_the_uniform_limit():
    assert _binomial_mod_entropy(10_000_000, 0.25, 4) == pytest.approx(math.log(4))
    assert _binomial_mod_entropy(0, 0.25, 4) == 0.0


# --------------------------------------------------------------------------------------
# Builder: both label routes agree
# --------------------------------------------------------------------------------------


def test_builder_writes_both_schemas_and_the_two_label_routes_agree(tmp_path):
    from datasets import load_from_disk

    from data.data_collators import labels_from_span_markers

    common = [
        sys.executable, "scripts/build_symbolic_dataset.py",
        "--task", "recall", "far_copy",
        "--seq_len", "512", "--min_gap", "128", "--key_len", "2", "--value_len", "2",
        "--span_len", "8", "--n_distractors", "3", "--n_train", "8", "--n_eval", "4",
    ]
    diag = tmp_path / "diag"
    lm = tmp_path / "lm"
    for out, extra in ((diag, []), (lm, ["--lm_columns_only"])):
        proc = subprocess.run(
            common + ["--out_dir", str(out / "ds"), "--manifest", str(out / "m.json")] + extra,
            cwd=REPO, text=True, capture_output=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    man = json.loads((diag / "m.json").read_text())
    assert [s["name"] for s in man["sources"]] == ["sym_recall", "sym_far_copy"]
    assert man["symbolic_meta"]["vocab_size"] == 4 + len(CONTROL_NAMES)
    assert man["label_policy"] == "answer_only"
    for src in man["sources"]:
        assert src["floor_nats_per_supervised_token"] == pytest.approx(math.log(4), abs=1e-5)

    lm_man = json.loads((lm / "m.json").read_text())
    assert lm_man["label_policy"] == "span_markers"
    start, end = lm_man["symbolic_meta"]["markers"]

    for task in ("recall", "far_copy"):
        d = load_from_disk(str(diag / "ds" / task / "eval"))
        l = load_from_disk(str(lm / "ds" / task / "eval"))
        assert set(d.column_names) == {"input_ids", "labels", "gap"}
        assert set(l.column_names) == {"input_ids", "attention_mask", "special_tokens_mask"}
        for i in range(len(d)):
            ids = d[i]["input_ids"]
            # The marker route must recover exactly the stored labels, minus the `end` marker,
            # which closes the span and is itself a (legitimately) supervised token.
            via_markers = labels_from_span_markers(ids, start, end)
            stored = d[i]["labels"]
            sup_stored = [j for j, x in enumerate(stored) if x != -100]
            sup_marker = [j for j, x in enumerate(via_markers) if x != -100]
            assert sup_marker == sup_stored + [sup_stored[-1] + 1]
            assert all(via_markers[j] == stored[j] for j in sup_stored)
        assert all(x == 1 for x in l[0]["attention_mask"])
        assert sum(l[0]["special_tokens_mask"]) >= 4
