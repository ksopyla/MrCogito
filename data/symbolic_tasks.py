"""Symbolic long-context tasks over a tiny alphabet, with closed-form information floors.

Why this module exists
----------------------
E22 killed a concept-array bet using natural-text cross-entropy, where the *entire* prize for
far context is ~0.05 nats at pilot scale (E18's reach ablation) — the objective barely paid for
the channel, and the pre-registered gate was unreachable by construction. These generators
remove that ambiguity. Every row is a sequence over a small symbolic alphabet (DNA-like) in
which the supervised tokens are *determined* by evidence at a controlled distance and
*independent* of everything inside a local window, so:

- the prize is large and exact — `ln(n_symbols)` nats per supervised token, not 0.05;
- the floor for any model whose receptive field excludes the evidence is computable in closed
  form (`floor_nats`), so "the channel carried information" becomes a provable statement rather
  than a comparison against a control that may itself be weak;
- nothing is memorisable: keys, values, spans and chains are drawn fresh per row, so a model
  cannot bake the mapping into its weights.

Tasks, and the mechanism each one isolates
------------------------------------------
- `recall`     — content addressing. One of several key/value blocks is queried at the end.
                 The array must be *addressed* by content, not by position.
- `far_copy`   — channel bandwidth. A span from far back must be reproduced verbatim; sweeping
                 `span_len` measures how many bits the channel actually carries.
- `chain`      — composition over slots. `a->b`, `b->c`, `c->d` are scattered in random order,
                 so answering requires several dependent lookups *inside* the latent space.
- `count`      — aggregation. Report the count of a queried symbol (mod `count_mod`) over the
                 whole sequence. Unlike the others, this cannot be solved by retrieving one
                 site: it needs a running statistic. This is the one task where a compressive
                 bottleneck should have a *structural advantage* over exact attention, which
                 must re-derive the statistic from the whole prefix every time.

Role in the research program
----------------------------
A falsifier, not a success criterion. A compressive channel that cannot do these tasks cannot
be useful at 1M-10M context, and that can be established cheaply. The converse does not hold:
solving them says nothing on its own about whether the channel helps on language.

Conventions
-----------
`labels` are *aligned* with `input_ids` (`labels[i] == input_ids[i]` on supervised positions,
`-100` elsewhere) and the model does the shift, matching `scripts/build_copy_task_dataset.py`
and `data.data_collators.labels_from_span_markers`. Rows are exactly `seq_len` tokens. The
supervised span is framed by the `answer` / `end` control ids, so rows can also be trained
through `--loss_span_markers` with no `labels` column, keeping the plain LM shard schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

TASKS: tuple[str, ...] = ("recall", "far_copy", "chain", "count")

# Control symbols, in the order they are laid out above the content alphabet.
CONTROL_NAMES: tuple[str, ...] = (
    "bos",
    "eos",
    "keymark",   # opens a key/value block (recall)
    "spanmark",  # opens a span to be copied (far_copy)
    "hop",       # opens an `x -> y` edge (chain)
    "query",     # opens the question
    "answer",    # opens the supervised span (doubles as the START marker)
    "end",       # closes the supervised span (doubles as the END marker)
)


@dataclass(frozen=True)
class SymbolicVocab:
    """Id layout: `n_symbols` content symbols at `sym_lo`, then the control ids.

    `sym_lo` exists so symbolic rows can be mapped into a reserved slice of a real tokenizer's
    id space when they are mixed into a text corpus (as `build_retrieval_mix_dataset.py` does
    with its key range). For pure-symbolic runs leave it at 0 and the vocabulary is
    `n_symbols + 8` ids wide.
    """

    n_symbols: int = 4
    sym_lo: int = 0

    def __post_init__(self) -> None:
        if self.n_symbols < 2:
            raise ValueError(f"n_symbols must be >= 2, got {self.n_symbols}")
        if self.sym_lo < 0:
            raise ValueError(f"sym_lo must be >= 0, got {self.sym_lo}")

    @property
    def symbol_ids(self) -> np.ndarray:
        return np.arange(self.sym_lo, self.sym_lo + self.n_symbols, dtype=np.int64)

    def control(self, name: str) -> int:
        try:
            return self.sym_lo + self.n_symbols + CONTROL_NAMES.index(name)
        except ValueError as exc:  # pragma: no cover - programming error
            raise KeyError(f"unknown control symbol {name!r}; have {CONTROL_NAMES}") from exc

    @property
    def vocab_size(self) -> int:
        return self.sym_lo + self.n_symbols + len(CONTROL_NAMES)


@dataclass(frozen=True)
class SymbolicTaskConfig:
    """One task instance. `min_gap` is the contract that makes the floor exact.

    Every row guarantees `gap >= min_gap + 1`, where `gap` is the distance from the last token
    of the evidence to the first supervised token. The first supervised token is predicted from
    position `answer_start - 1`, whose raw window of width `w` covers
    `[answer_start - w, answer_start - 1]`, so the evidence is invisible exactly when
    `w <= min_gap` — for both `dec_local="block"` (segment reset, reach at most `dec_segment`)
    and `dec_local="swa"` (sliding window `dec_segment`). Set `min_gap >= dec_segment`.
    """

    task: str = "recall"
    seq_len: int = 2048
    n_symbols: int = 4
    sym_lo: int = 0
    min_gap: int = 1024
    key_len: int = 4
    value_len: int = 4
    span_len: int = 32
    n_distractors: int = 7
    hops: int = 3
    count_mod: int = 4

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.min_gap < 1:
            raise ValueError("min_gap must be >= 1")
        if self.task == "count":
            if self.count_mod < 2:
                raise ValueError("count_mod must be >= 2")
            if self.count_mod > self.n_symbols:
                raise ValueError(
                    f"count_mod ({self.count_mod}) must be <= n_symbols ({self.n_symbols}) so the "
                    "answer fits in one symbol"
                )
        if self.task == "chain" and self.hops < 2:
            raise ValueError("chain needs hops >= 2 to require composition")
        need = self.tail_len + self.evidence_len + 1  # + BOS
        if need + self.min_gap > self.seq_len:
            raise ValueError(
                f"seq_len {self.seq_len} too short for task {self.task!r}: needs "
                f"{need} tokens of structure + min_gap {self.min_gap}"
            )

    @property
    def vocab(self) -> SymbolicVocab:
        return SymbolicVocab(n_symbols=self.n_symbols, sym_lo=self.sym_lo)

    @property
    def answer_len(self) -> int:
        """Number of supervised tokens per row."""
        return {
            "recall": self.value_len,
            "far_copy": self.span_len,
            "chain": self.key_len,
            "count": 1,
        }[self.task]

    @property
    def query_len(self) -> int:
        """Tokens between the `query` and `answer` markers."""
        return {"recall": self.key_len, "far_copy": 0, "chain": self.key_len, "count": 1}[self.task]

    @property
    def tail_len(self) -> int:
        # query + query tokens + answer + answer tokens + end + eos
        return 1 + self.query_len + 1 + self.answer_len + 1 + 1

    @property
    def evidence_len(self) -> int:
        """Total tokens of evidence blocks placed in the body."""
        if self.task == "recall":
            return (self.n_distractors + 1) * (1 + self.key_len + self.value_len)
        if self.task == "far_copy":
            return 1 + self.span_len
        if self.task == "chain":
            return (self.hops + self.n_distractors) * (1 + 2 * self.key_len)
        return 0  # count has no localized evidence

    @property
    def answer_start(self) -> int:
        return self.seq_len - self.tail_len + 1 + self.query_len + 1


@dataclass
class SymbolicRow:
    input_ids: np.ndarray
    labels: np.ndarray
    gap: int
    answer_start: int
    answer_len: int
    meta: dict = field(default_factory=dict)


def _sample_distinct_tuples(rng: np.random.Generator, n: int, width: int, n_symbols: int) -> list[tuple[int, ...]]:
    """`n` distinct symbol tuples, so keys/nodes never collide inside a row."""
    if n_symbols**width < n:
        raise ValueError(f"cannot draw {n} distinct tuples of width {width} over {n_symbols} symbols")
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    while len(out) < n:
        cand = tuple(int(x) for x in rng.integers(0, n_symbols, size=width))
        if cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def _place_blocks(
    rng: np.random.Generator, blocks: Sequence[Sequence[int]], lo: int, hi: int
) -> list[int]:
    """Non-overlapping random offsets for `blocks` inside `[lo, hi)`, in a random order.

    Offsets are returned per block (aligned with `blocks`). Raises if they cannot fit, which is
    already excluded by `SymbolicTaskConfig.__post_init__`.
    """
    total = sum(len(b) for b in blocks)
    slack = (hi - lo) - total
    if slack < 0:
        raise ValueError(f"blocks of total length {total} do not fit in [{lo}, {hi})")
    # Distribute the slack into len(blocks)+1 gaps uniformly (stars and bars), which spreads the
    # evidence over the whole body instead of clustering it.
    cuts = np.sort(rng.integers(0, slack + 1, size=len(blocks)))
    consumed, offsets = 0, []
    for i, block in enumerate(blocks):
        offsets.append(lo + int(cuts[i]) + consumed)
        consumed += len(block)
    return offsets


def generate_row(cfg: SymbolicTaskConfig, rng: np.random.Generator) -> SymbolicRow:
    """One row of exactly `cfg.seq_len` tokens, with the answer determined by far evidence."""
    v = cfg.vocab
    S, A = cfg.seq_len, cfg.n_symbols
    ids = np.empty(S, dtype=np.int64)
    # iid uniform filler: nothing about the answer is predictable from local statistics.
    ids[:] = v.sym_lo + rng.integers(0, A, size=S)
    ids[0] = v.control("bos")

    tail_start = S - cfg.tail_len
    answer_start = cfg.answer_start
    evidence_hi = answer_start - cfg.min_gap  # evidence must end strictly before this

    def sym(t: Sequence[int]) -> list[int]:
        return [v.sym_lo + int(x) for x in t]

    meta: dict = {}
    if cfg.task == "recall":
        n_items = cfg.n_distractors + 1
        keys = _sample_distinct_tuples(rng, n_items, cfg.key_len, A)
        values = [tuple(int(x) for x in rng.integers(0, A, size=cfg.value_len)) for _ in range(n_items)]
        blocks = [[v.control("keymark"), *sym(k), *sym(val)] for k, val in zip(keys, values)]
        offsets = _place_blocks(rng, blocks, 1, evidence_hi)
        for off, block in zip(offsets, blocks):
            ids[off : off + len(block)] = block
        pick = int(rng.integers(0, n_items))
        query_tokens, answer_tokens = sym(keys[pick]), sym(values[pick])
        evidence_end = offsets[pick] + len(blocks[pick]) - 1
        meta = {"n_items": n_items, "picked": pick}

    elif cfg.task == "far_copy":
        span = tuple(int(x) for x in rng.integers(0, A, size=cfg.span_len))
        block = [v.control("spanmark"), *sym(span)]
        (off,) = _place_blocks(rng, [block], 1, evidence_hi)
        ids[off : off + len(block)] = block
        query_tokens, answer_tokens = [], sym(span)
        evidence_end = off + len(block) - 1

    elif cfg.task == "chain":
        # Only *sources* must be distinct, so every edge has one unambiguous target. The terminal
        # node is a pure target and is drawn iid uniform, which keeps the floor exactly
        # `ln(A)` per token — a distinct-pool draw would leak a little information.
        sources = _sample_distinct_tuples(rng, cfg.hops + cfg.n_distractors, cfg.key_len, A)
        terminal = tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len))
        chain = [*sources[: cfg.hops], terminal]
        edges = [(chain[i], chain[i + 1]) for i in range(cfg.hops)]
        edges += [
            (src, tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len)))
            for src in sources[cfg.hops :]
        ]
        # Shuffle so the chain is not in reading order: it must be resolved by content.
        order = rng.permutation(len(edges))
        blocks = [[v.control("hop"), *sym(edges[i][0]), *sym(edges[i][1])] for i in order]
        offsets = _place_blocks(rng, blocks, 1, evidence_hi)
        for off, block in zip(offsets, blocks):
            ids[off : off + len(block)] = block
        query_tokens, answer_tokens = sym(chain[0]), sym(chain[-1])
        # The binding constraint is the *last* hop the model still needs, i.e. the latest-placed
        # chain edge; anything earlier is further away.
        chain_positions = [
            offsets[j] + len(blocks[j]) - 1 for j, i in enumerate(order) if int(i) < cfg.hops
        ]
        evidence_end = max(chain_positions)
        meta = {"hops": cfg.hops}

    else:  # count
        target = int(rng.integers(0, A))
        body = ids[1:tail_start]
        total = int(np.count_nonzero(body == v.sym_lo + target))
        residue = total % cfg.count_mod
        query_tokens, answer_tokens = [v.sym_lo + target], [v.sym_lo + residue]
        # The statistic spans the whole body, so the nearest evidence is adjacent: `gap` is not a
        # meaningful quantity here and the floor comes from the unseen *portion* of the body.
        evidence_end = tail_start - 1
        meta = {"target": target, "total": total, "residue": residue}

    tail = [
        v.control("query"),
        *query_tokens,
        v.control("answer"),
        *answer_tokens,
        v.control("end"),
        v.control("eos"),
    ]
    assert len(tail) == cfg.tail_len, (len(tail), cfg.tail_len)
    ids[tail_start:] = tail
    assert ids[answer_start] == answer_tokens[0]

    labels = np.full(S, -100, dtype=np.int64)
    labels[answer_start : answer_start + cfg.answer_len] = ids[answer_start : answer_start + cfg.answer_len]

    return SymbolicRow(
        input_ids=ids,
        labels=labels,
        gap=int(answer_start - evidence_end),
        answer_start=int(answer_start),
        answer_len=int(cfg.answer_len),
        meta=meta,
    )


def iter_rows(cfg: SymbolicTaskConfig, n_rows: int, seed: int) -> Iterator[SymbolicRow]:
    rng = np.random.default_rng(seed)
    for _ in range(n_rows):
        yield generate_row(cfg, rng)


# --------------------------------------------------------------------------------------
# Information floors
# --------------------------------------------------------------------------------------

_BINOM_EXACT_LIMIT = 200_000


def _binomial_mod_entropy(n: int, p: float, m: int) -> float:
    """Entropy in nats of `Binomial(n, p) mod m`.

    For large `n` the residue is exponentially close to uniform, so `ln(m)` is returned above a
    cutoff rather than summing a million log-gamma terms.
    """
    if n <= 0:
        return 0.0
    if n > _BINOM_EXACT_LIMIT:
        return math.log(m)
    k = np.arange(n + 1)
    from scipy.special import gammaln  # local import: only the floor path needs scipy

    log_pmf = (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)
    )
    pmf = np.exp(log_pmf - log_pmf.max())
    pmf /= pmf.sum()
    probs = np.zeros(m)
    np.add.at(probs, k % m, pmf)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def floor_nats(cfg: SymbolicTaskConfig, window: int) -> float:
    """Cross-entropy floor, in nats per supervised token, for a model whose raw receptive field
    at the answer is the last `window` tokens and which has no other route to the evidence.

    This is the number a segment-confined control (arm C) must sit at, and the number a working
    concept channel must beat. It is exact, not estimated — which is precisely what E22's S1
    gate lacked (see `docs/4_Research_Notes/e22_root_cause_20260912.md` §7.1).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if cfg.task == "count":
        # The first answer token is predicted from position `answer_start - 1`, so the visible
        # tokens are `[answer_start - window, answer_start - 1]`. Body tokens outside that are
        # unobserved and contribute `Binomial(n_far, 1/A)` to the count.
        tail_start = cfg.seq_len - cfg.tail_len
        body_len = tail_start - 1
        visible_body = max(0, tail_start - max(1, cfg.answer_start - window))
        n_far = max(0, body_len - visible_body)
        return _binomial_mod_entropy(n_far, 1.0 / cfg.n_symbols, cfg.count_mod)
    if window > cfg.min_gap:
        # The evidence may fall inside the window for some rows; the floor is then row-dependent
        # and this function no longer bounds anything. Callers must keep `window <= min_gap`.
        return 0.0
    return math.log(cfg.n_symbols)


def chance_accuracy(cfg: SymbolicTaskConfig) -> float:
    """Per-supervised-token accuracy of a model that cannot see the evidence."""
    if cfg.task == "count":
        return 1.0 / cfg.count_mod
    return 1.0 / cfg.n_symbols
