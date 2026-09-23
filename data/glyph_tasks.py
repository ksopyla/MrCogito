"""Typed-vocab capability tasks with *structured* (not iid) noise.

Why this module exists
----------------------
`data/symbolic_tasks.py` is the DNA A=4 instrument: exact floor `ln(4)`, chance 25%,
iid uniform filler. That is the right control for *channel bandwidth*. It is a toy
noise model — a local LM cannot "find the filler plausible", so it does not test
"ignore language-like distractors".

This family (Glyph) is the second instrument. Vocab 16 or 32 is *typed* (digits,
letters, brackets, operators, a four-word closed lexicon, role markers), not 32
arbitrary ids. Filler is drawn from a known process a local model would like
(Markov over letters/words, a valid Dyck fragment, a well-formed digit-op expression).
The answer is still uniquely determined by a stack/DFA/scan verifier, so recovered
bits stay well-defined. DNA is untouched.

Tasks and the mechanism each isolates
--------------------------------------
- `copy_span`     — positional INDEX / bandwidth, but the haystack is structured.
- `reverse`       — same span, emit it backwards (Delétang Reverse String / Olsson reverse).
- `every_k`       — emit tokens at indices 0, k, 2k, … of the marked span (selective indexing).
- `filter_mod`    — emit the digits ≡ 0 (mod m) in order (MAD selective copy / predicate filter).
- `dyck_close`    — unmatched Dyck-2 stack; emit the unique closing sequence (32 only).
- `fact_markov`   — keyed recall whose haystack is Markov, not iid (BABILong Adapt).
- `story_fact`    — same recall with a closed word as the key (32 only; TinyStories Adapt).
- `chain_ordered_noise` / `chain_shuffled_noise` — DNA hops inside structured filler;
                     shuffled plants a second independent chain as the distractor.

The DNA suite remains the exact-floor control. Do not score E18 here until a matched
dense control hits 75% (same protocol as `data/bapo_ladder.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from data.symbolic_tasks import SymbolicRow, _place_blocks, _sample_distinct_tuples

GLYPH_TASKS: tuple[str, ...] = (
    "copy_span",
    "reverse",
    "every_k",
    "filter_mod",
    "dyck_close",
    "fact_markov",
    "story_fact",
    "chain_ordered_noise",
    "chain_shuffled_noise",
)

GLYPH_RETRIEVAL_TASKS: frozenset[str] = frozenset(GLYPH_TASKS)

WIDTH_16 = 16
WIDTH_32 = 32
NOISE_KINDS: tuple[str, ...] = ("markov", "dyck", "arith", "mixed", "iid")

# Schnabel et al. 2025 class, plus the Chomsky / MAD mechanism the rung actually stresses.
BAPO_CLASS: dict[str, str] = {
    "copy_span": "easy-index-bandwidth",
    "reverse": "easy-index-permute",
    "every_k": "easy-index-select",
    "filter_mod": "easy-selective-copy",
    "dyck_close": "easy-stack-bounded",
    "fact_markov": "easy-match2-structured",
    "story_fact": "easy-match2-words",
    "chain_ordered_noise": "easy-dfa",
    "chain_shuffled_noise": "hard-reachability",
}

# Width-16 layout (16 ids). Dyck-2 / words / hop / ops live only on 32.
_DIGITS_16 = ("d0", "d1", "d2", "d3")
_LETTERS_16 = ("a", "b", "c", "d")
_BRACKETS_16 = ("lparen", "rparen")
_ROLES_16 = ("bos", "eos", "query", "answer", "end", "mark")

# Width-32 layout (exactly 32 ids). Typed on purpose so a local LM can model each class.
# 8 digits + 8 letters + 4 brackets + 1 op + 4 words + 7 roles = 32.
_DIGITS_32 = ("d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7")
_LETTERS_32 = ("a", "b", "c", "d", "e", "f", "g", "h")
_BRACKETS_32 = ("lparen", "rparen", "lbrack", "rbrack")
_OPS_32 = ("plus",)
_WORDS_32 = ("cat", "dog", "red", "blue")
_ROLES_32 = ("bos", "eos", "query", "answer", "end", "mark", "hop")

_BRACKET_PAIRS_16: tuple[tuple[str, str], ...] = (("lparen", "rparen"),)
_BRACKET_PAIRS_32: tuple[tuple[str, str], ...] = (("lparen", "rparen"), ("lbrack", "rbrack"))


def _layout(width: int) -> tuple[str, ...]:
    if width == WIDTH_16:
        return _DIGITS_16 + _LETTERS_16 + _BRACKETS_16 + _ROLES_16
    if width == WIDTH_32:
        return _DIGITS_32 + _LETTERS_32 + _BRACKETS_32 + _OPS_32 + _WORDS_32 + _ROLES_32
    raise ValueError(f"width must be {WIDTH_16} or {WIDTH_32}, got {width}")


@dataclass(frozen=True)
class GlyphVocab:
    """Closed typed alphabet. Ids are 0..width-1; `sym_lo` reserved for a future mix-in."""

    width: int = WIDTH_32
    sym_lo: int = 0

    def __post_init__(self) -> None:
        if self.width not in (WIDTH_16, WIDTH_32):
            raise ValueError(f"width must be 16 or 32, got {self.width}")
        if self.sym_lo < 0:
            raise ValueError(f"sym_lo must be >= 0, got {self.sym_lo}")

    @property
    def names(self) -> tuple[str, ...]:
        return _layout(self.width)

    def id(self, name: str) -> int:
        try:
            return self.sym_lo + self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown glyph {name!r} at width {self.width}; have {self.names}") from exc

    def control(self, name: str) -> int:
        """DNA-compatible alias: role tokens are the controls."""
        return self.id(name)

    def ids(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.id(n) for n in names], dtype=np.int64)

    @property
    def digit_names(self) -> tuple[str, ...]:
        return _DIGITS_16 if self.width == WIDTH_16 else _DIGITS_32

    @property
    def letter_names(self) -> tuple[str, ...]:
        return _LETTERS_16 if self.width == WIDTH_16 else _LETTERS_32

    @property
    def word_names(self) -> tuple[str, ...]:
        return () if self.width == WIDTH_16 else _WORDS_32

    @property
    def opener_names(self) -> tuple[str, ...]:
        return tuple(p[0] for p in self.bracket_pairs)

    @property
    def closer_names(self) -> tuple[str, ...]:
        return tuple(p[1] for p in self.bracket_pairs)

    @property
    def bracket_pairs(self) -> tuple[tuple[str, str], ...]:
        return _BRACKET_PAIRS_16 if self.width == WIDTH_16 else _BRACKET_PAIRS_32

    @property
    def digit_ids(self) -> np.ndarray:
        return self.ids(self.digit_names)

    @property
    def letter_ids(self) -> np.ndarray:
        return self.ids(self.letter_names)

    @property
    def word_ids(self) -> np.ndarray:
        return self.ids(self.word_names) if self.word_names else np.array([], dtype=np.int64)

    @property
    def opener_ids(self) -> np.ndarray:
        return self.ids(self.opener_names)

    @property
    def closer_ids(self) -> np.ndarray:
        return self.ids(self.closer_names)

    @property
    def op_ids(self) -> np.ndarray:
        return self.ids(_OPS_32) if self.width == WIDTH_32 else np.array([], dtype=np.int64)

    @property
    def closer_of(self) -> dict[int, int]:
        return {self.id(o): self.id(c) for o, c in self.bracket_pairs}

    @property
    def entity_ids(self) -> np.ndarray:
        """cat / dog — story keys. Empty on width 16."""
        if self.width != WIDTH_32:
            return np.array([], dtype=np.int64)
        return self.ids(("cat", "dog"))

    @property
    def vocab_size(self) -> int:
        return self.sym_lo + self.width

    def digit_value(self, tok: int) -> int:
        lo = int(self.digit_ids[0])
        hi = lo + len(self.digit_ids)
        if not (lo <= int(tok) < hi):
            raise ValueError(f"token {tok} is not a digit in {self.digit_names}")
        return int(tok) - lo

    def digit_token(self, value: int) -> int:
        if not (0 <= value < len(self.digit_ids)):
            raise ValueError(f"digit value {value} out of range for {self.digit_names}")
        return int(self.digit_ids[value])


@dataclass(frozen=True)
class GlyphTaskConfig:
    """One Glyph rung. `min_gap` is the same contract as DNA: floor is exact iff
    the raw window is `<= min_gap`. `n_symbols` is the *answer-class* size (not the
    full typed vocab), so `data.symbolic_tasks.floor_nats` duck-types this config.
    """

    task: str = "reverse"
    seq_len: int = 128
    min_gap: int = 16
    width: int = WIDTH_32
    noise: str = "markov"
    span_len: int = 16
    key_len: int = 2
    value_len: int = 4
    hops: int = 2
    n_distractors: int = 1
    n_decoys: int = 0
    n_keep: int = 8
    k: int = 2
    modulus: int = 3
    family: str = "glyph"
    sym_lo: int = 0

    def __post_init__(self) -> None:
        if self.task not in GLYPH_TASKS:
            raise ValueError(f"task must be one of {GLYPH_TASKS}, got {self.task!r}")
        if self.noise not in NOISE_KINDS:
            raise ValueError(f"noise must be one of {NOISE_KINDS}, got {self.noise!r}")
        if self.min_gap < 1:
            raise ValueError("min_gap must be >= 1")
        if self.task in {"dyck_close", "story_fact", "chain_ordered_noise", "chain_shuffled_noise"}:
            if self.width != WIDTH_32:
                raise ValueError(f"{self.task} needs width 32 (typed hop/words/Dyck-2)")
        if self.task == "every_k":
            if self.k < 2:
                raise ValueError("every_k needs k >= 2")
            if self.span_len % self.k != 0:
                raise ValueError(f"span_len {self.span_len} must be divisible by k={self.k}")
            if self.k >= len(self.vocab.digit_ids):
                raise ValueError(f"k={self.k} must be a digit in the vocab")
        if self.task == "filter_mod":
            if self.modulus < 2:
                raise ValueError("modulus must be >= 2")
            if not (1 <= self.n_keep < self.span_len):
                raise ValueError("filter_mod needs 1 <= n_keep < span_len")
            keepers = self._keeper_values()
            if len(keepers) < 2:
                raise ValueError(
                    f"modulus {self.modulus} over {len(self.vocab.digit_ids)} digits has "
                    f"{len(keepers)} keeper class(es); need >= 2 so the floor is not degenerate"
                )
        if self.task in {"chain_ordered_noise", "chain_shuffled_noise"} and self.hops < 2:
            raise ValueError("chain needs hops >= 2")
        if self.task == "dyck_close" and len(self.vocab.bracket_pairs) < 2:
            raise ValueError("dyck_close needs Dyck-2")
        need = self.tail_len + self.evidence_len + 1
        if need + self.min_gap > self.seq_len:
            raise ValueError(
                f"seq_len {self.seq_len} too short for glyph {self.task!r}: needs "
                f"{need} tokens of structure + min_gap {self.min_gap}"
            )

    def _keeper_values(self) -> list[int]:
        n_d = len(self.vocab.digit_ids)
        return [v for v in range(n_d) if v % self.modulus == 0]

    @property
    def vocab(self) -> GlyphVocab:
        return GlyphVocab(width=self.width, sym_lo=self.sym_lo)

    @property
    def n_letters(self) -> int:
        return len(self.vocab.letter_ids)

    @property
    def n_digits(self) -> int:
        return len(self.vocab.digit_ids)

    @property
    def n_symbols(self) -> int:
        """Answer-class size. Duck-types DNA `floor_nats` / `chance_accuracy`."""
        if self.task in {"copy_span", "reverse", "every_k"}:
            return self.n_letters
        if self.task == "filter_mod":
            return len(self._keeper_values())
        if self.task == "dyck_close":
            return len(self.vocab.bracket_pairs)
        if self.task in {"fact_markov", "story_fact"}:
            return self.n_digits
        # chain terminals are letter tuples drawn iid, same as DNA
        return self.n_letters

    @property
    def answer_len(self) -> int:
        return {
            "copy_span": self.span_len,
            "reverse": self.span_len,
            "every_k": self.span_len // self.k,
            "filter_mod": self.n_keep,
            "dyck_close": self.span_len,
            "fact_markov": self.value_len,
            "story_fact": self.value_len,
            "chain_ordered_noise": self.key_len,
            "chain_shuffled_noise": self.key_len,
        }[self.task]

    @property
    def query_len(self) -> int:
        return {
            "copy_span": 0,
            "reverse": 0,
            "every_k": 1,          # the digit k, shown to the decoder
            "filter_mod": 1,       # the modulus, shown to the decoder
            "dyck_close": 0,
            "fact_markov": self.key_len,
            "story_fact": 1,        # the entity word
            "chain_ordered_noise": self.key_len,
            "chain_shuffled_noise": self.key_len,
        }[self.task]

    @property
    def tail_len(self) -> int:
        return 1 + self.query_len + 1 + self.answer_len + 1 + 1

    @property
    def evidence_len(self) -> int:
        if self.task in {"copy_span", "reverse", "every_k", "filter_mod", "dyck_close"}:
            return 1 + self.span_len
        if self.task == "fact_markov":
            kv = 1 + self.key_len + self.value_len
            return (self.n_distractors + 1) * kv
        if self.task == "story_fact":
            blk = 1 + 1 + self.value_len
            return (self.n_distractors + 1) * blk
        hop = 1 + 2 * self.key_len
        n_true = self.hops
        n_dist = self.n_distractors
        return (n_true + n_dist) * hop

    @property
    def bapo_class(self) -> str:
        return BAPO_CLASS[self.task]

    @property
    def answer_start(self) -> int:
        return self.seq_len - self.tail_len + 1 + self.query_len + 1

    @property
    def answer_alphabet(self) -> np.ndarray:
        """Ids the supervised tokens are drawn from (uniform)."""
        v = self.vocab
        if self.task in {"copy_span", "reverse", "every_k", "chain_ordered_noise", "chain_shuffled_noise"}:
            return v.letter_ids
        if self.task == "filter_mod":
            return np.array([v.digit_token(x) for x in self._keeper_values()], dtype=np.int64)
        if self.task == "dyck_close":
            return v.closer_ids
        return v.digit_ids


# --------------------------------------------------------------------------------------
# Structured noise
# --------------------------------------------------------------------------------------


def _sticky_markov(rng: np.random.Generator, n: int, alphabet: np.ndarray, stay: float = 0.55, step: float = 0.30) -> np.ndarray:
    """Known bigram process: stay / step-to-next-cyclic / jump. A local LM can fit this."""
    alphabet = np.asarray(alphabet, dtype=np.int64)
    m = int(len(alphabet))
    if n <= 0:
        return np.array([], dtype=np.int64)
    if m < 2:
        raise ValueError("Markov alphabet needs >= 2 symbols")
    out = np.empty(n, dtype=np.int64)
    idx = int(rng.integers(0, m))
    out[0] = alphabet[idx]
    for i in range(1, n):
        u = float(rng.random())
        if u < stay:
            pass
        elif u < stay + step:
            idx = (idx + 1) % m
        else:
            idx = int(rng.integers(0, m))
        out[i] = alphabet[idx]
    return out


def _dyck_seq(
    rng: np.random.Generator, n: int, openers: np.ndarray, closers: np.ndarray, pad: int
) -> np.ndarray:
    """A valid Dyck string of even length `n` or `n-1` (odd n pads with `pad`)."""
    out = np.empty(n, dtype=np.int64)
    n_pair = n if n % 2 == 0 else n - 1
    n_types = len(openers)
    stack: list[int] = []
    pos = 0
    while pos < n_pair:
        depth = len(stack)
        left = n_pair - pos
        can_close = depth > 0
        can_open = (left - 1) >= (depth + 1)
        if depth == 0 or not can_close:
            do_open = True
        elif not can_open:
            do_open = False
        else:
            do_open = bool(rng.random() < 0.5)
        if do_open:
            t = int(rng.integers(0, n_types))
            stack.append(t)
            out[pos] = int(openers[t])
        else:
            t = stack.pop()
            out[pos] = int(closers[t])
        pos += 1
    if stack:
        raise RuntimeError("Dyck generator left a non-empty stack")
    if n % 2:
        out[-1] = int(pad)
    return out


def _arith_seq(rng: np.random.Generator, n: int, digits: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Well-formed `d (op d)*`. Never ends on an operator. Values are irrelevant."""
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        out[i] = int(digits[int(rng.integers(0, len(digits)))])
        i += 1
        if i >= n:
            break
        if i == n - 1:
            out[i] = int(digits[int(rng.integers(0, len(digits)))])
            break
        out[i] = int(ops[int(rng.integers(0, len(ops)))])
        i += 1
    return out


def _noise_alphabet(cfg: GlyphTaskConfig) -> np.ndarray:
    """Tokens a local model is allowed to predict in the haystack (never controls)."""
    v = cfg.vocab
    if cfg.width == WIDTH_32:
        return np.concatenate([v.letter_ids, v.word_ids])
    return v.letter_ids


def fill_noise_run(ids: np.ndarray, lo: int, hi: int, cfg: GlyphTaskConfig, rng: np.random.Generator) -> str:
    """Fill `[lo, hi)` with one structured process. Returns the kind actually used."""
    n = hi - lo
    if n <= 0:
        return cfg.noise
    v = cfg.vocab
    kind = cfg.noise
    if kind == "mixed":
        choices = ["markov", "dyck"]
        if cfg.width == WIDTH_32:
            choices.append("arith")
        kind = str(rng.choice(choices))
    if kind == "arith" and (cfg.width != WIDTH_32 or len(v.op_ids) == 0):
        kind = "markov"
    pad = int(v.letter_ids[0])
    if kind == "iid":
        alphabet = _noise_alphabet(cfg)
        ids[lo:hi] = alphabet[rng.integers(0, len(alphabet), size=n)]
    elif kind == "markov":
        ids[lo:hi] = _sticky_markov(rng, n, _noise_alphabet(cfg))
    elif kind == "dyck":
        ids[lo:hi] = _dyck_seq(rng, n, v.opener_ids, v.closer_ids, pad)
    elif kind == "arith":
        ids[lo:hi] = _arith_seq(rng, n, v.digit_ids, v.op_ids)
    else:
        raise ValueError(f"unknown noise {kind}")
    return kind


def _fill_holes(ids: np.ndarray, cfg: GlyphTaskConfig, rng: np.random.Generator) -> list[tuple[int, int, str]]:
    """Structured-noise each contiguous run of unset (-1) body tokens."""
    kinds: list[tuple[int, int, str]] = []
    i = 0
    S = len(ids)
    while i < S:
        if ids[i] >= 0:
            i += 1
            continue
        j = i
        while j < S and ids[j] < 0:
            j += 1
        kind = fill_noise_run(ids, i, j, cfg, rng)
        kinds.append((i, j, kind))
        i = j
    return kinds


def _is_valid_dyck(seq: Sequence[int], closer_of: dict[int, int]) -> bool:
    openers = set(closer_of)
    closers = set(closer_of.values())
    stack: list[int] = []
    for t in seq:
        t = int(t)
        if t in openers:
            stack.append(t)
        elif t in closers:
            if not stack or closer_of[stack[-1]] != t:
                return False
            stack.pop()
        else:
            # pad letter at odd length: allowed only as a trailing pad, treated as break
            if stack:
                return False
    return not stack


def _is_valid_arith(seq: Sequence[int], digits: set[int], ops: set[int]) -> bool:
    """Well-formed filler: digits and ops, never starts/ends on an op, never two ops in a row."""
    seq = [int(x) for x in seq]
    if len(seq) == 0:
        return True
    if seq[0] not in digits or seq[-1] not in digits:
        return False
    prev_op = False
    for t in seq:
        if t in digits:
            prev_op = False
            continue
        if t in ops:
            if prev_op:
                return False
            prev_op = True
            continue
        return False
    return True


# --------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------


def _as_letter_ids(v: GlyphVocab, t: Sequence[int]) -> list[int]:
    letters = v.letter_ids
    return [int(letters[int(x)]) for x in t]


def generate_row(cfg: GlyphTaskConfig, rng: np.random.Generator) -> SymbolicRow:
    """One row of exactly `cfg.seq_len` tokens. Answer determined by far evidence."""
    v = cfg.vocab
    S = cfg.seq_len
    ids = np.full(S, -1, dtype=np.int64)
    ids[0] = v.control("bos")

    tail_start = S - cfg.tail_len
    answer_start = cfg.answer_start
    evidence_hi = answer_start - cfg.min_gap

    meta: dict = {"family": "glyph", "noise": cfg.noise, "width": cfg.width}
    query_tokens: list[int] = []
    answer_tokens: list[int] = []
    evidence_end = 0

    if cfg.task in {"copy_span", "reverse", "every_k", "filter_mod", "dyck_close"}:
        block, query_tokens, answer_tokens = _span_block(cfg, rng)
        (off,) = _place_blocks(rng, [block], 1, evidence_hi)
        ids[off : off + len(block)] = block
        evidence_end = off + len(block) - 1
        meta["span_off"] = int(off)

    elif cfg.task == "fact_markov":
        n_items = cfg.n_distractors + 1
        keys = _sample_distinct_tuples(rng, n_items, cfg.key_len, cfg.n_letters)
        values = [tuple(int(x) for x in rng.integers(0, cfg.n_digits, size=cfg.value_len)) for _ in range(n_items)]
        blocks = [
            [v.control("mark"), *_as_letter_ids(v, k), *[v.digit_token(x) for x in val]]
            for k, val in zip(keys, values)
        ]
        offsets = _place_blocks(rng, blocks, 1, evidence_hi)
        for off, block in zip(offsets, blocks):
            ids[off : off + len(block)] = block
        pick = int(rng.integers(0, n_items))
        query_tokens = _as_letter_ids(v, keys[pick])
        answer_tokens = [v.digit_token(x) for x in values[pick]]
        evidence_end = offsets[pick] + len(blocks[pick]) - 1
        meta.update({"n_items": n_items, "picked": pick})

    elif cfg.task == "story_fact":
        entities = [int(x) for x in v.entity_ids]
        n_items = cfg.n_distractors + 1
        if n_items > len(entities):
            raise ValueError(f"story_fact n_distractors+1={n_items} > n_entities {len(entities)}")
        used = [int(x) for x in rng.permutation(entities)[:n_items]]
        values = [tuple(int(x) for x in rng.integers(0, cfg.n_digits, size=cfg.value_len)) for _ in range(n_items)]
        blocks = [[v.control("mark"), used[i], *[v.digit_token(x) for x in values[i]]] for i in range(n_items)]
        offsets = _place_blocks(rng, blocks, 1, evidence_hi)
        for off, block in zip(offsets, blocks):
            ids[off : off + len(block)] = block
        pick = int(rng.integers(0, n_items))
        query_tokens = [used[pick]]
        answer_tokens = [v.digit_token(x) for x in values[pick]]
        evidence_end = offsets[pick] + len(blocks[pick]) - 1
        meta.update({"n_items": n_items, "picked": pick, "entity": int(used[pick])})

    else:  # chain_*
        query_tokens, answer_tokens, evidence_end, chain_meta = _emit_glyph_chain(
            cfg, rng, ids, evidence_hi, shuffle=cfg.task == "chain_shuffled_noise"
        )
        meta.update(chain_meta)

    tail = [
        v.control("query"),
        *query_tokens,
        v.control("answer"),
        *answer_tokens,
        v.control("end"),
        v.control("eos"),
    ]
    assert len(tail) == cfg.tail_len, (len(tail), cfg.tail_len, cfg.task)
    ids[tail_start:] = tail
    assert ids[answer_start] == answer_tokens[0]

    hole_kinds = _fill_holes(ids, cfg, rng)
    meta["noise_runs"] = [(int(a), int(b), k) for a, b, k in hole_kinds]
    assert int(ids.min()) >= 0, "unfilled holes"
    assert int(ids.max()) < v.vocab_size

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


def _span_block(cfg: GlyphTaskConfig, rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    v = cfg.vocab
    mark = v.control("mark")
    if cfg.task in {"copy_span", "reverse", "every_k"}:
        span = [int(x) for x in rng.choice(v.letter_ids, size=cfg.span_len)]
        block = [mark, *span]
        if cfg.task == "copy_span":
            return block, [], list(span)
        if cfg.task == "reverse":
            return block, [], list(reversed(span))
        # every_k: 0-indexed every k-th; k is shown as a digit in the query
        answer = list(span[0 :: cfg.k])
        assert len(answer) == cfg.answer_len
        return block, [v.digit_token(cfg.k)], answer

    if cfg.task == "filter_mod":
        keepers = cfg._keeper_values()
        others = [x for x in range(cfg.n_digits) if x not in keepers]
        keep_pos = rng.choice(cfg.span_len, size=cfg.n_keep, replace=False)
        mask = np.zeros(cfg.span_len, dtype=bool)
        mask[keep_pos] = True
        span_vals = np.empty(cfg.span_len, dtype=np.int64)
        span_vals[mask] = rng.choice(keepers, size=cfg.n_keep)
        span_vals[~mask] = rng.choice(others, size=cfg.span_len - cfg.n_keep)
        span = [v.digit_token(int(x)) for x in span_vals]
        answer = [v.digit_token(int(x)) for x in span_vals[np.sort(keep_pos)]]
        # keep in *order of appearance*, not shuffled keep_pos
        answer = [tok for tok in span if v.digit_value(tok) % cfg.modulus == 0]
        assert len(answer) == cfg.n_keep
        return [mark, *span], [v.digit_token(cfg.modulus)], answer

    # dyck_close: span_len unmatched openers (a valid Dyck prefix). Unique closers = reverse match.
    openers = list(int(x) for x in v.opener_ids)
    unmatched = [int(openers[int(rng.integers(0, len(openers)))]) for _ in range(cfg.span_len)]
    closer_of = v.closer_of
    answer = [closer_of[t] for t in reversed(unmatched)]
    return [mark, *unmatched], [], answer


def _emit_glyph_chain(
    cfg: GlyphTaskConfig,
    rng: np.random.Generator,
    ids: np.ndarray,
    evidence_hi: int,
    *,
    shuffle: bool,
) -> tuple[list[int], list[int], int, dict]:
    """Letter-tuple hops. Distractors prefer a *second independent chain* when they fit."""
    v = cfg.vocab
    A = cfg.n_letters
    hop_id = v.control("hop")
    n_true = cfg.hops
    n_dist = cfg.n_distractors
    sources = _sample_distinct_tuples(rng, n_true + n_dist, cfg.key_len, A)
    terminal = tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len))
    chain = [*sources[:n_true], terminal]
    chain_edges = [(chain[i], chain[i + 1]) for i in range(n_true)]

    dist_edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    second_chain = False
    if n_dist >= n_true:
        # Second independent chain occupying n_true of the distractor budget.
        term2 = tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len))
        chain2 = [*sources[n_true : n_true + n_true], term2]
        dist_edges.extend((chain2[i], chain2[i + 1]) for i in range(n_true))
        extra = sources[n_true + n_true :]
        dist_edges.extend(
            (src, tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len))) for src in extra
        )
        second_chain = True
    else:
        dist_edges = [
            (src, tuple(int(x) for x in rng.integers(0, A, size=cfg.key_len)))
            for src in sources[n_true:]
        ]

    def blk(src: Sequence[int], dst: Sequence[int]) -> list[int]:
        return [hop_id, *_as_letter_ids(v, src), *_as_letter_ids(v, dst)]

    if shuffle:
        edges = chain_edges + dist_edges
        order = rng.permutation(len(edges))
        blocks = [blk(*edges[int(i)]) for i in order]
        offsets = _place_blocks(rng, blocks, 1, evidence_hi)
        chain_positions = [
            offsets[j] + len(blocks[j]) - 1 for j, i in enumerate(order) if int(i) < n_true
        ]
        for off, block in zip(offsets, blocks):
            ids[off : off + len(block)] = block
    else:
        chain_blocks = [blk(s, d) for s, d in chain_edges]
        dist_blocks = [blk(s, d) for s, d in dist_edges]
        chain_need = sum(len(b) for b in chain_blocks)
        dist_need = sum(len(b) for b in dist_blocks)
        slack = (evidence_hi - 1) - chain_need - dist_need
        slack_chain = max(0, slack // 2)
        chain_hi = 1 + chain_need + slack_chain
        chain_off = _place_blocks(rng, chain_blocks, 1, chain_hi)
        dist_off = _place_blocks(rng, dist_blocks, chain_hi, evidence_hi) if dist_blocks else []
        for off, block in zip(chain_off, chain_blocks):
            ids[off : off + len(block)] = block
        for off, block in zip(dist_off, dist_blocks):
            ids[off : off + len(block)] = block
        chain_positions = [off + len(b) - 1 for off, b in zip(chain_off, chain_blocks)]

    return (
        _as_letter_ids(v, chain[0]),
        _as_letter_ids(v, chain[-1]),
        max(chain_positions),
        {"hops": cfg.hops, "shuffled": shuffle, "second_chain": second_chain},
    )


def iter_rows(cfg: GlyphTaskConfig, n_rows: int, seed: int) -> Iterator[SymbolicRow]:
    rng = np.random.default_rng(seed)
    for _ in range(n_rows):
        yield generate_row(cfg, rng)


# --------------------------------------------------------------------------------------
# Verifier = the proof the task is solvable
# --------------------------------------------------------------------------------------


def expected_answer(cfg: GlyphTaskConfig, ids: np.ndarray) -> list[int]:
    """Reconstruct the answer from the prefix alone. This *is* the solvability proof."""
    v = cfg.vocab
    mark = v.control("mark")
    answer_start = cfg.answer_start
    body = ids[:answer_start]

    if cfg.task in {"copy_span", "reverse", "every_k", "filter_mod", "dyck_close"}:
        hits = np.flatnonzero(body == mark)
        if len(hits) != 1:
            raise AssertionError(f"{cfg.task}: expected one mark, found {len(hits)}")
        i = int(hits[0])
        span = [int(x) for x in ids[i + 1 : i + 1 + cfg.span_len]]
        if cfg.task == "copy_span":
            return span
        if cfg.task == "reverse":
            return list(reversed(span))
        if cfg.task == "every_k":
            return span[0 :: cfg.k]
        if cfg.task == "filter_mod":
            return [t for t in span if v.digit_value(t) % cfg.modulus == 0]
        closer_of = v.closer_of
        stack: list[int] = []
        for t in span:
            if t in closer_of:
                stack.append(int(t))
            elif t in set(closer_of.values()):
                if not stack or closer_of[stack[-1]] != t:
                    raise AssertionError("dyck prefix is illegal")
                stack.pop()
            else:
                raise AssertionError(f"non-bracket {t} in dyck span")
        return [closer_of[t] for t in reversed(stack)]

    if cfg.task == "fact_markov":
        q = [int(x) for x in ids[answer_start - 1 - cfg.key_len : answer_start - 1]]
        found: list[list[int]] = []
        for i in np.flatnonzero(body == mark):
            k = [int(x) for x in ids[i + 1 : i + 1 + cfg.key_len]]
            if k == q:
                found.append([int(x) for x in ids[i + 1 + cfg.key_len : i + 1 + cfg.key_len + cfg.value_len]])
        if len(found) != 1:
            raise AssertionError(f"fact_markov: query matched {len(found)} facts")
        return found[0]

    if cfg.task == "story_fact":
        q = int(ids[answer_start - 2])
        found = []
        for i in np.flatnonzero(body == mark):
            if int(ids[i + 1]) == q:
                found.append([int(x) for x in ids[i + 2 : i + 2 + cfg.value_len]])
        if len(found) != 1:
            raise AssertionError(f"story_fact: entity matched {len(found)} facts")
        return found[0]

    hop = v.control("hop")
    k = cfg.key_len
    edges: dict[tuple[int, ...], tuple[int, ...]] = {}
    for i in np.flatnonzero(body == hop):
        src = tuple(int(x) for x in ids[i + 1 : i + 1 + k])
        dst = tuple(int(x) for x in ids[i + 1 + k : i + 1 + 2 * k])
        if src in edges:
            raise AssertionError("duplicate hop source")
        edges[src] = dst
    node = tuple(int(x) for x in ids[answer_start - 1 - k : answer_start - 1])
    for _ in range(cfg.hops):
        if node not in edges:
            raise AssertionError("chain broke")
        node = edges[node]
    return list(node)


def verify_row(cfg: GlyphTaskConfig, row: SymbolicRow) -> bool:
    got = [int(x) for x in row.input_ids[row.answer_start : row.answer_start + cfg.answer_len]]
    return expected_answer(cfg, row.input_ids) == got


# --------------------------------------------------------------------------------------
# Floors — same contract as DNA, answer-class size as n_symbols
# --------------------------------------------------------------------------------------


def floor_nats(cfg: GlyphTaskConfig, window: int) -> float:
    """Identical contract to DNA: ln(|answer class|) when the window cannot see evidence."""
    from data.symbolic_tasks import floor_nats as dna_floor

    return dna_floor(cfg, window)  # duck-typed: retrieval + cfg.n_symbols


def chance_accuracy(cfg: GlyphTaskConfig) -> float:
    return 1.0 / cfg.n_symbols


def chance_entropy_nats(cfg: GlyphTaskConfig) -> float:
    return math.log(cfg.n_symbols)


def prize_nats(cfg: GlyphTaskConfig) -> float:
    return cfg.answer_len * chance_entropy_nats(cfg)


def prize_bits(cfg: GlyphTaskConfig) -> float:
    return prize_nats(cfg) / math.log(2)


__all__ = [
    "BAPO_CLASS",
    "GLYPH_RETRIEVAL_TASKS",
    "GLYPH_TASKS",
    "GlyphTaskConfig",
    "GlyphVocab",
    "NOISE_KINDS",
    "WIDTH_16",
    "WIDTH_32",
    "chance_accuracy",
    "chance_entropy_nats",
    "expected_answer",
    "fill_noise_run",
    "floor_nats",
    "generate_row",
    "iter_rows",
    "prize_bits",
    "prize_nats",
    "verify_row",
]
