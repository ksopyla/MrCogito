"""Shared schema, length ladder, and recipes for the CogitoProbe series.

Four families, one claim each. Rows are already tokenized over a *verified* atom
table of the E18/E22 tokenizer (SmolLM3 = Llama-3 vocab). ``input_ids`` are composed
from those atoms — they are never produced by re-tokenizing a glued surface string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FAMILIES: tuple[str, ...] = ("bits", "bind", "arith", "props")
SPLITS: tuple[str, ...] = ("train", "validation", "test")
LENGTH_LADDER: tuple[int, ...] = (1024, 4096, 8192, 16384, 32768)
VARIANTS: tuple[str, ...] = ("scaled", "fixed")  # density-matched vs prize-matched

HUB_NAMESPACE = "ksopyla"
HUB_IDS: dict[str, str] = {
    "bits": "ksopyla/cogito-probe-bits",
    "bind": "ksopyla/cogito-probe-bind",
    "arith": "ksopyla/cogito-probe-arith",
    "props": "ksopyla/cogito-probe-props",
}
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_SEED = 20260916
IGNORE_INDEX = -100

# Pilot = small-but-real public sample. Full = documented production recipe.
PILOT_COUNTS: dict[int, tuple[int, int, int]] = {
    # seq_len: (train, val, test)
    1024: (128, 32, 32),
    4096: (64, 16, 16),
    8192: (48, 16, 16),
    16384: (32, 12, 12),
    32768: (24, 8, 8),
}
FULL_COUNTS: dict[int, tuple[int, int, int]] = {
    1024: (4096, 256, 256),
    4096: (2048, 256, 256),
    8192: (1024, 128, 128),
    16384: (768, 128, 128),
    32768: (512, 128, 128),
}

FAMILY_CLAIMS: dict[str, str] = {
    "bits": (
        "A fixed latent set of C slots recovers at most ~C·k bits of unique prefix "
        "facts; accuracy falls once n_query·log2(|V|) exceeds that budget. Length "
        "at matched prize (variant=fixed) isolates haystack distance from capacity."
    ),
    "bind": (
        "Concepts bind (entity, attribute, value) tuples. A bag-of-tokens or single "
        "document embedding cannot answer who-has-X or one-hop friend queries when "
        "entities share the same attribute vocabulary."
    ),
    "arith": (
        "The bottleneck stores a compositional AST (or a sufficient set of node "
        "values + Dyck-3 match state), not a bag of digits. Eval-only is a shortcut "
        "control (~log2|result| bits) and must not be the success metric."
    ),
    "props": (
        "A fixed latent set carries the atomic propositions of a document, not the "
        "n-gram statistics of fluent filler. Shuffling filler must not change "
        "answers; shuffling propositions must."
    ),
}

# Arith: keep every node value in [-MAX_ABS, MAX_ABS] so answers are short digit strings.
ARITH_MAX_ABS = 99
ARITH_ATOMS: tuple[str, ...] = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*",
    "(", ")", "[", "]", "{", "}",
)


@dataclass(frozen=True)
class ProbeRecipe:
    """Per-family knobs that scale with seq_len. ``fixed`` holds the 1k counts at every length."""

    n_items: int          # facts / entities / expressions / propositions
    n_query: int          # packed supervised answers
    n_filler_target: int  # how much of seq_len should be non-evidence (approx)
    extra: dict[str, Any] = field(default_factory=dict)


def rung_name(seq_len: int) -> str:
    return f"seq{seq_len}"


def split_counts(scale: str, seq_len: int) -> tuple[int, int, int]:
    table = PILOT_COUNTS if scale == "pilot" else FULL_COUNTS
    if seq_len not in table:
        raise ValueError(f"seq_len {seq_len} not in ladder {LENGTH_LADDER}")
    return table[seq_len]


def expected_split_totals(
    scale: str,
    n_variants: int = 2,
    lengths: tuple[int, ...] | list[int] | None = None,
) -> dict[str, int]:
    """Row counts after the builder splits each rung budget across variants."""
    n_var = max(n_variants, 1)
    train = validation = test = 0
    for seq_len in (lengths if lengths is not None else LENGTH_LADDER):
        n_train, n_val, n_test = split_counts(scale, seq_len)
        train += max(n_train // n_var, 1) * n_var
        validation += max(n_val // n_var, 1) * n_var
        test += max(n_test // n_var, 1) * n_var
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "n_rows": train + validation + test,
    }


def recipe_for(family: str, seq_len: int, variant: str) -> ProbeRecipe:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if seq_len < 64:
        raise ValueError(f"seq_len {seq_len} is too short")
    # Unit tests and custom rungs reuse 1k knobs; official ladder uses the table.
    knob_len = seq_len if seq_len in LENGTH_LADDER else 1024

    scaled = knob_len != 1024 and variant == "scaled"
    # 1k baseline item counts. ``scaled`` multiplies with length; ``fixed`` does not.
    base_items = {"bits": 16, "bind": 8, "arith": 4, "props": 8}[family]
    base_query = {"bits": 8, "bind": 8, "arith": 4, "props": 6}[family]
    mult = {1024: 1, 4096: 2, 8192: 4, 16384: 8, 32768: 16}[knob_len] if scaled else 1
    n_items = base_items * mult
    n_query = min(base_query * (mult if family != "arith" else max(mult // 2, 1)), n_items)
    if family == "arith":
        n_query = min(4 if variant == "fixed" else min(8, 2 * mult), 8)
    extra: dict[str, Any] = {}
    if family == "bits":
        extra["redundancy"] = 1
        extra["n_values"] = 32
        extra["n_keys"] = max(64, n_items * 2)
    elif family == "bind":
        extra["n_colors"] = 8
        extra["n_places"] = 8
        extra["n_jobs"] = 8
        extra["n_hops"] = max(2, n_query // 4)
    elif family == "arith":
        extra["max_depth"] = 3
        extra["star_prob"] = 0.2
        extra["n_subexpr"] = max(2, n_query // 2)
        extra["n_match"] = 1
        extra["include_eval"] = True  # shortcut control, not the success metric
    elif family == "props":
        extra["n_colors"] = 8
        extra["n_places"] = 8
        extra["n_objects"] = 12
        extra["n_agents"] = max(8, n_items)
    return ProbeRecipe(
        n_items=n_items,
        n_query=n_query,
        n_filler_target=max(seq_len // 4, 16),
        extra=extra,
    )


@dataclass
class ProbeRow:
    id: str
    family: str
    task: str
    variant: str
    seq_len: int
    split: str
    seed: int
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    text: str
    context: str
    query: str
    answer: str
    n_tokens: int
    prize_bits: float
    gap: int
    answer_start: int
    answer_end: int
    evidence_end: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "family": self.family,
            "task": self.task,
            "variant": self.variant,
            "seq_len": self.seq_len,
            "rung": rung_name(self.seq_len),
            "split": self.split,
            "seed": self.seed,
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
            "text": self.text,
            "context": self.context,
            "query": self.query,
            "answer": self.answer,
            "n_tokens": self.n_tokens,
            "prize_bits": float(self.prize_bits),
            "gap": int(self.gap),
            "answer_start": int(self.answer_start),
            "answer_end": int(self.answer_end),
            "evidence_end": int(self.evidence_end),
            "meta": json.dumps(self.meta, sort_keys=True),
        }


LM_COLUMNS = ("input_ids", "attention_mask", "labels")
HF_COLUMNS = (
    "id",
    "family",
    "task",
    "variant",
    "seq_len",
    "rung",
    "split",
    "seed",
    "input_ids",
    "labels",
    "attention_mask",
    "text",
    "context",
    "query",
    "answer",
    "n_tokens",
    "prize_bits",
    "gap",
    "answer_start",
    "answer_end",
    "evidence_end",
    "meta",
)

FamilyName = Literal["bits", "bind", "arith", "props"]
