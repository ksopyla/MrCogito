"""Named BAPO rungs over `data/symbolic_tasks.py` (DNA) and `data/glyph_tasks.py`.

DNA A=4 + iid filler is the exact-floor bandwidth instrument. Glyph is the second
family (typed vocab 16/32, structured noise). A rung is a (scale, task) pair with a
frozen config and the architecture-facing contract: the dense decoder-only control
must reach `SOLVABLE_ACC` before any compressed-channel number is interpreted. Scales:

  tiny         128 tokens        CPU, minutes; the solvability proof lives here
  tiny_wide    256 tokens        CPU bridge
  bridge       512 tokens        first GPU INDEX scale (right-align; window < gap)
  bridge_1k    1024 tokens        GPU INDEX scale-up
  medium       4k / 16k         GPU, <100M; advertised 4k spread is K1 until dense ≥ 75%
  large        32k / 128k       GPU, still <100M params

The other agent's tiny Arm-A 100% map (`symbolic_arm_a_100pct_limits_20260913.md`) owns the
`perceiver_concept` far_copy exam at seq=128. This ladder is the *scale-up* of that DNA
framework onto E18 vs matched dense transformers, plus a Glyph family the DNA suite
does not cover. Do not score E18 on Glyph until dense ≥ 75%.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.glyph_tasks import (
    BAPO_CLASS as GLYPH_BAPO_CLASS,
    GLYPH_TASKS,
    GlyphTaskConfig,
    chance_accuracy as glyph_chance_accuracy,
    floor_nats as glyph_floor_nats,
    generate_row as generate_glyph_row,
    prize_bits as glyph_prize_bits,
)
from data.symbolic_tasks import (
    AGGREGATION_TASKS,
    BAPO_CLASS,
    RETRIEVAL_TASKS,
    TASKS,
    SymbolicTaskConfig,
    chance_accuracy,
    floor_nats,
    generate_row as generate_dna_row,
    prize_bits,
)


SOLVABLE_ACC = 0.75  # dense control must clear this before a rung is interpretable
USER_CORE_TASKS: tuple[str, ...] = (
    "far_copy",       # far copy / bandwidth
    "recall",         # recalling facts from far past
    "select",         # important facts vs noise
    "chain_ordered",  # A→B→C→D in order
    "chain",          # same hops, shuffled
)
HARD_TASKS: tuple[str, ...] = ("unique", "match3", "count", "majority")
TINY_PROOF_TASKS: tuple[str, ...] = USER_CORE_TASKS  # what the CPU calibration always runs


@dataclass(frozen=True)
class Recipe:
    """A named (generator task + overrides) pair. Display name may differ from `task`."""

    name: str
    task: str
    overrides: dict
    note: str = ""
    family: str = "dna"


# Rungs whose dense control has been shown to hit SOLVABLE_ACC at tiny H=128 / packed
# answers. Use these for E18 scoring and for the first GPU scale-up. Default generator
# configs (2–3 MATCH2 items, shuffled chain) stay as harder hunts — do not score E18 there
# until a new S0 hunt passes.
CALIBRATED_RECIPES: dict[str, Recipe] = {
    "far_copy": Recipe("far_copy", "far_copy", {}, "INDEX / positional copy"),
    "recall_single": Recipe(
        "recall_single",
        "recall",
        {"n_distractors": 0},
        "one planted key; content addressing without a MATCH2 contrast set",
    ),
    "select_1decoy": Recipe(
        "select_1decoy",
        "select",
        {"n_distractors": 0, "n_decoys": 1},
        "keymark vs one decoy — a type cue, not MATCH2",
    ),
    "chain_ordered": Recipe("chain_ordered", "chain_ordered", {}, "in-order DFA hops"),
}

# Explicitly uncalibrated at tiny H=128 / ≤4000 steps. Kept as named hunts so we do not
# silently score E18 on an ill-posed rung.
UNCALIBRATED_AT_TINY: dict[str, Recipe] = {
    "recall": Recipe("recall", "recall", {}, "default MATCH2 (2 distractors); dense ~30% @3200"),
    "select": Recipe("select", "select", {}, "multi-item + decoys; dense ~39% @3200"),
    "chain": Recipe("chain", "chain", {}, "shuffled REACHABILITY; dense ~34% @3200"),
    "chain_shuffled": Recipe(
        "chain_shuffled",
        "chain",
        {"n_distractors": 0},
        "no distractors; dense still ~34% @4000 — composition wall, not an E18 kill",
    ),
}

# E30 limit exams (2026-09-22, `e30_limits` / `e30_broad` on Odra+Polonez, 31M).
# These were ad-hoc flag combos; registering them freezes the configs so the next
# run reproduces the same exam instead of a near-miss. All are hunts until a dense
# S0 ≥ 75% exists at the same (scale, recipe) — see the small-model protocol.
E30_LIMIT_RECIPES: dict[str, Recipe] = {
    "lookup_1key": Recipe(
        "lookup_1key",
        "recall",
        {"n_distractors": 0, "key_len": 8, "value_len": 8},
        "E30 single lookup: 1 planted key, long key+value (64-bit prize "
        "at bridge_1k pack 32). Dense 0 bits @1024 — exam kill until S0.",
    ),
    "lookalike": Recipe(
        "lookalike",
        "select",
        {"n_distractors": 0, "n_decoys": 3, "key_len": 8, "value_len": 8},
        "E30 lookalike: keymark fact + 3 lookalike decoys (64-bit prize at "
        "bridge_1k pack 32). Full read ~99%; notebooks partial.",
    ),
    "chain_4hop": Recipe(
        "chain_4hop",
        "chain_ordered",
        {"hops": 4, "n_distractors": 0, "key_len": 8},
        "E30 in-order 4-hop chain (64-bit prize at bridge_1k pack 32). "
        "Passes for e30 @1024 (40 bits/77%); shuffled variant is unsolved "
        "by dense — do not score shuffled here.",
    ),
}

# Plot / CSV display order. Recipe names first, then generator names, then BAPO-hard extras,
# then the Glyph family (typed vocab, structured noise).
TASK_DISPLAY_ORDER: tuple[str, ...] = (
    "far_copy",
    "recall_single",
    "recall",
    "select_1decoy",
    "select",
    "chain_ordered",
    "chain_4hop",
    "lookup_1key",
    "lookalike",
    "chain_shuffled",
    "chain",
    "unique",
    "match3",
    "count",
    "majority",
    "copy_span",
    "reverse",
    "every_k",
    "filter_mod",
    "dyck_close",
    "fact_markov_single",
    "fact_markov",
    "story_fact",
    "chain_ordered_noise",
    "chain_shuffled_noise",
)

# Glyph (typed 16/32, structured noise). Uncalibrated until a dense S0 hits SOLVABLE_ACC.
# DNA remains the exact-floor control; these rungs are the second family, not a replacement.
GLYPH_RECIPES: dict[str, Recipe] = {
    "copy_span": Recipe(
        "copy_span", "copy_span", {"noise": "markov"},
        "INDEX / positional copy in a Markov haystack", family="glyph",
    ),
    "reverse": Recipe(
        "reverse", "reverse", {"noise": "markov"},
        "Delétang reverse / Olsson reverse; positional + permutation", family="glyph",
    ),
    "every_k": Recipe(
        "every_k", "every_k", {"noise": "markov", "k": 2},
        "selective indexing: keep indices 0, k, 2k, …; k is shown in the query", family="glyph",
    ),
    "filter_mod": Recipe(
        "filter_mod", "filter_mod", {"noise": "markov", "modulus": 3},
        "MAD selective copy: digits ≡ 0 (mod m) in order; m shown in the query", family="glyph",
    ),
    "dyck_close": Recipe(
        "dyck_close", "dyck_close", {"noise": "markov", "width": 32},
        "Dyck-2 unmatched stack → unique closers (width 32)", family="glyph",
    ),
    "fact_markov_single": Recipe(
        "fact_markov_single", "fact_markov", {"n_distractors": 0, "noise": "markov"},
        "one planted key in Markov filler — content addressing without MATCH2", family="glyph",
    ),
    "fact_markov": Recipe(
        "fact_markov", "fact_markov", {"noise": "markov"},
        "MATCH2 in Markov filler (BABILong Adapt)", family="glyph",
    ),
    "story_fact": Recipe(
        "story_fact", "story_fact", {"n_distractors": 0, "noise": "markov", "width": 32},
        "TinyStories Adapt: closed-word key, packed digit payload", family="glyph",
    ),
    "chain_ordered_noise": Recipe(
        "chain_ordered_noise", "chain_ordered_noise", {"noise": "markov", "width": 32},
        "in-order DFA hops in structured filler", family="glyph",
    ),
    "chain_shuffled_noise": Recipe(
        "chain_shuffled_noise", "chain_shuffled_noise",
        {"n_distractors": 0, "noise": "markov", "width": 32},
        "shuffled hops; hunt, not a gate, until dense ≥ 75%", family="glyph",
    ),
}

GLYPH_CORE_RECIPES: tuple[str, ...] = (
    "copy_span",
    "reverse",
    "every_k",
    "filter_mod",
    "dyck_close",
    "fact_markov_single",
)


def resolve_recipe(name: str) -> Recipe:
    if name in CALIBRATED_RECIPES:
        return CALIBRATED_RECIPES[name]
    if name in UNCALIBRATED_AT_TINY:
        return UNCALIBRATED_AT_TINY[name]
    if name in E30_LIMIT_RECIPES:
        return E30_LIMIT_RECIPES[name]
    if name in GLYPH_RECIPES:
        return GLYPH_RECIPES[name]
    if name in TASKS:
        return Recipe(name, name, {}, "")
    if name in GLYPH_TASKS:
        return Recipe(name, name, {}, family="glyph")
    known = sorted(
        set(CALIBRATED_RECIPES) | set(UNCALIBRATED_AT_TINY) | set(E30_LIMIT_RECIPES)
        | set(TASKS) | set(GLYPH_RECIPES) | set(GLYPH_TASKS)
    )
    raise ValueError(f"unknown recipe {name!r}; expected one of {known}")

# Which config field is the supervised span. Growing it is how packed CE gets a gradient
# (Arm-A: span=8 stayed at chance, span=32 hit 99%). count/majority are 1-token by design.
_ANSWER_FIELD: dict[str, str] = {
    "far_copy": "span_len",
    "recall": "value_len",
    "select": "value_len",
    "unique": "value_len",
    "match3": "value_len",
    "chain": "key_len",
    "chain_ordered": "key_len",
}

_GLYPH_ANSWER_FIELD: dict[str, str] = {
    "copy_span": "span_len",
    "reverse": "span_len",
    "every_k": "span_len",
    "filter_mod": "n_keep",
    "dyck_close": "span_len",
    "fact_markov": "value_len",
    "story_fact": "value_len",
    "chain_ordered_noise": "key_len",
    "chain_shuffled_noise": "key_len",
}


@dataclass(frozen=True)
class Scale:
    name: str
    seq_len: int
    min_gap: int
    # Decoder / E18 stack window. Must be <= min_gap so retrieval floors stay exact.
    local_window: int
    n_symbols: int = 4
    key_len: int = 2
    value_len: int = 2
    span_len: int = 8
    n_distractors: int = 3
    hops: int = 2
    count_mod: int = 4
    n_decoys: int = 4
    n_duplicates: int = 2


SCALES: dict[str, Scale] = {
    "tiny": Scale("tiny", seq_len=128, min_gap=16, local_window=16, span_len=32, n_distractors=2, hops=2, n_decoys=3, n_duplicates=2, key_len=2, value_len=4),
    "tiny_wide": Scale("tiny_wide", seq_len=256, min_gap=32, local_window=32, span_len=16, n_distractors=3, hops=3, n_decoys=6, n_duplicates=3),
    # First published "beyond tiny" DNA rungs. local_window < min_gap so e18_local is a real
    # leak control (K2). Spread placement at these lengths is a content-addressing hunt;
    # INDEX S0 uses --evidence_align right (fixed offset = min_gap+1).
    "bridge": Scale("bridge", seq_len=512, min_gap=64, local_window=16, span_len=32, n_distractors=2, hops=2, n_decoys=3, n_duplicates=2, key_len=2, value_len=4),
    "bridge_1k": Scale("bridge_1k", seq_len=1024, min_gap=64, local_window=16, span_len=32, n_distractors=2, hops=2, n_decoys=3, n_duplicates=2, key_len=2, value_len=4),
    "medium": Scale("medium", seq_len=4096, min_gap=1024, local_window=256, span_len=32, n_distractors=7, hops=3, n_decoys=12, n_duplicates=4, key_len=4, value_len=4),
    "medium_16k": Scale("medium_16k", seq_len=16384, min_gap=4096, local_window=1024, span_len=32, n_distractors=7, hops=4, n_decoys=16, n_duplicates=6, key_len=4, value_len=4),
    "large": Scale("large", seq_len=32768, min_gap=8192, local_window=1024, span_len=32, n_distractors=7, hops=4, n_decoys=16, n_duplicates=6, key_len=4, value_len=4),
    "large_128k": Scale("large_128k", seq_len=131072, min_gap=16384, local_window=1024, span_len=32, n_distractors=7, hops=5, n_decoys=24, n_duplicates=8, key_len=4, value_len=4),
}


def _scale_kwargs(sc: Scale) -> dict:
    return dict(
        seq_len=sc.seq_len,
        n_symbols=sc.n_symbols,
        min_gap=sc.min_gap,
        key_len=sc.key_len,
        value_len=sc.value_len,
        span_len=sc.span_len,
        n_distractors=sc.n_distractors,
        hops=sc.hops,
        count_mod=sc.count_mod,
        n_decoys=sc.n_decoys,
        n_duplicates=sc.n_duplicates,
    )


def target_answer_len(scale: str | Scale) -> int:
    """Supervised tokens we try to pack so CE can train. ~32–64 bits at A=4."""
    sc = SCALES[scale] if isinstance(scale, str) else scale
    if sc.seq_len <= 128:
        return 16
    if sc.seq_len <= 512:
        return 24
    return 32


def pack_overrides(scale: str | Scale, task: str, target: int | None = None) -> dict:
    """Grow the supervised span toward `target` tokens without unpacking the scale default.

    Shrinks decoys/distractors only if the target cannot fit otherwise. Never drops
    `hops` below 2, `n_decoys` below 1 on `select`, or `n_distractors` below 1 on
    tasks that need a contrast set. Returns the kwargs to merge into `config_for`.
    """
    if task in GLYPH_TASKS:
        return _glyph_pack_overrides(scale, task, target)
    sc = SCALES[scale] if isinstance(scale, str) else scale
    field = _ANSWER_FIELD.get(task)
    if field is None:
        return {}
    target = target if target is not None else target_answer_len(sc)
    base = _scale_kwargs(sc)
    current = int(base[field])
    want = max(current, target)

    def fits(extra: dict) -> bool:
        try:
            SymbolicTaskConfig(task=task, **{**base, **extra})
            return True
        except ValueError:
            return False

    extra: dict = {}
    if fits({field: want}):
        extra[field] = want
        return extra

    # Shrink contrast-set size so a packed answer can still sit behind min_gap.
    shrink_keys: list[tuple[str, int]] = []
    if task == "select":
        shrink_keys = [("n_decoys", 1), ("n_distractors", 1)]
    elif task == "unique":
        shrink_keys = [("n_duplicates", 1)]
    elif task in {"recall", "match3", "chain", "chain_ordered"}:
        shrink_keys = [("n_distractors", 1)]

    for key, lo in shrink_keys:
        while base[key] > lo:
            base[key] -= 1
            extra[key] = base[key]
            if fits({**extra, field: want}):
                extra[field] = want
                return extra

    # Largest packed answer that still fits after shrinking.
    for n in range(want, current - 1, -1):
        if fits({**extra, field: n}):
            extra[field] = n
            return extra
    return extra


def _glyph_scale_kwargs(sc: Scale, task: str) -> dict:
    target = target_answer_len(sc)
    n_keep = min(sc.span_len - 1, max(4, target))
    n_dist = min(sc.n_distractors, 1) if task == "story_fact" else sc.n_distractors
    return dict(
        task=task,
        seq_len=sc.seq_len,
        min_gap=sc.min_gap,
        width=32,
        noise="markov",
        span_len=sc.span_len,
        key_len=sc.key_len,
        value_len=sc.value_len,
        hops=sc.hops,
        n_distractors=n_dist,
        n_keep=n_keep,
        k=2,
        modulus=3,
    )


def _glyph_pack_overrides(scale: str | Scale, task: str, target: int | None = None) -> dict:
    sc = SCALES[scale] if isinstance(scale, str) else scale
    target = target if target is not None else target_answer_len(sc)
    base = _glyph_scale_kwargs(sc, task)
    k = int(base["k"])

    def fits(extra: dict) -> bool:
        try:
            GlyphTaskConfig(**{**base, **extra})
            return True
        except ValueError:
            return False

    extra: dict = {}
    if task == "every_k":
        want_span = max(int(base["span_len"]), target * k)
        if want_span % k:
            want_span += k - (want_span % k)
        if fits({"span_len": want_span}):
            return {"span_len": want_span}
        for n in range(want_span, k - 1, -k):
            if fits({"span_len": n}):
                return {"span_len": n}
        return extra

    field = _GLYPH_ANSWER_FIELD.get(task)
    if field is None:
        return {}
    current = int(base[field])
    want = max(current, target)
    if task == "filter_mod":
        extra = {"span_len": max(int(base["span_len"]), want + 1)}
        if fits({**extra, "n_keep": want}):
            extra["n_keep"] = want
            return extra
        for n in range(want, 0, -1):
            cand = {**extra, "n_keep": n, "span_len": max(int(base["span_len"]), n + 1)}
            if fits(cand):
                return cand
        return extra
    if fits({field: want}):
        extra[field] = want
        return extra
    shrink_lo = 0
    if task in {"fact_markov", "chain_ordered_noise", "chain_shuffled_noise", "story_fact"}:
        while base["n_distractors"] > shrink_lo:
            base["n_distractors"] -= 1
            extra["n_distractors"] = base["n_distractors"]
            if fits({**extra, field: want}):
                extra[field] = want
                return extra
    for n in range(want, 0, -1):
        if fits({**extra, field: n}):
            extra[field] = n
            return extra
    return extra


def is_glyph_task(task: str) -> bool:
    return task in GLYPH_TASKS or task in GLYPH_RECIPES


def config_for(scale: str | Scale, task: str, *, pack: bool = True, **over):
    """Build a DNA `SymbolicTaskConfig` or a Glyph config, depending on `task`."""
    sc = SCALES[scale] if isinstance(scale, str) else scale
    if task in GLYPH_TASKS:
        kw = _glyph_scale_kwargs(sc, task)
        if pack:
            kw.update(_glyph_pack_overrides(sc, task))
        kw.update(over)
        return GlyphTaskConfig(**kw)
    kw = dict(task=task, **_scale_kwargs(sc))
    if pack:
        kw.update(pack_overrides(sc, task))
    kw.update(over)
    return SymbolicTaskConfig(**kw)


def generate_row_for(cfg, rng):
    """Dispatch to DNA or Glyph. Probe / tests should use this, not a raw import."""
    if isinstance(cfg, GlyphTaskConfig) or getattr(cfg, "family", "dna") == "glyph":
        return generate_glyph_row(cfg, rng)
    return generate_dna_row(cfg, rng)


def rung_card(scale: str | Scale, task: str, **over) -> dict:
    """The numbers a spec / plot / gate can cite for one (scale, task)."""
    sc = SCALES[scale] if isinstance(scale, str) else scale
    cfg = config_for(sc, task, **over)
    window = sc.local_window
    glyph = isinstance(cfg, GlyphTaskConfig)
    bapo = GLYPH_BAPO_CLASS[task] if glyph else BAPO_CLASS[task]
    prize = glyph_prize_bits(cfg) if glyph else prize_bits(cfg)
    floor = glyph_floor_nats(cfg, window) if glyph else floor_nats(cfg, window)
    chance = glyph_chance_accuracy(cfg) if glyph else chance_accuracy(cfg)
    family = "glyph" if glyph else ("aggregation" if task in AGGREGATION_TASKS else "retrieval")
    return {
        "scale": sc.name,
        "task": task,
        "bapo_class": bapo,
        "family": family,
        "seq_len": cfg.seq_len,
        "min_gap": cfg.min_gap,
        "local_window": window,
        "n_symbols": cfg.n_symbols,
        "answer_len": cfg.answer_len,
        "prize_bits": prize,
        "floor_nats": floor,
        "chance_acc": chance,
        "solvable_acc": SOLVABLE_ACC,
        "bits_per_input_token_ceiling": prize / cfg.seq_len,
        "bytes_per_input_token_ceiling": prize / 8.0 / cfg.seq_len,
        "packed_answer_len": cfg.answer_len,
        "target_answer_len": target_answer_len(sc),
        "width": getattr(cfg, "width", 4),
        "noise": getattr(cfg, "noise", "iid"),
    }


def ladder_cards(scale: str, tasks: tuple[str, ...] | None = None) -> list[dict]:
    tasks = tasks or TASKS
    return [rung_card(scale, t) for t in tasks]


__all__ = [
    "AGGREGATION_TASKS",
    "CALIBRATED_RECIPES",
    "E30_LIMIT_RECIPES",
    "GLYPH_CORE_RECIPES",
    "GLYPH_RECIPES",
    "GLYPH_TASKS",
    "HARD_TASKS",
    "RETRIEVAL_TASKS",
    "Recipe",
    "SCALES",
    "SOLVABLE_ACC",
    "TASK_DISPLAY_ORDER",
    "TINY_PROOF_TASKS",
    "UNCALIBRATED_AT_TINY",
    "USER_CORE_TASKS",
    "config_for",
    "generate_row_for",
    "is_glyph_task",
    "ladder_cards",
    "pack_overrides",
    "resolve_recipe",
    "rung_card",
    "target_answer_len",
]
