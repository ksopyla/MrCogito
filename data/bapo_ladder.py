"""Named BAPO rungs over `data/symbolic_tasks.py`.

A rung is a (scale, task) pair with a frozen `SymbolicTaskConfig` and the architecture-facing
contract: the dense decoder-only control must reach `SOLVABLE_ACC` on that config before any
compressed-channel number is interpreted. Scales:

  tiny        64–256 tokens     CPU, minutes; the solvability proof lives here
  medium      4k and 16k       GPU, <100M params
  large        32k and 128k     GPU, still <100M params

The other agent's tiny Arm-A 100% map (`symbolic_arm_a_100pct_limits_20260913.md`) owns the
`perceiver_concept` far_copy exam at seq=128. This ladder is the *scale-up* of that DNA
framework onto E18 vs matched dense transformers, with BAPO-hard tasks the original four
generators did not cover.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.symbolic_tasks import (
    AGGREGATION_TASKS,
    BAPO_CLASS,
    RETRIEVAL_TASKS,
    TASKS,
    SymbolicTaskConfig,
    chance_accuracy,
    floor_nats,
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


def config_for(scale: str | Scale, task: str, *, pack: bool = True, **over) -> SymbolicTaskConfig:
    sc = SCALES[scale] if isinstance(scale, str) else scale
    kw = dict(task=task, **_scale_kwargs(sc))
    if pack:
        kw.update(pack_overrides(sc, task))
    kw.update(over)
    return SymbolicTaskConfig(**kw)


def rung_card(scale: str | Scale, task: str, **over) -> dict:
    """The numbers a spec / plot / gate can cite for one (scale, task)."""
    sc = SCALES[scale] if isinstance(scale, str) else scale
    cfg = config_for(sc, task, **over)
    window = sc.local_window
    return {
        "scale": sc.name,
        "task": task,
        "bapo_class": BAPO_CLASS[task],
        "family": "aggregation" if task in AGGREGATION_TASKS else "retrieval",
        "seq_len": cfg.seq_len,
        "min_gap": cfg.min_gap,
        "local_window": window,
        "n_symbols": cfg.n_symbols,
        "answer_len": cfg.answer_len,
        "prize_bits": prize_bits(cfg),
        "floor_nats": floor_nats(cfg, window),
        "chance_acc": chance_accuracy(cfg),
        "solvable_acc": SOLVABLE_ACC,
        "bits_per_input_token_ceiling": prize_bits(cfg) / cfg.seq_len,
        "bytes_per_input_token_ceiling": prize_bits(cfg) / 8.0 / cfg.seq_len,
        "packed_answer_len": cfg.answer_len,
        "target_answer_len": target_answer_len(sc),
    }


def ladder_cards(scale: str, tasks: tuple[str, ...] | None = None) -> list[dict]:
    tasks = tasks or TASKS
    return [rung_card(scale, t) for t in tasks]


__all__ = [
    "AGGREGATION_TASKS",
    "HARD_TASKS",
    "RETRIEVAL_TASKS",
    "SCALES",
    "SOLVABLE_ACC",
    "TINY_PROOF_TASKS",
    "USER_CORE_TASKS",
    "config_for",
    "ladder_cards",
    "pack_overrides",
    "rung_card",
    "target_answer_len",
]
