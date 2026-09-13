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
    "tiny": Scale("tiny", seq_len=96, min_gap=16, local_window=16, span_len=32, n_distractors=2, hops=2, n_decoys=3, n_duplicates=2, key_len=2, value_len=4),
    "tiny_wide": Scale("tiny_wide", seq_len=256, min_gap=32, local_window=32, span_len=16, n_distractors=3, hops=3, n_decoys=6, n_duplicates=3),
    "medium": Scale("medium", seq_len=4096, min_gap=1024, local_window=256, span_len=32, n_distractors=7, hops=3, n_decoys=12, n_duplicates=4, key_len=4, value_len=4),
    "medium_16k": Scale("medium_16k", seq_len=16384, min_gap=4096, local_window=1024, span_len=32, n_distractors=7, hops=4, n_decoys=16, n_duplicates=6, key_len=4, value_len=4),
    "large": Scale("large", seq_len=32768, min_gap=8192, local_window=1024, span_len=32, n_distractors=7, hops=4, n_decoys=16, n_duplicates=6, key_len=4, value_len=4),
    "large_128k": Scale("large_128k", seq_len=131072, min_gap=16384, local_window=1024, span_len=32, n_distractors=7, hops=5, n_decoys=24, n_duplicates=8, key_len=4, value_len=4),
}


def config_for(scale: str | Scale, task: str, **over) -> SymbolicTaskConfig:
    sc = SCALES[scale] if isinstance(scale, str) else scale
    kw = dict(
        task=task,
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
    kw.update(over)
    return SymbolicTaskConfig(**kw)


def rung_card(scale: str | Scale, task: str) -> dict:
    """The numbers a spec / plot / gate can cite for one (scale, task)."""
    sc = SCALES[scale] if isinstance(scale, str) else scale
    cfg = config_for(sc, task)
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
    "rung_card",
]
