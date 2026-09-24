"""MrCogito capability suite — the standard exam ladder every new architecture runs.

Why this module exists
----------------------
Every architecture so far was scored on ad-hoc flag combinations, so results were hard to
compare and easy to misread (see the E30 review). This module freezes one ladder:

* **Levels** of increasing difficulty (learns at all → carries a fact → picks signal over a
  lookalike → long reach → multi-step reasoning → language-like noise → BAPO-hard stretch).
* **Cells** = one exam at one length, each a frozen probe configuration
  (`verification/bapo_capability_probe.py`, no fork) with its expected prize in bits.
* **Sizes** 5M / 10M / 30M / 50M: the width-matched 4-layer family the E30 ledger used
  (1 local layer, 1 global read, 2 local layers; head_dim 64; 1 KV head), so past runs are
  directly comparable.
* **Policy**: step size and budget per (size, length), taken from the small-model protocol.
* **References**: every scored past result on these exact cells (dense = full model, E18 =
  full read on the same platform, E21 = averaged notebook, E30 = sliding-window notebook).

The runner (`scripts/run_capability_suite.py`) turns (arches × sizes × tier) into probe jobs
and GPU launch scripts; the scorecard (`analysis/capability_scorecard.py`) turns their JSON
into per-level pass/fail, comparisons with the references, and a scale-up verdict.
Spec: `docs/engineering_specs/capability_suite.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SUITE_VERSION = "2026-09-24.v1"
PASS_ACC = 0.75            # answer-token accuracy that counts as a pass (SOLVABLE_ACC)
CEILING_FRACTION = 0.75    # "matches the ceiling" = ≥ 0.75 × the dense model's bits on the same run


# --------------------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Size:
    name: str
    hidden: int
    head_dim: int = 64
    kv_heads: int = 1
    pre_layers: int = 1
    global_layers: int = 1
    stack_layers: int = 2
    max_params: int = 0
    dense_params_m: float = 0.0   # measured with the probe factory (DNA vocab)

    def probe_args(self) -> list[str]:
        return [
            "--hidden", str(self.hidden), "--head_dim", str(self.head_dim),
            "--kv_heads", str(self.kv_heads), "--pre_layers", str(self.pre_layers),
            "--global_layers", str(self.global_layers), "--stack_layers", str(self.stack_layers),
            "--max_params", str(self.max_params),
        ]


SIZES: dict[str, Size] = {
    "5m": Size("5m", hidden=384, max_params=7_000_000, dense_params_m=5.11),
    "10m": Size("10m", hidden=512, max_params=12_000_000, dense_params_m=8.97),
    "30m": Size("30m", hidden=960, max_params=40_000_000, dense_params_m=31.00),
    "50m": Size("50m", hidden=1216, max_params=60_000_000, dense_params_m=49.54),
}
SIZE_ORDER = ("5m", "10m", "30m", "50m")


# --------------------------------------------------------------------------------------
# Levels and cells
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    number: int
    name: str
    question: str


LEVELS: dict[int, Level] = {
    0: Level(0, "Learns at all", "Short books (128 tokens): can the architecture learn to copy and to look up a fact at all?"),
    1: Level(1, "Carries a fact", "Medium books (256–512): does a whole fact survive the channel?"),
    2: Level(2, "Picks signal over a lookalike", "A decoy block with the same shape: does it keep the right fact and ignore the lookalike?"),
    3: Level(3, "Long reach", "1024–2048 tokens: does retrieval survive a long book?"),
    4: Level(4, "Multi-step reasoning", "Follow a 4-hop chain of facts through a long book."),
    5: Level(5, "Language-like noise", "Facts hidden in plausible Markov 'text' filler instead of random letters."),
    6: Level(6, "BAPO-hard stretch", "Tasks no bounded channel should solve easily (shuffled chains, uniqueness). Reported, never gating."),
}


@dataclass(frozen=True)
class Cell:
    id: str
    level: int
    recipe: str
    scale: str
    what: str
    prize_bits: float
    overrides: tuple[tuple[str, object], ...] = ()
    stretch: bool = False          # reported, not part of the level gate
    family: str = "dna"            # dna | glyph

    @property
    def seq_len(self) -> int:
        from data.bapo_ladder import SCALES

        return int(dict(self.overrides).get("seq_len", SCALES[self.scale].seq_len))

    def probe_args(self) -> list[str]:
        out = ["--scale", self.scale, "--recipe", self.recipe]
        for k, v in self.overrides:
            out += [f"--{k}", str(v)]
        return out


def _c(id, level, recipe, scale, what, prize, stretch=False, family="dna", **over):
    return Cell(id, level, recipe, scale, what, float(prize), tuple(sorted(over.items())), stretch, family)


CELLS: tuple[Cell, ...] = (
    # L0 — learns at all (128 tokens)
    _c("L0.copy-128", 0, "far_copy", "tiny", "copy a marked 32-letter span", 64),
    _c("L0.lookup-128", 0, "recall_single", "tiny", "look up one planted fact (16-letter value)", 32),
    # L1 — carries a fact (256–512)
    _c("L1.copy-256", 1, "far_copy", "tiny_wide", "copy a marked 24-letter span", 48),
    _c("L1.lookup-256", 1, "recall_single", "tiny_wide", "look up one fact, 24-letter value", 48),
    _c("L1.copy-512", 1, "far_copy", "bridge", "copy a 32-letter span (fixed offset)", 64, evidence_align="right"),
    _c("L1.lookup-512", 1, "recall_single", "bridge", "look up one fact, 24-letter value (fixed offset)", 48,
       evidence_align="right"),
    # L2 — picks signal over a lookalike
    _c("L2.lookalike-128", 2, "select_1decoy", "tiny", "the fact vs one same-shaped decoy", 32),
    _c("L2.lookalike-1k", 2, "select_1decoy", "bridge_1k", "the fact vs one decoy, 1024 tokens", 64),
    # L3 — long reach
    _c("L3.lookup-1k", 3, "recall_single", "bridge_1k", "one fact anywhere in 1024 tokens, 32-letter value", 64),
    _c("L3.lookup-2k", 3, "recall_single", "bridge_1k", "one fact anywhere in 2048 tokens", 64, seq_len=2048),
    _c("L3.copy-1k", 3, "far_copy", "bridge_1k", "copy a 32-letter span from anywhere in 1024 tokens", 64,
       stretch=True),
    # L4 — multi-step reasoning
    _c("L4.chain-1k", 4, "chain_ordered", "bridge_1k", "follow 4 in-order hops, 1024 tokens", 64, hops=4),
    _c("L4.chain-2k", 4, "chain_ordered", "bridge_1k", "follow 4 in-order hops, 2048 tokens", 64, hops=4,
       seq_len=2048),
    # L5 — language-like noise (Glyph; uncalibrated until a dense ceiling is recorded)
    _c("L5.fact-512", 5, "fact_markov_single", "bridge", "one fact in Markov 'text' filler", 72, family="glyph"),
    _c("L5.story-512", 5, "story_fact", "bridge", "a fact keyed by a word, in word-like filler", 72, family="glyph"),
    _c("L5.chain-512", 5, "chain_ordered_noise", "bridge", "in-order hops inside structured filler", 72,
       family="glyph"),
    _c("L5.fact-1k", 5, "fact_markov_single", "bridge_1k", "one fact in Markov filler, 1024 tokens", 96,
       family="glyph"),
    # L6 — BAPO-hard stretch (never gating)
    _c("L6.shuffled-1k", 6, "chain_shuffled", "bridge_1k", "2 hops given out of order", 64, stretch=True, hops=2),
    _c("L6.unique-256", 6, "unique", "tiny_wide", "report the one fact that appears once", 48, stretch=True),
)
CELL_BY_ID = {c.id: c for c in CELLS}


# --------------------------------------------------------------------------------------
# Policy: step size and budget
# --------------------------------------------------------------------------------------

# (size, length bucket) → (lr, warm_residuals, measured_in_ledger)
_LR: dict[tuple[str, str], tuple[float, bool, bool]] = {
    ("5m", "short"): (1e-3, False, True), ("5m", "mid"): (3e-4, True, False), ("5m", "long"): (1e-4, True, False),
    ("10m", "short"): (1e-3, False, True), ("10m", "mid"): (3e-4, True, False), ("10m", "long"): (1e-4, True, False),
    ("30m", "short"): (3e-4, True, True), ("30m", "mid"): (3e-4, True, True), ("30m", "long"): (1e-4, True, True),
    ("50m", "short"): (2e-4, True, False), ("50m", "mid"): (2e-4, True, False), ("50m", "long"): (5e-5, True, True),
}


def _bucket(seq_len: int) -> str:
    return "short" if seq_len <= 256 else ("mid" if seq_len <= 512 else "long")


def lr_for(size: str, cell: Cell) -> tuple[float, bool, bool]:
    """(learning rate, warm residuals, was this combination measured in the ledger)."""
    return _LR[(size, _bucket(cell.seq_len))]


@dataclass(frozen=True)
class Budget:
    steps: int
    k1_mult: int
    batch: int
    eval_every: int


def budget_for(cell: Cell) -> Budget:
    """Advertised budget; the probe extends it once when eval CE is still falling."""
    L = cell.seq_len
    if L <= 256:
        return Budget(800, 4, 32, 100)
    if L <= 512:
        return Budget(1200, 4, 32, 100)
    if L <= 1024:
        return Budget(1200, 4, 8, 100)
    return Budget(1200, 4, 4, 100)


EVAL_ROWS = 256  # the E30 ledger used 16–32 rows; 256 keeps the accuracy SE near ±1–2 points


# --------------------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    name: str
    levels: tuple[int, ...]
    seeds: int
    include_stretch: bool
    purpose: str


TIERS: dict[str, Tier] = {
    "screen": Tier("screen", (0, 1, 2), 1, False,
                   "Does a new architecture learn and carry facts at all? One seed, hours on one GPU."),
    "standard": Tier("standard", (0, 1, 2, 3, 4), 2, False,
                     "The comparison run: every gating level up to reasoning, two seeds."),
    "full": Tier("full", (0, 1, 2, 3, 4, 5, 6), 3, True,
                 "Claim run: all levels incl. language-like noise and stretch cells, three seeds."),
}


def cells_for(tier: str, levels: Optional[tuple[int, ...]] = None, cell_ids: Optional[tuple[str, ...]] = None):
    t = TIERS[tier]
    if cell_ids:
        return [CELL_BY_ID[i] for i in cell_ids]
    lv = levels if levels is not None else t.levels
    return [c for c in CELLS if c.level in lv and (t.include_stretch or not c.stretch)]


# Arch-specific probe flags, applied whenever that arch is in the job. They only affect
# their own arch (the probe ignores E21 flags on E30 and vice versa).
ARCH_FLAGS: dict[str, tuple[str, ...]] = {
    "e21": ("--message_identity_slots",),          # the fair frozen-mean concat control
    "e30": ("--swp_n_heads", "8", "--swp_query_dim", "128"),
}
DEFAULT_CONTROLS = ("dense",)


# --------------------------------------------------------------------------------------
# References (past scored results on exactly these cells)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reference:
    cell: str
    size: str
    arch: str
    bits: float
    note: str = ""
    source: str = ""


_R31 = "run_reports/e30_30m_gpu_odra_polonez_20260922.md"
_RLIM = "run_reports/e30_length_hardness_limits_20260922.md"
_RCOV = "run_reports/e30_coverage_and_breadth_20260922.md"
_RCPU = "run_reports/e30_tiny_5m_9m_capability_20260921.md"


def _refs(cell, size, source, note="", **bits):
    return [Reference(cell, size, a, float(b), note, source) for a, b in bits.items()]


REFERENCES: tuple[Reference, ...] = tuple(
    # 30M — GPU hunt (dense/e18/e21 at 1e-3 for 128-token rows; e30 at 3e-4; 256/512 at 3e-4 warm)
    _refs("L0.copy-128", "30m", _R31, dense=62.9, e18=63.0, e21=42.5, e30=59.8)
    + _refs("L0.lookup-128", "30m", _R31, dense=31.2, e18=11.4, e21=14.7, e30=28.0)
    + _refs("L2.lookalike-128", "30m", _R31, dense=31.4, e18=31.3, e21=8.8, e30=27.8)
    + _refs("L1.copy-256", "30m", _R31, dense=47.3, e18=47.2, e21=42.3, e30=46.1)
    + _refs("L1.lookup-256", "30m", _R31, dense=47.1, e18=47.2, e21=40.9, e30=46.9)
    + _refs("L1.copy-512", "30m", _R31, dense=64.0, e18=64.0, e21=61.4, e30=63.1)
    + _refs("L1.lookup-512", "30m", _R31, dense=48.0, e18=48.0, e21=44.0, e30=47.6)
    # 30M — 1024/2048 limits (step 1e-4, warm; "full read" there is the dense model)
    + _refs("L2.lookalike-1k", "30m", _RLIM, "4800 steps", dense=62.6, e21=18.7, e30=25.9)
    + _refs("L3.lookup-1k", "30m", _RLIM, "4800 steps", dense=0.0, e21=0.0, e30=25.7)
    + _refs("L3.lookup-2k", "30m", _RLIM, "256-token pages; 128-token pages extended: 25 bits",
            dense=0.0, e21=0.0, e30=8.3)
    + _refs("L3.copy-1k", "30m", _RCOV, dense=2.0, e21=9.0, e30=3.0)
    + _refs("L4.chain-1k", "30m", _RLIM, "e30 needed 9600 steps", dense=63.3, e21=4.0, e30=40.1)
    + _refs("L4.chain-2k", "30m", _RCOV, "128-token pages", dense=63.0, e21=0.0, e30=0.0)
    + _refs("L6.shuffled-1k", "30m", _RCOV, dense=0.0, e21=0.0, e30=0.0)
    # 50M (H=1216)
    + _refs("L2.lookalike-1k", "50m", _RLIM, dense=62.9, e21=16.6, e30=25.3)
    + _refs("L3.lookup-1k", "50m", _RLIM, dense=0.0, e21=0.0, e30=24.3)
    + _refs("L4.chain-1k", "50m", _RCOV, "step 5e-5; e21 after a doubled budget", dense=63.0, e21=46.0, e30=42.0)
    # 10M (H=512, CPU, 1e-3) and 5M (H=384) — 128-token rows
    + _refs("L0.copy-128", "10m", _RCPU, "CPU, 1e-3", e21=13.64, e30=36.96)
    + _refs("L0.lookup-128", "10m", _RCPU, "CPU, 1e-3", e21=4.84, e30=12.30)
    + _refs("L0.copy-128", "5m", _RCPU, "CPU; e30 at 1e-3, e21 at 3e-3 extended", e21=37.10, e30=11.95)
    + _refs("L0.lookup-128", "5m", _RCPU, "CPU; e30 at 1e-3, e18/e21 at 3e-3", e18=25.35, e21=11.63, e30=3.76)
)


def references_for(cell: str, size: str) -> dict[str, Reference]:
    return {r.arch: r for r in REFERENCES if r.cell == cell and r.size == size}


# --------------------------------------------------------------------------------------
# Scale-up rule (used by the scorecard; explained in the spec)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleRule:
    min_size_for_verdict: str = "30m"
    min_frontier_level: int = 2        # must pass every gating cell of L0..L2
    trend_tolerance_bits: float = 2.0  # bits may dip this much between sizes and still count as flat
    min_speed_vs_dense: float = 0.5    # tokens/sec relative to dense at the same size
    beats_past_on_hard: int = 1        # ≥ this many L3/L4 cells where it ties or beats the best past arch


SCALE_RULE = ScaleRule()


__all__ = [
    "ARCH_FLAGS", "Budget", "CELLS", "CELL_BY_ID", "CEILING_FRACTION", "Cell", "DEFAULT_CONTROLS",
    "EVAL_ROWS", "LEVELS", "Level", "PASS_ACC", "REFERENCES", "Reference", "SCALE_RULE", "SIZES",
    "SIZE_ORDER", "SUITE_VERSION", "Size", "TIERS", "Tier", "budget_for", "cells_for", "lr_for",
    "references_for",
]
