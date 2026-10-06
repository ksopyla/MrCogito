"""Capability checks v4 — the one leveled series of learning-capability tasks (source of truth).

Every task is a synthetic exam from the shared engine (`data/bapo_ladder.py` generates it,
`verification/bapo_capability_probe.py` trains and scores it). v4 organizes them by *what the model must
do* (levels C0–C7, each building on the previous), with two separate dials measured the same way on every
level: input length (train at 1–2k, read up to 128k with no more training) and model size (5M → 50M).
Process and rules: docs/engineering_specs/capability_checks.md (skill `capability-checks`).

Rules every task obeys (the author's decisions, 2026-10-04):
  * **from scratch** — random init, our own architectures; never a checkpoint from another run or a
    pretrained model;
  * **an in-run curriculum is allowed only when it is written here** (`curriculum`), identical for every
    architecture, starting from random init;
  * **first-letter accuracy** is the score (the first answer letter is predicted with no answer letters in
    context; later letters can be copied once the first identifies the candidate); pass = median over
    seeds 0, 1, 2 ≥ 75 %; the dense model trains next to the candidate as the ceiling;
  * **exams with several same-shaped candidates** (the parallel chains, the keyed lookups) are scored on the
    **picked candidate** (`score="candidate"`): the answer is decoded greedily and the planted candidate
    nearest to it must be the asked one (probe `answer_exact.candidate`). There the first letter has a
    guessing floor of ~40 % (answer with any candidate: the commonest first letter among them wins), the
    picked candidate 1 / #candidates, and a lossy copy of the right fact still counts. Every task states its guessing floor (`floor`, of its own score) next to chance
    (E33a diagnosis, 2026-10-05: docs/4_Research_Notes/e33a_loop_diagnosis_20261004.md);
  * **flawed tasks** stay listed with their reason so nobody reuses them as evidence; they never gate,
    never enter a level and are crossed out on the dashboard.

A task's **recipe** is its exam (`recipe` + `args`), its training budget (`train`) and its written
curriculum (`curriculum`), all three the same for every architecture and the dense ceiling. `train` empty =
the suite v3 budget of the size (`evaluation/capability_suite.py`). Changing any of the three is a protocol
change: it applies to every model, bumps `VERSION` and gets a line in the spec's calibration log.

`status`: `active` (frozen recipe, run it) · `calibrating` (the from-scratch recipe — step size, budget or
curriculum — is not fixed yet; see the calibration study in the spec) · `flawed` (do not use as evidence).
"""
from __future__ import annotations

from dataclasses import dataclass

VERSION = "v4-draft-2026-10-06.2"
PASS_ACC = 0.75
SEEDS = (0, 1, 2)
LADDER = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)

# Training budgets (probe flags; micro-batch splitting `--grad_accum` is memory-only and left to the runner).
# k1_mult: the run may extend to k1_mult x steps while it has not reached the pass mark.
TRAIN_1K = ("--seq_len", "1024", "--steps", "1200", "--k1_mult", "4", "--batch", "32", "--lr", "1e-4")
TRAIN_1K_X4 = ("--seq_len", "1024", "--steps", "4800", "--k1_mult", "4", "--batch", "32", "--lr", "1e-4")


@dataclass(frozen=True)
class Level:
    id: str
    name: str
    question: str
    measures: str


LEVELS: tuple[Level, ...] = (
    Level("C0", "Carry", "Does a span of letters survive the trip from far back?",
          "Copy a marked span. The simplest proof that information flows through the architecture at all."),
    Level("C1", "Address", "Can it find one fact by its key?",
          "One key → value fact sits somewhere in a book of random letters; the question names the key. "
          "With one fact the block marker alone finds it (locate); the keyed tasks plant 4 same-shaped facts "
          "or edges, so only the key in the question picks the right one (content addressing)."),
    Level("C2", "Discriminate", "Does it pick the right fact over look-alikes?",
          "Lookup, but the book also holds same-shaped facts with other keys (1 → 8 decoys). The read must "
          "be precise, not 'the most fact-like thing'. Decoys open with their own marker, so the marker, not "
          "the key, can separate them; key addressing is checked by the keyed C1 tasks and C3."),
    Level("C3", "Hold many", "Can it keep many facts when it does not know the question yet?",
          "8 or 16 facts are planted and the question asks about one. The memory is written before the "
          "question is seen, so it must keep all of them (capacity)."),
    Level("C4", "Compose", "Can it follow a chain of facts in order?",
          "Facts link A → B → C …; the question gives the start, the answer is the end. Hops are listed in "
          "order, 4 → 8 of them. The chain is the first k hop blocks in reading order, so a reader that "
          "counts hop markers can answer without using the keys (a memory without slot order cannot)."),
    Level("C5", "Reason", "Can it follow a chain whose hops are shuffled among decoy chains?",
          "Four chains of the same length, hops shuffled; only the start node in the question says which "
          "chain is the answer, so every hop must really be followed. Every chain runs one edge past the "
          "asked node (no chain-end guess), and the picked candidate is scored. The research frontier."),
    Level("C6", "Aggregate", "Can it answer a question about the whole book?",
          "The answer depends on all facts at once (the one fact that appears once; a count), not on one "
          "lookup."),
    Level("C7", "Language-like", "Do the skills survive plausible 'text' filler?",
          "The same exams with a typed vocabulary and Markov 'text' as filler instead of random letters. "
          "Real text follows as C8 when its generator exists."),
)
LEVEL_BY_ID = {lv.id: lv for lv in LEVELS}


@dataclass(frozen=True)
class Task:
    id: str
    level: str | None          # None for flawed tasks
    name: str
    recipe: str                # engine recipe (`--recipe`)
    args: tuple[str, ...]      # extra probe flags that define the exam
    train_len: int
    prize_bits: float | None   # information content of one answer
    measures: str
    status: str = "active"     # active | calibrating | flawed
    curriculum: str | None = None
    ladder: bool = False       # read at every LADDER length ≥ train_len after training
    chance: float = 0.25       # DNA letters (A/C/G/T); Glyph rows use a typed vocabulary
    flaw: str | None = None
    family: str = "dna"
    score: str = "first"       # first: first-letter accuracy · candidate: the picked candidate (probe `answer_exact`)
    floor: float | None = None  # guessing floor of `score` when it is above chance (measured on the exam's rows)
    train: tuple[str, ...] = ()  # training budget every architecture uses (empty: the suite v3 budget)


_L = True
TASKS: tuple[Task, ...] = (
    # C0 — carry
    Task("C0.copy-128", "C0", "copy 32 letters, 128-token book", "far_copy", ("--scale", "tiny"), 128, 64,
         "copy a marked 32-letter span"),
    Task("C0.copy-256", "C0", "copy 24 letters, 256-token book", "far_copy", ("--scale", "tiny_wide"), 256, 48,
         "copy a marked 24-letter span"),
    Task("C0.copy-512", "C0", "copy 32 letters, 512-token book (fixed offset)", "far_copy",
         ("--scale", "bridge", "--evidence_align", "right"), 512, 64, "copy a 32-letter span at a fixed offset"),
    Task("C0.copy-1k", "C0", "copy 32 letters, 1024-token book", "far_copy", ("--scale", "bridge_1k"), 1024, 64,
         "copy a 32-letter span from anywhere in 1024 tokens", ladder=_L),
    # C1 — address
    Task("C1.lookup-128", "C1", "one fact, 128-token book", "recall_single", ("--scale", "tiny"), 128, 32,
         "look up one planted fact (16-letter value)"),
    Task("C1.lookup-256", "C1", "one fact, 256-token book", "recall_single", ("--scale", "tiny_wide"), 256, 48,
         "look up one fact, 24-letter value"),
    Task("C1.lookup-512", "C1", "one fact, 512-token book (fixed offset)", "recall_single",
         ("--scale", "bridge", "--evidence_align", "right"), 512, 48, "look up one fact at a fixed offset"),
    Task("C1.lookup-1k", "C1", "one fact, 1024-token book", "recall_single", ("--scale", "bridge_1k"), 1024, 64,
         "one fact anywhere in 1024 tokens, 32-letter value", ladder=_L),
    Task("C1.lookup-2k", "C1", "one fact, 2048-token book", "recall_single",
         ("--scale", "bridge_1k", "--seq_len", "2048"), 2048, 64, "one fact anywhere in 2048 tokens", ladder=_L),
    Task("C1.lookup-16k", "C1", "one fact, trained up to a 16k book", "recall_single",
         ("--scale", "bridge_1k"), 16384, 64,
         "one fact anywhere in up to 16k tokens; tests whether longer training books carry to 128k",
         curriculum="one run from random init: 2k (C1.lookup-2k recipe) → 8k (1500 steps, batch 16, "
                    "step 5e-5) → 16k (1000 steps, batch 8, step 5e-5)", ladder=_L),
    Task("C1.keyed4-1k", "C1", "1 of 4 facts by its key, 1024 tokens", "recall",
         ("--scale", "bridge_1k", "--n_distractors", "3", "--key_len", "4"), 1024, 64,
         "4 same-shaped facts; only the key in the question picks one (first exam that needs the key)",
         status="calibrating", score="candidate", floor=0.25, ladder=_L, train=TRAIN_1K),
    Task("C1.edge-1k", "C1", "1 of 4 edges by its start node, 1024 tokens", "chain_parallel",
         ("--scale", "bridge_1k", "--hops", "1", "--chain_overhang", "1", "--key_len", "16"), 1024, 32,
         "the parallel chains with one hop: the start node picks the edge (keyed lookup in chain format; "
         "stage 1 of the C5 curriculum)", status="calibrating", score="candidate", floor=0.125, ladder=_L,
         train=TRAIN_1K_X4),
    # C2 — discriminate
    Task("C2.lookalike-128", "C2", "fact vs 1 look-alike, 128 tokens", "select_1decoy", ("--scale", "tiny"), 128, 32,
         "the fact vs one same-shaped decoy"),
    Task("C2.lookalike-1k", "C2", "fact vs 1 look-alike, 1024 tokens", "select_1decoy", ("--scale", "bridge_1k"),
         1024, 64, "the fact vs one decoy", ladder=_L),
    Task("C2.decoy8-1k", "C2", "fact among 8 look-alikes, 1024 tokens", "select",
         ("--scale", "bridge_1k", "--n_decoys", "8", "--n_distractors", "0"), 1024, 64,
         "pick the fact among 8 same-shaped decoys", status="calibrating", ladder=_L),
    # C3 — hold many
    Task("C3.recall8-1k", "C3", "recall 1 of 8 facts, 1024 tokens", "recall",
         ("--scale", "bridge_1k", "--n_distractors", "7", "--key_len", "4"), 1024, 64,
         "keep 8 facts (4-letter keys) without knowing which one will be asked", status="calibrating", ladder=_L,
         floor=0.44, train=TRAIN_1K_X4),
    Task("C3.recall16-1k", "C3", "recall 1 of 16 facts, 1024 tokens", "recall",
         ("--scale", "bridge_1k", "--n_distractors", "15", "--key_len", "4"), 1024, 64,
         "keep 16 facts; precision of addressing among many similar slots", status="calibrating", ladder=_L,
         floor=0.39),
    # C4 — compose (in-order)
    Task("C4.chain4-1k", "C4", "in-order 4-hop chain, 1024 tokens", "chain_ordered",
         ("--scale", "bridge_1k", "--hops", "4"), 1024, 64, "follow 4 hops listed in order", ladder=_L),
    Task("C4.chain4-2k", "C4", "in-order 4-hop chain, 2048 tokens", "chain_ordered",
         ("--scale", "bridge_1k", "--hops", "4", "--seq_len", "2048"), 2048, 64,
         "follow 4 in-order hops in a 2k book (no architecture, dense included, learns it at the v3 budget)",
         status="calibrating", ladder=_L),
    Task("C4.chain8-1k", "C4", "in-order 8-hop chain, 1024 tokens", "chain_ordered",
         ("--scale", "bridge_1k", "--hops", "8"), 1024, 64, "follow 8 in-order hops", status="calibrating",
         ladder=_L),
    # C5 — reason (parallel chains; 16-letter nodes so the overhang fits in 1024 tokens)
    *(Task(f"C5.pchain{h}-1k", "C5", f"parallel {h}-hop chain among 3 decoy chains, 1024 tokens", "chain_parallel",
           ("--scale", "bridge_1k", "--hops", str(h), "--chain_overhang", "1", "--key_len", "16",
            "--hop_count_in_question"), 1024, 32,
           f"follow {h} shuffled hops; 4 chains of the same length, only the start node names the right one; "
           f"every chain runs one edge past the asked node; the question states the hop count ({h} hop markers)",
           status="calibrating", ladder=_L, score="candidate", floor=fl, train=TRAIN_1K_X4,
           curriculum=f"candidate (to be fixed by calibration): one run from random init, the same exam at 1 hop → "
                      + " → ".join(str(k) for k in range(2, h + 1))
                      + " hops (hop count in every question), each stage at the task's budget, optionally with "
                      "half its rows at the previous hop count (`--replay_hops`)")
      for h, fl in ((2, 0.083), (3, 0.0625), (4, 0.05))),
    # C6 — aggregate
    Task("C6.unique-256", "C6", "the fact that appears once, 256 tokens", "unique", ("--scale", "tiny_wide"), 256, 48,
         "report the one fact that appears once (no key to look up)"),
    Task("C6.unique-1k", "C6", "the fact that appears once, 1024 tokens", "unique", ("--scale", "bridge_1k"), 1024,
         None, "report the one fact that appears once in 1024 tokens", status="calibrating", ladder=_L),
    Task("C6.count-1k", "C6", "count, 1024 tokens", "count", ("--scale", "bridge_1k"), 1024, 2,
         "a whole-book count, one answer token (2-bit prize: weak signal, report with care)",
         status="calibrating", ladder=_L),
    # C7 — language-like filler (Glyph: typed vocabulary, chance 1/8)
    Task("C7.fact-512", "C7", "one fact in Markov 'text', 512 tokens", "fact_markov_single", ("--scale", "bridge"),
         512, 72, "one fact in Markov 'text' filler", chance=0.125, family="glyph"),
    Task("C7.story-512", "C7", "a fact keyed by a word, 512 tokens", "story_fact", ("--scale", "bridge"), 512, 72,
         "a fact keyed by a word, in word-like filler", chance=0.125, family="glyph"),
    Task("C7.chain-512", "C7", "in-order hops in structured filler, 512 tokens", "chain_ordered_noise",
         ("--scale", "bridge"), 512, 72, "in-order hops inside structured filler", chance=0.125, family="glyph"),
    Task("C7.fact-1k", "C7", "one fact in Markov 'text', 1024 tokens", "fact_markov_single", ("--scale", "bridge_1k"),
         1024, 96, "one fact in Markov filler, 1024 tokens", chance=0.125, family="glyph"),
    # Flawed — kept so they are recognised, never used as evidence
    Task("X.shuffled2-1k", None, "shuffled 2-hop chain, no decoy chains", "chain_shuffled",
         ("--scale", "bridge_1k", "--hops", "2"), 1024, 64, "2 hops given out of order", status="flawed",
         flaw="shortcut: the answer is the only node that is never the start of a hop, so it is found without "
              "following any hop (use C5 parallel chains)"),
    Task("X.shuffled3-1k", None, "shuffled 3-hop chain, no decoy chains", "chain",
         ("--scale", "bridge_1k", "--hops", "3", "--n_distractors", "0"), 1024, 64, "3 hops given out of order",
         status="flawed", flaw="same no-hop shortcut as the shuffled 2-hop chain (use C5 parallel chains)"),
    *(Task(f"X.pchain{h}-1k-v1", None, f"parallel {h}-hop chain, 32-letter nodes, first letter", "chain_parallel",
           ("--scale", "bridge_1k", "--hops", str(h)), 1024, None, f"the E31b / E33a parallel {h}-hop exam",
           status="flawed", floor=fl,
           flaw=f"guessable: answering with any link target scores {fl:.0%} on the first letter (no hop followed), "
                "and picking a chain end 25 % of whole answers; every arm, dense included, sat on that floor "
                "(use C5: overhang, picked candidate)")
      for h, fl in ((2, 0.44), (3, 0.41), (4, 0.39))),
    Task("X.match3-1k", None, "the fact planted three times", "match3", ("--scale", "bridge_1k"), 1024, None,
         "report the fact that appears three times", status="flawed",
         flaw="at 1k the book holds the triple plus only 2 single facts: averaging all facts gives the majority "
              "letter at each position, no matching needed"),
    Task("X.majority-1k", None, "majority letter of the book", "majority", ("--scale", "bridge_1k"), 1024, 2,
         "the most frequent answer in the book", status="flawed",
         flaw="about 2/3 of the book is the winner, so any sample answers it"),
)
TASK_BY_ID = {t.id: t for t in TASKS}
FLAWED = {t.id: t.flaw for t in TASKS if t.status == "flawed"}

# ---- packages: which tasks a check runs --------------------------------------------------------------
PACKAGES = {
    "screen": {"sizes": ("10m",), "seeds": (0,), "levels": ("C0", "C1", "C2"), "ladder": False,
               "use": "does it learn at all — a few GPU-hours"},
    "core": {"sizes": ("30m",), "seeds": SEEDS, "levels": ("C0", "C1", "C2", "C3", "C4", "C5"), "ladder": True,
             "use": "the full check every variant gets; the no-harm comparison uses it"},
    "frontier": {"sizes": ("30m",), "seeds": SEEDS, "levels": ("C5", "C6"), "ladder": True,
                 "use": "deeper tasks for the current research question; they join core once stable"},
    "language": {"sizes": ("30m",), "seeds": SEEDS, "levels": ("C7",), "ladder": False,
                 "use": "the same skills in plausible filler; real text (C8) when it exists"},
}


# ---- legacy: map every past result onto a v4 task, with an honest match label -------------------------
# match: same (same exam and from-scratch protocol; re-scored on first letter) · settings-differ (same exam,
# from scratch, other step size / budget / rows) · curriculum-differs (from scratch, but through an
# unlisted schedule, e.g. started from the run's own lookup-2k weights) · not-from-scratch (another run's
# checkpoint: excluded from capability claims) · flawed.
SUITE_V3 = {  # suite v3 cell → v4 task (all from scratch with frozen settings)
    "L0.copy-128": "C0.copy-128", "L1.copy-256": "C0.copy-256", "L1.copy-512": "C0.copy-512",
    "L3.copy-1k": "C0.copy-1k", "L0.lookup-128": "C1.lookup-128", "L1.lookup-256": "C1.lookup-256",
    "L1.lookup-512": "C1.lookup-512", "L3.lookup-1k": "C1.lookup-1k", "L3.lookup-2k": "C1.lookup-2k",
    "L2.lookalike-128": "C2.lookalike-128", "L2.lookalike-1k": "C2.lookalike-1k", "L4.chain-1k": "C4.chain4-1k",
    "L4.chain-2k": "C4.chain4-2k", "L5.fact-512": "C7.fact-512", "L5.story-512": "C7.story-512",
    "L5.chain-512": "C7.chain-512", "L5.fact-1k": "C7.fact-1k", "L6.unique-256": "C6.unique-256",
    "L6.shuffled-1k": "X.shuffled2-1k",
}
BATTERY_V1 = {  # length-battery slot (analysis/capability_board.parse_job) → v4 task
    "lookup_2k": "C1.lookup-2k", "lookup_16k": "C1.lookup-16k", "chain_2k": "C4.chain4-2k",
    "recall8_1k": "C3.recall8-1k", "recall16_1k": "C3.recall16-1k", "decoy8_1k": "C2.decoy8-1k",
    "unique_1k": "C6.unique-1k", "match3_1k": "X.match3-1k", "chain8_1k": "C4.chain8-1k",
    "pchain2_1k": "X.pchain2-1k-v1", "pchain3_1k": "X.pchain3-1k-v1", "pchain4_1k": "X.pchain4-1k-v1",
    "shuf2_1k": "X.shuffled2-1k", "shuf3_1k": "X.shuffled3-1k", "count_1k": "C6.count-1k",
    "majority_1k": "X.majority-1k",
}
NOT_FROM_SCRATCH_ARMS = ("loopft",)  # E33a arm fine-tuned from E31's lookup weights


def legacy_suite(cell: str) -> tuple[str | None, str]:
    task = SUITE_V3.get(cell)
    if task is None:
        return None, "unmapped"
    return task, "flawed" if task in FLAWED else "same"


def legacy_battery(variant: str, slot: str) -> tuple[str | None, str]:
    """A length-battery / study job (already parsed into variant + slot) → (v4 task, match)."""
    task = BATTERY_V1.get(slot)
    if task is None:
        return None, "unmapped"                      # e.g. curriculum intermediate stages (lookup_8k)
    if task in FLAWED:
        return task, "flawed"
    if any(variant.endswith("_" + a) for a in NOT_FROM_SCRATCH_ARMS):
        return task, "not-from-scratch"
    if slot in ("lookup_2k", "lookup_16k"):          # trained from random init (+ the C1.lookup-16k stages)
        return task, "settings-differ" if slot == "lookup_2k" else "same"
    if variant == "dense":                           # dense hard exams: from scratch at 1k, v1 settings
        return task, "settings-differ"
    return task, "curriculum-differs"                # E31b / E33a: started from the run's own lookup-2k weights


__all__ = ["TRAIN_1K", "TRAIN_1K_X4", "BATTERY_V1", "FLAWED", "LADDER", "LEVELS", "LEVEL_BY_ID", "PACKAGES", "PASS_ACC", "SEEDS",
           "SUITE_V3", "TASKS", "TASK_BY_ID", "VERSION", "Level", "Task", "legacy_battery", "legacy_suite"]
