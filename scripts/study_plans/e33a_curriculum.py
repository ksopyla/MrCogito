"""E33a hop curriculum (2026-10-05): does the read–think–reread loop learn hop 2 once hop 1 exists?

Diagnosis: docs/4_Research_Notes/e33a_loop_diagnosis_20261004.md. Every E33a arm sat on the link-target guess
because none learned the first hop. Here every arm starts from the same E31 single-read lookup weights (the step
where E31 has learned content addressing before: recall16 88 % from them) and climbs the guess-proof parallel
chains (one edge of overhang, 16-letter nodes, scored on the picked candidate):

    edge (1 hop) → pchain2 → pchain3        per arm, each stage from the previous stage's weights

Arms (e31_li_m1, same start, same data per seed): the loop with per-round progress exits, the loop with answer
exits, the single read (1 round), and a second seed of the progress loop. No lookup replay: the question is the
mechanism, not no-harm. A mechanism test, not a capability claim (not from scratch). The dense ceiling for this
schedule is the tracking session's `calibrate_reasoning` phase.

    uv run python scripts/run_study_queue.py --plan e33a_curriculum --phase hops --out Cache/study/e33a_curriculum \\
        --host polonez --gpus 0 1 2 3 --mode scripts
(the start checkpoint is linked into the out folder as `lookup_start`, see ROOT_CKPT)
"""
from __future__ import annotations

from scripts.study_plans.e30_vs_e31 import BASE, E33A_L1K, E33A_LOOP, E33A_LOOP_ANS  # noqa: F401

ROOT_CKPT = "lookup_start"  # symlink → the E31 single-read lookup-2k run (len_lookup_e31_li_m1_s0)
CHAIN = ["--chain_overhang", "1", "--key_len", "16"]
STAGES = (("edge", 1), ("pchain2", 2), ("pchain3", 3))
ARMS = (
    ("loop", 1, E33A_LOOP),
    ("loopans", 1, E33A_LOOP_ANS),
    ("r1", 1, ["--loop_rounds", "1"]),
    ("loop", 2, E33A_LOOP),
)
EXAMS = {name: ["--recipe", "chain_parallel", "--hops", str(h)] for name, h in STAGES}


def _arm(tag: str, seed: int, flags: list[str]) -> list[dict]:
    out, prev = [], ROOT_CKPT
    for exam, _h in STAGES:
        name = f"e33cur_{exam}_{tag}_s{seed}"
        out.append({"name": name, "args": [*E33A_L1K, *EXAMS[exam], *CHAIN, "--arch", "e31_li_m1", "--seed", str(seed),
                                           *flags],
                    "init": prev, "ladder": None, "cost": 1.6, "save": True})
        prev = name
    return out


def jobs(phase: str) -> list[dict]:
    if phase == "hops":
        return [j for tag, seed, flags in ARMS for j in _arm(tag, seed, flags)]
    raise SystemExit(f"unknown phase {phase!r} (hops)")
