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

ROOT_CKPT = "lookup_start"  # symlink → an E31 single-read lookup-2k run that learned lookup (len_lookup_e31_li_m1_s2, 99 %; s0 never did)
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


# ---- hop2_diag (2026-10-06): why the second hop is not learned ------------------------------------------------
# A linear probe on the 1-hop loop checkpoint: after round 1 the state names node 1 by its *first* letters (99 / 91 /
# 69 %, chance from letter ~8), while the question's start node is held by its *last* letters (the local window),
# so the memory is addressed by the tail of a source name and round 1 returns the head of the target name: the
# second lookup has nothing to match. Diagnostics (not capability evidence), all from random init on the C5 pchain2
# exam as written (hop count in the question, overhang, 4x budget), one knob changed each:
#   one-token names (64 symbols, key_len 1)  — no head/tail mismatch, the decision is the whole answer
#   4-letter names                           — next to tracking's dense k4 16x run
#   written path (n1 then n2), single read   — two 1-hop lookups, the second queried by the node just written
#   → then the loop takes over the written step: final-only answers from the path weights (Coconut-style stage)
# Loop arms halve the micro-batch (same effective batch). Every job writes a flow log.
ONE_TOKEN = ["--n_symbols", "64", "--key_len", "1"]
LOOP_MICRO = ["--grad_accum", "2"]


def _diag(name, arch, extra, *, init=None, cost=2.0):
    from scripts.study_plans.e30_vs_e31 import task_job

    j = task_job("C5.pchain2-1k", arch, 0, prefix="diag33")
    j.update(name=name, init=init, cost=cost, save=True)
    j["args"] = j["args"] + extra + ["--flow_log", "@out"]
    return j


def hop2_diag_jobs() -> list[dict]:
    loop = [*E33A_LOOP, *LOOP_MICRO]
    path = "diag33_path_r1_s0"
    return [
        _diag("diag33_tok1_loop_s0", "e31_li_m1", [*ONE_TOKEN, *loop], cost=6.4),
        _diag("diag33_tok1_r1_s0", "e31_li_m1", [*ONE_TOKEN, "--loop_rounds", "1"], cost=2.2),
        _diag("diag33_tok1_dense_s0", "dense", ONE_TOKEN, cost=1.0),
        _diag(path, "e31_li_m1", ["--chain_answer_path", "--loop_rounds", "1"], cost=2.4),
        # the loop replaces the written step: same exam, final node only, from the path-trained weights, 1x budget
        _diag("diag33_path_to_loop_s0", "e31_li_m1", [*loop, "--steps", "1200"], init=path, cost=1.8),
        _diag("diag33_k4_loop_s0", "e31_li_m1", ["--key_len", "4", *loop], cost=6.4),
    ]


def jobs(phase: str) -> list[dict]:
    if phase == "hops":
        return [j for tag, seed, flags in ARMS for j in _arm(tag, seed, flags)]
    if phase == "hop2_diag":
        return hop2_diag_jobs()
    raise SystemExit(f"unknown phase {phase!r} (hops, hop2_diag)")
