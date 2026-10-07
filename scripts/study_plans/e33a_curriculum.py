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


def _diag(name, arch, extra, *, init=None, cost=2.0, seed=0):
    from scripts.study_plans.e30_vs_e31 import task_job

    j = task_job("C5.pchain2-1k", arch, seed, prefix="diag33")
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


# ---- hop2_name (2026-10-06): the whole-name loss --------------------------------------------------------------
# hop2_diag: with the bridge written (n1 then n2) the single read finds n2 at 95 %; asked for n2 only, the loop's
# round 1 still finds n1 (first letter 100 %) but round 2 stays at the floor. The whole-name loss asks every loop
# state to name its node completely at the decision position, so round 2 has the tail to query by. Starts (linked
# into the out folder): `diag33_path_r1_s0` (hop2_diag, written path) and `e33cur_edge_loop_s1` (loop, 1 hop).
NAME = ["--loop_name_aux", "0.5"]


def hop2_name_jobs() -> list[dict]:
    loop = [*E33A_LOOP, *LOOP_MICRO, *NAME, "--steps", "1200"]
    return [
        _diag("diag33_path_to_loop_name_s0", "e31_li_m1", loop, init="diag33_path_r1_s0", cost=1.8),
        _diag("diag33_edge_to_pchain2_name_mix_s0", "e31_li_m1",
              [*loop, "--replay_recipe", "chain_parallel", "--replay_hops", "1", "--replay_frac", "0.5"],
              init="e33cur_edge_loop_s1", cost=1.8),
    ]


# ---- hop2_short (2026-10-07): names short enough for one read --------------------------------------------------
# hop2_name: the whole-name loss lifted node 1's letters 2–4 in the round-1 state (54 / 90 / 48 → 97 / 96 / 79 %) but
# letters 5–16 stay at chance: one read at the decision position returns ~4 letters (the memory is otherwise read a
# few letters per answer position). 2 hops rose from 10.9 to 17.6 % (floor 8.3). So: names of 4 and 8 letters,
# with the 1-hop stage first (direct 2-hop from scratch never learned hop 1), then 2 hops with the whole-name loss
# and half the rows at 1 hop, 4x budget. Start: the E31 single-read lookup-2k weights that learned lookup (Odra
# `len_lookup_e31_li_m1_s1`, 98 %), linked into the out folder as `lookup_start`.
def hop2_short_jobs() -> list[dict]:
    loop = [*E33A_LOOP, *LOOP_MICRO]
    mix = ["--replay_recipe", "chain_parallel", "--replay_hops", "1", "--replay_frac", "0.5"]
    out = []
    for k in (4, 8):
        edge = f"diag33_edge_k{k}_loop_s1"
        out += [
            _diag(edge, "e31_li_m1", ["--key_len", str(k), "--hops", "1", *loop, "--steps", "1200"],
                  init=ROOT_CKPT, cost=1.6, seed=1),
            _diag(f"diag33_pchain2_k{k}_name_mix_s1", "e31_li_m1", ["--key_len", str(k), *loop, *NAME, *mix],
                  init=edge, cost=6.0, seed=1),
        ]
    return out


# ---- hop2_ablate (2026-10-07): what made the second hop work -------------------------------------------------
# hop2_short: the loop learned 2 hops with 4- and 8-letter names (picked 97.8 / 97.7 %, floor 8.3; round 1 → n1
# 99–100 %, round 2 → n2 98–99 %; a sudden jump at ~0.7k / ~1.2k steps). Same recipe (1-hop stage → 2 hops with
# half the rows at 1 hop, 8-letter names, seed 1, from the E31 lookup weights), one ingredient removed each:
#   single read (no loop) · loop without the whole-name loss; then the reach: 3 hops from the 2-hop loop, and
#   16-letter names through the same recipe (the earlier 16-letter runs had a different 1-hop stage).
def hop2_ablate_jobs() -> list[dict]:
    loop = [*E33A_LOOP, *LOOP_MICRO]
    mix1 = ["--replay_recipe", "chain_parallel", "--replay_hops", "1", "--replay_frac", "0.5"]
    k8 = ["--key_len", "8"]
    edge_r1 = "diag33_edge_k8_r1_s1"
    edge16 = "diag33_edge_k16_loop_s1"
    return [
        _diag(edge_r1, "e31_li_m1", [*k8, "--hops", "1", "--loop_rounds", "1", "--steps", "1200"], init=ROOT_CKPT,
              cost=0.6, seed=1),
        _diag("diag33_pchain2_k8_r1_mix_s1", "e31_li_m1", [*k8, "--loop_rounds", "1", *mix1], init=edge_r1,
              cost=2.2, seed=1),
        _diag("diag33_pchain2_k8_noname_mix_s1", "e31_li_m1", [*k8, *loop, *mix1], init="diag33_edge_k8_loop_s1",
              cost=2.0, seed=1),
        _diag("diag33_pchain3_k8_name_mix_s1", "e31_li_m1",
              [*k8, "--hops", "3", *loop, *NAME, "--replay_recipe", "chain_parallel", "--replay_hops", "2",
               "--replay_frac", "0.5"], init="diag33_pchain2_k8_name_mix_s1", cost=3.0, seed=1),
        _diag(edge16, "e31_li_m1", ["--hops", "1", *loop, "--steps", "1200"], init=ROOT_CKPT, cost=1.6, seed=1),
        _diag("diag33_pchain2_k16_name_mix_s1", "e31_li_m1", [*loop, *NAME, *mix1], init=edge16, cost=4.0, seed=1),
    ]


# Recipe A (final answer only, tracking 2026-10-07): progress exits and the name loss both use the path, which dense
# and the single read do not get. The fair loop arm: exits trained on the answer, no name loss, same schedule.
def hop2_recipe_a_jobs() -> list[dict]:
    loop_a = ["--loop_rounds", "4", "--loop_exit_aux", "0.3", "--loop_exit_targets", "answer", *LOOP_MICRO]
    mix1 = ["--replay_recipe", "chain_parallel", "--replay_hops", "1", "--replay_frac", "0.5"]
    return [_diag("diag33_pchain2_k8_ansexit_mix_s1", "e31_li_m1", ["--key_len", "8", *loop_a, *mix1],
                  init="diag33_edge_k8_loop_s1", cost=2.0, seed=1)]


# ---- v4_path2 (2026-10-07): the frozen C5.path2-1k exam, E31 and E33 from random init -----------------------------
# C5.path2-1k is active (dense seeds 0–2 pass). Exactly the written recipe for every model: 1 hop at TRAIN_1K_X4, then
# 2 hops at the task's TRAIN_1K_X16 with half the rows at 1 hop. E33's loop exits are trained on the answer (here the
# written path), no per-round node targets (recipe B as written). Board names: v4_C5_path2-1k_h{k}_<variant>_s<seed>.
PATH2_ARMS = (("e31_li_m1", []),
              ("e33a_loop", ["--loop_rounds", "4", "--loop_exit_aux", "0.3", "--loop_exit_targets", "answer",
                             *LOOP_MICRO]))


def v4_path2_jobs(seeds=(0, 1, 2)) -> list[dict]:
    from evaluation.capability_tasks import TRAIN_1K_X4
    from scripts.study_plans.e30_vs_e31 import task_job

    out = []
    for variant, flags in PATH2_ARMS:
        for seed in seeds:
            prev = None
            for k in (1, 2):
                j = task_job("C5.path2-1k", "e31_li_m1", seed, prefix="v4")
                j["name"] = f"v4_C5_path2-1k_h{k}_{variant}_s{seed}"
                j["args"] = j["args"] + ["--hops", str(k)] + (list(TRAIN_1K_X4) if k == 1 else
                                                             ["--replay_recipe", "chain_parallel", "--replay_hops",
                                                              "1", "--replay_frac", "0.5"]) + flags
                j.update(init=prev, save=True, cost=(1.5 if variant == "e31_li_m1" else 4.0))
                prev = j["name"]
                out.append(j)
    return out


# ---- recipe_a_scratch (2026-10-07): the loop from random init on recipe A ---------------------------------------
# Recipe A (C5.pchain2, final answer only, no intermediate targets): the 4-layer dense (12.1 %) and the 8-layer dense
# (15.2 %, floor 8.3) fail it from random init; the loop learned it from E31 lookup weights (94.5 %). Same schedule as
# the 8-layer dense run (calibrate_c5_deep_a): 8-letter names, 1 hop at TRAIN_1K_X4 → 2 hops at 19.2k steps with half
# the rows at 1 hop. Loop exits trained on the answer. The E31 single read on the same schedule is the no-loop control.
def recipe_a_scratch_jobs() -> list[dict]:
    from evaluation.capability_tasks import TRAIN_1K_X4
    from scripts.study_plans.e30_vs_e31 import task_job

    arms = [("e33a_loop", s, ["--loop_rounds", "4", "--loop_exit_aux", "0.3", "--loop_exit_targets", "answer",
                              *LOOP_MICRO]) for s in (0, 1, 2)]
    arms.append(("e31_li_m1", 0, []))
    out = []
    for variant, seed, flags in arms:
        h1 = f"recA_C5_pchain2-1k_h1_{variant}_s{seed}"
        j1 = task_job("C5.pchain2-1k", "e31_li_m1", seed, prefix="recA")
        j1.update(name=h1, save=True, cost=2.5 if variant == "e33a_loop" else 1.0,
                  args=j1["args"] + ["--key_len", "8", "--hops", "1", *TRAIN_1K_X4, *flags, "--flow_log", "@out"])
        j2 = task_job("C5.pchain2-1k", "e31_li_m1", seed, prefix="recA")
        j2.update(name=f"recA_C5_pchain2-1k_h2_{variant}_s{seed}", init=h1, save=True,
                  cost=3.5 if variant == "e33a_loop" else 1.5,
                  args=j2["args"] + ["--key_len", "8", "--steps", "19200", "--k1_mult", "1", "--replay_recipe",
                                     "chain_parallel", "--replay_hops", "1", "--replay_frac", "0.5", *flags,
                                     "--flow_log", "@out"])
        out += [j1, j2]
    return out


def jobs(phase: str) -> list[dict]:
    if phase == "hops":
        return [j for tag, seed, flags in ARMS for j in _arm(tag, seed, flags)]
    if phase == "hop2_diag":
        return hop2_diag_jobs()
    if phase == "hop2_name":
        return hop2_name_jobs()
    if phase == "hop2_short":
        return hop2_short_jobs()
    if phase == "hop2_ablate":
        return hop2_ablate_jobs()
    if phase == "hop2_recipe_a":
        return hop2_recipe_a_jobs()
    if phase == "v4_path2":
        return v4_path2_jobs()
    if phase == "recipe_a_scratch":
        return recipe_a_scratch_jobs()
    raise SystemExit(f"unknown phase {phase!r} (hops, hop2_diag, hop2_name, hop2_short, hop2_ablate)")
