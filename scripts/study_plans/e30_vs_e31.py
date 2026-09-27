"""E30 vs E31 limits study (2026-09-27). Spec: docs/experiments_specs/ahead/E31b_e30_vs_e31_limits.md

Phases (see the spec for the questions each answers):
  ratio  — tokens per latent / per reader entry, 6 → 64, E31 and E30, lookup-1k and recall8-1k
  length — lookup 2k → 8k → 16k and chain (from lookup weights) 2k → 8k, ladder to 128k
  hard   — harder / multi-hop exams at 1k from each arch's lookup-2k weights, ladder 1k → 128k
  dense  — dense ceiling on the hard exams at 1k (training length only)
  odra / polonez — the host split used for the first wave (seed 1 on Odra, seed 0 on Polonez)

All arms share the E31 platform (30M, H 960, 4 layers, 128-d token embedding, no n-grams)
and the windowed global read (`--message_raw_window 256`: the memory is the only long path).
"""
from __future__ import annotations

BASE = [
    "--scale", "bridge_1k", "--no-skip_uncalibrated", "--no-dense_first",
    "--hidden", "960", "--head_dim", "64", "--kv_heads", "1",
    "--pre_layers", "1", "--global_layers", "1", "--stack_layers", "2", "--max_params", "40000000",
    "--warm_residuals", "--eval_every", "100", "--amp", "auto",
    "--swp_n_heads", "8", "--swp_query_dim", "128", "--token_embedding_dim", "128", "--ngram_orders", "none",
    "--message_raw_window", "256",
]

LADDER_FULL = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
LADDER_1K = [1024, 2048, 4096, 8192, 16384]
LADDER_1K_FULL = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Training length settings (the suite's 30m policy; 2k needs step 5e-5).
L1K = ["--seq_len", "1024", "--steps", "1200", "--k1_mult", "4", "--batch", "32", "--grad_accum", "1",
       "--eval_rows", "256", "--lr", "1e-4"]
L2K = ["--seq_len", "2048", "--steps", "1200", "--k1_mult", "6", "--batch", "32", "--grad_accum", "2",
       "--eval_rows", "256", "--lr", "5e-5"]
# Curriculum stages as in the E31 ladder runs (sdpa fits to 16k; flex is needed from 32k).
L8K = ["--seq_len", "8192", "--steps", "1500", "--k1_mult", "1", "--batch", "16", "--grad_accum", "4",
       "--eval_rows", "128", "--lr", "5e-5"]
L16K = ["--seq_len", "16384", "--steps", "1000", "--k1_mult", "1", "--batch", "8", "--grad_accum", "4",
        "--eval_rows", "128", "--lr", "5e-5"]
# Fine-tuning from lookup weights at 1k (content addressing already learned).
L1K_FT = ["--seq_len", "1024", "--steps", "1200", "--k1_mult", "4", "--batch", "32", "--grad_accum", "1",
          "--eval_rows", "256", "--lr", "5e-5"]

EXAMS = {
    "lookup": ["--recipe", "recall_single"],
    "recall8": ["--recipe", "recall", "--n_distractors", "7", "--key_len", "4"],
    "chain": ["--recipe", "chain_ordered", "--hops", "4"],
    # hard / multi-hop
    "recall16": ["--recipe", "recall", "--n_distractors", "15", "--key_len", "4"],
    "decoy8": ["--recipe", "select", "--n_decoys", "8", "--n_distractors", "0"],
    "shuf2": ["--recipe", "chain", "--hops", "2", "--n_distractors", "0"],
    "shuf3": ["--recipe", "chain", "--hops", "3", "--n_distractors", "0"],
    "chain8": ["--recipe", "chain_ordered", "--hops", "8"],
    # shuffled chain among 3 decoy chains: no pure-target shortcut (shuf2/shuf3 have one)
    "pchain2": ["--recipe", "chain_parallel", "--hops", "2"],
    "pchain3": ["--recipe", "chain_parallel", "--hops", "3"],
    "unique": ["--recipe", "unique"],
    "match3": ["--recipe", "match3"],
    "count": ["--recipe", "count"],
    "majority": ["--recipe", "majority"],
}
HARD = ("recall8", "recall16", "decoy8", "shuf2", "shuf3", "chain8", "unique", "match3", "count", "majority")

LI_ARCHES = ("e30_li", "e31_li", "e31_li_m1")

# Ratio arms: (name, arch, flags). tok/latent = stride / K; tok/entry = stride / (K·m) (E30: m = 1).
RATIO_ARMS = [
    # E31: tokens per latent at one reader entry per latent (the memory size E30 has at K32)
    ("e31_K32m1", "e31_li", ["--lm_latents", "32", "--lm_reader_tokens", "1"]),               # 6 / 6
    ("e31_K16m1", "e31_li", ["--lm_latents", "16", "--lm_reader_tokens", "1"]),               # 12 / 12
    ("e31_K8m1", "e31_li", ["--lm_latents", "8", "--lm_reader_tokens", "1"]),                 # 24 / 24
    ("e31_K4m1", "e31_li", ["--lm_latents", "4", "--lm_reader_tokens", "1"]),                 # 48 / 48
    ("e31_K4m1_s256", "e31_li", ["--lm_latents", "4", "--lm_reader_tokens", "1", "--lm_stride", "256"]),  # 64 / 64
    ("e31_W512K8m1", "e31_li", ["--lm_window", "512", "--lm_stride", "384", "--lm_latents", "8",
                                "--lm_reader_tokens", "1"]),                                  # 48 / 48, long window
    # E31: write compression vs read bandwidth (same tokens per latent, more reader entries)
    ("e31_K8m4", "e31_li", ["--lm_latents", "8", "--lm_reader_tokens", "4"]),                 # 24 / 6
    ("e31_K4m8", "e31_li", ["--lm_latents", "4", "--lm_reader_tokens", "8"]),                 # 48 / 6
    ("e31_W512K8m8", "e31_li", ["--lm_window", "512", "--lm_stride", "384", "--lm_latents", "8",
                                "--lm_reader_tokens", "8"]),                                  # 48 / 6, long window
    ("e31_K32m5", "e31_li", ["--lm_latents", "32", "--lm_reader_tokens", "5"]),               # 6 / 1.2 (E31 default)
    # E30: tokens per slot (window 256 pinned, no auto-fit)
    ("e30_K32", "e30_li", ["--swp_bank_size", "32", "--swp_window", "256", "--swp_stride", "192", "--no-swp_auto_fit"]),
    ("e30_K16", "e30_li", ["--swp_bank_size", "16", "--swp_window", "256", "--swp_stride", "192", "--no-swp_auto_fit"]),
    ("e30_K8", "e30_li", ["--swp_bank_size", "8", "--swp_window", "256", "--swp_stride", "192", "--no-swp_auto_fit"]),
    ("e30_K4", "e30_li", ["--swp_bank_size", "4", "--swp_window", "256", "--swp_stride", "192", "--no-swp_auto_fit"]),
    ("e30_K4_s256", "e30_li", ["--swp_bank_size", "4", "--swp_window", "256", "--swp_stride", "256", "--no-swp_auto_fit"]),
    ("e30_W512K8", "e30_li", ["--swp_bank_size", "8", "--swp_window", "512", "--swp_stride", "384", "--no-swp_auto_fit"]),
]

# Estimated GPU-hours (0.3–1 s/step, one extension allowed)
COST = {"1k": 0.7, "2k": 3.5, "8k": 1.8, "16k": 2.2, "ladder": 0.3}


def _job(name, arch, seed, length, exam, extra=(), *, init=None, ladder=None, cost=None):
    return {
        "name": name,
        "args": [*length, *EXAMS[exam], "--arch", arch, "--seed", str(seed), *extra],
        "init": init,
        "ladder": ladder,
        "cost": cost,
    }


def ratio_jobs(seeds=(0,)):
    out = []
    for seed in seeds:
        for exam in ("lookup", "recall8"):
            for name, arch, flags in RATIO_ARMS:
                out.append(_job(f"ratio_{exam}_{name}_s{seed}", arch, seed, L1K, exam, flags,
                                ladder=LADDER_1K, cost=COST["1k"] + 0.1))
        out.append(_job(f"ratio_recall8_dense_s{seed}", "dense", seed, L1K, "recall8", cost=0.4))
    return out


def length_jobs(arches=("e30_li", "e31_li_m1"), seeds=(0, 1), chain_arches=("e30_li", "e31_li_m1", "e31_li")):
    out = []
    for arch in sorted(set(arches) | set(chain_arches)):
        for seed in seeds:
            base = f"len_lookup_{arch}_s{seed}"
            if arch in arches:
                out += [
                    _job(base, arch, seed, L2K, "lookup", ladder=LADDER_FULL, cost=COST["2k"] + COST["ladder"]),
                    _job(f"{base}_b8k", arch, seed, L8K, "lookup", init=base, ladder=LADDER_FULL,
                         cost=COST["8k"] + COST["ladder"]),
                    _job(f"{base}_b16k", arch, seed, L16K, "lookup", init=f"{base}_b8k", ladder=LADDER_FULL,
                         cost=COST["16k"] + COST["ladder"]),
                ]
            if arch in chain_arches:
                ch = f"len_chain_{arch}_s{seed}"
                out += [
                    _job(ch, arch, seed, L2K, "chain", init=base, ladder=LADDER_FULL, cost=COST["2k"] + COST["ladder"]),
                    _job(f"{ch}_b8k", arch, seed, L8K, "chain", init=ch, ladder=LADDER_FULL,
                         cost=COST["8k"] + COST["ladder"]),
                ]
    return out


def hard_jobs(arches=LI_ARCHES, seeds=(0,), exams=HARD):
    out = []
    for arch in arches:
        for seed in seeds:
            for exam in exams:
                out.append(_job(f"hard_{exam}_{arch}_s{seed}", arch, seed, L1K_FT, exam,
                                init=f"len_lookup_{arch}_s{seed}", ladder=LADDER_1K_FULL,
                                cost=COST["1k"] + COST["ladder"]))
    return out


def dense_jobs(seeds=(0,), exams=HARD):
    return [_job(f"dense_{exam}_s{seed}", "dense", seed, L1K, exam, ladder=[1024, 2048], cost=0.4)
            for seed in seeds for exam in exams]


def jobs(phase: str) -> list[dict]:
    if phase == "ratio":
        return ratio_jobs()
    if phase == "length":
        return length_jobs()
    if phase == "hard":
        return hard_jobs()
    if phase == "dense":
        return dense_jobs()
    if phase == "pchain_s0":  # parallel-chain exams, seed 0 (Polonez, after its first queue)
        return hard_jobs(seeds=(0,), exams=("pchain2", "pchain3")) + dense_jobs(exams=("pchain2", "pchain3"))
    if phase == "odra":  # seed 1: E31_li lookup/chain weights already live here
        return ratio_jobs() + length_jobs(seeds=(1,)) + hard_jobs(seeds=(1,))
    if phase == "polonez":  # seed 0 (E31_li seed-0 lookup weights live here) + dense ceiling
        return length_jobs(seeds=(0,)) + dense_jobs()
    raise SystemExit(f"unknown phase {phase!r}")
