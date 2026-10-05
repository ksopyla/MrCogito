"""E30 vs E31 limits study (2026-09-27). Spec: docs/experiments_specs/ahead/E31b_e30_vs_e31_limits.md

Phases (see the spec for the questions each answers):
  ratio  — tokens per latent / per reader entry, 6 → 64, E31 and E30, lookup-1k and recall8-1k
  length — lookup 2k → 8k → 16k and chain (from lookup weights) 2k → 8k, ladder to 128k
  hard   — harder / multi-hop exams at 1k from each arch's lookup-2k weights, ladder 1k → 128k
  dense  — dense ceiling on the hard exams at 1k (training length only)
  odra / polonez — the host split used for the first wave (seed 1 on Odra, seed 0 on Polonez)
  battery_<tag> — the standard length & hard battery (the E31b protocol as one function,
           `battery_jobs`) for a variant registered in BATTERY_VARIANTS; every new variant runs it

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
    "pchain4": ["--recipe", "chain_parallel", "--hops", "4"],
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

# Wave 2: seeds at the ratio boundary + hyperparameters at 24 and 48 tokens per latent
RATIO2_SEEDED = [a for a in RATIO_ARMS if a[0] in ("e31_K16m1", "e31_K8m1", "e30_K16", "e30_K8")]
_K4 = ["--lm_latents", "4", "--lm_reader_tokens", "1"]
_K8 = ["--lm_latents", "8", "--lm_reader_tokens", "1"]
RATIO2_HP = [
    # 48 tokens per latent (the E32 coarse regime)
    ("e31_K4m1_D1024", "e31_li", [*_K4, "--lm_latent_dim", "1024", "--max_params", "45000000"]),
    ("e31_K4m1_r3", "e31_li", [*_K4, "--lm_rounds", "3"]),
    ("e31_K4m1_enc3", "e31_li", [*_K4, "--lm_enc_layers", "3"]),
    ("e31_K4m1_nocomp", "e31_li", [*_K4, "--no-lm_competition"]),
    ("e31_K4m1_h16", "e31_li", [*_K4, "--lm_heads", "16"]),
    ("e31_W128K2m1", "e31_li", ["--lm_window", "128", "--lm_stride", "96", "--lm_latents", "2",
                                "--lm_reader_tokens", "1"]),                     # 48, short window
    ("e31_K4m8_D1024", "e31_li", ["--lm_latents", "4", "--lm_reader_tokens", "8", "--lm_latent_dim", "1024", "--max_params", "45000000"]),
    # 24 tokens per latent
    ("e31_K8m1_D1024", "e31_li", [*_K8, "--lm_latent_dim", "1024", "--max_params", "45000000"]),
    ("e31_K8m1_nocomp", "e31_li", [*_K8, "--no-lm_competition"]),
    ("e31_K8m1_r3", "e31_li", [*_K8, "--lm_rounds", "3"]),
]


# E33a: tied loop over [global read + local] × 4, exits through the untied answer layer.
E33A_LOOP = ["--loop_rounds", "4", "--loop_exit_aux", "0.3", "--loop_exit_targets", "progress"]  # phase e33a (as launched)
# General exits: every loop predicts the answer (= the next token in text). No synthetic-only labels, so the
# E31-protocol checks and the suite stay fair to E31 and the recipe carries over to text training.
E33A_LOOP_ANS = ["--loop_rounds", "4", "--loop_exit_aux", "0.3", "--loop_exit_targets", "answer"]
E33A_CHAIN = ["--key_len", "8", "--replay_recipe", "recall_single", "--replay_frac", "0.25"]
E33A_PCHAIN_LADDER = [1024, 4096, 16384]
# Same effective batch as the study (32 rows); 4 loops + 3 exits need smaller micro-batches on a 3090
# (Odra smoke 3 Oct: 1k at micro 16 = 15.0 GB, 1.1 s/step; 2k at micro 8 = 15.4 GB, 3.0 s/step).
E33A_L1K = [*L1K_FT[:L1K_FT.index("--grad_accum") + 1], "2", *L1K_FT[L1K_FT.index("--grad_accum") + 2:]]
E33A_L2K = [*L2K[:L2K.index("--grad_accum") + 1], "4", *L2K[L2K.index("--grad_accum") + 2:]]
E33A_LOOKUP_NO_HARM = [{"lengths": [1024, 2048, 8192, 32768], "args": ["--recipe", "recall_single"],
                        "out": "ladder_lookup.json", "rows": 128}]


def _e33a_chain(tag, init, flags, seed, *, arch="e31_li_m1"):
    """pchain2 → pchain3 → pchain4 at 1k from `init`; lookup no-harm ladder on the last stage."""
    out, prev = [], init
    for hops in (2, 3, 4):
        name = f"e33a_pchain{hops}_{tag}_s{seed}"
        out.append({**_job(name, arch, seed, E33A_L1K, f"pchain{hops}", [*E33A_CHAIN, *flags],
                           init=prev, ladder=E33A_PCHAIN_LADDER, cost=2.2),
                    "rows": 128, "ladders": E33A_LOOKUP_NO_HARM if hops == 4 else None})
        prev = name
    return out


def e33a_jobs():
    arch = "e31_li_m1"
    out = []
    # ★ the loop from step 0: lookup at 2k with the loop on, then the chain curriculum (seeds 1, 2)
    for seed in (1, 2):
        lk = f"e33a_lookup_loop_s{seed}"
        out.append(_job(lk, arch, seed, E33A_L2K, "lookup", E33A_LOOP, ladder=[2048, 8192, 32768], cost=6.0))
        out += _e33a_chain("loop", lk, E33A_LOOP, seed)
    lk1 = "e33a_lookup_loop_s1"
    # the same loop fine-tuned from the past single-read checkpoint (does learning the loop from the start matter?)
    out += _e33a_chain("loopft", f"len_lookup_{arch}_s1", E33A_LOOP, 1)
    # single-read control from the past checkpoint, same replay
    out += _e33a_chain("r1", f"len_lookup_{arch}_s1", ["--loop_rounds", "1"], 1)
    # mechanism arms, branching from the loop-trained lookup weights (seed 1)
    loop_ans = [*E33A_LOOP[:-1], "answer"]
    out += _e33a_chain("loopans", lk1, loop_ans, 1)
    out += _e33a_chain("loopfrz", lk1, [*E33A_LOOP, "--freeze_writer"], 1)
    out += _e33a_chain("loopinj", lk1, [*E33A_LOOP, "--loop_inject", "prelude"], 1)
    # wide core (read + both local layers, head only after) needs its own loop-trained lookup stage
    wide = [*E33A_LOOP, "--loop_span", "2"]
    out.append(_job("e33a_lookup_wide_s1", arch, 1, E33A_L2K, "lookup", wide, ladder=[2048, 8192, 32768], cost=6.0))
    out += _e33a_chain("wide", "e33a_lookup_wide_s1", wide, 1)
    return out


def _accum(settings, mult):
    i = settings.index("--grad_accum") + 1
    return [*settings[:i], str(int(settings[i]) * mult), *settings[i + 1:]]


# The length & hard battery: the E31b protocol as one reusable function. Every new variant runs it
# (process: docs/engineering_specs/capability_checks.md, skill `capability-checks`), so its ladders
# compare like for like with `e31_li_m1`. Scored on first letter, 128 rows per ladder length.
BATTERY_VERSION = "2026-09-27.v1"  # bump on any change to an exam, stage, ladder or row count
BATTERY_HARD = ("recall8", "recall16", "decoy8", "unique", "match3", "chain8")
BATTERY_RECALL_8K = ("recall8", "recall16")


def battery_jobs(tag, arm, arch, seeds=(1, 2), flags=(), *, l1k=L1K_FT, l2k=L2K, accum=1, train_root=True,
                 rows=128):
    """The E31b protocol for one variant, per seed, all from the variant's own lookup-2k weights:
      root     `{tag}_lookup_{arm}_s{seed}` — lookup at 2k from scratch (skip with train_root=False when
               an earlier phase already trained it under that name);
      length   lookup 8k → 16k stages and the in-order 4-hop chain at 2k → 8k, each laddered 2k → 128k;
      hard     recall8, recall16, decoy8, unique, match3, chain8 at 1k, laddered 1k → 128k;
      recall   the 8k stage for recall8 / recall16, laddered 2k → 128k.
    `accum` multiplies only the 8k/16k micro-batch split (memory, same effective batch). Job names
    follow `{tag}_{exam}_{arm}_s{seed}` so the ledger can group them by variant."""
    out = []
    for seed in seeds:
        lk = f"{tag}_lookup_{arm}_s{seed}"
        if train_root:
            out.append({**_job(lk, arch, seed, l2k, "lookup", flags, ladder=LADDER_FULL, cost=6.0), "rows": rows})
        ch = f"{tag}_chain_{arm}_s{seed}"
        out += [
            {**_job(f"{lk}_b8k", arch, seed, _accum(L8K, accum), "lookup", flags, init=lk,
                    ladder=LADDER_FULL, cost=3.8), "rows": rows},
            {**_job(f"{lk}_b16k", arch, seed, _accum(L16K, accum), "lookup", flags, init=f"{lk}_b8k",
                    ladder=LADDER_FULL, cost=4.5), "rows": rows},
            {**_job(ch, arch, seed, l2k, "chain", flags, init=lk, ladder=LADDER_FULL, cost=6.5), "rows": rows},
            {**_job(f"{ch}_b8k", arch, seed, _accum(L8K, accum), "chain", flags, init=ch,
                    ladder=LADDER_FULL, cost=3.8), "rows": rows},
        ]
        for exam in BATTERY_HARD:
            name = f"{tag}_hard_{exam}_{arm}_s{seed}"
            out.append({**_job(name, arch, seed, l1k, exam, flags, init=lk, ladder=LADDER_1K_FULL, cost=2.2),
                        "rows": rows})
            if exam in BATTERY_RECALL_8K:
                out.append({**_job(f"{name}_b8k", arch, seed, _accum(L8K, accum), exam, flags, init=name,
                                   ladder=LADDER_FULL, cost=3.8), "rows": rows})
    return out


# One entry per variant that runs the battery: `--phase battery_<tag>` (kwargs of battery_jobs).
# Add a variant here — never copy the job list into a new function.
BATTERY_VARIANTS = {
    # E33a (loop from step 0): roots trained in phase `e33a`; micro-batches halved for 4 loops + exits
    # flags: general answer exits (every loop predicts the answer, = the next token in text), so the checks
    # stay fair to E31 and carry over to text; phase `e33a` itself was launched with progress exits
    "e33a": dict(tag="e33a", arm="loop", arch="e31_li_m1", flags=E33A_LOOP_ANS, l1k=E33A_L1K, l2k=E33A_L2K,
                 accum=2, train_root=False),
}


def e33a_e31b_jobs(seeds=(1, 2)):
    """The battery on E33a, from each seed's loop-trained lookup-2k weights. Same exams, steps and ladders
    as e31_li_m1 in E31b; micro-batches halved (memory only, same effective batch). Also queues the second
    seed of the general (answer-exit) reasoning arm."""
    return (_e33a_chain("loopans", "e33a_lookup_loop_s2", E33A_LOOP_ANS, 2)
            + battery_jobs(seeds=seeds, **BATTERY_VARIANTS["e33a"]))


def ratio2_jobs():
    out = []
    for exam in ("lookup", "recall8"):
        for seed in (1, 2):
            for name, arch, flags in RATIO2_SEEDED:
                out.append(_job(f"ratio_{exam}_{name}_s{seed}", arch, seed, L1K, exam, flags,
                                ladder=LADDER_1K, cost=COST["1k"] + 0.1))
        for name, arch, flags in RATIO2_HP:
            out.append(_job(f"ratio_{exam}_{name}_s0", arch, 0, L1K, exam, flags,
                            ladder=LADDER_1K, cost=COST["1k"] + 0.1))
    return out


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
    if phase == "wave2_s0":  # fair dense ceiling (from its own lookup-1k weights) + chain retry
        dl = "dense_lookup1k_s0"
        out = [_job(dl, "dense", 0, L1K, "lookup", ladder=[1024, 2048], cost=0.5)]
        out += [_job(f"dense_ft_{e}_s0", "dense", 0, L1K_FT, e, init=dl, ladder=[1024, 2048], cost=0.5)
                for e in ("recall8", "recall16", "decoy8", "pchain2", "pchain3", "chain8", "unique")]
        k12 = [a if a != "6" or L2K[i - 1] != "--k1_mult" else "12" for i, a in enumerate(L2K)]
        out.append({"name": "len_chain_e31_li_s0_k12", "args": [*k12, *EXAMS["chain"], "--arch", "e31_li",
                    "--seed", "0"], "init": "len_lookup_e31_li_s0", "ladder": LADDER_FULL, "cost": 7.0})
        return out
    if phase == "pchain_s1":  # parallel-chain exams, seed 1 (Odra, after its first queue)
        return hard_jobs(seeds=(1,), exams=("pchain2", "pchain3"))
    if phase == "ratio2":
        return ratio2_jobs()
    if phase == "ratio3":  # read bandwidth at 24 tokens per latent: K8 m4 seeds
        arms = [a for a in RATIO_ARMS if a[0] in ("e31_K8m4",)]
        return [_job(f"ratio_{e}_{n}_s{sd}", a, sd, L1K, e, f, ladder=LADDER_1K, cost=0.8)
                for e in ("lookup", "recall8") for sd in (1, 2) for n, a, f in arms]
    if phase == "hard_s0":  # replicate the writer split on seed 0 (Polonez)
        return hard_jobs(arches=("e30_li", "e31_li"), seeds=(0,),
                         exams=("recall8", "recall16", "decoy8", "unique"))
    if phase == "ord":  # order-preserving, length-invariant slot positions (in-order chain needs order)
        return length_jobs(arches=("e30_ord", "e31_ord_m1"), seeds=(1,), chain_arches=("e30_ord", "e31_ord_m1"))
    if phase == "e33":  # read → update → read loop on the parallel chain (lookup → pchain2 → pchain3)
        out = []
        for arch in ("e30_li", "e31_li_m1"):
            for R in (1, 4):
                rr = ["--message_read_rounds", str(R)]
                p2, p3 = f"e33_pchain2_{arch}_R{R}_s1", f"e33_pchain3_{arch}_R{R}_s1"
                out += [_job(p2, arch, 1, L1K_FT, "pchain2", rr, init=f"len_lookup_{arch}_s1",
                             ladder=LADDER_1K, cost=1.2),
                        _job(p3, arch, 1, L1K_FT, "pchain3", rr, init=p2, ladder=LADDER_1K, cost=1.2)]
        return out
    if phase == "e33b":  # the same loop test with 8-letter nodes (a node fits in 1–2 latents)
        out = []
        for arch in ("e30_li", "e31_li_m1"):
            for R in (1, 4):
                rr = ["--message_read_rounds", str(R), "--key_len", "8"]
                p2, p3 = f"e33b_pchain2_k8_{arch}_R{R}_s1", f"e33b_pchain3_k8_{arch}_R{R}_s1"
                out += [_job(p2, arch, 1, L1K_FT, "pchain2", rr, init=f"len_lookup_{arch}_s1",
                             ladder=LADDER_1K, cost=1.2),
                        _job(p3, arch, 1, L1K_FT, "pchain3", rr, init=p2, ladder=LADDER_1K, cost=1.2)]
        return out
    if phase == "ord_s0":  # replicate the order-preserving arms (e31_ord_m1 chain worked on seed 1)
        return length_jobs(arches=("e30_ord", "e31_ord_m1"), seeds=(0, 2), chain_arches=("e30_ord", "e31_ord_m1"))
    if phase == "mix":  # split slot-key RoPE: content at QUERY + order at a scaled distance
        return length_jobs(arches=("e30_mix", "e31_mix_m1"), seeds=(1,), chain_arches=("e30_mix", "e31_mix_m1"))
    if phase == "e33c":  # supervised Q-loop: read round r predicts chain node r+1 (8-letter nodes)
        out = []
        for arch in ("e30_li", "e31_li_m1"):
            rr = ["--message_read_rounds", "4", "--round_aux", "0.5", "--key_len", "8"]
            p2, p3 = f"e33c_pchain2_k8_{arch}_R4aux_s1", f"e33c_pchain3_k8_{arch}_R4aux_s1"
            out += [_job(p2, arch, 1, L1K_FT, "pchain2", rr, init=f"len_lookup_{arch}_s1", ladder=LADDER_1K, cost=1.2),
                    _job(p3, arch, 1, L1K_FT, "pchain3", rr, init=p2, ladder=LADDER_1K, cost=1.2)]
        return out
    if phase == "e33a":  # read–think–reread loop (spec E33a_reread_loop.md)
        return e33a_jobs()
    if phase == "e33a_e31b":  # the E31b protocol on the looped model (no capability lost; length to 128k)
        return e33a_e31b_jobs()
    if phase == "e33a_lookup16k_s0":  # third seed of C1.lookup-16k (capability checks v4 core: seeds 0–2)
        lk = "e33a_lookup_loop_s0"
        return [_job(lk, "e31_li_m1", 0, E33A_L2K, "lookup", E33A_LOOP_ANS, ladder=[2048, 8192, 32768], cost=6.0),
                *[j for j in e33a_e31b_jobs(seeds=(0,)) if j["name"].startswith(lk + "_b")]]
    if phase.startswith("battery_"):  # the standard length & hard battery for a registered variant
        tag = phase.removeprefix("battery_")
        if tag not in BATTERY_VARIANTS:
            raise SystemExit(f"no battery variant {tag!r}; add it to BATTERY_VARIANTS")
        return battery_jobs(**BATTERY_VARIANTS[tag])
    if phase == "recall_len":  # does the 8k stage carry multi-fact recall to length (as it did lookup)?
        out = []
        for arch in ("e30_li", "e31_li_m1", "e31_li"):
            for exam in ("recall8", "recall16"):
                src = f"hard_{exam}_{arch}_s1"
                out.append(_job(f"{src}_b8k", arch, 1, L8K, exam, init=src, ladder=LADDER_FULL, cost=2.2))
        return out
    if phase == "calibrate_reasoning":
        # Migration step 3 for the reasoning exams (v4-draft-2026-10-05): the guess-proof C5 parallel chains
        # (chain overhang, 16-letter nodes, scored on the picked candidate), their 1-hop stage C1.edge-1k, the
        # keyed lookup C1.keyed4-1k, and C6.count. Dense only, seed 0, from random init: the cheapest recipe
        # dense passes (>= 75 % on the task's own score) is frozen; then the champion runs it.
        x4 = [*L1K[:L1K.index("--steps") + 1], "4800", *L1K[L1K.index("--steps") + 2:]]  # 4x budget (k1 4 -> <= 19.2k)
        guessproof = ["--chain_overhang", "1", "--key_len", "16"]

        def lr(length, value):
            return [*length[:length.index("--lr") + 1], value]

        def pchain(h):
            return [*guessproof, "--hops", str(h)]

        return [
            # C1 keyed stages (the first exams where the key is needed)
            _job("cal_edge_dense_s0", "dense", 0, L1K, "pchain2", pchain(1), cost=0.5),
            _job("cal_keyed4_dense_s0", "dense", 0, L1K, "recall8", ["--n_distractors", "3"], cost=0.5),
            # C5.pchain2: v3 budget, 4x budget at two step sizes, and the written 1 hop -> 2 hops curriculum
            _job("cal_pchain2_dense_x1_s0", "dense", 0, L1K, "pchain2", pchain(2), cost=0.5),
            _job("cal_pchain2_dense_x4_s0", "dense", 0, x4, "pchain2", pchain(2), cost=1.3),
            _job("cal_pchain2_dense_lr3e-4_x4_s0", "dense", 0, lr(x4, "3e-4"), "pchain2", pchain(2), cost=1.3),
            _job("cal_edge_to_pchain2_dense_s0", "dense", 0, x4, "pchain2", pchain(2), init="cal_edge_dense_s0",
                 cost=1.3),
            # C5.pchain3: from scratch at 4x, and the curriculum continued (1 -> 2 -> 3 hops)
            _job("cal_pchain3_dense_x4_s0", "dense", 0, x4, "pchain3", pchain(3), cost=1.3),
            _job("cal_edge_to_pchain3_dense_s0", "dense", 0, x4, "pchain3", pchain(3),
                 init="cal_edge_to_pchain2_dense_s0", cost=1.3),
            # C6.count (dense at chance under v1 settings)
            _job("cal_count_dense_lr1e-4_x4_s0", "dense", 0, x4, "count", cost=1.3),
            _job("cal_count_dense_lr3e-4_x4_s0", "dense", 0, lr(x4, "3e-4"), "count", cost=1.3),
        ]
    if phase in ("lookup16k_root", "lookup16k_stages"):
        # C1.lookup-16k (written curriculum 2k → 8k → 16k from random init) where seeds are missing:
        # e31_li_m1 seed 2 and the dense ceiling seeds 0–2. Split in two bursts (Polonez cooldown between).
        out = (length_jobs(arches=("e31_li_m1",), seeds=(2,), chain_arches=())
               + length_jobs(arches=("dense",), seeds=(0, 1, 2), chain_arches=()))
        for j in out:
            if j["args"][j["args"].index("--arch") + 1] == "dense":
                j["ladder"] = [L for L in j["ladder"] if L <= 32768]  # dense reads cost grows quadratically
        stage = phase == "lookup16k_stages"
        return [j for j in out if bool(j.get("init")) == stage]
    if phase == "odra":  # seed 1: E31_li lookup/chain weights already live here
        return ratio_jobs() + length_jobs(seeds=(1,)) + hard_jobs(seeds=(1,))
    if phase == "polonez":  # seed 0 (E31_li seed-0 lookup weights live here) + dense ceiling
        return length_jobs(seeds=(0,)) + dense_jobs()
    raise SystemExit(f"unknown phase {phase!r}")
