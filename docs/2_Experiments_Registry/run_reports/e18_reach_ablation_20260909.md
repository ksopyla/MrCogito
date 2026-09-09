# E18 — reach ablation: does the single global read carry long-range information? (2026-09-09)

**Date:** 2026-09-09 · **Machine:** Polonez, GPU 0, no training · **Git:** `1a89736` (probe), `515820e` (global_positions knob)
**Spec:** [E18](../../experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md) · stage A/B/dense: [e18_pilot_stageA_20260907.md](e18_pilot_stageA_20260907.md)
**Artifacts:** `Cache/eval/e18/reach/*.json` on Polonez (per-row bucket CE for every window) · job `Cache/jobs/e18_reach.sh` · log `Cache/logs/e18_reach.log`

---

## Why this measurement

P3 as written compares position-bucket CE across two *runs* (8k stage vs 32k stage). That comparison is
confounded twice: (1) by optimizer history — stage B warm-started weights only, at Muon lr 0.01 with a
500-step re-warmup (launcher bug, fixed in `ec6d99b`) and no decay, and ended **uniformly ~2% worse at
every position including [0,2k)**, an optimization regression, not a context result; (2) by the intrinsic
"later in a book is easier" gradient, which makes CE fall with position whether or not the model uses far
context (the dense 8k control shows the same fall).

`PerceiverARLM.reach_override(W)` removes both: **same tokens, same weights, only the global read's window
changes** (`full` → `swa(W)`). Δ(W) = CE(W) − CE(full), paired per row, with a standard error. Positions
p < W see identical keys and must give Δ = 0 exactly (unit-tested; observed 0.0000 on flex/bf16 too).

**Instrument checks (all passed):** 39 CPU tests + the CUDA flex ≡ sdpa test on Polonez; the P2 copy
model as positive control at the 16k copy offset:

| global read window | copy token accuracy (200 rows, 32k) |
|---|---|
| full | 0.99994 |
| 16384 (covers the offset) | 0.99994 |
| **16382** (two tokens short) | **0.0040** |
| 8192 | 0.0040 |

The window semantics hold to the token on real hardware, and the 6-layer × 2048 local stack cannot
bridge 16k on its own: **the global read is the retrieval channel** when the task is retrieval.

## Results — natural text

**8 PG-19 rows ≥ 32k, buckets by position (paired Δ vs full, mean ± se; + = worse without reach):**

| model / window | [0,2k) | [2k,8k) | [8k,16k) | [16k,32k) |
|---|---|---|---|---|
| **stage A** (8k-trained perceiver), global read → 512 | +0.0004 ±0.0007 | +0.0007 ±0.0009 | +0.0007 ±0.0010 | +0.0008 ±0.0007 |
| stage A, → 2048 | 0 | −0.0000 ±0.0003 | −0.0009 ±0.0003 | −0.0011 ±0.0005 |
| stage A, → 8192 | 0 | 0 | −0.0003 ±0.0001 | −0.0010 ±0.0004 |
| **stage B** (32k continuation), → 512 | +0.0008 ±0.0003 | −0.0000 ±0.0005 | −0.0002 ±0.0008 | −0.0009 ±0.0006 |
| stage B, → 2048 | 0 | −0.0004 ±0.0002 | −0.0008 ±0.0004 | **−0.0015 ±0.0005** |
| **dense control**, *all 14 layers* → 512 | +0.085 ±0.020 | +0.326 ±0.067 | +0.415 ±0.087 | +0.298 ±0.120 |
| dense, all → 2048 | 0 | **+0.099 ±0.017** | **+0.094 ±0.034** | −0.032 ±0.059 |
| dense, all → 8192 | 0 | 0 | −0.008 ±0.010 | **−0.093 ±0.027** |

**64 mixed rows ≥ 16k (higher power):**

| model / window | [0,2k) | [2k,8k) | [8k,16k) |
|---|---|---|---|
| stage A, global read → 512 | −0.0002 ±0.0001 | +0.0003 ±0.0001 | +0.0002 ±0.0001 |
| stage A, → 2048 | 0 | +0.0002 ±0.0000 | +0.0003 ±0.0000 |
| dense, all → 512 | +0.063 ±0.008 | +0.105 ±0.005 | +0.072 ±0.006 |
| dense, all → 2048 | 0 | **+0.035 ±0.003** | +0.006 ±0.004 |
| dense, all → 8192 | 0 | 0 | **−0.019 ±0.002** |

Absolute CE (stage A, full): 3.192 / 2.940 / 2.645 / 2.611 on the 32k rows; 3.801 / 4.035 / 4.040 on the 16k rows.

**Per-token tail (stage A, 64 rows, 1.05M tokens; `SA_reach16k_tail.json`)** — rules out a sparse retrieval
signal hiding behind the ~0 mean: read → 512: 0.91% of tokens worse by > 0.1 nats, **0.89% better** by
> 0.1 nats (worst-1% mean +0.14, max +1.37, min −1.57); read → 2048: 0.26% worse vs 0.25% better. A used
channel would be asymmetric (many worse, few better); this is symmetric perturbation.

## Reading

1. **The bottom global read contributes ≈ 0.0003 nats (0.01%) to language-model loss at 8k–32k**, on
   both checkpoints and both row sets. The whole P1 parity with dense (3.790 vs 3.786) is carried by the
   12 × 2048 local stack, whose chained reach (~24.6k) covers the pilot context.
2. **The prize exists.** The dense model extracts **+0.035 nats (0.9%)** from direct access to keys 2k–8k
   back and **+0.10 nats (2.6%)** from reach 512→8k. That is the magnitude P3's ≥ 2% asks for, and the
   perceiver's single read captures none of it — it does not need to at this geometry, because the
   stack already has it.
3. **Beyond the training length the dense model's direct reach is harmful** (−0.019 ± 0.002 at
   [8k,16k) when capped at 8k; −0.093 at [16k,32k)): RoPE extrapolation failure. The perceiver has no
   such term — its windows never exceed the training range — which is what the stage-A "CE keeps falling
   past 8k" observation actually was.
4. **Stage B's far global read is slightly harmful** (−0.0015 ± 0.0005 at [16k,32k), 3σ, when capped at
   2k): softmax dilution over 16k+ keys carrying no useful signal. Small now; a warning for 1M.
5. P3 is **unmeasurable at this geometry**, not failed. Stage B's uniform degradation is an optimizer
   artifact (see stage A report); even a clean 32k stage could not have shown the global read's value
   with a 24.6k-reach stack inside a 32k context.

## What it does not settle

Two hypotheses are both consistent with (1)–(2): **H1** the bottom read is redundant here and becomes
load-bearing when the stack cannot reach (256k+); **H2** a single read at depth 1, with queries and keys
formed by one SWA-512 layer, can only do lexical matching (copy) and never semantic retrieval for LM.
The dense control cannot separate them. A geometry where the stack's reach ≪ context can.

## Hardening pre-checks on the P2 mechanism (tiny CPU study, 2026-09-09)

`verification/e18_copy_tiny.py --task copy --pre_window 16 --value_embed_layers 0,1` (0.98M params, context
130, AdamW 3e-3, single seed — indicative only). Eval CE / token accuracy on plain copy (fixed offset):

| variant | @950 | @1450 | @2950 | reading |
|---|---|---|---|---|
| baseline: RoPE read at the bottom | 2.20 / 0.17 | 4.64 / 0.10 (spike) | **1.11 / 0.51** | learns; noisy at this lr |
| `--global_logit_scale log` (SSMax on the read) | 2.03 / 0.22 | 1.68 / 0.27 | — | ≥ baseline, no spike; **safe to enable** |
| `--global_positions 2` (read at mid-depth) | 3.63 / 0.05 | 2.26 / 0.19 | — | learns, slower start; positional copy survives a deep read |
| `--global_nope` (content-only read) | 5.54 / 0.015 | 5.54 / 0.015 | 5.22 / 0.022 | **floor for ~1,900 steps**, then a slow descent |

The P2 copy mechanism is **position-based**: with RoPE the read attends at a fixed relative offset and
reads the value embedding; without RoPE it must build an induction circuit through the pre-encoder,
which only starts forming after ~1,900 steps here and is ~10× slower. So `PAR_GLOBAL_NOPE` is **not
free** — keep RoPE on the read for stage 1 of the main run, and treat NoPE as a stage-2/3 option only
after a pilot-scale copy check (a 6-layer 32k copy run with `PAR_GLOBAL_NOPE=True`, ~3 GPU-h). The
log-length scale costs nothing on this task and is the cheaper anti-dilution measure for 1M.

## Iteration 2 — geometry arms (seq 8k, stack window N=256, 0.5B tokens, 125M)

Launched 2026-09-09 14:11 UTC (`Cache/jobs/e18_geometry_arms.sh`, A → C → B). Same data, optimizer and
batch as stage A; only `PAR_BLOCK=256` (chained stack reach ≈ 3.1k, so 62% of the 8k context is beyond
the stack) and the read's presence/position differ.

### Arm A — bottom global read (the E18 design) — `perceiver_ar_perceiver_H768L1g1s12N256_20260909_141145`

Eval loss (256 rows, seq 8k) every 0.05B tokens: 5.541 4.893 4.619 4.462 4.360 4.276 4.205 4.156 4.122
**4.090**. Against stage A (N=2048) at equal tokens: 0.2B **4.462 vs 4.386** (+1.7%), 0.4B **4.156 vs
4.065** (+2.2%) — the N=256 geometry costs ~2% overall (12 layers lose their direct 256–2048 context;
one read cannot replace twelve). Throughput 19.8k tok/s (vs 28k at N=2048: window-256 masks are mostly
partial 128-blocks in flex, the slow path; irrelevant at the main run's N=4096). 7.0 GPU-h.

**Reach ablation, 64 rows (paired Δ vs full, + = worse without reach):**

| read window | [0,2k) | [2k,8k) | [8k,16k) (beyond training length) |
|---|---|---|---|
| → 256 | +0.0292 ±0.0012 | **+0.0343 ±0.0010** | +0.0320 ±0.0008 |
| → 512 | +0.0210 ±0.0011 | +0.0303 ±0.0009 | +0.0280 ±0.0008 |
| → 2048 | 0 | **+0.0239 ±0.0009** | +0.0225 ±0.0007 |
| → 8192 | 0 | 0 | **+0.0217 ±0.0007** |

Per-token tail (read → 256): **27.5% of tokens worse by > 0.1 nats vs 17.5% better**, worst-1% mean
+1.03 nats, max +5.1 — an asymmetric, heavy tail: a used retrieval channel (contrast stage A: 0.9% vs
0.9%, symmetric).

**Reading.** (i) **H1 confirmed at first order:** when the stack cannot reach, the same bottom read that
was worth 0.0002 nats in stage A becomes worth **0.024 nats for 2k–8k reach** (120×) and is used on a
quarter of all tokens. (ii) **Magnitude vs dense:** the dense control extracts 0.035 nats from 2k–8k
direct access at *every* layer; the single bottom read recovers ~⅔ of that. The missing third is the
H2 residual (shallow queries / one layer) that arm B tests. (iii) **Extrapolation:** at positions
8k–16k, beyond the 8k training length, the read's access to keys > 8k back is worth +0.022 nats — the
read generalises past its training range, which the dense control did not (its far reach *hurt* there).
(iv) Net value still to be read from arm C: the ablation bounds the read's trained-in worth at ~0.03
nats (0.8%), so the pre-registered "A beats C by ≥ 1%" may land at the edge; the 3σ reach-Δ condition is
met by 25σ.

### Arm C — no global read — running since 21:15 UTC · Arm B — read at mid-depth — queued

## Next (iteration 2, needs a go)

Three matched 125M arms at seq 8k, **stack window N=256** (chained reach ≈ 3.1k, so 62% of the context is
beyond the stack), 0.5B tokens each, same data/optimizer as stage A: (A) global read at the bottom
(index 1, the E18 design); (B) global read at **mid-depth** (`PAR_GLOBAL_POSITIONS=7`, queries formed by
half the stack); (C) no global read (`PAR_GLOBAL_LAYERS=0 NUM_LAYERS=13`, same 14 layers/params).
Readouts: eval loss at equal tokens vs the dense 8k control and stage A; the reach probe on each; position
buckets on the 16k rows. Pre-registered reading: A ≈ C → H2 at this depth, B decides the main-run
placement; A < C by ≥ 1% with reach-Δ > 0 → H1, keep the design and harden the read for 1M.
