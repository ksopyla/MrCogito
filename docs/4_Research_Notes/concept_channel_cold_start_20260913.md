# The concept channel has a series cold start — and the array *can* be addressed (2026-09-13)

**Class:** dated root-cause diagnosis (append-only ledger — do not rewrite the results below).
**Instrument:** `data/symbolic_tasks.py` + `verification/symbolic_channel_probe.py`
([suite spec](../engineering_specs/symbolic_long_context_suite.md)). CPU only, ~1 h total.
**Follows:** [E22 root cause](e22_root_cause_20260912.md), which left three candidate failures
entangled. **Feeds:** E23 (`docs/experiments_specs/ahead/E23_exclusive_concept_channel.md`).

## TL;DR

1. On a task whose information floor is **exact**, the concept array **does** carry addressable
   far content. This is the first positive evidence in this family — E22 only ever demonstrated a
   document embedding plus 0.05 nats of far marginal.
2. It carries it **badly**: 15% of what raw access to the same evidence recovers.
3. The write and the mask are **not** the bottleneck. The evidence reaches the slots and the mask
   exposes them. The **read** is what fails.
4. A large part of that failure is an **initialisation artefact, not an architecture limit.** The
   concept path has two zero-init residual gates in series, and the gradient that would make
   either one useful is proportional to the other. Breaking them recovers **17× more information
   at matched steps** (3/3 seeds), with high variance in how fast the channel escapes.

## Why a synthetic instrument was needed

E22 spent 0.44B tokens and two GPUs to learn that its array carried 0.05 nats of far content, and
that this was close to *all there was to carry* — E18's reach ablation had already measured a
dense model of the same width extracting only 0.035–0.105 nats from keys 2k–8k back. The gate
asked for 0.30. So the run could not distinguish:

- (1) the objective does not pay for the channel;
- (2) the read cannot address the array;
- (3) the write blurs the content before it can be addressed.

All three have the same root: **we were measuring a channel with data whose long-range
information content we did not know.** The symbolic tasks fix that by construction — the
supervised tokens are determined by evidence at a controlled distance and independent of
everything a local window can see, so the prize is `ln(alphabet)` and the floor is derived.

## Setup

`far_copy`: a 32-symbol span over a 4-symbol alphabet is marked early in a 128-token row and must
be reproduced verbatim after a marker at the end. `dec_segment 32`, `min_gap 32` (so the decoder's
raw window provably cannot reach the span), `r = 8`, 4 decoder layers, ~1.3M params, 3000 steps,
batch 32, AdamW one-cycle 3e-3. Floor **1.3863** nats, chance accuracy **0.250**.

## Result 1 — three arms, both controls green

| arm | route to the evidence | CE (nats) | vs floor | accuracy |
|---|---|---|---|---|
| **D** | full raw access (`dec_segment = seq_len`) | **0.0000** | −1.3863 | **1.000** |
| **C** | segment-confined, no array | **1.3863** | +0.0000 | **0.250** |
| **A** | the array only (`exclusive` scope) | **1.1726** | −0.2137 | **0.423** |

Arm C is pinned at the floor to four decimals across all 3000 steps, so the task does not leak;
arm D solves it outright, so the task is learnable at this size. Only with both of those does
arm A's number mean anything — and it means the channel works, partially.

Attribution on arm A's own weights: removing the array raises CE to **1.8458** at chance accuracy
(*worse* than the floor — the model has reorganised around the array and is lost without it), and
`far`-slots-only equals `real` to four decimals, confirming that under exclusive scope the array
is a purely long-range channel.

**Read against arm D, the channel recovers 15.4% of the available information** (0.2137 of 1.3863
nats) at 42% accuracy against 100%. It is real and it is weak.

## Result 2 — the write and the mask are fine; the read is not

Measured on the `recall` configuration (128-token rows, `r = 8` so 16 slots, `dec_segment 32`,
`min_gap 32`, the queried value 8 symbols long at positions 43–50, answer at 118, gap 68).
Perturbing *only* those 8 far value symbols and re-encoding:

- the slot array moves by `max |Δz| = 1.68` against an array norm of 45 — **the write carries the
  evidence**;
- the answer position queries with `raw_span(117) = 21`, so its visibility threshold is
  `cpos < 96`, admitting slots 0–11 (block ends 7, 15, … 95). The perturbed value sits in slots 5
  and 6 — **the mask exposes exactly the slots that hold it**;
- yet at initialisation the answer logits are **bit-identical** under the perturbation
  (`max |Δlogits| = 0.000e+00`), for arm A just as for the blind arm C.

That last line is the whole story: the content is present and visible, and the model is
structurally incapable of noticing at step 0.

## Result 3 — two zero-init gates in series

The family zero-inits every residual-writing projection (muP-like, modded-nanogpt). For
self-attention that is free. On the concept path it is not, because two such gates sit between the
evidence and the loss:

- `pooler.wo` — the learned-query branch of the write, and the **only order-sensitive part of
  it**. The rest of the pooler is a mean over the block, which is an order-free bag of symbols and
  therefore cannot represent a span at all.
- `xattn.wo` — the read's output projection.

The coupling is the problem. The gradient into the read's query/key projections is proportional to
`xattn.wo`, and so is the gradient into `pooler.wo`. At zero:

- the read cannot become **selective**, because selectivity has no gradient until `wo` is nonzero;
- `wo` only grows along the direction the *current* (near-uniform) attention supplies, which is
  the **average of the visible slots**;
- and the write cannot become **order-sensitive**, because that gradient is zero too.

So the only escape route available early is "consume the mean of the slots" — which is precisely
a **document embedding**. That is a mechanistic explanation for what E22 actually measured: 0.17
nats of document-level content and 0.05 nats of far marginal is what a channel looks like when it
never escapes its cold start inside the training budget.

### Measurement

`xattn_wo_init_std = pooler_wo_init_std = 0.02` versus the default `0.0`, arm A, matched seeds,
**at step 2000**:

| seed | default | both gates seeded | 
|---|---|---|
| 0 | 1.3275 (−0.0588) | **0.8780 (−0.5083)** |
| 1 | 1.3862 (−0.0001) | **1.3308 (−0.0555)** |
| 2 | 1.3863 (−0.0000) | **0.9542 (−0.4321)** |
| mean information recovered | 0.0196 nats | **0.3320 nats** |

Direction replicates **3/3**. At 3000 steps seed 0 ends at 1.1726 (default) versus 0.8424
(seeded) — 0.214 versus 0.544 nats, i.e. 15% versus 39% of what arm D recovers. Under the seeded
init, removing the array costs far more (CE 3.76–8.70 versus the default's 1.3863, which is just
the floor again), i.e. the model becomes genuinely dependent on the channel rather than
incidentally helped by it.

**Caveat, stated plainly:** escape time is highly variable (seed 0 departs the floor around step
500, seed 1 around 1250, seed 2 around 750), so the *magnitude* is not pinned — between 0.06 and
0.51 nats at 2000 steps. What replicates is the direction and the removal of the long plateau.
These are 1.3M-parameter CPU runs; the effect size must be re-measured at E23 scale before any
claim rests on it.

## Laws this adds

Extending the three laws in [the E22 root cause](e22_root_cause_20260912.md#laws) and the two
added there:

- **A compressive channel needs a warm read.** Zero-init is safe for a projection whose input is
  already in the residual stream, and unsafe for one whose input is an external memory: the
  gradient that makes the read selective is proportional to the read's own output weight, so the
  channel's only early escape is the content-free average of its slots. Whenever a new read of a
  new memory is added, seed its output projection.
- **Order-free writes cannot be rescued downstream.** Mean pooling over a block discards order, so
  no read can recover a span from it. The order-sensitive part of the write must be live at
  initialisation, not gated behind the read.

## What this changes for E23

- Set `--pcl_xattn_wo_init_std 0.02 --pcl_pooler_wo_init_std 0.02` on arm A. The knobs default to
  `0.0`, so every E22 checkpoint still loads and reproduces bit-for-bit.
- Run the CPU pre-flight gate before committing 36 GPU-h. It costs minutes and it would have
  caught this.
- **Prognosis to hold honestly:** the read *can* address the array, so E23's K1 ("the read is
  simply incapable") is unlikely to trigger. But the channel recovers well under half of what raw
  access does even on a task built to be maximally favourable to it, so S1 on natural text — where
  the prize is 0.05 nats rather than 1.39 — remains a hard ask. The realistic expectation is a
  small positive on S1b/S2 and a marginal S1.
- The next mechanism question is **bandwidth**, not existence: sweep `far_copy --span_len` to map
  the bits-per-slot curve, and compare against `count`, where a compressive summary should have a
  structural advantage over exact attention rather than paying a tax.
