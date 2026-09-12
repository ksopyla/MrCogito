# E18 family verdict — one global read: free, retrieval-capable by position, but not a content retriever

**Date:** 2026-09-12 (decisive control) · covers 2026-09-07 → 2026-09-12
**Machine:** Polonez (4× RTX 3090)
**Specs:** [E18](../../experiments_specs/done_failed/E18_perceiver_ar_v2_baseline.md) · [E18b](../../experiments_specs/done_failed/E18b_retrieval_trained_read.md) · [E18c](../../experiments_specs/ahead/E18c_concept_compressed_read.md) (blocked)
**Prior reports:** [stage A](e18_pilot_stageA_20260907.md) · [reach ablation + geometry arms](e18_reach_ablation_20260909.md)
**Git:** `1a89736` (reach probe) → `a591877` (warm-start fix)
**Artifacts:** `Cache/eval/e18/`, `Cache/eval/e18/reach/`, `Cache/eval/e18b/` on Polonez

---

## Goal

Close the E18 family with a single evidence-based verdict. E18 asked whether one full-causal
attention layer reading a contextualized history beats a dense transformer on long-context loss.
When that failed, E18b asked the re-scoped question: can that read be *taught* to retrieve, making
"retrieval-grade long context at a 1 KB/token cache" the product claim instead.

## Runs

| run id | arm | what it tested |
|---|---|---|
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943` | stage A | 125M, 8k, 1.0B tokens |
| `perceiver_ar_dense_H768L1g1s12N2048_20260907_193351` | dense control | matched, 8k, 1.0B tokens |
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260908_145256` | stage B | 32k continuation (optimizer regression, superseded) |
| `perceiver_ar_perceiver_H768L1g1s6N2048_20260908_230306` | P2 | plain copy @32k, offset 16k |
| `perceiver_ar_perceiver_H768L1g1s12N256_20260909_141145` | A | N=256, read at layer 1 |
| `perceiver_ar_perceiver_H768L1g0s13N256_20260909_211550` | C | N=256, **no read** |
| `perceiver_ar_perceiver_H768L1g1@7s12N256_20260910_040940` | B | N=256, read at layer 7 |
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260910_184948` | R | 32k, LM + 5% retrieval |
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_021311` | 0 | 32k, LM only (protocol control) |
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_130626` | R2 | R + value embedding on the read |
| `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_210251` | T | **100%** retrieval |
| `perceiver_ar_dense_H768L1g1s12N2048_20260912_062402` | **DT** | **dense, 100% retrieval — the decisive control** |

## Findings

### 1. The architecture is free at short context (P1 ✅, P4 ✅)
Eval loss 3.790 vs dense 3.786 at 1.0B tokens; throughput 1.02× dense. No quality tax to adopt.

### 2. The read does exact *positional* retrieval perfectly (P2 ✅)
Plain copy at 32k, offset 16k: **99.9998%** token accuracy. Restricting the read's window to 16,382
— two tokens short of the offset — drops it to **0.4%**. The window semantics hold to the token and
the read is unambiguously the retrieval channel for this task.

### 3. The read contributes ~nothing to language-model loss (P3 ❌)
New instrument: `PerceiverARLM.reach_override` + `--probe reach`, a paired ablation that shortens
only the read on a trained model, same tokens and weights. Positions below the window are
bit-identical (unit-tested), so Δ is a clean measurement.

| model | Δ from cutting the read's reach to 2k, at [2k,8k) |
|---|---|
| stage A (N=2048) | +0.0002 nats, per-token tail symmetric (0.9% worse / 0.9% better) |
| arm A (N=256, read needed) | **+0.024 nats**, 25σ, 27% of tokens worse vs 17% better |
| dense (all layers cut) | +0.035 nats — the prize exists |

**Arm C settles it:** a model with *no read at all* scores **4.091** against arm A's **4.090**. The
read is *used* when the local stack cannot reach, but a read-free stack recovers the same loss.
Arm B (read at layer 7) also reaches 4.090 and is depended on 5.7× less, so placement is not the
lever for LM loss either. Natural text does not supervise long-range addressing.

### 4. The read cannot be taught content-addressed retrieval (E18b S1 ❌, S2 ❌)
Task: 32k rows of real text with 8–24 items; each item is `KEY value` at the source and
`KEY START value END` ≥1024 tokens later, loss on the value. Labels are derived by the collator from
reserved span markers, so rows keep the LM shard schema. Passkey is never trained on — it is the
held-out transfer probe.

| arm | task share | first-token acc | passkey @32k |
|---|---|---|---|
| 0 | none (never saw the task) | 2.4% | 0.0 |
| R | 5% | 4.2% | 0.0 |
| R2 | 5% + value embedding on the read | 4.4% | 0.0 |
| T | **100%** (~20× the practice) | **4.49%** | 0.0 |
| **DT — dense, same task, same protocol** | 100% | **99.33%** | **0.725** |

Arm T's task loss plateaued at **3.904** (perplexity 49.6); DT reached **0.070** (perplexity 1.07).
Two hypotheses were tested and killed: the missing value embedding on the retrieving layer (R2's LM
trajectory matched R to five decimals; an 8M-parameter table left no trace) and supervision volume
(20× more practice moved first-token accuracy by 0.09 points).

**DT is the decisive control.** A 125M model at 32k learns this task to 99.3% and *transfers* to
passkey at 72.5% from a 0% baseline, without ever seeing that format. So the task is not flawed and
the benchmark is not too hard at this scale: **the architecture is the cause.**

### 5. The extension protocol was broken, and the fix is worth ~12%
Stage B warm-started weights only at Muon 0.01 with a 500-step re-warmup (launcher shadowing bug,
fixed in `ec6d99b`) and no decay; it degraded the model **uniformly at every position**, which is an
optimizer signature, not a context result. Arm 0 reran the same stage correctly:

| position | stage A (8k) | stage B (broken) | arm 0 (fixed) |
|---|---|---|---|
| [0, 2k) | 3.192 | 3.265 | **2.876** |
| [8k, 16k) | 2.645 | 2.712 | **2.319** |
| [16k, 32k) | 2.611 | 2.678 | **2.279** |

Encoded as a hard rule in the AWS plan (step 6): never restart a converged model at peak lr with a
fresh optimizer; resume state, or warm-start at ≤20% peak lr with decay.

## Interpretation

E18's headline hypothesis — better prediction from long context through one read — is **falsified**,
and this agrees with Perceiver AR's own reported "no gain past 2k" and with the literature, where even
full attention buys only 1–3% of loss. E18b's re-scoped retrieval claim is **also falsified for the
read as designed**, now with a proper control rather than by inference.

The most useful reframing came from reading the architecture as what it structurally is: an
encoder-decoder. Everything below the read produces the cached keys and values; the read is the
cross-attention; everything above is the decoder. In those terms the pilot is a **1-layer encoder**,
1 cross-attention, 12-layer decoder — against Perceiver AR's 0-layer encoder and dense's effective
14. We asked a one-layer encoder to produce keys discriminative enough to be content-addressed, and
queries one layer deep to address them. This is the same defect BiXT attacks in the Perceiver family
(there, tokens never deepen while latents do), though BiXT's bidirectional mechanism does not
transfer literally because E18 has no latent array.

**What survives:** dense-parity short-context quality, a 1 KB/token cache (23× under dense), exact
positional retrieval, length extrapolation of the read beyond its training range (arm A helped at
8k–16k where dense's far reach *hurt*), a reusable paired-ablation instrument, and the protocol fix.

## Decision

- **E18 → `done_failed`**: the long-context-loss hypothesis is falsified at pilot scale.
- **E18b → `done_failed`**: the retrieval claim is falsified for a read at layer 1; DT provides the
  pre-registered decisive control.
- **Do not launch the AWS main run** on either spec as written.
- **E18c (concept-compressed read) stays blocked**: it compresses a *functional* retrieval channel,
  and we do not have one.

## Future directions

1. **Read at mid-depth — the one cheap open question.** `PAR_GLOBAL_POSITIONS=7` makes everything
   below the read a 7-layer encoder, deepening queries *and* keys together, with **zero change to the
   cache** (still one layer, 1 KB/token) and no LM cost (arm B). Staged as
   `Cache/jobs/e18b_mid_taskonly.sh`, ~1.4 GPU-h, not launched. Caveat: the warm start was trained
   with the read at layer 1, so a win is strong and a loss is ambiguous.
2. **Two reads (layers 1 and 7)** if mid-depth fails — tests whether match-then-retrieve needs two
   hops. 2 KB/token, still 11× under dense.
3. **Looped weight-tied encoder.** Encoder cost is paid once per token at prefill; decoder cost is
   paid per generated token. For huge-prompt / short-answer workloads, deepening or looping the
   encoder is the cheap direction and adds no parameters. Prior art in our own ledger is negative
   (E16 shared-depth recurrence, `done_failed`) but in a different setting.
4. **Mechanistic probe.** Behavioural accuracy cannot show *why* the circuit failed. A probe that
   inspects whether the read's attention at `START` peaks at the source position would settle
   "no circuit" vs "wrong circuit" directly.
5. **Platform reuse regardless.** The E19 latent write-back and E21 agent-message work need a
   well-defined one-layer K/V interface, not a retrieval circuit. `prefix_kv()` and
   `write_back_proj` are built and tested; E19/E21 do not depend on this verdict.

*Related: [`master_experiment_log.md`](../master_experiment_log.md), [`agenda.md`](../../1_Strategy_and_Plans/agenda.md), [reach ablation report](e18_reach_ablation_20260909.md)*
