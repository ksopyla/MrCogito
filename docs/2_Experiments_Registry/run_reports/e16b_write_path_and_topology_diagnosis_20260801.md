# E16b — Write-path & memory-topology diagnosis: why the shared concept memory goes dead, and the 4-bank per-layer fix

**Date:** 2026-08-01
**Checkpoint:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850/checkpoint-7900` (E16b)
**Sister reports:** [free-run generation quality vs base Gemma](e16b_generation_quality_assessment_20260801.md) (the decoding failure + Layer-0 probe) · [E16b mechanism success](e16b_longctx_muon_1b_20260725.md)
**Theory lens:** [Reasoning bandwidth / BAPO](../../literature_review/reasoning_bandwidth_information_flow.md)
**Motivates:** new spec **E17 — 4-bank per-global-layer concept memory** (this report is its analysis backing).

---

## TL;DR

E16b's free-run repetition is the visible symptom of one root cause: **the concept
*write* path is dead, so the concept memory carries almost no input-specific
information, and the read of that near-constant memory becomes a generation attractor.**

The writes are dead for **two compounding reasons**:

1. **Cold-start gradient starvation** (cheap diagnostic, done 2026-08-01): a write only
   gets gradient *through later reads*; at init the reads are closed (gate 0.01), so the
   write gate gets ≈0 signal and never opens. Reads and writes had the *same* optimizer
   and LR (both 1-D scalars → AdamW @ 2e-4 under Muon), so this is not a config/LR bug.
2. **The shared-depth memory topology** (the structural cause): one 128-slot memory,
   written by all four global layers (5/11/17/23) with a *tied* writer, makes writes
   **"altruistic"** — a write at depth *d* can only help depths *>*d and future blocks,
   never the writing layer itself — so the write gate never gets a clean "self" gradient,
   and four depth-specialized features fight over one accumulator. This is the user's
   "lower vs higher layers carry different information" intuition, and it is already
   written into the E13 hypothesis.

**Fix:** give each of the four global layers **its own concept bank** (4-bank per-layer
memory) → each write becomes "selfish" (a layer reads the bank it just wrote) → clean
gradient → writes open → the memory carries real prefix content → the constant-bias
attractor disappears. This is a **new spec, E17** (not an edit to E13).

**BAPO caveat (important):** per-layer memory raises *nominal* `a`-bandwidth, which BAPO
Thm 10 says does **not** raise *effective* reasoning bandwidth. E17 is therefore justified
as a **write-dynamics / generation-stability fix** (open the writes → restore effective
`a` → kill the degeneration), **not** as a reasoning-capacity upgrade. Its success metrics
must attribute any win to "writes opened," not "more slots."

---

## 1. How the concept memory actually works in E16b

- **One** shared concept state `z [B, 128, H]` (128 slots). Initialized from a learned
  `concept_init [128, H]` at the start of each sequence.
- **Four** global layers — Gemma indices **5, 11, 17, 23** (the `full_attention` layers
  in the 5-sliding : 1-global pattern; sliding_window = concept_block = K = 512).
- Per 512-token block, the 26 layers run in order. At each global layer:
  1. **READ** — the layer cross-attends into `z` with its *own* Q/K/V/O (LoRA-adapted),
     adding `tanh(read_gate)·read` to its hidden state. Each global layer has its own
     read-gate scalar.
  2. **WRITE** — a **shared (tied)** `ConceptWriteHead` (one BiXT + norms — *same weights
     for all four depths*) updates the *same* `z`:
     `z ← z + tanh(depth_alphas[d]) · sandwich(BiXT(z, this_block_hidden))`.
     Each depth has its own write-gate scalar; the summary operator is shared.
- `z` (refined 4× per block) then carries into the next block.

So: **4 global layers, ONE shared 128-slot memory, ONE tied writer, only per-depth scalar
gates differ.** (Code: `nn/backbone_concept_lm.py` — `_forward_shared_depth_block`,
`ConceptWriteHead`, `GlobalLayerWithConceptRead`.)

---

## 2. Why the writes are dead — two compounding causes

### 2a. Cold-start gradient starvation (verified, read-only diagnostic)

W&B final gates: **read gates 0.85–0.88 (wide open); write gates ±0.05 (dead).**

Under E16b's Muon optimizer, `nn/muon.py:88` routes every `ndim≥2` matrix to Muon (lr
0.01) and every <2-D param to an AdamW fallback (`muon_adamw_lr=2e-4`). The read gates
(0-D) and the write gates (`depth_alphas`, shape `[4]`) are **both 1-D scalars → both
AdamW @ 2e-4 — same optimizer, same LR.** The `concept_memory_lr` differential-LR path is
adam-only (`concept_pretraining_trainer.py:69`) and was off (`CONCEPT_MEMORY_LR=""`), so it
played no role. → The writes **were** trained, at the same LR as the reads.

Same optimization, opposite outcome ⇒ the cause is **gradient magnitude**:

- A write only matters through *later reads* (`∂loss/∂z ∝ how open the reads are`). At
  init both gates = 0.01, so reads barely pass gradient into `z` → the write gate gets
  ≈0 signal → it cannot grow. (Chicken-and-egg.)
- The write is `tanh(α)·summary`; the gradient to `α` is itself scaled by the write's tiny
  magnitude (self-attenuation).
- The single-block graph tie-in `loss + 0.0·z.sum()` feeds gradient to the write
  *weights* but **zero** to `depth_alphas`, so the gate scalar relies entirely on the
  recurrent read path.

**Corroboration that it is not an LR problem:** E16 (the short-ctx pilot) used AdamW with
a **dedicated** concept-memory LR of 3e-4 and *still* failed to establish causal writes
(beyond-1024 deltas ≈ +0.0005). Giving the writes their own LR did not fix it.

### 2b. The shared-depth topology (the structural cause — the user's intuition)

Even if the cold-start were solved, the shared topology structurally starves the writes:

- **Reads are selfish, writes are altruistic.** A read at layer 11 changes layer 11's
  *own* prediction → strong, direct gradient → read gates open. A write at layer 5 only
  changes what layers 11/17/23 and future blocks read → indirect gradient, routed through
  those later reads → much weaker.
- **A write at depth *d* cannot help depths ≤ *d* within the block.** Layer 5 reads `z`
  *before* this block's writes begin, so it reads the previous block's state; its own write
  is only ever consumed by layers 11/17/23 and the next block. The "self" coupling that
  would give the write gate a clean gradient **does not exist** in the shared design.
- **Multi-depth conflict in one accumulator.** A single tied writer must summarize
  layer-5 features (local/syntactic) *and* layer-23 features (semantic/abstract) into the
  same 128 slots. Those objectives partially conflict → gradients on the shared writer
  cancel, and the accumulator is a tug-of-war between depths.

This is exactly the "different depths carry different information, so don't force them
through one memory" argument — and it is already the stated rationale of
[E13](../../experiments_specs/ahead/E13_layerwise_recurrent_kv_memory.md): *"each depth can
preserve and retrieve long-range information in its own representation space rather than
forcing all layers to share one memory representation."*

### 2c. The generation consequence (the observed symptom)

Dead writes ⇒ `z ≈ concept_init` (a learned constant) plus a small input-dependent residue
(the ~5%-gate writes accumulated over the block recursion). Wide-open reads then inject a
**near-constant directional bias** into every global layer ⇒ under greedy argmax a
fixed-point / structured attractor (the FinePDF table/outline patterns in the mix). This
is the free-run degeneration quantified in the
[sister report](e16b_generation_quality_assessment_20260801.md): `zero` (concepts off) is
the only fluent mode; `frozen`/`real` degenerate; sampling escapes, repetition-penalty
doesn't.

---

## 3. The BAPO framing — and the caveat that shapes the fix

Map ([lit review](../../literature_review/reasoning_bandwidth_information_flow.md) §"Relevance to MrCogito"):

| BAPO channel | MrCogito realization | E16b status |
|---|---|---|
| `a` — prefix bandwidth (compressed summary, bits) | the concept set `z [B,C,H]`, read by cross-attn | **severely degraded** — `z` dominated by a non-informative constant |
| `b` — attention bandwidth (raw-token reach, tokens) | windowed self-attn, `context_window = K = 512` | intact (the 512 window) |
| effective `a` (is the channel used?) | Δshuffle/Δzero beyond-window CE gap | Δshuffle_beyond 2.47 — the model *uses* `z`'s small residue, but the constant dominates under free-run |

**E16b is a new route to the BAPO "effective-`a` collapse" failure** — not the usual
decoder-bypass (routing through `b`), but the concept channel itself carrying almost no
prefix information because the write path is dead.

**The caveat (BAPO Thm 10; lit review line 191):** *adding depth/heads/banks at a fixed
bottleneck does not raise effective bandwidth*; the review explicitly **demotes E11/E12/E13
(more read paths / depth) as the "wrong axis"** for reasoning, because they raise *nominal*
`a`, which Thm 10 says does not raise *effective* bandwidth. A 4-bank topology raises
nominal `a` (4×128 slots), so on the pure-capacity axis BAPO would predict **no reasoning
gain**.

**Resolution — what E17 is and is not:**

- E17 is **not** a reasoning-capacity bet. It is a **write-dynamics / generation-stability
  bet**: per-layer banks make each write "selfish" (a layer reads the bank it just wrote)
  → clean gradient → writes open → `z` carries real prefix content → the constant-bias
  attractor disappears and effective `a` is restored.
- The benefit, if any, must be **attributed to "the write gates opened," not "there are
  more slots."** E17's success metrics are therefore write-gate magnitudes + per-bank
  effective-`a` + free-run diversity — *not* reasoning-task scores. A "win" that improves
  CE while the per-layer write gates stay dead would be the Thm-10 "wrong axis" failure
  mode and must be read as such.
- **BAPO↔Pfau note (design constraint, not E17's target):** per-layer recurrent banks are
  Markov across blocks (`z_{ℓ,b} → z_{ℓ,b+1}`, not re-readable). Pfau's hidden-computation
  barrier ([lit review](../../literature_review/reasoning_bandwidth_information_flow.md)
  lines 138–143) warns this bites *multi-step latent reasoning*. E17 targets the
  write/generation failure, not multi-step reasoning; re-readability of the bank trajectory
  is a future concern (and is what E08 Concept-Flow is for).

---

## 4. The structural fix — 4-bank per-global-layer concept memory (E17)

**One changed variable vs E16b:** replace the single shared `z [B,128,H]` with **four
independent banks** `z₅, z₁₁, z₁₇, z₂₃ [B,128,H]`, one per global layer. Each global layer
reads only its own bank and writes only its own bank (writer tied or per-layer — a spec
decision). Everything else (backbone, LoRA, K, C=128, objective, data, optimizer) held
fixed vs E16b.

**Why it fixes both causes at once:**

- **Breaks the cold-start (2a):** layer ℓ now reads `z_ℓ` that it wrote last block → the
  read directly depends on the write → the write gate gets a clean, strong "self" gradient
  from step 1. No more "writes need reads open, reads closed at init" deadlock.
- **Removes the multi-depth conflict (2b):** no shared accumulator; each depth summarizes
  into its own representation space. The altruistic-write asymmetry collapses (each layer's
  write feeds its own next-block read).

**What it does not claim:** a reasoning-bandwidth gain (BAPO Thm 10). The claim is
narrower and falsifiable: *per-layer banks open the write gates and recover non-degenerate
free-run generation, at unchanged nominal compute.*

---

## 5. E13 vs E17 — write a new spec, do not edit E13

- **E13** (`ahead/E13_layerwise_recurrent_kv_memory.md`) is the *full* per-layer design —
  one memory per **all 26** layers, via the E12 KV-prefix read interface; status *draft,
  gated on E12*, never run. It is a different changed-variable (memory at every layer +
  a new read interface) and a bigger bet.
- **E17** is the cheaper intermediary: **4 banks at exactly the 4 global layers**, reusing
  E16b's existing read/write interface at those layers. E13's own "Explicitly not E13"
  section already says this 4-bank topology *"needs its own spec because it changes number
  of memory banks and injection depth together."*

→ **Author E17 as a new spec; leave E13 unchanged.** Next free ID is **E17** (E01–E16
taken).

---

## 6. Recommendations

1. **Spec E17 (4-bank per-global-layer memory)** via `experiment-design` — the structural
   fix. Success metrics: per-layer write gates open (≥0.05 by ~25M tokens), per-bank
   Δshuffle/Δstatic > 0.01 beyond-window, and **free-run distinct-1/REP-3 within ~0.1 of
   base Gemma** (the actual symptom). Kill if per-layer write gates stay dead like E16b's.
2. **Cheap falsification pre-check (1-i):** before the full E17 run, a short resume from
   `checkpoint-7900` with `WRITE_GATE_INIT=0.3` under the *current shared* topology. If
   even that cannot open the write gates, the cold-start is deeper than init and E17's
   "selfish gradient" argument is the right place to look; if it *does* open them but
   generation still degenerates, that isolates the topology (2b) as the remaining cause and
   strengthens the E17 case.
3. **Do not** sell E17 as a reasoning fix; instrument it as a write-dynamics +
   generation-stability fix per the BAPO caveat. Reasoning bandwidth remains the E08
   (Concept-Flow / decomposition) axis.

*See also:* [generation quality report](e16b_generation_quality_assessment_20260801.md)
(the decoding failure + Layer-0 probe), [E13](../../experiments_specs/ahead/E13_layerwise_recurrent_kv_memory.md)
(full per-layer design), [BAPO review](../../literature_review/reasoning_bandwidth_information_flow.md).
