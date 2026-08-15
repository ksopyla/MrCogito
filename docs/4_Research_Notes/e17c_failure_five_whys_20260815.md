# E17c failure: five whys (carry dropout trains a block-boundary gist)

**Date:** 2026-08-15
**Status:** Permanent research note (eval-day diagnosis; not a new experiment spec)
**Run:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241` · best `checkpoint-2370`
**Related:** [E17c spec](../experiments_specs/done_failed/E17c_depth_private_working_memory.md) ·
[report](../2_Experiments_Registry/run_reports/e17c_depth_private_working_memory_20260815.md) ·
[recurrent-memory review](../literature_review/recurrent_memory_transformers.md) ·
[anti-collapse note](../literature_review/concept_bottleneck_collapse_mitigation.md)

---

## What failed (and what did not)

E17c's *registered* 300M carryless gate **passed**: first-64 Δpermutation **0.594** CI [0.543, 0.645].
The cell is not inert. It failed as a **generation-memory platform**: RankMe **6.75** (bank 1 **1.84**),
normal-context Δpermutation_beyond **0.013**, free-run `real≈shuffle` @256 **0.23/0.53**.

A 16-batch held-out diagnostic (`…_why_position_bins.json`) splits that 0.59 by *where* in the
new block it lives. Carryless Δpermutation (mean ± std):

| Intra-block bin | Carry dropped | Carry present |
|---|---|---|
| 0–64 | **0.603 ± 0.150** | 0.014 ± 0.010 |
| 64–128 | 0.123 ± 0.038 | 0.014 ± 0.012 |
| 128–256 | 0.056 ± 0.015 | 0.012 ± 0.006 |
| 256–512 | 0.026 ± 0.009 | 0.012 ± 0.004 |
| 64–end | 0.050 ± 0.011 | 0.013 ± 0.004 |

Absolute CE, first 64 tokens: with carry **2.005**; carryless real **2.842**; carryless perm **3.445**.
Concepts recover ~0.60 nats of a **0.84 nat** gap versus just keeping the previous 512 tokens.
After ~64 tokens of the *new* block, local context has refilled and banks barely matter.

---

## Five whys

### Why 1 — Why is E17c not a working-memory platform?

Because the only place concepts beat a permutation is the **first ~64 tokens after a block
boundary when the previous K-carry is artificially removed**. Generation and ordinary eval
never remove that carry, so they sit in the ~0.013 nats regime. RankMe collapse means even
the carryless path is a low-rank gist, not a 128-slot workspace.

### Why 2 — Why doesn't that gist transfer when the carry is present?

Because the K=512 token path is strictly better. First-64 CE with carry is 2.00 vs 2.84
carryless-with-real-banks. Training applies carry dropout with p=0.5, so half the steps can
ignore concepts entirely. `BackboneConceptLM.generate` always re-encodes the growing prefix
with `carry_policy="normal"`. Train/test mismatch is the same pattern as conditioning dropout
that is not applied at inference: the shortcut remains at decode.

Evidence in code: dropout is `self.training and memory_carry_dropout`, not used in
`generate()` (`nn/backbone_concept_lm.py`).

### Why 3 — Why did only bank 0 / layer 5 absorb the pressure?

Because the dedicated read is **added into the residual after global layer 5**:
`h := h' + tanh(read_gate) * CrossAttn(h', z_0)`. Layers 11/17/23 then see a hidden state
that already contains the gist. Permuting banks 1–3 on the carryless first-64 task yields
Δ **0.029 / 0.013 / 0.032** vs bank 0 **0.380**. Update gates stayed *open*
(0.84 / 0.78 / 0.28 / 0.85), so the unused banks are not "closed" — they overwrite a
collapsed state (bank 1 RankMe 1.84). This is the stacked-recurrence failure Block-Recurrent
Transformer avoided by using **one** recurrent layer, not four independent cells on a residual
highway ([recurrent-memory review](../literature_review/recurrent_memory_transformers.md)).

### Why 4 — Why did RankMe collapse by the first eval (~30M tokens)?

Because the rewarded task is a **low-dimensional previous-block summary**, and the write is
high-rate `gated_replace` (`z ← (1-g)z + g·BiXT(z, h_block)`, mean g≈0.8). 128 slots are
given no reason to specialize. `concept_losses` (VICReg / orthogonality) **raise ValueError
on the backbone family** (`training/concept_pretraining_args.py`) — the anti-collapse tools
from E05 are unwired here. Jing et al. (arXiv:2110.09348) + VICReg (arXiv:2105.04906) are
the matching collapse story; see [anti-collapse note](../literature_review/concept_bottleneck_collapse_mitigation.md).
Infini-attention's independent reproduction also warns that repeated compressive overwrite
of a small state degrades long-context use
([HF Infini-attention writeup](https://huggingface.co/blog/infini-attention)).

### Why 5 — Why did carry dropout not create working memory in the first place?

Because dropping the *previous* K tokens only creates a memory demand until the **current
block refills the local window**. At offset t of the new block the model already sees t
current tokens (+ BOS). After t≈64 those tokens dominate next-token CE on FinePDFs.
The training recipe then **upweights exactly that 64-token region by 4×**, so the optimizer
is paid to solve the gist-at-boundary problem and is not paid to keep state useful at
offsets 128–512. Ordinary causal LM on long PDFs does not need a fact from 2k tokens ago
once 64–512 local tokens are visible — the same reason E14/E15 synthetic recall was
introduced (and failed) as a different objective family.

Carry dropout is a real anti-bypass for *block starts*. It is not an anti-bypass for
*long-range content inside a block*.

---

## What is missing (the information constraint, not another gate init)

1. **A constraint that stays active after the local window fills.** Mask/drop current-block
   tokens, or cap the visible local span (e.g. last 64 tokens + concepts), so banks must
   carry content at offsets 128–512. Pressuring 512 tokens while leaving those tokens
   visible will not do this — the fill-in curve already shows why.
2. **Train/test match.** Either decode with the same carry drop, or train at dropout 1.0
   so there is no "ignore memory" half of the mixture. p=0.5 + always-carry generate is
   classifier-free training without CFG at sample time.
3. **Anti-collapse on backbone concepts.** VICReg is implemented and banned for this family.
4. **Depth job allocation.** Four residual-injected banks without isolating the post-read
   residual makes banks 1–3 redundant. Block-Recurrent's single recurrent layer is the
   published pattern; alternatively, later banks must query a *pre-injection* stream.
5. **An objective that needs retrieved content, not topic continuation.** FinePDFs CE at a
   boundary is "keep writing this document." Associative recall / MQAR / needle tasks ask
   for a specific earlier span. E14/E15 tried a synthetic version and died on protocol;
   the need remains.

---

## Improve without a major refactor (ranked by whether they attack Why 5)

These are config or small-wiring changes. They are **not** a substitute for a new
architectural bet; they test whether the diagnosis is right.

| Lever | Attacks | Expected | Kill |
|---|---|---|---|
| **Current-block span drop** (extend existing carry-dropout: also mask p∈[0.3,0.6] of the *current* block, TSDAE-style) | Why 5 | Carryless Δperm stays high into bins 128–512; RankMe less free to collapse to a 64-token gist | After ~30M, bin 256–512 Δperm still <0.05 |
| **Cap visible local span to 64** in the windowed mask (keep K=512 writes) | Why 5 | Same as above, even *with* carry present (Δbeyond should rise) | Δbeyond still ~0.01 at 50–100M |
| `MEMORY_CARRY_DROPOUT=1.0` + generate with `drop_after_first` | Why 2 | `real` beats `shuffle` under matched decode; will **not** by itself lift bin 256–512 | real≈shuffle even under dropped carry |
| Wire `CONCEPT_LOSSES=vicreg` into backbone (today it hard-errors) | Why 4 | RankMe holds ≥19.2 | RankMe still <10 with vicreg on |
| `WRITE_UPDATE_GATE_INIT=0.05` or a max-update clamp | Why 4 | Lower overwrite, maybe less collapse | Gates still →0.8 by 30M or RankMe still dead |
| `CONCEPT_BLOCK=128` | Why 5, weakly | More boundaries; first-64 is half the block | Fill-in curve just rescales; Δbeyond still tiny |
| `MEMORY_PRESSURE_TOKENS=512` **alone** | — | **Unlikely**: tokens 64–512 already have local context under carry drop | Do not spend a 300M run on this alone |
| 1B continuation / LR retune | — | Will not fix transfer | Already killed |

Do **not** launch 1B. Do **not** split this into five micro-A/Bs as the research program.
If a short smoke is run, the single coherent claim to test is: **keep memory necessary after
the local window fills** (row 1 or 2). VICReg is additive insurance, not the bet.

---

## Diagnostic artifacts

- Position bins: `Cache/Evaluation_reports/…133241_why_position_bins.json`
- Carryless perm gate: `…133241_best_perm_gate.json`
- Geometry: `…133241_best_concept_analysis.json`

*Related: E17c run report, `nn/backbone_concept_lm.py` (carry dropout, gated_replace, residual read), `training/concept_pretraining_args.py` (vicreg unwired).*
