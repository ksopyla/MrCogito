# E23 — Exclusive concept channel: make the array the only route, and make the objective pay for it

- **Status:** draft 2026-09-12 (written from the E22 diagnosis; awaiting KS approval) · not launched
- **Serves:** Vision priorities 1–2 and 4 — a concept array that carries *facts* from far context (the
  memory a reasoning core needs) and, by construction, the receiver-side of an agent-to-agent latent
  message: the decoder has no raw access to what the array summarises. Evidence and reasoning:
  [E22 root cause](../../4_Research_Notes/e22_root_cause_20260912.md) · [E22 report](../../2_Experiments_Registry/run_reports/e22_pilot_verdict_20260912.md).
- **Implementation plan:** `E23_exclusive_concept_channel_plan.md` *(to be authored by `implementation-plan`)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-12 · closed —

> One coherent bet on the E22 platform, changing exactly the two things the E22 diagnosis isolates
> and nothing else. E22 proved the array is live and diverse (RankMe 265) and that a CE-trained
> decoder reads it as a *document embedding* (0.17 nats, redundant across slots) plus 0.05 nats of far
> facts — because natural-text CE is worth ≈ 0.05 nats beyond 1k tokens at this scale, and because the
> mask let every token also read its own segment's slots. E23 (i) closes the channel in both
> directions (`concept_xattn_scope=exclusive`) and (ii) trains under an objective in which a
> substantial share of the loss sits on tokens whose targets are *determined* by far content. The claim
> is that with these two changes the same encoder → array → latent → decoder becomes a
> content-addressable memory. Controls exist to falsify that, not to hedge it. r, c, K, hierarchical
> pooling and the receiver-only message probe are registered post-signal iterations.

## Hypothesis
If the E22 `perceiver_concept` model (6-layer SWA-512 encoder → one pooled slot per 16 tokens →
4-layer causal latent transformer → 8-layer decoder confined to 1024-token segments) is trained
**(a)** with cross-attention that admits only slots ending *before* the token's raw segment, so the
array is the only route for everything it carries, and **(b)** under an objective where ≈ 40% of the
loss mass sits on tokens whose targets are determined by content ≥ 1 segment back — far-repeat
tokens in natural text weighted ×8, plus 30% (by tokens) dense-label long-range rows (keyed recall,
far span copy, multi-hop variable tracking) — then at 0.5B tokens the far-slot marginal on natural
text rises from E22's 0.05 to ≥ 0.10 nats and to ≥ 0.5 nats on far-repeat tokens, keyed-recall
first-token accuracy through the array reaches ≥ 50% (E22 4.8%, E18b-R 4.2%, dense 99.3%), and
held-out passkey at 32k reaches ≥ 0.5 — **because** a channel learns the codebook of the source it is
trained on: with I(far context; target) ≈ 0 under natural-text CE the optimal code is the document
marginal (what E22 learned); with I(far; target) large on a substantial fraction of tokens and no
other route to that information, the optimal code is the conditional — slot-specific content —
and E22 already showed the array *holds* slot-specific content (half of each slot's energy) that the
decoder simply never had a reason to read.

## Builds-on
- **Foundation:** `nn/perceiver_concept_lm.py` unchanged except the config knob already merged
  (`concept_xattn_scope`, `raw_span`, `near`/`far` overrides — commit `0f4b4f9`); `training/train_concept_pretraining.py`
  via `scripts/train_concept_pretraining_multigpu.sh` (`PCL_*`); `data/packed_dataset.py`;
  `scripts/build_retrieval_mix_dataset.py` (E18b keyed-recall rows, `--min_gap`), `scripts/build_copy_task_dataset.py`;
  the `perceiver_ar` eval layer (`evaluation/long_context_probes.py --probe concept|suite|tasks`,
  `scripts/eval_perceiver_ar_suite.sh`). Thin wrapper `scripts/launch_e23.sh` (pins the protocol,
  delegates; no fork).
- **Init / checkpoint:** random init, from scratch. **Not** warm-started from E22 arm A: its decoder is
  trained to read same-segment slots and the document marginal — exactly the prior E23 removes.
- **Baseline to beat:** (1) **arm C** — the same model with `concept_mode=none` (segment-only decoder,
  63M compute) trained on the *same E23 objective*: it cannot retrieve past its segment, so it is the
  floor for every long-range metric and shows what the weighted objective buys on its own;
  (2) **arm D** — the dense 18-layer `perceiver_ar` control on the same objective (E22's was lost;
  E18b's dense control reached 99.3% keyed recall and passkey 0.725 at 32k on a 100%-task mix):
  the ceiling and the "task is learnable here" check; (3) anchors: E22 arm A far marginal 0.05,
  keyed recall 4.8%, passkey 0.0; E18b-R 4.2%.
- **Materially new vs the ledger:** no ledger run has had a *two-sided* closed concept channel (E05/E18
  had raw bypasses; E22 had a slot bypass), and none has trained a concept array on an objective where
  far content carries a substantial share of the loss on *natural text* (far-repeat weighting is new;
  E18b's recall rows sat on a 1-layer encoder with a single read and no latent stack, at 5% and then
  100% of the mix). E23 is the first test of "deep-contextualised positional slots + transformer over
  slots + retrieval-worth supervision" together — the combination the E18 verdict named and never ran.

## The architectural bet
```
tokens x[0:S]  (S = 32768, packed documents, doc_ids)
  ─► TinyHashedEmbedding ─► ENCODER 6 × Block(swa 512, causal, doc-masked)                 h_enc
  ─► POOL: slot j = mean(h_enc[block j]) + XAttn(q_learned → block j), r = 16, c = 1        z0 [B, S/16, d]
  ─► LATENT 4 × Block(full causal over slots, doc-masked, RoPE at block-end positions)       z
  ─► DECODER 8 × [ self-attn confined to the 1024-token segment
                   → cross-attn to { z_j : end(j) < segment_start(t), doc(z_j) = doc(t) } ∪ {null}   ◄── (a) exclusive
                   → SwiGLU ]
  ─► lm_head; loss = Σ_t w_t · CE_t / Σ_t w_t                                                         ◄── (b) pays for far content
        w_t = 8   if t is a far-repeat token: its (4-gram context, target) pair occurred earlier in the
                  same document *only* before the token's raw segment (copyable from far, not from near)
        w_t = 1   otherwise on natural-text rows
        dense-label rows (30% of tokens): loss only on the answer spans (E18b marker protocol)
```
**(a)** is `PCL_CONCEPT_XATTN_SCOPE=exclusive` — one config value; segment 0 tokens see only the null
slot, which is the built-in zero for every ablation. **(b)** is a collator-side per-token weight
computed from ids (one hash pass per document, O(S)) plus three dense-label row sources merged into
the manifest by token share: keyed recall 15% (E18b builder, `--min_gap 4096`, values 1–64 tokens),
far span copy 7.5% (a 32–256-token span from ≥ 4096 tokens back must be reproduced after a marker),
multi-hop variable tracking 7.5% (2–3-hop chains, hops ≥ 2048 tokens apart — the reasoning probe in
training form). Natural text 70% (the E22 mix: PG-19 / FinePDFs / FineWeb-Edu / stack-edu-py).
**Controls share every line** except the one they remove: arm C deletes the cross-attention; arm D is
the dense 18-layer transformer. Both train on the identical weighted objective and manifest.

**Sizes:** as E22 (d 768, 6 q-heads × 128, 2 kv-heads, SwiGLU 2048; compute ≈ 128M / 63M / 128M for
A / C / D). Cache at inference: the array (96 B/token) + one 1024-token segment per decoder layer.

## Why this is not a safe retread
It is not "E22 with more recall data": E18b showed volume alone (5% → 100%) does nothing when the
channel has a bypass or a shallow encoder; E23 changes the *mask* so the array is the only route, and
changes *which natural-text tokens carry the loss* so the channel is paid for on ordinary documents,
not only on synthetic rows. It is not an r/c/K knob sweep — those are post-signal. Cross-domain hooks:
rate-distortion — a code trained on a source with zero far mutual information is a code for the
marginal; you get the conditional code only by training on a source where the conditional carries
information (Shannon); and the testing effect in memory research — episodic memory consolidates under
*retrieval practice*, not passive exposure. Frontier anchor: LCLM (arXiv:2606.09659) puts loss on
uncompressed tokens interleaved with compressed spans for the same reason; E23 goes further by
selecting, inside natural text, the tokens whose information *is* in the compressed span.

## Success criteria (set BEFORE running; Odra 3×3090, 32k, 0.5B tokens per arm)
- **S1 memory on natural text:** paired ablation (`--probe concept`, modes `real | none | shuffled`;
  under exclusive scope `far` ≡ `real`) on ≥ 64 held-out PG-19 rows ≥ 32k: **CE(none) − CE(real) ≥ 0.10
  nats** on tokens with ≥ 4096 of history (2× E22's far marginal) and **≥ 0.5 nats on far-repeat tokens**;
  CE(shuffled) − CE(real) ≥ 0.5 × Δ_none on far-repeat tokens (content, not prior). Segment-0 Δ must be
  0.00 ± 0.01 in every mode (built-in control; a nonzero value is a bug, not a result).
- **S2 useful, not just used:** arm A ≤ **0.90 ×** arm C CE on far-repeat tokens ≥ 4096; held-out
  keyed-recall first-token accuracy (`--probe tasks`) ≥ **50%** and ≥ 0.5 × arm D's.
- **S3 transfer to unseen formats (the reasoning-side probes):** passkey @32k with the needle ≥ 4096
  before the query ≥ **0.5**; multikey ≥ 0.3; vt 2-hop ≥ 0.2 (all zero in E22; dense DT passkey 0.725).
- **S4 price:** arm A natural-text CE (all tokens) within **3%** of arm D; on [0, 1k) within **2%** of
  arm C (exclusivity costs nothing locally — E22 measured the near-slot marginal at 0.03).
- **S5 geometry (diagnostic):** RankMe(z) ≥ 128; adjacent-slot cosine reported (E22: 265 / 0.69).

## Kill criteria (set BEFORE running)
- **K1 read cannot address the array:** at 50% of budget, keyed-recall first-token < **10%** while arm D
  ≥ 60% → stop. Next step is a different *read* (two-stage: query the latent stack, then the slots; or
  per-slot K/V of block-end states), **not** r / c / K — E22 cleared the encoder, pooler and array geometry.
- **K2 objective pays, channel does not deliver:** at 50% of budget arm A − arm C on far-repeat tokens
  ≤ 0.02 nats → same path as K1.
- **K3 weighting instability:** eval loss rising over 3 consecutive evals or a > 2× loss spike attributable to
  far-repeat rows → restart once at weight ×3; if repeated, drop the weighting and record it.
- **K4:** throughput < 40% of arm D at 32k → fix the kernel path first.
- Any arm > 14 GPU-h without an evaluable checkpoint → stop and debug.

## Plan
- **Data:** `e23_longmix_32k_manifest.json` on Polonez (then NAS → Odra): the E22 natural-text sources at
  70% of tokens + keyed recall 15% (`scripts/build_retrieval_mix_dataset.py --min_gap 4096`) + far
  span copy 7.5% + multi-hop vt 7.5% (two new small builders, same LM-shard schema and marker
  protocol). Written as a new manifest via `scripts/write_manifest_variant.py` — never
  `dataset_mix_weight_override`. Far-repeat weights are computed in the collator (no stored labels).
  **Pre-launch instrument (zero GPU) — measured 2026-09-12 on Polonez** (E22 mix train shards,
  segment 1024, 4-gram context + target, most recent earlier occurrence before the segment start):
  far-repeat share PG-19 **1.14%** of tokens (1.28% beyond 4k; near-repeat 1.01%), FinePDFs **5.69%**
  (near 9.01%), FineWeb-Edu 0.50%, keyed-recall rows 9.02%. Token-weighted natural text ≈ **2.5%**
  (3-gram: ≈ 3.7%). At w = 8 that is ≈ **17%** of the natural-text loss (3-gram: 23%), and with the
  30% dense-label rows ≈ **40%** of the total loss on far-determined tokens — the share the
  Hypothesis assumes. Pin `--far_repeat_ngram 4`, `--far_repeat_weight 8`.
- **Compute:** Odra 3×3090 for arms A and C (≈ 12 GPU-h each at 32k, E22 calibration: 560 steps ≈ 6.5 h
  for 0.44B); Polonez 4×3090 for arm D after the `goodwrite_ml` move completes.
- **Steps / epochs:** 0.5B tokens per arm (≈ 640 steps), effective batch 24 packed 32k rows ≈ 0.79M tokens, Muon 0.01 / AdamW 2e-4 / wd
  0.1 / clip 0.5, 100-step warmup, cosine to 10% — the E22 schedule, unchanged.
- **Launch:** `bash scripts/launch_e23.sh` (arm A = `PCL_CONCEPT_XATTN_SCOPE=exclusive FAR_REPEAT_WEIGHT=8`
  on the E23 manifest) · `E23_ARM=C bash scripts/launch_e23.sh` · `E23_ARM=dense bash scripts/launch_e23.sh`.
  Eval: `scripts/eval_perceiver_ar_suite.sh` (`--probe concept` now reports `near`/`far` too; on an
  exclusive model `far` must equal `real` to 1e-3 — a second built-in check).
- **New foundation code (reusable, config-selectable):** far-repeat per-token loss weights in the
  causal-LM collator (`--far_repeat_weight`, `--far_repeat_ngram 4`, window = `dec_segment`) consumed by
  the per-token loss path both families already expose; far-copy and vt training-row builders; the
  `launch_e23.sh` wrapper. No change to `nn/perceiver_concept_lm.py` beyond `0f4b4f9`.
- **Registered post-signal iterations (only after S1–S2 pass):** latent repeats K ∈ {2, 4} (the
  reasoning-bandwidth curve on vt hops); r ∈ {8, 32}; hierarchical r = 16 / 256 for the 128k–1M stage;
  **receiver-only decoding** — score a query segment given *only* the array of a different context
  window (the agent-to-agent message probe, E21's question on this platform); **exclusive-scope
  ablation** — arm A0 = exclusive scope on the plain E22 objective, to attribute the gain between the
  mask and the objective (the diagnosis predicts the mask alone changes little).

## Result
<Filled in AFTER, by experiment-track. Link out; do not paste full results here.>
- Run ids: `<A>`, `<C>`, `<D>` · WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
