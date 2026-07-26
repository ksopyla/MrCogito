# E09 — Gated recurrent concept memory (concepts that evolve during decoding)

- **Status:** canceled — superseded before a standalone run when its recurrence mechanism was
  folded into the E10 pretrained-backbone concept-memory design
- **Serves:** the "concept embeddings as latent reasoning state" invariant (recursive/iterative
  refinement of concepts is first-class) and the generation-coherence gap surfaced in the
  E05/E02-long generation review — concepts are currently a *frozen prefix snapshot*
  (`ConceptCausalDecoderStack.forward` reads `concepts` via cross-attention and never reassigns it),
  so beyond the decoder's K-window the model has no evolving memory of its own output.
- **Implementation plan:** [E09_recurrent_concept_memory_plan.md](E09_recurrent_concept_memory_plan.md)
  *(drafted 2026-07-05; the HOW)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-07-05

> One experiment = one hypothesis = one changed variable. Implementation is args/config over the
> shared E05 entrypoint, never a new fork. The spec is frozen once a run starts.

## Hypothesis
If the C=128 concepts become a **gated, recurrent state updated from each decoded window** of K=128
tokens (vs the frozen prefix-snapshot of E05), then **beyond-window suffix cross-entropy will drop
and free-generation repetition will fall**, because the decoder gains an *evolving* memory of its
own output instead of a static snapshot plus a K-token sliding window. (Falsifiable: if beyond-window
suffix-CE does not improve over the frozen control, the hypothesis is wrong.)

## Builds-on
- **Foundation:** `ConceptEncoderForConditionalLM` (BiXT encoder + windowed causal AR decoder), the
  shared `scripts/launch_e05.sh` → `scripts/train_perceiver_denoise_multigpu.sh` entrypoint, and the
  pretokenized `smollm3_inspired_2k_e05` manifest. **No new fork.**
- **Init / checkpoint:** random init, **seed 42 — identical to E05 Adam** for all shared parameters;
  the new write op is **zero-init (gate α = 0)**, so step-0 of E09 reproduces E05 Adam byte-for-byte
  (clean stability property; divergence is attributable solely to the write op learning). No
  warm-start (warm-start would add a variable vs the E05 Adam control).
- **Baseline to beat:** E05 Adam `concept_ar_prefix_H768L6C128D4_20260629_093840` /
  `checkpoint-69142` — windowed K=128, 0.5 ep `smollm3_inspired_2k_e05`, eval_loss 3.829,
  within-sample RankMe 37.67, beyond-window Δshuffle 0.39, free-running generations are repetition
  loops (token-F1 0.149). This run **is** the frozen control arm.

## The single change
Concept memory mode: `frozen` → `gated_recurrent`.

- `frozen` (E05 / control): concepts computed once from the prefix, then read-only at every decoder
  layer/step (current behavior).
- `gated_recurrent` (E09): after each decoded window of K=128 tokens, **reuse BiXT's concept-update
  direction** (`BiXTCrossAttention` `lat ← tok`, `nn/concept_encoder.py:432`, with
  `update_tokens=False`) to refresh the concepts from the window's tokens:
  `concepts ← concepts + α · sandwich_norm(BiXT_lat←tok(concepts, window_tokens))`, where `α` is a
  **zero-init learned gate** and the window tokens are re-embedded via the encoder's token-embedding
  path (a new `ConceptEncoder.embed_tokens` helper — the same space BiXT compresses in the encoder).
  This makes concept *creation* (encoder, from prefix) and concept *maintenance* (decoder loop, from
  generated windows) the **same operation**. Trained via BPTT across windows, **block-causal**
  (concepts predicting block *b* incorporate only blocks *<b*, so there is no suffix leak).

The zero-init gate + sandwich-RMSNorm + BPTT are the **faithful implementation** of "gated recurrent"
(a bare write collapses — Infini-attention, `docs/literature_review/recurrent_memory_transformers.md`
§D), not independent variables. **Everything else is held fixed vs E05 Adam:** arch H768 / T256 / L6 /
C128 / D4, SwiGLU + RMSNorm + RoPE, windowed K=128 decoder, SmolLM2-135M tokenizer, prefix→suffix
objective, `smollm3_inspired_2k_e05` data, seed 42, AdamW optimizer, 0.5-ep budget, effective batch 72.

**Relationship to E08 (concept-flow reasoner):** complementary, not duplicate. E08 iterates concepts
*within a single reasoning pass*; E09 iterates concepts *across generation windows*. The two compose
and share the anti-collapse machinery.

## Success criteria (set BEFORE running)
- **PRIMARY (decisive):** beyond-window suffix cross-entropy (suffix positions ≥ K=128) of the
  treatment **lower than the E05 Adam control by ≥ 0.3 nats** at the final checkpoint. *(Reuses the
  existing `_teacher_forced_ce_window`, `nn/concept_encoder_perceiver.py:1638`; the trainer logs it
  as `suffix_ce_beyond_window`.)*
- **SECONDARY (generation):** on a fixed prompt battery, free-running **distinct-2 ≥ control** and
  **repetition-rate ≤ control** (measured via the generation notebook).
- **MUST-NOT-REGRESS (safety):** within-sample concept RankMe **≥ 0.8 × control** (no collapse);
  STS-B zero-shot **not worse than control by > 0.03**.

## Kill criteria (set BEFORE running)
- **Stage 0 (pre-training, cheap, ~1 h, no GPU training):** run the beyond-window diagnostic on E05
  and E02-long (suffix-CE vs suffix position; repetition/distinct-n vs generated length, in the
  generation notebook). **If beyond-window suffix-CE is essentially flat** (no "wall"), the
  hypothesis is falsified → **do not run Stage 1.**
- **Stage 1 (training):** stop if (a) treatment beyond-window suffix-CE has not dropped below the
  control by **step 40,000**, or (b) within-sample RankMe collapses below **0.5 × control** at any
  eval (recurrent-memory collapse — the Infini-attention failure mode).

## Plan
- **Data:** `smollm3_inspired_2k_e05` (reuse the existing pretokenized manifest — same as E05; no
  re-tokenization). Collator unchanged (`DataCollatorForPrefixGeneration`); the block split is
  internal to the model forward.
- **Compute:** Odra (3× RTX 3090); est. **~68 GPU-h** for the single treatment arm (matched to E05
  Adam's 68.2 GPU-h; the control arm already exists, so no extra compute for it). Stage 0: ~1 h, runs
  on macOS + a quick Odra eval pass.
- **Steps / epochs:** **0.5 ep / 69,142 steps**; effective batch 72 (per-device 8 × accum 3 × 3 GPU);
  LR **5e-5** AdamW; `max_grad_norm 0.5`; warmup 2000; cosine — all matched to E05 Adam.
- **Launch:**
  ```bash
  CONCEPT_MEMORY_MODE=gated_recurrent CONCEPT_MEMORY_BLOCK=128 \
    OPTIMIZER=adam SKIP_PRETOKENIZE=1 bash scripts/launch_e05.sh
  ```
  (New env vars `CONCEPT_MEMORY_MODE`/`CONCEPT_MEMORY_BLOCK` passed through the launcher; default
  `frozen` leaves every prior launch byte-identical. Concept update cadence = every K=128 decoded
  tokens, matching the decoder window.)
- **New foundation code (reusable, via `research-implement` — NOT a fork):**
  1. A config-selectable **gated recurrent concept memory** path on `ConceptEncoderForConditionalLM`,
     selected by `concept_memory_mode` on `ConceptEncoderConfig` (`frozen` default |
     `gated_recurrent`); `frozen` leaves all prior checkpoints byte-for-byte unchanged. The write op
     **reuses `BiXTCrossAttention` (`update_tokens=False`)** — no new attention module; new params
     are only the write-BiXT instance + a zero-init gate scalar `α` + a sandwich-RMSNorm.
  2. A small `ConceptEncoder.embed_tokens(token_ids)` helper (refactor of the first lines of
     `ConceptEncoder.forward`) so the write op's token side reuses the encoder's embedding space.
  3. A block-recurrent `encode_decode_loss` path (gated by `concept_memory_mode`): suffix split into
     blocks of `CONCEPT_MEMORY_BLOCK`=128, decoded **block-causal with a one-block window carry**
     (so the K=128 sliding window stays continuous across block boundaries), concepts refreshed via
     the BiXT write op between blocks; gradients flow across blocks automatically.
  4. Trainer eval logs suffix-CE within/beyond window (reuses `_teacher_forced_ce_window`) — the
     primary metric. Free-generation inference (the notebook) extends symmetrically: generate K
     tokens → write concepts → repeat.

## Risks
- **Collapse (the main risk).** Iterating a small latent memory can degrade it (Infini-attention HF
  reproduction: long-context PPL *worsens* with more compressions). Mitigated by zero-init gate +
  sandwich-RMSNorm + the RankMe kill-gate.
- **Block-decode + window-carry complexity.** The decoder must run block-causal with a one-block
  carry so the K=128 sliding window stays continuous across boundaries (a naive per-block decode
  would drop the previous window and break local fluency). Main implementation subtlety; mitigated
  by reusing the existing windowed `decode_logits` per block + gradient checkpointing.
- **~1.5–2× step time** (16 block-decodes vs one full forward). Acceptable (decoder is the cheap
  part; encoder BiXT dominates); a KV-cache optimization is a follow-up, not this spec.
- **0.5 ep may be too short for the write op to learn.** Mitigated by the cheap, sensitive primary
  metric (beyond-window suffix-CE) and the step-40k kill-gate. A longer run would be a follow-up
  spec, not scope creep here.

## Result
No standalone E09 run was launched. The design was superseded by E10, which reused its
recurrent BiXT write mechanism on the pretrained-backbone platform.

## References
- `docs/literature_review/recurrent_memory_transformers.md` — the read/write-memory axis
  (Block-Recurrent Transformer §B, RMT §B, Infini-attention §D, Coconut §B).
- `nn/concept_encoder_perceiver.py` `ConceptCausalDecoderStack.forward` / `encode_decode_loss` /
  `_teacher_forced_ce_window` — the frozen-read path this extends and the metric it reuses.
- `nn/concept_encoder.py` `BiXTCrossAttention` — the write op (lat←tok, `update_tokens=False`).
- E05 spec (`E05_windowed_decoder_concept_memory.md`) — the baseline protocol.
