# E01 — Concept-conditioned autoregressive decoder (from scratch)

- **Status:** done (training 2026-06-13 · evaluation 2026-06-14)
- **Serves:** the new **encoder→AR-decoder** Current focus in [agenda.md](../../1_Strategy_and_Plans/agenda.md). First step of the planned series (see agenda "Series roadmap"). Bridges Vision SG1 (concept quality) → SG2 (text generation): the encoder still produces concepts we can probe for quality, but the decoder now *generates* autoregressively instead of reconstructing in parallel.
- **Implementation plan:** [E01_concept_ar_decoder_plan.md](E01_concept_ar_decoder_plan.md) *(the HOW)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-06 · closed 2026-06-14

> One experiment = one hypothesis = one changed variable. Implementation is a config
> (`--decoder_type causal_ar`) over the shared `train_perceiver_denoise.py` entrypoint and its
> launcher — NOT a new fork. The spec is **frozen once the run starts**; results live in the
> registry + run report, not here.

## Hypothesis
If we condition a **from-scratch autoregressive Transformer decoder** on the concept bottleneck
(causal self-attention over target tokens **+** cross-attention to the `C` concepts as memory) and
train it as a denoising autoencoder, then the model will reconstruct/generate text autoregressively
**while genuinely using the concepts** — measured by a concept-ablation gap **ΔCE ≥ 0.5 nats** and
concept **effective rank > 32/128**, beating the parallel-decoder denoise baseline's collapsed
geometry (rank 10.6/128) — because an AR decoder removes the per-position conditional-independence
ceiling of the position decoder, and **decoder-input word-dropout** blocks the "copy from my own left
context" shortcut that otherwise causes posterior collapse (decoder ignores the latent).

## Builds-on
- **Foundation (reuse + extend, no fork):**
  - Reuse as-is: `ConceptEncoder` + `BiConceptEncoderLayer`/`BiXTCrossAttention` (BiXT, token↔concept
    asymmetry) — `nn/concept_encoder.py`; `DataCollatorForTSDAE` — `data/data_collators.py`;
    `load_and_preprocess_text_dataset` — `data/dataset_preprocess.py`; the shared entrypoint
    `training/train_perceiver_denoise.py` and launcher `scripts/train_perceiver_denoise_multigpu.sh`.
  - New **reusable, config-selectable** components in `nn/concept_encoder_perceiver.py`:
    `ConceptCausalDecoderLayer`, `ConceptCausalDecoderStack`, model `ConceptEncoderForConditionalLM`
    (selected via `config.decoder_type="causal_ar"`). Keep the existing `PerceiverDecoderStack`
    (position decoder) intact as the frozen STS-B / representation probe.
- **Init / checkpoint:** random init (train from scratch — the explicit goal).
- **Baseline to beat:** `perceiver_denoise_H512L6C128D3_20260308_220324` — eff. rank **10.61/128**,
  zero-shot STS-B **0.607**, recon CE eval **1.869**. Broader collapsed-rank history: 5–10/128.
  *(E01 is a new modern reference line, not a perfectly-controlled ablation of this run — see below.)*

## The single change
**The decoder.** Replace the parallel, position-only cross-attention decoder
(`p(x|concepts)=Πₚ p(xₚ|concepts,p)`, non-generative) with an **autoregressive concept-conditioned
Transformer decoder** trained with next-token cross-entropy + decoder-input word-dropout.

> **Scope honesty (read this).** E01 is the **first run of a new line**, so alongside the decoder it
> also fixes the **modern baseline choices** the series will build on, all config-selectable with
> backward-compatible defaults (old checkpoints still load):
> - data **MiniPile → FineWeb-Edu** `sample-10BT`;
> - tokenizer **ModernBERT → SmolLM2** (vocab ~49,152; matches the E04 warm-start);
> - activation **GELU → SwiGLU** (`hidden_act=silu`);
> - normalization **LayerNorm → RMSNorm** (`norm_type=rmsnorm`);
> - **RoPE in the decoder's causal self-attention** (`decoder_pos_type=rope`); encoder keeps learned
>   absolute token positions, concepts stay orderless (RoPE is ill-defined on orderless concepts);
> - a **slight scale bump** (~61–76M → ~135M, ≈ SmolLM2-135M).
>
> These are *baseline configuration*, not a controlled ablation — the old baseline's numbers are
> **reference anchors**, not a like-for-like control (note: reconstruction CE is **not comparable across
> vocabularies**, so success #4 is recalibrated to the SmolLM2 vocab; rank / STS-B / concept-ablation ΔCE
> stay valid). From E01 onward, every later experiment changes **exactly one** knob vs E01 (see series
> roadmap).

**Resolved (frozen for E01) — training objective = (A) AR denoising reconstruction.** Encoder sees
TSDAE-deleted input → concepts; decoder AR-reconstructs the clean sequence (`DataCollatorForTSDAE`,
reused as-is). Lowest-risk upgrade of the current denoise objective: "confirm the bottleneck + now it
generates." **(B) prefix→suffix AR generation** is deferred to **E02** (strongest semantic pressure;
the AR decoder is the materially-new ingredient vs the previously-failed random-init prefix track).

### Posterior collapse — the central risk, and how we prevent + detect it
A strong AR decoder with teacher forcing can learn `p(xₜ|x₍₎ₜ)` from its own gold left context and make
the concepts optional (Bowman et al. 2016, *Generating Sentences from a Continuous Space*). The **copy
path matters more than raw decoder size**, so we attack it on four fronts:
- **Lean decoder vs heavy encoder/bottleneck:** decoder **L4 < encoder L6**, decoder `intermediate_size`
  ≤ encoder's. The bottleneck should be the expensive part, not the decoder.
- **Decoder-input word-dropout `p=0.4`** (Bowman's fix): blanks gold left-context tokens via a **learned
  "dropout" embedding** (one `nn.Parameter`, substituted at the embedding level — *not* a `[MASK]` token,
  which SmolLM2 lacks), forcing fallback onto concepts.
- **Concepts are the decoder's only route to target content** (no encoder-token skip connection).
- **Detect it three ways** at eval (see success #1): concept **zeroing**, concept **shuffling** (stronger —
  breaks instance-specific info while keeping concept statistics), and a **no-concept floor** (same decoder,
  concepts zeroed throughout = pure-LM baseline the full model must beat by a margin).

> Honest caveat: reconstruction + teacher forcing has an inherently *weak* need for the latent, so a small
> ΔCE in E01 could mean "task too easy from left context," not "concepts are bad." That is precisely why
> **E02 (prefix→suffix) is the decisive semantic test** — the decoder must produce content absent from its
> left context. E01 proves the *plumbing* (concepts → AR text works, concepts are used); E02 proves *semantics*.

## Success criteria (set BEFORE running)
1. **Concept usage (primary — addresses posterior collapse):** concept-ablation **ΔCE ≥ 0.5 nats** —
   both **zeroing** and **shuffling** concepts at eval raise next-token CE by ≥ 0.5 vs the intact model;
   the intact model beats the **no-concept floor** by ≥ 0.5 nats. (Shuffle is the stronger test.)
2. **De-collapse:** concept **effective rank > 32/128** (`analysis/run_concept_analysis.py`), clearly
   above the 5–10/128 collapsed history.
3. **Semantics:** zero-shot **STS-B Pearson ≥ 0.62** (≥ prior best 0.607) from mean-pooled concepts.
4. **Generation sanity:** eval next-token **CE well below random** (random ≈ `ln(vocab)` ≈ 10.8 for the
   SmolLM2 vocab; target a clear margin, e.g. **CE < ~4.0** — recalibrate once the first eval lands since
   the vocab changed) **and** AR reconstructions are qualitatively coherent on a held-out sample.

## Kill criteria (set BEFORE running)
- By **25% of the step budget**: if concept-ablation **ΔCE < 0.1** *and* effective rank **< 10/128**
  → AR decoder ignores concepts (posterior collapse), **stop**.
- By **25% of the step budget**: if eval CE has not dropped below **~6.0** (not learning) → **stop**.
- Compute cap: **> ~80 GPU-hours** without clearing the not-learning gate → **stop**.

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`; subsample to ~1–2B tokens (cap doc
  count to fit the holdout-split loader). **Append `<|endoftext|>` (eos) to every document** in
  preprocessing so the decoder learns to stop.
- **Tokenizer:** **`HuggingFaceTB/SmolLM2-135M`** (vocab 49,152). Base tokenizer has `<|endoftext|>`
  (bos=eos=unk) but **no distinct pad and no `[MASK]`** → set **`pad_token=eos_token`** and keep
  vocab at 49,152 for clean SmolLM2 warm-start/alignment. Padding is positional
  (`attention_mask`, labels `-100`), so real eos tokens remain trainable; embedding `padding_idx` is
  disabled when pad aliases eos. `[MASK]` is not needed (TSDAE deletes via `attention_mask`;
  word-dropout uses a learned embedding).
- **Model (~135M):** `hidden_size=768`, `token_embedding_dim=256` (asymmetry kept), encoder
  `num_hidden_layers=6` (BiXT), `concept_num=128`, **decoder `decoder_num_layers=4` (causal AR, lean by
  design — see posterior collapse)**, `intermediate_size=2048` (decoder ≤ this), `hidden_act=silu`
  (SwiGLU), `norm_type=rmsnorm`, `decoder_pos_type=rope`, `max_seq_length=512`.
- **Compute:** Polonez / Odra, 3–4× RTX 3090, bf16, AdamW-fused + cosine (as in the reference launcher).
- **Steps / epochs:** ~1 epoch over the subsample (≈ prior run's step budget); confirm against kill gates.
- **Launch (env-var overrides on the shared launcher):**
  ```bash
  DECODER_TYPE=causal_ar \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 DECODER_WORD_DROPOUT=0.4 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code (reusable, via `research-implement`):** `ConceptCausalDecoderLayer` +
  `ConceptCausalDecoderStack` (manual q/k/v + **RoPE** on decoder self-attn, SDPA `is_causal=True`) +
  `ConceptEncoderForConditionalLM` in `nn/concept_encoder_perceiver.py`; config fields `decoder_type`
  ("perceiver_posonly" default | "causal_ar"), `decoder_word_dropout` (0.0 default), `norm_type`
  ("layernorm" default | "rmsnorm"), `decoder_pos_type` ("learned" default | "rope"); wire `hidden_act`
  through `ACT2FN` (SwiGLU) and a `build_norm(norm_type)` helper in encoder + decoder; concept-ablation
  eval metric + register the new family in `analysis/run_concept_analysis.py` `MODEL_CLASSES` and
  `evaluation/concept_eval_routing.py`. Preserve the checkpoint eval contract.

> **Amendment 2026-06-11 (before the full run; warm-up evidence).** The 0.3-epoch warm-up
> `concept_ar_H768L6C128D4_20260607_172931` (Polonez) exposed a train/eval protocol mismatch:
> with word-dropout **p=0.4** the decoder specialized to blanked inputs — clean-input eval CE
> *rose* 6.8→9.0 while train CE fell to 3.1, and offline diagnosis on the last checkpoint showed
> CE 13.9 (clean inputs, above random) vs **0.49** under the train-matched word-dropout condition
> (gap 13.4 nats; Δzero even turned negative). Changes for the full run, all evidence-driven:
> - **`DECODER_WORD_DROPOUT` 0.4 → 0.2** (the copy-path guard stays, the distribution shift shrinks);
> - eval now logs **`ce_intact_wd` / `gap_clean_vs_wd`** (train-matched word-dropout CE) so the
>   mismatch is measured, not hidden;
> - **seeded eval collator** (deterministic TSDAE deletions on the held-out set — stable
>   `eval_loss`, fair best-checkpoint selection);
> - `run_concept_analysis.py` label masking fixed for pad=eos tokenizers (positional, not by id).
> Success criterion #4's "eval CE < ~4.0" is judged on the **matched-condition** `ce_intact_wd`
> (and the clean-input CE must not diverge upward); other gates unchanged.
>
> **Amendment 2026-06-11 (second, same day — loss bug).** Pre-E02 code review found a
> **double-shift** in the AR loss: decoder inputs were shift-right-ed *and* the CE shifted
> logits/labels again, so every target `x_t` was predicted from context ending at `x_{t-2}`
> (skip-one objective). This inflated every CE number from the warm-up and explains the
> at-chance no-concept floor. Fixed (`_teacher_forced_ce`, single shift); the full E01 run was
> restarted from scratch on the fixed code. Warm-up CE values are **not comparable** with the
> fixed run; rank / STS-B / ΔCE-as-a-gate remain conceptually valid but will be re-measured.

## Result
- Run id: `concept_ar_H768L6C128D4_20260613_185955`
- WandB: [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_H768L6C128D4_20260613_185955)
- Run report: `docs/2_Experiments_Registry/run_reports/e01_concept_ar_decoder_20260614.md`
- Verdict: **mixed** — AR plumbing confirmed and decoder uses concepts at step 4000 (Δshuffle 1.50, Δzero 1.48 ≥ 0.5 gate; success #1 and #4 pass at checkpoint-4000), but eval CE rises monotonically after the first checkpoint (overfitting), rank collapses from 14.64 → 4.64 by end-of-training, and best STS-B 0.556 misses the 0.62 gate (successes #2 and #3 fail). The reconstruction objective cannot sustain concept quality over a full epoch on FineWeb-Edu.
