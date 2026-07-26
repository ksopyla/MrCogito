# E07 — Sentence-gap / boundary-only long-span infilling

- **Status:** canceled — rejected before implementation/run because it was an auxiliary
  objective change rather than the selected architecture-level direction
- **Serves:** the [agenda](../../1_Strategy_and_Plans/agenda.md) collapse focus. A pure **objective** change that makes the target intrinsically global: mask whole sentences (or long interior spans) and regenerate them from the concepts, with the decoder denied the within-span tokens — so the early span positions cannot be served by local context (the same mechanism behind E02's strong early-Δ, generalized to interior, bidirectional spans). Grounded in SpanBERT boundary-only [1907.10529] and PEGASUS gap-sentence generation [1912.08777]; the literature is explicit that **long spans help only when within-span tokens are hidden** (T5 μ=10 ≤ μ=3 otherwise).
- **Implementation plan:** E07_sentence_gap_infilling_plan.md *(to author after approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-14 · closed —

> One changed variable vs E02: the data **objective/collator** — sentence-gap masking instead of prefix→suffix split. Same model/decoder/encoder/data corpus.

## Hypothesis
If we train the encoder→concepts→AR-decoder to **regenerate masked whole sentences** (gap-sentence generation: select ~k% of sentences, remove them from the encoder input, target = the removed sentences), with the decoder having **no access to the within-gap tokens**, then early-position concept-ablation Δ and manifold RankMe will exceed E02's, because reconstructing a removed sentence requires aggregating the surrounding document **through the concepts** — local continuation cannot supply a span the model never sees.

## Builds-on
- **Foundation (reuse + extend):** `ConceptEncoderForConditionalLM` (AR decoder, prefix/suffix forward already supports "encoder sees context, decoder generates disjoint target"), shared entrypoint. New: a **`DataCollatorForSentenceGap`** (reusable collator) producing `(encoder_input = doc with k% sentences removed, target = removed sentences)`. Sentence boundaries via the existing boundary-token logic in `DataCollatorForPrefixGeneration`.
- **Init / checkpoint:** random init.
- **Baseline to beat:** **E02** (prefix→suffix; STS-B 0.702, early-Δzero 1.43) — same family, different masking.

## The single change
**Objective/collator:** prefix→suffix split → **sentence-gap infilling** (remove k% of sentences, regenerate them). Mask ratio `k` (start ~25–30%) is the dial on how much global aggregation is forced. Model, decoder, encoder, tokenizer, corpus, seq-len held at E02.

## Success criteria (set BEFORE running)
- **Primary:** early-position **Δzero ≥ E02 (≥ 1.43)** AND manifold **RankMe ≥ E02**.
- **Co-primary:** zero-shot **STS-B ≥ 0.70** (≥ E02 — sentence-level reconstruction should retain/improve semantics).

## Kill criteria (set BEFORE running)
- By 25% budget: if early-Δzero and manifold RankMe are both **≤ E02** at matched steps → the sentence-gap objective is not better than prefix→suffix → stop.
- If the gap task is too hard (eval CE near random, no descent) → reduce mask ratio `k` once; if still flat, stop.

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu` `sample-10BT`, SmolLM2-135M tokenizer — **deliberately == E02** so the only change vs the E02 baseline is the *objective* (using a long-doc corpus here would confound objective-vs-data; long-doc distribution is E05's variable, not E07's). E07 is a global-aggregation test, **not** a long-range-memory test, so moderate-length multi-sentence text suffices. Safeguard against degenerate short docs: **filter to documents with ≥ 4 sentences** (so a 25–30% gap removes ≥ 1 whole sentence with real surrounding context); drop docs below that. A **long-doc follow-up (E07b)** on FinePDFs/LongBlocks (from `playground/long_dataset_seq_len_analysis.ipynb`) is a separate single-variable step *after* E07 wins on FineWeb-Edu — or it composes naturally into E05.
- **Compute:** Odra/Polonez 3–4× RTX 3090, bf16. Comparable cost to E02.
- **Steps / epochs:** matched to E02 (0.3-epoch gate → 1 epoch).
- **Launch (after the `DataCollatorForSentenceGap` lands; new `OBJECTIVE_VARIANT=sentence_gap`):**
  ```bash
  EXPERIMENT_ID=E07 DECODER_TYPE=causal_ar OBJECTIVE_VARIANT=sentence_gap GAP_SENTENCE_RATIO=0.30 \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 CONCEPT_NUM=128 \
  INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  SEED=42 NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=2 \
  uv run bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code:** `DataCollatorForSentenceGap` (reusable, in `data/data_collators.py`) + an `OBJECTIVE_VARIANT=sentence_gap` route in the entrypoint (mirrors the existing `prefix_suffix` branch — the model's prefix/suffix forward is reused unchanged).

## Result
No run was launched. The design was deliberately canceled before implementation.
