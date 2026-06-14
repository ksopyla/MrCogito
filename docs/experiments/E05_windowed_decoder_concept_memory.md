# E05 — Windowed decoder + concepts as cross-window memory (long context)

- **Status:** draft (largest lift; gate behind E04's result; needs an implementation-plan before code)
- **Serves:** the [agenda](../1_Strategy_and_Plans/agenda.md) collapse focus **and** the long-context / "10M-token" Vision. Fuses the two strong levers: a **local sliding window for fluency** + **the 128 concepts as the only carrier of cross-window information**. Structurally bypass-proof in the way the literature says actually works (Gist tokens [2304.08467], ICAE [2307.06945], AutoCompressor [2305.14788]) — the raw out-of-window tokens are **not reachable** except through the concepts.
- **Implementation plan:** E05_windowed_decoder_concept_memory_plan.md *(to author after approval — non-trivial)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-14 · closed —

> One changed variable: the decoder may attend only to the **last K tokens + the concepts** (vs full causal context). Longer sequence length on long, coherent documents is the **enabling condition** for that variable to bite (at 512 tokens almost all dependencies are within any reasonable K), staged in the Plan — not a second hypothesis.

## Hypothesis
If the decoder's causal self-attention is restricted to a **local window of the last K tokens** (e.g. K=128) while cross-window context is available **only** through the 128 concepts, trained on **long coherent documents** at sequence length ≥ 4k, then concepts will become a **genuine long-range memory** — manifold RankMe and early/long-range concept-ablation Δ will rise well above the full-context AR baseline, and long-range prediction (CE on tokens whose true dependency lies >K back) will improve **only when concepts are intact** — because any dependency beyond K *cannot* be served by the local window and must flow through the bottleneck.

## Builds-on
- **Foundation (reuse + extend):** `ConceptEncoderForConditionalLM` (causal AR decoder) + `ConceptCausalDecoderStack` — extend the decoder's `scaled_dot_product_attention` with a **sliding-window causal mask** (reusable `decoder_context_window` config; `None` = current full causal, preserving E01/E02). Shared entrypoint + launcher. Long-doc data pipeline (new loader / packing-aware collator).
- **Init / checkpoint:** random init (Stage A); optionally warm-start from the best E04/E03 checkpoint later.
- **Baseline to beat:** the same-architecture **full-context AR** run at the same long seq-len/data (window=∞), i.e. a matched window-ON/OFF pair; plus E02 (STS-B 0.702) as the semantic reference.

## The single change
**Decoder local context = last-K window** (vs full causal). Implemented as `decoder_context_window=K` on the existing causal decoder. Encoder, concepts (C128), tokenizer, objective held fixed. The enabling condition (seq-len 4k+, long-doc corpus) moves with it and is staged below.

## Success criteria (set BEFORE running)
- **Primary (long-range memory exists):** on tokens with true dependency >K back, **CE(window, concepts intact) < CE(window, concepts zeroed) by ≥ 0.5 nats**, and this gap **grows with sequence length** (4k vs 512). Manifold **RankMe(windowed) ≥ RankMe(full-context) + 8**.
- **Co-primary:** zero-shot **STS-B ≥ 0.65** (semantics not sacrificed for memory).
- **Sanity:** within-window CE comparable to the full-context baseline (the window doesn't break local fluency).

## Kill criteria (set BEFORE running)
- By 25% budget: if the intact-vs-zeroed long-range CE gap is **< 0.2 nats** (concepts not used as memory) → stop.
- If long-context training is unstable (OOM at target seq-len after batch calibration, or NCCL/throughput collapse) → stop, fall back to a shorter seq-len staging.

## Plan
- **Data (from `playground/long_dataset_seq_len_analysis.ipynb`, 2026-06-14):** large-scale pretraining candidate **FinePDFs-100BT** (varied, real long tail; most <8k) for Stage A; ultra-long tails from **LongBlocks** (books/arXiv/Wiki/Stack, many >32k) and **OpenThoughts3** for stress tests. **Avoid packing unrelated short docs** (creates fake long-range signal — confirmed concern in the compression literature). Re-run the length analysis at 1k–10k rows before fixing the mix.
- **Compute:** Polonez preferred (256GB RAM, 4×3090) for long-context memory/throughput; bf16; expect large one-time tokenization.
- **Steps / epochs:** staged — **(0)** calibrate batch at seq 4k; **(1)** short window-ON/OFF pair at seq 4k on FinePDFs to check the long-range CE gap; **(2)** scale seq-len (8k+) / data if the gate clears.
- **Launch (after the foundation `decoder_context_window` lands):**
  ```bash
  EXPERIMENT_ID=E05 DECODER_TYPE=causal_ar DECODER_CONTEXT_WINDOW=128 \
  MAX_SEQ_LENGTH=4096 HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=reconstruction TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=<finepdfs-100bt> SEED=42 NUM_EPOCHS=<staged> \
  PER_DEVICE_BATCH_SIZE=<calibrated> GRADIENT_ACCUMULATION_STEPS=<calibrated> \
  uv run bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code:** (1) `decoder_context_window` sliding-window causal mask in `ConceptCausalDecoderStack` (reusable; default `None` = unchanged). (2) long-document data loader / length-aware batching for FinePDFs/LongBlocks. (3) a long-range-dependency ablation metric (intact-vs-zeroed CE bucketed by dependency distance) in the eval/analysis path.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>` · WandB: <link> · Run report: `<...>`
- Verdict: promising | mixed | regression | killed — <one line>
