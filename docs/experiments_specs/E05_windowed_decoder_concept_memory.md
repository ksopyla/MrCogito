# E05 — Windowed decoder + concepts as cross-window memory (long context)

- **Status:** foundation IMPLEMENTED 2026-06-18 (`decoder_context_window` sliding-window mask + `long_2k_base_v1` dataset mix + beyond-window ablation metric; smoke-green, 7 unit tests). **Not yet launched** (gate behind E04, which is running on Odra). Scope set to **seq-len 2K** + a **dataset mix** per the 2026-06-17/18 discussion.
- **Serves:** the [agenda](../1_Strategy_and_Plans/agenda.md) collapse focus **and** the long-context / "10M-token" Vision. Fuses the two strong levers: a **local sliding window for fluency** + **the 128 concepts as the only carrier of cross-window information**. Structurally bypass-proof in the way the literature says actually works (Gist tokens [2304.08467], ICAE [2307.06945], AutoCompressor [2305.14788]) — the raw out-of-window tokens are **not reachable** except through the concepts.
- **Implementation plan:** [E05_windowed_decoder_concept_memory_plan.md](E05_windowed_decoder_concept_memory_plan.md) *(written 2026-06-18; foundation built)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-14 · closed —

> One changed variable: the decoder may attend only to the **last K tokens + the concepts** (vs full causal context). Longer sequence length (**2K**) on long, coherent documents is the **enabling condition** for that variable to bite (at 512 tokens almost all dependencies are within any reasonable K), staged in the Plan — not a second hypothesis.

## Hypothesis
If the decoder's causal self-attention is restricted to a **local window of the last K tokens** (default **K=256**) while cross-window context is available **only** through the 128 concepts, trained on **long coherent documents** at sequence length **2048**, then concepts will become a **genuine long-range memory** — manifold RankMe and beyond-window concept-ablation Δ will rise well above the full-context AR baseline, and prediction on positions beyond the window's reach (`t ≥ decoder_num_layers·(K−1)`) will improve **only when concepts are intact** — because any dependency past the window's effective receptive field *cannot* be served locally and must flow through the bottleneck.

> **Effective-window caveat (found during implementation, 2026-06-18):** sliding-window attention's receptive field grows with depth — L stacked window-K layers reach ≈ `L·(K−1)` tokens back (Mistral-SWA behaviour, verified in `tests/test_e05_windowed_decoder.py::test_window_receptive_field_grows_with_depth`). With `decoder_num_layers=4`, `K=256` the local field is ≈ 1020 tokens, so at seq-len 2048 only the second half of each sequence is genuinely forced through the concepts. Choose K (and the beyond-window metric boundary) against `L·(K−1)`, not K alone. To force concepts harder, lower K or `decoder_num_layers`.

## Builds-on
- **Foundation (reuse + extend):** `ConceptEncoderForConditionalLM` (causal AR decoder) + `ConceptCausalDecoderStack` — the decoder's `scaled_dot_product_attention` now takes a **sliding-window causal mask** built from `decoder_context_window` (`None` = full causal, flash `is_causal` path; E01/E02/E03 unchanged). Shared entrypoint + launcher. Multi-dataset mix loader (`load_and_preprocess_dataset_mix`, registry `DATASET_MIXES`).
- **Init / checkpoint:** random init (Stage A); optionally warm-start from the best E04/E03 checkpoint later.
- **Baseline to beat:** the same-architecture **full-context AR** run at the **same 2K seq-len + same mix** (`decoder_context_window=None`), i.e. a **matched window-ON/OFF pair** on identical data/seed/budget; plus E02 (STS-B 0.702) / E02-long (0.714) as the semantic reference.

## The single change
**Decoder local context = last-K window** (vs full causal). Implemented as `decoder_context_window=K` on the existing causal decoder. Encoder, concepts (C128), tokenizer, objective, data and seq-len held fixed across the A/B. The enabling condition (seq-len **2048**, long-doc mix) is shared by **both** arms, so it is not a second variable within the experiment.

## Success criteria (set BEFORE running)
- **Primary (long-range memory exists):** on **beyond-window** positions (`t ≥ K`, reported as `concept_ablation/delta_*_beyond_window`), **Δzero ≥ 0.5 nats AND Δshuffle ≥ 0.5 nats** for the windowed arm, and **windowed beyond-window Δ > control beyond-window Δ** (judged offline at a fixed boundary via `run_concept_analysis.py --ablation_window_k K`). Manifold **RankMe(windowed) ≥ RankMe(control) + 8**.
- **Co-primary:** zero-shot **STS-B ≥ 0.65** (semantics not sacrificed for memory; stretch ≥ 0.71 vs E02-long).
- **Sanity:** within-window CE comparable to the full-context baseline (the window doesn't break local fluency).

## Kill criteria (set BEFORE running)
- By 25% budget: if the windowed arm's beyond-window intact-vs-zeroed Δ is **< 0.2 nats** (concepts not used as memory) → stop.
- If long-context training is unstable (OOM at 2K after batch calibration, or NCCL/throughput collapse) → stop, fall back to a shorter seq-len / smaller mix.

## Plan
- **Data — `long_2k_base_v1` mix** (registered in `data/dataset_preprocess.py:DATASET_MIXES`; interleaved by sampling weight; **no packing of unrelated short docs**). Weights chosen from the 1k-row seqlen sample (`playground/training_dataset_catalog.ipynb`, SmolLM2 tokenizer):
  | Source | hf_id | weight | % docs > 2K | role |
  |---|---|---|---|---|
  | FinePDFs-100BT | `HuggingFaceFW/finepdfs_100BT` (parquet shards 0–7) | 0.50 | 34.2% | long-range backbone (real coherent docs) |
  | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` `sample-10BT` | 0.30 | 8.6% | quality web + continuity with E01–E04 |
  | FineMath-3+ | `HuggingFaceTB/finemath` `finemath-3plus` | 0.20 | 14.7% | coherent math/reasoning structure |
  Each source capped (`max_samples`) to bound disk/compute; short docs are kept un-packed (they simply don't exercise the window). Stress-test extensions for later stages (not in the launch mix): peS2o, LongBlocks, OpenThoughts3 (ultra-long, but SFT/multi-column).
- **Compute:** Polonez preferred (256 GB RAM, 4×3090) for the 2K memory/throughput; bf16; expect a large one-time tokenization of the mix.
- **Steps / epochs:** staged — **(0)** calibrate batch at seq 2K; **(1)** short **window-ON/OFF** pair at 2K on the mix to check the beyond-window Δ gate (~0.3 epoch warmup); **(2)** full 1-epoch run if it clears.
- **Launch (foundation already landed):**
  ```bash
  # windowed arm (E05)
  EXPERIMENT_ID=E05 DECODER_TYPE=causal_ar DECODER_CONTEXT_WINDOW=256 \
  DATASET_MIX=long_2k_base_v1 MAX_SEQ_LENGTH=2048 \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  SEED=42 NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=<calibrated> GRADIENT_ACCUMULATION_STEPS=<calibrated> \
  EVAL_BATCH_SIZE=4 LEARNING_RATE=3e-4 WARMUP_STEPS=500 LOGGING_STEPS=100 \
  EVAL_STEPS=1000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=3 DDP_TIMEOUT=7200 \
  uv run bash scripts/train_perceiver_denoise_multigpu.sh

  # matched control = same line WITHOUT DECODER_CONTEXT_WINDOW (full causal)
  ```
- **Foundation code (LANDED 2026-06-18):** (1) `decoder_context_window` sliding-window causal mask in `ConceptCausalDecoderLayer`/`ConceptCausalDecoderStack` (`build_sliding_window_causal_mask`; reusable, default `None`). (2) multi-dataset mix loader `load_and_preprocess_dataset_mix` + `DATASET_MIXES["long_2k_base_v1"]` + `--dataset_mix` / `DATASET_MIX` knob. (3) beyond-window concept-ablation metric (`concept_ablation_ce(..., window_k=K)` → `delta_*_beyond_window`, logged live for the windowed arm; offline via `run_concept_analysis.py --ablation_window_k K` for both arms). Tests: `tests/test_e05_windowed_decoder.py` (7, green).

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>` · WandB: <link> · Run report: `<...>`
- Verdict: promising | mixed | regression | killed — <one line>
