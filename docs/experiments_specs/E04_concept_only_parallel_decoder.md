# E04 — Concept-only parallel decoder (remove the AR bypass)

- **Status:** draft
- **Serves:** the [agenda](../1_Strategy_and_Plans/agenda.md) "attack concept collapse at the root" focus. Where E03 *adds pressure* (anchor) to make concepts richer, E04 *removes the escape hatch*: it tests whether the chronic collapse is driven by the **autoregressive decoder bypass** (teacher-forced local context lets the decoder reconstruct tokens without the bottleneck, so required rate through `z`→0). The cheapest decisive test of the root-cause diagnosis.
- **Implementation plan:** E04_concept_only_parallel_decoder_plan.md *(to author after approval)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-14 · closed —

> One experiment = one hypothesis = one changed variable. Implementation is a config
> (`DECODER_TYPE=perceiver_posonly`) over the shared entrypoint — NOT a new fork.

## Hypothesis
If we replace the causal AR decoder with a **position-query, concept-only parallel decoder** (output queries cross-attend to the 128 concepts; **no causal token self-attention**, so no teacher-forced local side-channel), holding the encoder, data, tokenizer, sequence length and reconstruction objective fixed, then the **per-sample representation manifold rank (RankMe) and early-position concept-ablation Δ will be ≥ the matched causal-AR baseline**, because the decoder can no longer reconstruct tokens from local context and must route all information through the concepts — testing whether the AR bypass (not capacity, not optimization) is the collapse driver, **without any auxiliary loss**.

## Builds-on
- **Foundation (reuse, no fork):** `ConceptEncoderForDenoisingPerceiver` (the maintained parallel/Perceiver-IO decoder, `nn/concept_encoder_perceiver.py`), `ConceptEncoder` + BiXT encoder, `DataCollatorForTSDAE`, shared entrypoint `training/train_perceiver_denoise.py` + launcher `scripts/train_perceiver_denoise_multigpu.sh`. Selected via `DECODER_TYPE=perceiver_posonly`.
- **Init / checkpoint:** random init (same as E01).
- **Baseline to beat:** the **E03 anchor-OFF control** (= fresh E01-recon causal-AR baseline, same data/seed/budget) on the **new metrics**; reference points: E01 STS-B 0.556, E02 STS-B 0.702 / early-Δzero 1.43; manifold RankMe of E01/E02 to be (re)computed with the new tooling on existing checkpoints.

## The single change
**Decoder architecture:** causal AR (`decoder_type=causal_ar`, token self-attention) → **concept-only parallel** (`decoder_type=perceiver_posonly`, position-only queries, cross-attend concepts, no token self-attention). Everything else identical to the E03 control: FineWeb-Edu `sample-10BT`, SmolLM2-135M tokenizer, `hidden_size=768`, `token_embedding_dim=256`, encoder `L6`, `C128`, `max_seq=512`, objective `reconstruction`, `deletion_rate=0.6`, no anchor, no concept losses.

> **Verify (foundation check, not a fork):** the `perceiver_posonly` decoder queries must be **position-only** — assert no input-token embedding leaks into the queries (that would be a different leakage). If the current path adds input embeddings, add a config flag `decoder_query_mode=position_only` (reusable, default preserves old behaviour).

## Success criteria (set BEFORE running)
- **Primary:** manifold **RankMe(E04) ≥ RankMe(control) + 8** AND **early-position Δzero(E04) ≥ Δzero(control)** (concepts more necessary once the bypass is gone).
- **Co-primary:** zero-shot **STS-B(E04) ≥ STS-B(control)** (removing the bypass should not cost semantics; ≥ 0.607 prior best is a stretch goal).
- Reconstruction CE is allowed to be **worse** than the AR baseline (parallel decode is less fluent) — this is a representation probe, not a generation run.

## Kill criteria (set BEFORE running)
- By **25% of the step budget:** if manifold RankMe **and** early-Δzero are both **≤ control**, the bypass is not the driver → stop and reweight toward E03/E06.
- If eval loss diverges or NaNs in the first 2 evals → stop, debug.

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu` `sample-10BT` (== E01/E03), SmolLM2-135M tokenizer.
- **Compute:** Odra 3× RTX 3090 (or Polonez when free), bf16. Cheap — no teacher, no extra head.
- **Steps / epochs:** matched to the E03 control (0.3-epoch warmup gate → 1-epoch if it clears).
- **Launch:**
  ```bash
  EXPERIMENT_ID=E04 DECODER_TYPE=perceiver_posonly \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm \
  OBJECTIVE_VARIANT=reconstruction DELETION_RATE=0.6 \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  SEED=42 NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=24 GRADIENT_ACCUMULATION_STEPS=2 \
  LEARNING_RATE=3e-4 WARMUP_STEPS=500 LOGGING_STEPS=100 EVAL_STEPS=1000 SAVE_STEPS=1000 \
  SAVE_TOTAL_LIMIT=3 DDP_TIMEOUT=3600 uv run bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code (if any):** likely none (config only). Possibly a small reusable `decoder_query_mode=position_only` flag if the verification finds query-side token leakage.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>` · WandB: <link> · Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
