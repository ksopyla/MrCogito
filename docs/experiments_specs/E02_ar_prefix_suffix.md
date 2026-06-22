# E02 — Prefix-to-suffix AR generation objective

- **Status:** done (original training/eval 2026-06-13/14 · E02-long eval 2026-06-18 · Tier-2.5 pool probe 2026-06-20)
- **Serves:** the encoder->AR-decoder Current focus in [agenda.md](../1_Strategy_and_Plans/agenda.md). This is the second step in the AR series: keep the E01 AR foundation fixed, then test whether a stronger SODA-style objective forces more semantic concepts.
- **Implementation plan:** [E02_ar_prefix_suffix_plan.md](E02_ar_prefix_suffix_plan.md) *(the HOW)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-06-07 · closed 2026-06-14

> One experiment = one hypothesis = one changed variable. E02 changes the training
> objective only: reconstruction -> prefix-to-suffix AR generation. Do not launch
> until the active E01 baseline has a final checkpoint score recorded below.

## Hypothesis
If the E01 concept-conditioned AR decoder is trained to generate a held-out suffix from concepts produced from only the prefix, then the concept bottleneck will carry more semantic information than AR denoising reconstruction, because the decoder cannot recover target content from its own gold left context or from encoder-visible target tokens.

## Builds-on
- **Foundation:** reuse `ConceptEncoder` + BiXT encoder, `ConceptCausalDecoderStack`, `ConceptEncoderForConditionalLM`, the shared `training/train_perceiver_denoise.py` entrypoint, and `scripts/train_perceiver_denoise_multigpu.sh`. Add only reusable, config-selectable prefix/suffix objective plumbing; no new training fork.
- **Init / checkpoint:** random init, same as E01. This isolates the objective against the E01 from-scratch AR baseline; warm-start remains E04.
- **Baseline to beat:** E01 full run, once recorded. The active E01 run
  `concept_ar_H768L6C128D4_20260607_172931` on Polonez is a **0.3-epoch warmup/plumbing run**, not
  the final baseline. Current early probe at epoch 0.080: eval CE **6.820**, concept ablation Δzero
  **3.12**, Δshuffle **1.35**. E02 may mirror this with a **0.3-epoch Odra warmup**, but the real E02
  verdict must compare a full E02 run against a full E01 run with matched budget.

## The single change
**The objective/data view.** Keep E01 architecture and modern baseline choices fixed, but change the batch contract from AR denoising reconstruction:

`encoder sees deleted/noisy full sequence -> decoder reconstructs the same full sequence`

to prefix-to-suffix AR generation:

`encoder sees clean prefix only -> concepts -> decoder autoregressively predicts suffix + eos`

Everything else is held fixed unless needed for 3-GPU Odra batch-size calibration: dataset family, tokenizer, hidden size, token embedding asymmetry, concept count, encoder depth, decoder depth, SwiGLU, RMSNorm, RoPE, optimizer family, and evaluation probes.

## Success criteria (set BEFORE running)
1. **Semantic quality:** zero-shot STS-B Pearson is at least **E01 final + 0.03** and **>= 0.65**. If E01 does not clear 0.62, E02 must still clear **0.62** to count as useful.
2. **De-collapse:** concept effective rank is at least **E01 final + 16** and **>= 48/128**.
3. **Concept usage:** concept-ablation Δshuffle on suffix CE is **>= 1.0 nats** at final eval, and Δzero is **>= 2.0 nats**.
4. **Trainability:** suffix next-token eval CE is clearly below random (`ln(vocab) ~= 10.8`) and reaches **< 6.0** by the first 25% checkpoint, with coherent held-out suffix samples.

## Kill criteria (set BEFORE running)
- By the first **25% checkpoint**: if suffix eval CE is **>= 8.0**, stop as not learning the objective.
- By the first **25% checkpoint**: if concept-ablation Δshuffle is **< 0.3** and effective rank is **< 16/128**, stop as posterior collapse / unusable concepts.
- By the **50% checkpoint**: if zero-shot STS-B is **< 0.58** and effective rank is still **< 24/128**, stop rather than spending the full Odra run.
- Compute cap: **~90 Odra GPU-hours** without clearing the trainability and concept-usage gates.

## Plan
- **Data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, same preprocessing/tokenizer family as E01. Split each sequence into prefix and suffix using sentence-boundary preference when possible; otherwise token split. Start with a tight continuation task: prefix ratio **0.35-0.45** (roughly 40% prefix / 60% suffix), suffix includes eos so the decoder learns to stop.
- **Tokenizer / format:** `HuggingFaceTB/SmolLM2-135M`; pad aliases eos as in E01. This is a
  **document-continuation** experiment, not an instruction/chat experiment: do **not** wrap FineWeb-Edu
  rows in ChatML / `<|im_start|>` / `<|im_end|>` templates. Use raw text split into prefix and suffix,
  with `<|endoftext|>` as the boundary/stop token. Prefix/suffix collator must support tokenizers with
  no `[CLS]`, `[SEP]`, or `[MASK]`.
- **Model:** same as E01: `hidden_size=768`, `token_embedding_dim=256`, encoder `num_hidden_layers=6`, `concept_num=128`, decoder `decoder_num_layers=4`, `intermediate_size=2048`, `hidden_act=silu`, `norm_type=rmsnorm`, `decoder_pos_type=rope`, `max_seq_length=512`.
- **Context length:** keep `max_seq_length=512` for E02 warmup + first full comparison so E01/E02 stay comparable and iteration remains cheap. Longer context (`1024+`) is a later one-variable experiment after the AR objective line is stable.
- **Compute:** Odra, 3x RTX 3090, bf16. Calibrate `PER_DEVICE_BATCH_SIZE` once after implementation; keep effective batch close to the matching E01 run despite 3 GPUs.
- **Steps / epochs:** first launch is a **0.3-epoch warmup** on Odra to prove the objective, loss, ablations, and samples work. If healthy, run a matched-budget full E02 run (likely **1-2 epochs**, decided after E01 full baseline).
- **Launch (env-var overrides on the shared launcher, after implementation):**
  ```bash
  DECODER_TYPE=causal_ar \
  HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 DECODER_NUM_LAYERS=4 \
  CONCEPT_NUM=128 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
  OBJECTIVE_VARIANT=prefix_suffix DELETION_RATE=0.0 DECODER_WORD_DROPOUT=0.0 \
  PREFIX_RATIO_MIN=0.35 PREFIX_RATIO_MAX=0.45 SPLIT_STRATEGY=sentence_boundary \
  TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
  DATASET_NAME=HuggingFaceFW/fineweb-edu DATASET_SUBSET=sample-10BT \
  NUM_EPOCHS=0.3 TRAIN_NUM_PROC=8 TEST_NUM_PROC=4 DATALOADER_NUM_WORKERS=4 SAVE_TOTAL_LIMIT=5 \
  bash scripts/train_perceiver_denoise_multigpu.sh
  ```
- **New foundation code (reusable, via `research-implement`):** extend the shared entrypoint with `objective_variant="prefix_suffix"`; add or adapt a prefix/suffix collator that works with eos-only causal tokenizers; teach `ConceptEncoderForConditionalLM` to accept `prefix_input_ids`/`prefix_attention_mask` plus `suffix_input_ids`/`labels`; compute suffix next-token CE and concept-ablation CE on suffix targets; expose `PREFIX_RATIO_MIN`, `PREFIX_RATIO_MAX`, and `SPLIT_STRATEGY` in the existing launcher. No new `train_*.py`.

## Result
- Run id: `concept_ar_prefix_H768L6C128D4_20260613_134159`
- WandB: [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260613_134159)
- Run report: `docs/2_Experiments_Registry/run_reports/e02_ar_prefix_suffix_20260614.md`
- Verdict: **mixed / positive** — STS-B 0.702 clears the 0.65 gate with +0.05 margin (success #1 pass; new project best, +0.095 over prior best 0.607). Suffix eval CE 3.52 < 6.0 (success #4 pass). But effective rank 11.57/128 fails the ≥ 48/128 de-collapse gate (success #2 fail) and suffix-CE Δshuffle 0.50 / Δzero 0.73 fail their gates (success #3 fail). The prefix→suffix objective is the better semantic objective — concepts are compact but semantically loaded. Concept geometry collapse remains the primary blocker; E03 is the targeted fix.

### Follow-up: 5-epoch extension (E02-long), evaluated 2026-06-18; Tier-2.5 probe 2026-06-20
- Run id: `concept_ar_prefix_H768L6C128D4_20260614_101305` (Polonez, 4× 3090, 5 epochs, eff batch 160)
- WandB: [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260614_101305) · Run report: `run_reports/e02_long_5epoch_20260618.md`
- Result: longer training **de-collapses** prefix→suffix concepts — slot rank rises 5.9→11.6→16.7 across 0.3/1/5 epochs (the inverse of E01 reconstruction's collapse). Upgraded metrics show genuinely healthy geometry (RankMe 245.9, anisotropy 0.32, mean concept cosine 0.124, 63 dims for 95% var). STS-B **0.714** (new project best, +0.012 over 1-ep E02) but plateaus despite 5× budget + richer geometry.
- Tier-2.5 frozen-encoder pool probe on ck-296000 (Polonez, 2026-06-20): SICK relatedness **mean P −0.203 / attention P 0.133** (Δ**+0.336**; Spearman Δ+0.311), showing distributed concept information hidden from mean pooling. PAWS remains mixed: mean acc/F1 **0.509 / 0.428**, attention acc/F1 **0.546 / 0.377** (accuracy +0.037, F1 −0.051).
- Key reframe: "concept collapse" is objective-dependent, not universal — prefix→suffix concepts improve with scale and should be the objective basis for E05. The attention probe supports the idea that richer concept geometry exists, but naive mean-pool readout underuses it.
