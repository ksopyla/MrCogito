# E05 — Windowed decoder + concepts as cross-window memory

- **Status:** foundation implemented 2026-06-18; docs reconciled 2026-06-25. **2026-06-26: first launch DIVERGED at step ~20 (LR 3e-4 / warmup 500 too hot for seq-2048 + windowed mask; grad_norm → 500k, beyond-window Δzero went negative — kill gate MET at 43% budget). Stopped. Fixed a latent AR-decoder padding-mask bug (suffix_attention_mask was discarded) and made LR/warmup/eval-steps env-overridable.** **2026-06-27: staged proving plan on Odra (3× 3090), mix `smollm3_inspired_2k_e05`, LR 1e-4 / warmup 1500 — (1) E05 1-epoch windowed arm with an early divergence kill-gate, then (2) E05-long 5-epoch matched A/B (windowed + full-causal control) to test de-collapse-with-scale and windowed > control on beyond-window Δ.** **Data-pipeline fix (2026-06-27): the mix is pretokenized via `scripts/pretokenize_mix.py` (the live `load_dataset` path can't cap DCLM's 27,838 `.jsonl.zst` files — ~190 h — and a huge DCLM doc killed a tokenize worker; pretokenize honors `max_shards` and adds a `PRETOKENIZE_MAX_CHARS` huge-doc guard + a `num_proc=1` fallback). Training loads the manifest via `PRETOKENIZED_MANIFEST` (instant `load_from_disk`). This pretokenize→manifest→train spine is the standard data path for all future phases (SFT, SFT+reasoning).**
- **Plan:** [E05_windowed_decoder_concept_memory_plan.md](E05_windowed_decoder_concept_memory_plan.md)
- **Owner:** Krzysztof Sopyla · opened 2026-06-14

## Hypothesis
Restrict decoder self-attention to the last **K=128** tokens; cross-window context flows **only** through 128 concepts. At seq-len **2048** with **prefix→suffix** (E02-long basis), concepts become genuine long-range memory — beyond-window ablation Δ rises above a matched full-context control.

Effective receptive field ≈ `L·(K−1)` = **508** tokens (L=4 decoder layers). Most of a 2K sequence is forced through concepts.

## The single change
`decoder_context_window=128` vs full causal (`None`). Everything else fixed across A/B.

## K is a fixed constant, not a scaling knob
**K=128 is a fixed coherence window, held constant across the long-context program.** It exists only to keep generated text locally fluent (token-to-token coherence within a sentence/clause); it is **not** a context window and must **not** grow with sequence length. This is load-bearing for the architecture's reason to exist:

- The vision (`vision_and_goals.md`) is **O(C·N)** — the concept count **C** scales with N, and the decoder is supposed to reason from concepts, not from a growing local window.
- If K grew with N, the decoder's self-attention would reintroduce an O(N·K) cost the concept bottleneck was built to avoid, and decoding would drift back to local copy instead of routing through concepts — defeating the experiment.
- So the thing that scales with N is **C**, not **K**. E05 holds both C=128 and K=128 fixed; later long-context runs raise N (and eventually C), never K.

The `decoder_context_window` config field remains (the control arm needs `None`), but it is a fixed hyperparameter per experiment, never auto-tuned to N.

## Builds-on
- E02-long semantic leader (STS-B 0.714, RankMe 246, prefix→suffix).
- `ConceptEncoderForConditionalLM` + `decoder_context_window` (sliding-window mask).
- Random init; matched **window-ON/OFF** pair on identical data/seed/budget.

## Launch config (staged proving on Odra)
| Knob | Value |
|------|-------|
| Objective | `prefix_suffix` |
| Mix | `smollm3_inspired_2k_e05` (recipe, incl. DCLM) — **pretokenized** via `scripts/pretokenize_mix.py` |
| Seq len | 2048 |
| K (fixed) | 128 |
| Arch | H768 / T256 / L6 / C128 / D4, SwiGLU + RMSNorm + RoPE |
| Tokenizer | SmolLM2-135M |
| Budget | **Staged proving:** (1) 1 epoch, windowed arm only + early divergence kill-gate; (2) E05-long 5 epoch, matched A/B (windowed + full-causal control) |
| LR / warmup | 1e-4 / 1500 (the 2026-06-26 fix; 3e-4 / 500 diverged) |
| Batch (per-device × accum) | 8 × 2 = effective 48 (calibrated 2026-06-27; ~12 GB/GPU plain-CE, throughput knee) |
| Server | Odra (3×3090) |
| Data path | **Pretokenize once → train from manifest.** The mix has huge sources (DCLM 27,838 `.jsonl.zst`; FinePDFs 28 GB) that the live `load_dataset` path cannot cap — it tried to download all 27,838 DCLM files (~190 h). `scripts/pretokenize_mix.py` honors `max_shards` (DCLM→35 files) and writes a manifest the trainer loads instantly via `PRETOKENIZED_MANIFEST`. Reusable across the 1-ep + 5-ep A/B arms. |

There is **no dedicated E05 launcher script**. The workflow is two env-var invocations of shared scripts: (1) `scripts/pretokenize_mix.py` (one-time per mix), (2) `scripts/train_perceiver_denoise_multigpu.sh` with `PRETOKENIZED_MANIFEST` (replaces the live `DATASET_MIX_RECIPE` path, which is retained only as a small-dataset fallback). The launcher already wires `PRETOKENIZED_MANIFEST` and `DECODER_CONTEXT_WINDOW` (passed only when set, so prior launches are unchanged).

### Data pipeline — pretokenize once (reusable for all E05 arms + future phases)
```bash
# One-time: tokenize the e05 mix. Honors max_shards (DCLM→35 files, FinePDFs→8); the
# PRETOKENIZE_MAX_CHARS guard pre-truncates gigantic web/PDF docs before the Fast tokenizer
# sees them, so a huge DCLM page can't OOM/crash a tokenize worker. Idempotent: sources
# already tokenized under datasets_tok/<name>/ are skipped on rerun.
source scripts/remote_paths.sh
uv run python scripts/pretokenize_mix.py \
  --mix smollm3_inspired_2k_e05 \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --max_seq_length 2048 --objective prefix_suffix \
  --download_workers 8 --jobs 1 --train_num_proc 8 --test_num_proc 4
# → writes $HF_DATASETS_CACHE/../datasets_tok/smollm3_inspired_2k_e05_manifest.json
```

### Training — loads the manifest (instant, no tokenization at train time)
```bash
# Shared arch env (set once per byobu window); only NUM_EPOCHS + the window differ per arm.
export DECODER_TYPE=causal_ar
export PRETOKENIZED_MANIFEST=/home/ksopyla/dev/hf_home/datasets_tok/smollm3_inspired_2k_e05_manifest.json
export MAX_SEQ_LENGTH=2048
export OBJECTIVE_VARIANT=prefix_suffix
export HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 CONCEPT_NUM=128
export DECODER_NUM_LAYERS=4 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu
export NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope
export TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M
export LEARNING_RATE=1e-4 WARMUP_STEPS=1500          # the 2026-06-26 fix (3e-4/500 diverged)
export PER_DEVICE_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2   # calibrated 2026-06-27: effective batch 48; ~12 GB/GPU, throughput knee

# --- Stage 1: E05 1-epoch, windowed arm (prove stable training + de-collapse) ---
EXPERIMENT_ID=E05 DECODER_CONTEXT_WINDOW=128 NUM_EPOCHS=1 \
  bash scripts/train_perceiver_denoise_multigpu.sh

# --- Stage 2a: E05-long 5-epoch, windowed arm ---
EXPERIMENT_ID=E05 DECODER_CONTEXT_WINDOW=128 NUM_EPOCHS=5 \
  bash scripts/train_perceiver_denoise_multigpu.sh

# --- Stage 2b: E05-long 5-epoch, matched control (full causal; omit the window) ---
EXPERIMENT_ID=E05 NUM_EPOCHS=5 \
  bash scripts/train_perceiver_denoise_multigpu.sh
```

Matched control (Stage 2b): identical to 2a but with `DECODER_CONTEXT_WINDOW` unset (the window
defaults to `None` = full causal). Run 2a and 2b on the same seed/budget; **all three arms load the
same pretokenized manifest**, so the A/B is on identical data.

### Phased training (future: SFT, SFT+reasoning) — same spine
This **pretokenize → manifest → train** pipeline is the standard data path for *all* phases, not just pretraining. The manifest format is objective-agnostic (tokenized dirs + weights + an `objective` field); each phase adds **one tokenize mode + one collator** plugged into the same spine:
- **Pretraining (now):** `--objective prefix_suffix|reconstruction` → tokenized text + `DataCollatorForPrefixGeneration` / `DataCollatorForTSDAE`.
- **SFT (Phase 4):** add an `instruction` tokenize mode (prompt+response, loss masked on the prompt) + a `DataCollatorForSFT`; `load_pretokenized_mix` + the `pretokenized_manifest` training path are reused unchanged.
- **SFT+reasoning (Phase 5):** a `reasoning` tokenize mode (CoT traces) on the same spine.

## Success / kill
- **Stage 1 early kill-gate (1 ep, windowed):** if grad_norm > 1e4, loss goes non-finite, or beyond-window Δzero < 0 within the first ~100 steps, stop — same divergence signature as 2026-06-26.
- **Stage 1 read (1 ep, windowed):** beyond-window Δzero & Δshuffle ≥ 0.3 nats (a 1-ep checkpoint is below the 5-ep target but must be positive and rising); STS-B ≥ 0.62; RankMe rises vs init. If Δ < 0.2 nats at 1 ep, do **not** spend the 5-ep budget — stop and diagnose.
- **Stage 2 primary (5 ep, A/B):** beyond-window Δzero & Δshuffle ≥ 0.5 nats **AND** windowed > control (`--ablation_window_k 128`); co-report the clean concept-only read at `--ablation_window_k 508`.
- **Stage 2 co-primary:** STS-B ≥ 0.65 (stretch 0.71 vs E02-long).
- **Stage 2 de-collapse-with-scale read:** RankMe / slot rank at 5 ep > at 1 ep (mirror the E02-long 5.9 → 11.6 → 16.7 rise across 0.3 / 1 / 5 ep).
- **Kill @ 25% budget (any arm):** beyond-window Δ < 0.2 nats → stop.

### Reading the gate: K-slice vs true local reach
The primary gate slices the ablation metric at `t ≥ K = 128` (`_teacher_forced_ce_window`). The decoder's *true* local receptive field is `L·(K−1) ≈ 508` (stacked window layers, L=4) — so positions in `[128, 508)` still have **partial** local access to far-back tokens, and only `t ≥ 508` is genuinely concept-only. The K-slice is the registered primary (it is where the window starts biting and where most positions sit), but to avoid over-claiming, also co-report a clean concept-only read on both checkpoints:

```bash
uv run python analysis/run_concept_analysis.py ... --ablation_window_k 508
```

The K-slice Δ must clear the primary threshold; the 508-slice Δ is the robustness check (fewer positions, noisier, but unconfounded by partial-local reach). If K-slice passes but 508-slice is flat, concepts are being used but only in the partial-local zone — a weaker, partial result to flag in experiment-track, not a clean win.

## Result
<Filled in AFTER by experiment-track.>
- Run id: `<run_id>` · WandB: <link>
- Verdict: —
