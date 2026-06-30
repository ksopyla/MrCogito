# E05 — Windowed decoder + concepts as cross-window memory

- **Status:** foundation implemented 2026-06-18; docs reconciled 2026-06-25. **2026-06-26: first launch DIVERGED at step ~20 (LR 3e-4 / warmup 500 too hot for seq-2048 + windowed mask; grad_norm → 500k, beyond-window Δzero went negative — kill gate MET at 43% budget). Stopped. Fixed a latent AR-decoder padding-mask bug (suffix_attention_mask was discarded) and made LR/warmup/eval-steps env-overridable.** **2026-06-27: staged proving plan on Odra (3× 3090), mix `smollm3_inspired_2k_e05`, LR 1e-4 / warmup 1500 — (1) E05 1-epoch windowed arm with an early divergence kill-gate, then (2) E05-long 5-epoch matched A/B (windowed + full-causal control) to test de-collapse-with-scale and windowed > control on beyond-window Δ.** **Data-pipeline fix (2026-06-27): the mix is pretokenized via `scripts/pretokenize_mix.py` (the live `load_dataset` path can't cap DCLM's 27,838 `.jsonl.zst` files — ~190 h — and a huge DCLM doc killed a tokenize worker; pretokenize honors `max_shards` and adds a `PRETOKENIZE_MAX_CHARS` huge-doc guard + a `num_proc=1` fallback). Training loads the manifest via `PRETOKENIZED_MANIFEST` (instant `load_from_disk`). This pretokenize→manifest→train spine is the standard data path for all future phases (SFT, SFT+reasoning).** **2026-06-28: the LR 1e-4 / warmup 1500 attempt (`concept_ar_prefix_H768L6C128D4_20260627_192407`) DIVERGED at step ~40k (epoch 0.19, 5.2B tokens seen) — eval_loss 3.32 → 4.03, grad_norm escalated 9 → 56 → 219 → 903 (post-clip 1.0 but bad-direction updates dominated), within-sample RankMe was healthy (59.8) before divergence then collapsed. Best checkpoint-40000 preserved; fast eval on Odra: Tier 0 ✓, Tier 1 within-sample RankMe 59.8 ✓, early-Δshuffle 0.85 ✓, beyond-window Δshuffle 0.35 (below Stage 2 0.5 target). Compute: 81.3 GPU-h / 17.78 kWh / 5.21B tokens (`compute/max_tokens_b`). Re-scoped to 0.5 epoch (~7B tokens, comparable to E02-long's 5-ep compute-hours) and retuned optimization: LR 5e-5, warmup 2000, `max_grad_norm=0.5`, per-device batch 12, eval_steps 4000.** **2026-06-30: attempt 3 (`concept_ar_prefix_H768L6C128D4_20260629_093840`) TRAINING COMPLETE — 0.5 ep / 10.2B tokens / 68.2 GPU-h / 18.24 kWh, no divergence. Batch 8 × accum 3 × 3 GPU = effective 72 (batches 12 and 10 both OOM'd — see run report). eval_loss fell monotonically 5.40 → 3.83 across 17 evals; pre-clip grad_norm held 0.4–0.55 through ~step 48k, then rose to 40–75 during the cosine-tail region without hurting eval_loss (opposite of attempt 2). Best checkpoint-68000 (eval_loss 3.829). Stage 1 read (geometry, Δshuffle_beyond, STS-B) pending eval.**
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
| Budget | **Staged proving:** (1) **0.5-epoch windowed arm** (≈ 7B tokens ≈ 110 GPU-h, comparable in compute-hours to E02-long's 5-epoch 24.5B-token / 290 GPU-h run — starts the rank-rises-with-scale regime at this architecture's seq-2048 token budget), with an early divergence kill-gate; (2) E05-long matched A/B (windowed + full-causal control) once 0.5-ep stability is proven |
| LR / warmup | **5e-5 / 2000** (the 2026-06-28 retune; LR 1e-4 / warmup 1500 diverged at step ~40k / epoch 0.19 — pre-clip grad_norm escalated 9 → 903 while cosine LR was still ~8.5e-5; pre-2026-06-26 LR 3e-4 / warmup 500 diverged at step ~20) |
| `max_grad_norm` | **0.5** (explicit; HF default 1.0 let bad-direction updates through during the 2026-06-28 divergence) |
| Batch (per-device × accum) | **12 × 2 = effective 72** (calibrated 2026-06-28: ~23 GB/GPU at seq 2048 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; was 8 × 2 = 48 before — bigger batch raises per-GPU power toward TDP and smooths gradients) |
| Eval steps | 4000 (was 2000; halves eval-idle GPU-2 stalls; checkpoint still every 2000 via `save_steps`) |
| Server | Odra (3×3090) |
| Data path | **Pretokenize once → train from manifest.** The mix has huge sources (DCLM 27,838 `.jsonl.zst`; FinePDFs 28 GB) that the live `load_dataset` path cannot cap — it tried to download all 27,838 DCLM files (~190 h). `scripts/pretokenize_mix.py` honors `max_shards` (DCLM→35 files) and writes a manifest the trainer loads instantly via `PRETOKENIZED_MANIFEST`. Reusable across the 0.5-ep + matched A/B arms. |

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

### Training — `scripts/launch_e05.sh` (the canonical E05 launcher; both A/B arms)
A thin wrapper that pins the E05 protocol (architecture, mix, objective, K=128, seq 2K) and the
token-matched effective batch, then `exec`s the generic `scripts/train_perceiver_denoise_multigpu.sh`
(which owns all training defaults + the accelerate invocation + the gated pretokenize phase, run
when `PRETOKENIZE_MIX` is set). The optimizer is selected by `OPTIMIZER=adam|muon` — the A/B single
variable; override any other knob by exporting it before the call.

```bash
# Adam arm (Odra). The 1-epoch run crashed and resumes from its last checkpoint:
SKIP_PRETOKENIZE=1 RESUME_FROM_CHECKPOINT=Cache/Training/<run>/checkpoint-<step> \
  OPTIMIZER=adam bash scripts/launch_e05.sh

# Muon arm (Polonez, fresh) — token-matched to the Adam arm: same seed/model/mix/effective-batch/
# epochs, only the optimizer + its LR differ. Pretokenize runs once on Polonez (its own manifest);
# bump workers for speed (output is identical): TRAIN_NUM_PROC=32 TEST_NUM_PROC=8.
OPTIMIZER=muon bash scripts/launch_e05.sh

# Matched full-causal control (unset the window).
SKIP_PRETOKENIZE=1 OPTIMIZER=adam DECODER_CONTEXT_WINDOW= bash scripts/launch_e05.sh
```

`launch_e05.sh` pins (2026-06-28 retune + 2026-06-29 dedup):
LR `5e-5` (adam) / `0.02` (muon matrix) + `MUON_ADAMW_LR=2e-3` fallback, warmup `2000`,
`MAX_GRAD_NORM=0.5`, per-device batch `8` (Odra, 3 GPU) / `6` (Polonez, 4 GPU) × grad-accum `3`
→ **effective batch 72 on both** (identical tokens/step), `EVAL_STEPS=4000`, `SAVE_STEPS=4000`,
`NUM_EPOCHS=0.5` (default — **MUST match the live Adam arm**; the resumed Odra run may be 1 epoch),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**A/B cleanliness caveat:** the Adam arm is the crash-then-resumed Odra run; the Muon arm is fresh
from seed 42. Identical init/data/tokens, but the resume is a noted confound on the Adam arm (resume
step, LR-schedule offset) — record it in the run report when interpreting the optimizer comparison.

**Note on `MAX_GRAD_NORM`:** `train_perceiver_denoise_multigpu.sh` wires `--max_grad_norm`
through the `MAX_GRAD_NORM` env-var (default 1.0 = HF Trainer default). The explicit 0.5 is the
2026-06-28 retune — the 2026-06-28 divergence had pre-clip grad_norm up to 903; post-clip 1.0
still let bad-direction updates dominate the late-run loss landscape, so the tighter 0.5 cap is
the second stability lever alongside the halved LR.

Matched control (Stage 2b): identical to 2a but with `DECODER_CONTEXT_WINDOW` unset (the window
defaults to `None` = full causal). Run 2a and 2b on the same seed/budget; **all three arms load the
same pretokenized manifest**, so the A/B is on identical data.

### Phased training (future: SFT, SFT+reasoning) — same spine
This **pretokenize → manifest → train** pipeline is the standard data path for *all* phases, not just pretraining. The manifest format is objective-agnostic (tokenized dirs + weights + an `objective` field); each phase adds **one tokenize mode + one collator** plugged into the same spine:
- **Pretraining (now):** `--objective prefix_suffix|reconstruction` → tokenized text + `DataCollatorForPrefixGeneration` / `DataCollatorForTSDAE`.
- **SFT (Phase 4):** add an `instruction` tokenize mode (prompt+response, loss masked on the prompt) + a `DataCollatorForSFT`; `load_pretokenized_mix` + the `pretokenized_manifest` training path are reused unchanged.
- **SFT+reasoning (Phase 5):** a `reasoning` tokenize mode (CoT traces) on the same spine.

## Success / kill
- **Stage 1 early kill-gate (0.5 ep, windowed):** if grad_norm > 1e4, loss goes non-finite, or beyond-window Δzero < 0 within the first ~100 steps, stop — same divergence signature as 2026-06-26 / 2026-06-28. **2026-06-28 addition:** also stop if eval_loss rises monotonically over **3 consecutive eval points** (12k steps at `EVAL_STEPS=4000`) — the 2026-06-28 divergence signature was eval_loss climbing from step 40k onward; catching that explicitly avoids burning GPU-hours on a known-failing trajectory.
- **Stage 1 read (0.5 ep, windowed):** beyond-window Δzero & Δshuffle ≥ 0.3 nats (a 0.5-ep checkpoint is below the matched-A/B target but must be positive and rising); STS-B ≥ 0.62; within-sample RankMe rises vs init. If Δ < 0.2 nats at 0.5 ep, do **not** spend the matched A/B budget — stop and diagnose.
- **Stage 2 primary (matched A/B, ≥1 ep):** beyond-window Δzero & Δshuffle ≥ 0.5 nats **AND** windowed > control (`--ablation_window_k 128`); co-report the clean concept-only read at `--ablation_window_k 508`.
- **Stage 2 co-primary:** STS-B ≥ 0.65 (stretch 0.71 vs E02-long).
- **Stage 2 de-collapse-with-scale read:** within-sample RankMe at matched-A/B > at 0.5 ep (mirror the E02-long 5.9 → 11.6 → 16.7 rise across 0.3 / 1 / 5 ep).
- **Kill @ 25% budget (any arm):** beyond-window Δ < 0.2 nats → stop.

### Reading the gate: K-slice vs true local reach
The primary gate slices the ablation metric at `t ≥ K = 128` (`_teacher_forced_ce_window`). The decoder's *true* local receptive field is `L·(K−1) ≈ 508` (stacked window layers, L=4) — so positions in `[128, 508)` still have **partial** local access to far-back tokens, and only `t ≥ 508` is genuinely concept-only. The K-slice is the registered primary (it is where the window starts biting and where most positions sit), but to avoid over-claiming, also co-report a clean concept-only read on both checkpoints:

```bash
uv run python analysis/run_concept_analysis.py ... --ablation_window_k 508
```

The K-slice Δ must clear the primary threshold; the 508-slice Δ is the robustness check (fewer positions, noisier, but unconfounded by partial-local reach). If K-slice passes but 508-slice is flat, concepts are being used but only in the partial-local zone — a weaker, partial result to flag in experiment-track, not a clean win.

## Result
<Filled in AFTER by experiment-track.>
- Attempt 2 (diverged): Run id `concept_ar_prefix_H768L6C128D4_20260627_192407` · WandB [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260627_192407) · Run report [e05_attempt2_diverged_20260628.md](../2_Experiments_Registry/run_reports/e05_attempt2_diverged_20260628.md). **Verdict: attempt 2 — DIVERGED (optimization failure, architecture sound).** Best checkpoint-40000 (eval_loss 3.317, within-sample RankMe 59.8, Δshuffle_beyond 0.35) clears the Stage 1 floor. Retuned (LR 5e-5 / clip 0.5 / batch 12) and re-scoped to 0.5 ep for attempt 3.
- Attempt 3 (completed): Run id `concept_ar_prefix_H768L6C128D4_20260629_093840` · WandB [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260629_093840) · Run report [e05_attempt3_completed_20260630.md](../2_Experiments_Registry/run_reports/e05_attempt3_completed_20260630.md). **Verdict: STAGE 1 PASS / STAGE 2 NOT YET MET.** Optimization win — LR 5e-5 + clip 0.5 + batch 8×accum3 (effective 72) is the proven-stable recipe for the K=128 windowed decoder (no divergence, 0.5 ep / 10.2B tokens / 68.2 GPU-h). Concept health OK: within-sample RankMe **37.67** (not collapsed), Δzero_beyond **6.99** (decoder reads concepts). **Semantic quality weak:** Δshuffle_beyond **0.39** (Stage 1 floor ≥0.3 ✓, Stage 2 target ≥0.5 ✗); **STS-B zero-shot 0.452 below both trivial floors** (token-embed-mean 0.486, teacher-hidden-mean 0.460); free-running generations are repetition loops (token-F1 0.149). SICK-R 0.183, SICK-E 0.634, PAWS 0.550/0.253, GLUE MRPC 0.669/0.778, GLUE STSB 0.354/0.341. Not an architectural dead-end (cf. attempt 2 divergence) — a "more training / stronger objective" signal. Matched A/B now justified. Two eval-script bugs fixed during eval: wandb tag truncation (`730e607`), SmolLM2 pad_token (`70e1fd2`).
