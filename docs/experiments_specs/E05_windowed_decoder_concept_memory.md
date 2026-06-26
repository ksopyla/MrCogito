# E05 — Windowed decoder + concepts as cross-window memory

- **Status:** foundation implemented 2026-06-18; docs reconciled 2026-06-25; **launching on Odra** (Polonez down). **2026-06-26: first launch DIVERGED at step ~20 (LR 3e-4 / warmup 500 too hot for seq-2048 + windowed mask; grad_norm → 500k, beyond-window Δzero went negative — kill gate MET at 43% budget). Stopped. Fixed a latent AR-decoder padding-mask bug (suffix_attention_mask was discarded) and made LR/warmup/eval-steps env-overridable. Sanity relaunch pending: LR 1e-4 / warmup 1500 / 0.05 epoch.**
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

## Launch config (windowed arm — Stage 1 warmup)
| Knob | Value |
|------|-------|
| Objective | `prefix_suffix` |
| Mix | `smollm3_inspired_2k_e05` (recipe, incl. DCLM) |
| Seq len | 2048 |
| K (fixed) | 128 |
| Arch | H768 / T256 / L6 / C128 / D4, SwiGLU + RMSNorm + RoPE |
| Tokenizer | SmolLM2-135M |
| Budget | 0.3 epoch warmup, then gate check |
| Server | Odra (3×3090) |
| Cache | `HF_DATASETS_CACHE` — first DDP run warms the HF `.map()` cache (tokenization runs under `main_process_first`); no separate pretokenize step. |

There is **no dedicated E05 launcher script**. Both arms are env-var invocations of the shared `scripts/train_perceiver_denoise_multigpu.sh`, which already wires `DECODER_CONTEXT_WINDOW` and `DATASET_MIX_RECIPE` (passed only when set, so all prior launches are unchanged).

```bash
# Windowed arm (K=128)
EXPERIMENT_ID=E05 \
DECODER_TYPE=causal_ar DECODER_CONTEXT_WINDOW=128 \
DATASET_MIX_RECIPE=smollm3_inspired_2k MAX_SEQ_LENGTH=2048 \
OBJECTIVE_VARIANT=prefix_suffix \
HIDDEN_SIZE=768 TOKEN_EMBEDDING_DIM=256 NUM_LAYERS=6 CONCEPT_NUM=128 \
DECODER_NUM_LAYERS=4 INTERMEDIATE_SIZE=2048 HIDDEN_ACT=silu \
NORM_TYPE=rmsnorm DECODER_POS_TYPE=rope \
TOKENIZER_NAME=HuggingFaceTB/SmolLM2-135M \
NUM_EPOCHS=0.3 PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8 \
DDP_TIMEOUT=14400 \
bash scripts/train_perceiver_denoise_multigpu.sh

# Matched control (Stage 1b): identical line, but omit DECODER_CONTEXT_WINDOW
# (keep DECODER_TYPE=causal_ar; the window defaults to None = full causal).
```

Matched control (Stage 1b): same invocation with `DECODER_CONTEXT_WINDOW` unset.

## Success / kill
- **Primary:** beyond-window Δzero & Δshuffle ≥ 0.5 nats; windowed > control (`--ablation_window_k 128`).
- **Co-primary:** STS-B ≥ 0.65 (stretch 0.71 vs E02-long).
- **Kill @ 25% budget:** beyond-window Δ < 0.2 nats → stop.

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
