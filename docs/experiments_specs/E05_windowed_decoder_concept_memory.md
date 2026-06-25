# E05 — Windowed decoder + concepts as cross-window memory

- **Status:** foundation implemented 2026-06-18; **launching 2026-06-25** on Odra (Polonez down).
- **Plan:** [E05_windowed_decoder_concept_memory_plan.md](E05_windowed_decoder_concept_memory_plan.md)
- **Owner:** Krzysztof Sopyla · opened 2026-06-14

## Hypothesis
Restrict decoder self-attention to the last **K=128** tokens; cross-window context flows **only** through 128 concepts. At seq-len **2048** with **prefix→suffix** (E02-long basis), concepts become genuine long-range memory — beyond-window ablation Δ rises above a matched full-context control.

Effective receptive field ≈ `L·(K−1)` = **508** tokens (L=4 decoder layers). Most of a 2K sequence is forced through concepts.

## Builds-on
- E02-long semantic leader (STS-B 0.714, RankMe 246, prefix→suffix).
- `ConceptEncoderForConditionalLM` + `decoder_context_window` (sliding-window mask).
- Random init; matched **window-ON/OFF** pair on identical data/seed/budget.

## The single change
`decoder_context_window=128` vs full causal (`None`). Everything else fixed across A/B.

## Launch config (windowed arm — Stage 1 warmup)
| Knob | Value |
|------|-------|
| Objective | `prefix_suffix` |
| Mix | `smollm3_inspired_2k` (recipe) |
| Seq len | 2048 |
| K | 128 |
| Arch | H768 / T256 / L6 / C128 / D4, SwiGLU + RMSNorm + RoPE |
| Tokenizer | SmolLM2-135M |
| Budget | 0.3 epoch warmup, then gate check |
| Server | Odra (3×3090) |
| Cache | `HF_DATASETS_CACHE` — training reuses HF `.map()` cache; pretokenize once before DDP |

```bash
bash scripts/launch_e05_odra.sh          # pretokenize + windowed warmup
SKIP_PRETOKENIZE=1 bash scripts/launch_e05_odra.sh   # re-run only
```

Matched control (Stage 1b): same script with `DECODER_CONTEXT_WINDOW=` unset.

## Success / kill
- **Primary:** beyond-window Δzero & Δshuffle ≥ 0.5 nats; windowed > control (`--ablation_window_k 128`).
- **Co-primary:** STS-B ≥ 0.65 (stretch 0.71 vs E02-long).
- **Kill @ 25% budget:** beyond-window Δ < 0.2 nats → stop.

## Result
<Filled in AFTER by experiment-track.>
- Run id: `<run_id>` · WandB: <link>
- Verdict: —
