# MrCogito — Research Agenda (living)

**Updated:** 2026-06-06 · The daily driver for *current* work. Overarching direction: [vision_and_goals.md](vision_and_goals.md). Results ledger: [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md). Specs: [../experiments/](../experiments/).

> This is **research / exploration** — the direction is genuinely open. This file
> stays small on purpose: how we work, the immediate focus, and a neutral record
> of what we've learned. It is **not** a committed multi-step plan, and nothing
> here is a final verdict.

## How we work (the process — this is the point)
- Go back to fundamentals. Make **small, well-defined increments** — one change at a time.
- One to a few active experiments, each with a frozen spec in `docs/experiments/<ID>.md` (hypothesis · builds-on · single change · success/kill criteria).
- Build on the **existing foundation**; reuse and extend it, don't fork a script per idea (see `.cursor/rules/experiment-discipline.mdc`).
- Treat every past run as **evidence that improved our understanding**, not as success or failure. Keep conclusions tentative.

## Guiding direction (open)
We still follow the [Vision](vision_and_goals.md): compress sequences into concepts and **reason in latent space**, working toward a multimodal / audio model eventually. *How* we get there is unsettled and under active exploration. Latent-space reasoning stays a central interest — likely explored with a different approach than before.

## Current focus
- **Shift to the encoder→AR-decoder paradigm.** Perceiver-*inspired* (keep BiXT + token↔concept
  asymmetry), re-confirm the bottleneck at a slightly bigger, modern scale, and address the two
  standing weaknesses: concept collapse and a decoder that cannot generate. First step:
  **[E01 — concept-conditioned autoregressive decoder](../experiments/E01_concept_ar_decoder.md)**
  (spec + [plan](../experiments/E01_concept_ar_decoder_plan.md)) — *draft, pending go-ahead before
  implementation.* All current decoders are parallel/non-AR (`p(x|concepts)=Πₚ p(xₚ|concepts,p)`);
  E01 adds a real AR decoder (causal self-attn over tokens + cross-attn to concepts) so the encoder
  produces concepts and the decoder *generates*.

### Series roadmap (plan-ahead; each step = ONE variable vs the prior, re-scoped as its own spec)
1. **E01 — AR decoder from scratch** *(this spec).* New **modern** baseline line: encoder→AR decoder,
   FineWeb-Edu, SmolLM2 tokenizer, SwiGLU + RMSNorm + RoPE(decoder), ~135M. Key new metric:
   concept-ablation ΔCE (is the decoder actually using concepts?).
2. **E02 — objective:** prefix→suffix AR generation (strongest semantic pressure; the AR decoder is the
   materially-new ingredient vs the previously-failed random-init prefix track).
3. **E03 — token↔concept asymmetry sweep:** vary `token_embedding_dim` (e.g. 128/256/512) vs E01.
4. **E04 — decoder warm-start (Flamingo-style):** gated cross-attention into a pretrained **SmolLM2-135M**
   AR decoder (concepts condition a strong LM); SmolLM2 tokenizer.
5. **E05 — reasoning:** recursive/weight-tied concept refinement between encoder and decoder
   (test-time compute scaling) — unparks the recursive family onto the AR foundation.
6. **Further knobs** (one-at-a-time, later): optimizer (AdamW → Muon/Lion trial), longer context,
   concept-count `C` scaling with `N`, RoPE in the encoder cross-attention (its own experiment —
   ill-defined on orderless concepts, needs care).

## What we've explored so far (evidence, not verdicts)
- **Reference baseline:** `perceiver_mlm_H512L6C128_20260208_211633` — just a comparison anchor (MRPC 82.7 / STS-B 0.650 via ViaDecoder; concept effective rank ~5/128). Not a target, not "good."
- **MLM + concept losses** (combined / kendall_gal / fixed): pushing concept diversity tended to cost downstream semantics — a tension worth remembering.
- **Diffusion (self-reconstruction, ELBO, VICReg) and prefix diffusion:** explored on MiniPile / WikiText-103; concept effective rank stayed low so far. **Set aside (parked), likely to revisit** — especially with warm-start. Code in `parked/`.
- **Perceiver denoise reconstruction:** strongest zero-shot STS-B so far (~0.607) with still-low-rank geometry and mixed supervised signal.
- Full history (with caveats): [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md); older roadmap + TODO diary in [5_Archive/](../5_Archive/).

## Not active right now (still part of the Vision)
Recursive concept refinement / latent reasoning, instruction SFT, long-context, and audio remain part of the long-term Vision — just not the current focus. Recursive and diffusion code is set aside in `parked/` (revivable). We won't plan these in detail until a direction is chosen.
