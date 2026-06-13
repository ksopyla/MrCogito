# MrCogito — Research Agenda (living)

**Updated:** 2026-06-13 · The daily driver for *current* work. Overarching direction: [vision_and_goals.md](vision_and_goals.md). Results ledger: [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md). Specs: [../experiments/](../experiments/).

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
- **Attack concept collapse at the root, on the encoder→AR-decoder foundation.** E01/E02 confirmed the
  AR plumbing; the chronic blocker behind *every* bet is still **concept collapse** (effective rank
  5–10/128 across ~60 runs). The next fundamental step is
  **[E03 — de-collapse via a frozen-encoder hidden-state anchor](../experiments/E03_concept_anchor_decollapse.md)**
  (spec + [plan](../experiments/E03_concept_anchor_decollapse_plan.md)): add an auxiliary loss where the
  128 concepts must reconstruct a **frozen SmolLM2-135M's per-token hidden states** (MSE), as **one
  variable vs E01**, run as a **matched anchor-ON/OFF pair** (the OFF arm is a fresh E01 baseline, so it
  needs neither Polonez nor E01-final). This is the shared "validate the bottleneck first" Stage-A that
  both the diffusion and recursion bets depend on (2026-06-13 scout evidence: Cosmos/LDLM/CALM).
- **Run state (2026-06-13):** E01 **ended on Polonez**; Polonez is **down (needs restart)** so its final
  rank/STS-B are not yet synced. **E02 (prefix→suffix) is active on Odra.** E03 queues after Odra frees
  or Polonez returns; it is cheap (frozen-teacher forward + lean MSE head).

### Series roadmap (plan-ahead; each step = ONE variable vs the prior, re-scoped as its own spec)
1. **E01 — AR decoder from scratch** *(ended on Polonez; final baseline pending sync).* Modern baseline
   line: encoder→AR decoder, FineWeb-Edu, SmolLM2 tokenizer, SwiGLU + RMSNorm + RoPE(decoder), ~135M.
   New metric: concept-ablation ΔCE.
2. **E02 — objective:** [prefix→suffix AR generation](../experiments/E02_ar_prefix_suffix.md)
   *(active on Odra).* Strongest semantic pressure on the AR foundation.
3. **E03 — de-collapse via frozen-encoder anchor** *(next; spec+plan written).* Per-token SmolLM2-h MSE
   distillation, auxiliary to E01 AR, matched ON/OFF pair. Gate: rank ≥ control+16 & ≥32/128, co-primary
   with zero-shot STS-B. **The shared Stage-A for the two bets below.**
4. **E04 — recursion (Ouro-style):** weight-tied concept refinement between encoder and decoder
   (test-time compute scaling). Use **Ouro** ([2510.25741](https://arxiv.org/abs/2510.25741)) **not TRM**
   (audited [2512.11847](https://arxiv.org/abs/2512.11847)). Only meaningful on **de-collapsed** concepts
   (needs E03) **and a depth-dependent reasoning bench** (needs the eval foundation below), else the
   step-curve is unfalsifiable.
5. **Diffusion decode — deferred staged program** (not the next experiment). Revive **only** as Stage-A
   (E03 anchor, recon-validated) → Stage-B **ELF-style flow + concept-conditioning dropout/CFG**.
   **CALM is *not* a diffusion decoder** (continuous-AR + energy head); a bare random-init AdaLN-Zero
   re-run repeats the 5 prior diffusion failures — do not.
6. **Engineering (parallel; not `E0NN` experiments):**
   - **Unified eval interface** — one orchestrator + report schema over the existing intrinsic probes
     (rank, anti-collapse, zero-shot STS-B, ΔCE) **now**; add a **`lighteval` (SmolLM3 list)** backend
     **later, once a checkpoint generates coherently**. Don't merge logic; unify the interface.
   - **Cross-tokenizer / dim embedding-transfer module** for warm-start — **FVT/OMP (vocab) + truncated
     SVD (dim)**; bare PCA covers only the dim leg. Build when the first warm-start run needs it (E03
     does not — it trains from scratch with a *frozen* teacher).
7. **Parked / not scheduled:** the **token↔concept asymmetry sweep** (`token_embedding_dim` 128/256/512,
   the former "E03") is demoted to a **P1-era E02 ablation**, not a headline experiment. Further knobs
   later (optimizer Muon/Lion, longer context, `C`-vs-`N` scaling, encoder-side RoPE).

## What we've explored so far (evidence, not verdicts)
- **Reference baseline:** `perceiver_mlm_H512L6C128_20260208_211633` — just a comparison anchor (MRPC 82.7 / STS-B 0.650 via ViaDecoder; concept effective rank ~5/128). Not a target, not "good."
- **MLM + concept losses** (combined / kendall_gal / fixed): pushing concept diversity tended to cost downstream semantics — a tension worth remembering.
- **Diffusion (self-reconstruction, ELBO, VICReg) and prefix diffusion:** explored on MiniPile / WikiText-103; concept effective rank stayed low so far. Code in `parked/`. **2026-06-13 lit scan (CALM/ELF/Cosmos/LDLM/Nemotron):** our 5 failures match a *known* failure mode — an **unvalidated bottleneck + decoder bypass**, not just bugs. Reviving needs a materially-new ingredient (frozen-encoder MSE anchor + concept-dropout/CFG, ideally warm-start), not a re-run. For "do concepts carry semantics?", **AR + ΔCE is the cleaner probe** than diffusion.
- **Recursive / latent-reasoning (Ouro/Huginn/TRM lit scan, 2026-06-13):** recurrent-depth is real but **task-selective** — gains show on multi-step/compositional benches, often flat on plain denoise/STS; **measurement is the bottleneck**. Use **Ouro**, not TRM (its ARC headline was audited down to ensemble + puzzle-ID lookup + shallow step-1). Only worth running on **de-collapsed** concepts (hence E03 first).
- **Perceiver denoise reconstruction:** strongest zero-shot STS-B so far (~0.607) with still-low-rank geometry and mixed supervised signal.
- **The shared missing ingredient (working hypothesis):** anchoring concepts to **frozen pretrained per-token hidden states** (MSE-to-h) is the de-collapse lever both lines need — and the one ingredient the SPRIND P1 matrix doesn't yet contain. E03 tests it directly.
- Full history (with caveats): [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md); older roadmap + TODO diary in [5_Archive/](../5_Archive/).

## Not active right now (still part of the Vision)
Recursive concept refinement is now **scheduled as E04** (gated on E03 de-collapse + the eval foundation), and diffusion decode is a **deferred staged program** (see roadmap) — both revivable from `parked/`. Instruction SFT, long-context, and audio remain long-term Vision only. Multi-agent latent communication stays the Stage-2 headline (see [team_brief](../sprind_frontier_ai/team_brief.md)).
