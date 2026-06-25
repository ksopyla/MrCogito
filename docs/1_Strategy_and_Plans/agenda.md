# MrCogito — Research Agenda (living)

**Updated:** 2026-06-20 · The daily driver for *current* work. Overarching direction: [vision_and_goals.md](vision_and_goals.md). Results ledger: [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md). Specs: [../experiments/](../experiments/).

> This is **research / exploration** — the direction is genuinely open. This file
> stays small on purpose: how we work, the immediate focus, and a neutral record
> of what we've learned. It is **not** a committed multi-step plan, and nothing
> here is a final verdict.

## How we work (the process — this is the point)
- Go back to fundamentals. Make **small, well-defined increments** — one change at a time.
- One to a few active experiments, each with a frozen spec in `docs/experiments_specs/<ID>.md` (hypothesis · builds-on · single change · success/kill criteria).
- Build on the **existing foundation**; reuse and extend it, don't fork a script per idea (see `.cursor/rules/experiment-discipline.mdc`).
- Treat every past run as **evidence that improved our understanding**, not as success or failure. Keep conclusions tentative.

## Guiding direction (open)
We still follow the [Vision](vision_and_goals.md): compress sequences into concepts and **reason in latent space**, working toward a multimodal / audio model eventually. *How* we get there is unsettled and under active exploration. Latent-space reasoning stays a central interest — likely explored with a different approach than before.

## Current focus
- **Move from collapse diagnosis to the next architecture test.** E01–E04 are done and evaluated. E02-long (STS-B 0.714, RankMe 246) remains the semantic leader; its Tier-2.5 probe (2026-06-20) confirms distributed concept information hidden from mean pooling (SICK mean −0.203 → attention 0.133, Δ+0.336; PAWS mixed). E03 anchor helps at 0.3 ep reconstruction but should stay an auxiliary lever, not the main research direction. E04 shows bypass removal improves geometry (within-sample RankMe 108, cross-sample 178) but not E02-level semantics.
- **Run state (2026-06-20):** E04 DONE on Odra; E02-long pool probe DONE on Polonez (`concept_ar_prefix_H768L6C128D4_20260614_101305`, ck-296000). **Odra free; Polonez free after eval.** Next main candidate: **E05 from scratch with prefix→suffix objective** — matched full-causal vs windowed-causal context, concepts as the only cross-window memory.
- **Queued (sequential — one experiment per server):**
  1. ~~**E03 matched control**~~ **DONE 2026-06-18**
  2. ~~**E04 parallel decoder**~~ **DONE 2026-06-20** — see "what we've explored".
  3. **E05 windowed decoder from scratch, prefix→suffix objective** — [(spec)](../experiments_specs/E05_windowed_decoder_concept_memory.md) · foundation implemented; spec/plan reconciled 2026-06-25 (K=128 fixed, `prefix_suffix`, `smollm3_inspired_2k` recipe); ready to launch.
  4. **Prefix→suffix + anchor (auxiliary ablation, not main focus):** useful only if E05 needs an anchor/control read, not the default next run.
  5. **Decoder-weakening ablation (sibling of E03):** same E01/E02 stack, single change `DECODER_WORD_DROPOUT=0.5`. Needs a frozen spec before the full run.

### Series roadmap (plan-ahead; each step = ONE variable vs the prior, re-scoped as its own spec)
1. **E01 — AR decoder from scratch** *(done 2026-06-14, mixed).* Modern baseline line: encoder→AR decoder, FineWeb-Edu, SmolLM2 tokenizer, SwiGLU + RMSNorm + RoPE(decoder), ~135M. Best at checkpoint-4000 (Δshuffle 1.50, STS-B 0.556); collapses to rank 4.64 by end. New metric established: concept-ablation ΔCE.
2. **E02 — objective:** [prefix→suffix AR generation](../experiments/E02_ar_prefix_suffix.md)
   *(done 2026-06-14, mixed/positive; STS-B 0.702 new project best).* Stronger semantic pressure than reconstruction; concepts are compact but semantically loaded.
3. **E03 — de-collapse via frozen-encoder anchor** *(anchor-ON warmup done 2026-06-15; control queued).*
   Gate (updated): judge on the **per-sample manifold RankMe + early-Δ**, co-primary with zero-shot
   STS-B (not the slot-mean rank, which understates the geometry). **The shared Stage-A for the two bets below.**
   Sibling cheap ablation: **decoder-weakening** (`DECODER_WORD_DROPOUT=0.5`) — see Queue above.
4. **E04 — concept-only parallel decoder** [(spec)](../experiments_specs/E04_concept_only_parallel_decoder.md)
   *(done 2026-06-20, mixed).* Parallel Perceiver-IO decoder removes AR bypass. Within-sample RankMe 108,
   cross-sample RankMe 178 (+27 vs E03 control); STS-B 0.532 > control 0.485 but << E02 0.702.
5. **E05 — windowed decoder + concepts as cross-window memory** [(spec)](../experiments_specs/E05_windowed_decoder_concept_memory.md) · [(plan)](../experiments_specs/E05_windowed_decoder_concept_memory_plan.md)
   *(largest lift; long-context program; **foundation implemented 2026-06-18**, E04 gate cleared).*
   Local window for fluency + concepts as the ONLY cross-window carrier (Gist/ICAE/AutoCompressor-style).
   **Scoped to seq-len 2K + a long-doc dataset mix** (`DATASET_MIX_RECIPE=smollm3_inspired_2k`: SmolLM3-inspired mix with
   explicit long-tail boosters, ~21% docs >2K). Single change = `DECODER_CONTEXT_WINDOW=128` (fixed) on `causal_ar` +
   `prefix_suffix` objective; matched window-ON/OFF pair; gate = beyond-window concept-ablation Δ
   (`run_concept_analysis.py --ablation_window_k 128`, co-reported at 508 for the concept-only read) + RankMe.
   **K is a fixed coherence window (128), never scaled to N** — the concept count C is what scales with N (per vision);
   raising K would reintroduce O(N·K) local decoding and defeat the bottleneck. Depth caveat: stacked window layers reach
   ≈ `L·(K−1)` ≈ 508 back; if the gate is weak, lower depth or raise seq-len — do not raise K. This is the 10M-token bet's Stage-A.
6. **E06 — latent-space prediction** [(spec)](../experiments/E06_latent_space_prediction.md)
   *(reuses E03 machinery).* Anchor promoted from auxiliary to primary objective (JEPA/data2vec/CPC) —
   learning signal entirely in representation space, no token bypass.
7. **E07 — sentence-gap / boundary-only infilling** [(spec)](../experiments/E07_sentence_gap_infilling.md)
   *(objective change vs E02).* Regenerate removed whole sentences; forces global aggregation through
   concepts (SpanBERT/PEGASUS). Sibling cheap ablation still available: decoder-weakening (`DECODER_WORD_DROPOUT=0.5`).
8. **E08 — recursion (Ouro-style)** *(renumbered from the old E04; gated on de-collapse + a depth bench).*
   Weight-tied concept refinement (test-time compute scaling). Use **Ouro** ([2510.25741](https://arxiv.org/abs/2510.25741))
   **not TRM** (audited [2512.11847](https://arxiv.org/abs/2512.11847)). Only meaningful on **de-collapsed**
   concepts and a depth-dependent reasoning bench, else the step-curve is unfalsifiable.
9. **Diffusion decode — deferred staged program** (not scheduled). Revive **only** as Stage-A
   (E03 anchor, recon-validated) → Stage-B **ELF-style flow + concept-conditioning dropout/CFG**.
   **CALM is *not* a diffusion decoder**; a bare random-init AdaLN-Zero re-run repeats the 5 prior
   diffusion failures — do not. (NB: masked-diffusion/MaskGIT now also a candidate bypass-free decoder for E04/E05.)
6. **Engineering (parallel; not `E0NN` experiments):**
   - **Canonical eval protocol** — tiers, what each measures, and when to run them, live in
     [evaluation_protocol.md](../3_Evaluations_and_Baselines/evaluation_protocol.md). Research track =
     Tiers L0–L4 (direction-finding); **`lighteval` is a separate external-comparability track**,
     deferred until the backbone is proven and we scale (SmolLM2-135M subset for ~135M; SmolLM3 list for
     1B–3B). Don't merge logic; unify the interface.
   - **Concept-information eval upgrade — DONE 2026-06-15** (eval-foundation, read-only). Fixes the
     core misalignment that semantic probes mean-pool away the concept structure: adds (1) within-sample
     concept RankMe as the PRIMARY de-collapse metric (slot-mean rank → secondary; cross-sample RankMe
     → relabeled embedding-diversity), (2) trivial-floor STS-B baselines (`--baseline`), (3) a
     frozen-encoder attention-pool probe (`--pool_mode attention`) that makes distributed-across-concepts
     info visible. GLUE full-finetune demoted from concept-content evidence. Spec:
     [concept_information_eval_upgrade.md](../engineering_specs/concept_information_eval_upgrade.md).
     **Follow-up (needs a GPU box):** re-run the probe + baselines on E01/E02/E03 best checkpoints to
     see whether the anchor buys distributed information that mean-pool was hiding.
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
- **Perceiver denoise reconstruction:** strongest zero-shot STS-B at the time (~0.607) with still-low-rank geometry and mixed supervised signal. Now superseded by E02.
- **E01 — AR denoising reconstruction (FineWeb-Edu, 1 epoch, 2026-06-14):** AR plumbing confirmed; decoder uses concepts early (Δshuffle 1.50 at step 4000). Eval CE rises monotonically thereafter (overfitting); rank collapses 14.64 → 4.64; best STS-B 0.556. Reconstruction + word-dropout insufficient to sustain concept quality over full training. Best checkpoint is an early checkpoint (4000 steps).
- **E02 — prefix→suffix AR generation (FineWeb-Edu, 1 epoch, 2026-06-14):** STS-B **0.702** — new project best, well above prior best (0.607) and E01-best (0.556). Prefix→suffix creates better semantic pressure than reconstruction. Rank still collapsed (11.57/128), suffix-CE ablation modest (Δshuffle 0.50). Key insight: compact geometry can coexist with high STS-B — the active subspace is semantically loaded even when rank is low.
- **Collapse root-cause + measurement reframe (2026-06-14):** deep dive (code + data + lit) concluded "concept collapse" is mostly **(a) a measurement artifact** and **(b) strong-AR-decoder posterior collapse**, not a capacity problem. (a) The headline "effective rank" SVDs the **batch-averaged** concepts, so it measures *slot redundancy*, not representation dimensionality; zero-shot STS-B **mean-pools 128 slots to one vector**, so it's nearly blind to slot rank. New per-sample manifold metric (RankMe) on a live E03 checkpoint: slot-rank **2.7** vs **RankMe ≈24**, anisotropy 0.28, 100% active slots — the usable geometry is far healthier than the slot-rank number implied. (b) The teacher-forced AR decoder bypasses the bottleneck via local context, so required rate through `z`→0 (Bowman 2015; Chen VLAE 2016; Alemi 2018). **Evidence:** E02 all-position ablation `Δzero=0.50` but **early-position `Δzero=1.43`/`Δshuffle=1.04`** (concepts strongly used where bypass is impossible); `gap_clean_vs_wd=0.037` rules out the word-dropout protocol artifact. New tooling (manifold RankMe, anisotropy, per-slot activity, early-Δ as primary gate) committed in `analysis/`. **Implication:** judge de-collapse on the per-sample manifold + early-Δ, not slot-mean rank; the levers are anchor (E03) + decoder weakening, not more concepts or longer training.
- **E03 anchor-ON warmup (FineWeb-Edu, 0.3 epoch, 2026-06-15):** All kill gates pass. Anchor MSE decreasing (0.512), AR CE stable (4.446 < E01-best 4.676), concept ablation healthy (Δshuffle 1.345, Δshuffle_early 3.342 — notably higher early-position signal than prior runs). Slot rank 10.34 held steady at 19k steps (no collapse seen unlike E01). STS-B 0.556 on par with E01-best on the same reconstruction objective. See [run report](../2_Experiments_Registry/run_reports/e03a_anchor_on_warmup_20260615.md).
- **E03 matched control (anchor-OFF, FineWeb-Edu, 0.3 epoch, eval 2026-06-18):** completes the matched pair. Reconstruction at 0.3 ep **collapses without the anchor** — slot rank peaks 17.4 (0.06 ep) then falls to 5.1, STS-B 0.485 (below E01-best 0.556), `gap_clean_vs_wd 1.677` (the decoder bypasses its collapsing concepts via local context). The anchor arm beats the control on every relative metric — RankMe 167 vs 150 (+16.7), STS-B 0.556 vs 0.485 (+0.071), AR CE 4.45 vs 4.79, and decisively gap_clean_vs_wd 0.128 vs 1.677 (13×). **But** absolute gates are unmet at 0.3 ep (STS-B < 0.62; slot rank +4.4 < +16) and the control's higher early-Δ (5.58 vs 3.34) is a *collapse symptom* (fewer directions used harder), not health. Verdict **mixed/promising**. See [run report](../2_Experiments_Registry/run_reports/e03_control_anchor_off_20260618.md).
- **E02-long — prefix→suffix, 5 epochs (FineWeb-Edu, Polonez, eval 2026-06-18; Tier-2.5 probe 2026-06-20):** **the most important reframe so far.** Longer prefix→suffix training **de-collapses** concepts: slot rank rises 5.9 → 11.6 → 16.7 across 0.3/1/5 epochs — the *opposite* of E01 reconstruction (rank 14.6 → 4.6 over 1 epoch). Upgraded metrics show genuinely healthy geometry (RankMe 245.9, anisotropy 0.32, mean concept cosine 0.124, 63 dims for 95% var). STS-B 0.714 (new project best, +0.012 over 1-ep E02) plateaus despite 5× budget + much richer geometry. Tier-2.5 confirms the extra structure is partly distributed across slots: SICK relatedness improves from mean **P −0.203** to attention **P 0.133** (Δ**+0.336**), while PAWS is mixed (accuracy +0.037, F1 −0.051). **Takeaway:** "concept collapse" is **objective-dependent, not universal** — prefix→suffix improves with scale and is the right objective basis for E05. See [run report](../2_Experiments_Registry/run_reports/e02_long_5epoch_20260618.md).
- **E04 — parallel Perceiver-IO decoder, reconstruction (FineWeb-Edu, Odra, eval 2026-06-20):** removes the AR bypass (no token self-attention). **Within-sample RankMe 107.8**, cross-sample RankMe 177.8 (+27 vs E03 control); STS-B 0.532 > control 0.485 but << E02 0.702. Tier-2.5 pool probe: SICK ΔPearson **+0.22** (mean −0.07 → attn 0.16) — distributed geometry partially hidden from mean pool; PAWS inconclusive; absolute semantics still weak. See [run report](../2_Experiments_Registry/run_reports/e04_parallel_decoder_20260620.md).
- **Anchor status after E03/E02-long:** anchoring concepts to frozen pretrained per-token hidden states helps reconstruction relative to a matched control, but it is an auxiliary/de-risking lever, not the main research direction. The architecture-first path is E05 from scratch with prefix→suffix.
- Full history (with caveats): [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md); older roadmap + TODO diary in [5_Archive/](../5_Archive/).

## Not active right now (still part of the Vision)
Recursive concept refinement is now **scheduled as E04** (gated on E03 de-collapse + the eval foundation), and diffusion decode is a **deferred staged program** (see roadmap) — both revivable from `parked/`. Instruction SFT, long-context, and audio remain long-term Vision only. Multi-agent latent communication stays the Stage-2 headline (see [team_brief](../sprind_frontier_ai/team_brief.md)).
