# MrCogito — Research Agenda (living)

**Updated:** 2026-08-14 · The daily driver for *current* work. Overarching direction: [vision_and_goals.md](vision_and_goals.md). Results ledger: [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md). Specs: [experiments_specs](../experiments_specs/).

> This is **research / exploration** — the direction is genuinely open. This file
> stays small on purpose: how we work, the immediate focus, and a neutral record
> of what we've learned. It is **not** a committed multi-step plan, and nothing
> here is a final verdict.

## How we work (the process — this is the point)
- Go back to fundamentals. Make **small, well-defined increments** — one change at a time.
- One to a few active experiments, each with a frozen spec in
  `docs/experiments_specs/ahead/<ID>.md` (hypothesis · builds-on · single change ·
  success/kill criteria).
- Build on the **existing foundation**; reuse and extend it, don't fork a script per idea
  (see `.cursor/rules/project-overview.mdc`).
- Treat every past run as **evidence that improved our understanding**, not as success or failure. Keep conclusions tentative.

## Guiding direction (open)
We still follow the [Vision](vision_and_goals.md): compress sequences into concepts and **reason in latent space**, working toward a multimodal / audio model eventually. *How* we get there is unsettled and under active exploration. Latent-space reasoning stays a central interest — likely explored with a different approach than before.

## Current focus
- **Current experiment — [E17c depth-private gated working memory](../experiments_specs/ahead/E17c_depth_private_working_memory.md)**
  ([implementation plan](../experiments_specs/ahead/E17c_depth_private_working_memory_plan.md)).
  E17c keeps E17's strictly block-causal four-bank topology and makes each depth a complete
  private memory cell: dedicated read projections, an untied BiXT writer, selective
  retain/replace dynamics, and causal carry dropout so prior-block information is sometimes
  available only through concepts. The decisive pre-registered gate is carryless first-64
  `Δpermutation ≥0.20` by 300M; a null at 100M (`<0.05`) kills early.
- **Why the previous next step changed (architecture/evaluation audit 2026-08-14):**
  E17b's mid-init writes briefly opened then closed, showing that another scalar-init cell
  does not address the learning incentive. More importantly, E16/E16b
  `shared_depth_recurrent` writes current-block information after an early global layer and
  rereads that state at later global layers in the **same block**; its BiXT writer is not
  token-causal. The large E16b/shared+0.3 teacher-forced ΔCE therefore cannot be treated as
  clean causal-memory evidence. Preserve those runs as historical results, but do not use
  shared-depth as the target topology. E17/E17b avoid this leak because each layer rereads
  only its own prior-block bank.
- **What E17/E17b actually established:** the per-layer family keeps free-run in prose
  (`real`@256 **0.20/0.60** vs shared E16b **0.04/0.94**) and is block-causal, but plain
  next-token CE does not sustain content-bearing writes (`Δshuffle_beyond≈0.004–0.006`).
  E17c therefore tests a working-memory **cell + information-pressure** hypothesis rather
  than per-layer + init 0.3. E17a's untied-additive writer remains a possible post-positive
  ablation, not a prerequisite.
- **Still open / still wanted:** [E08 Concept-Flow reasoner](../experiments_specs/ahead/E08_concept_flow_reasoner.md)
  (latent reasoning composition — preferably on a platform that already carries
  concepts), diffusion revive from `parked/` if a materially new ingredient appears,
  and design-only [E11](../experiments_specs/ahead/E11_memtoken_concept_memory.md) /
  [E12](../experiments_specs/ahead/E12_perlayer_kv_prefix_concepts.md) /
  [E13](../experiments_specs/ahead/E13_layerwise_recurrent_kv_memory.md).
  Priorities shifted; exploration stays multi-path.
- **Background references:** E02-long remains the from-scratch semantic reference
  (STS-B 0.714). E10–E16a / E14–E15 remain valid evidence about short-ctx / sparse
  recall regimes — lower priority for the next budget, not erased. E17 init-0.01
  ([done_success/E17](../experiments_specs/done_success/E17_four_bank_concept_memory.md) ·
  [1B gen](../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md))
  remains the per-layer free-run baseline.

### Series roadmap (genealogy; each step is one registered coherent bet)
1. **E01 — AR decoder from scratch** *(done 2026-06-14, mixed).*
2. **E02 — objective:** [prefix→suffix AR generation](../experiments_specs/done_success/E02_ar_prefix_suffix.md)
   *(done 2026-06-14, mixed/positive; STS-B 0.702).*
3. **E03 — de-collapse via frozen-encoder anchor** *(done).*
4. **E04 — concept-only parallel decoder** [(spec)](../experiments_specs/done_success/E04_concept_only_parallel_decoder.md)
   *(done 2026-06-20, mixed).*
5. **E05 — windowed decoder** [(spec)](../experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md)
   *(done_failed — used but semantically empty at <200M; motivated the Gemma pivot).*
6. **E10–E16a short-ctx Gemma line** *(done_failed / mixed — useful regime evidence; see ledger).*
7. **E16b long-ctx Muon** *(done_success 2026-07-25 — validated long-context path; follow-ups + other routes still open).*
8. **E17 four-bank per-layer (init 0.01)** *(done_success mixed 2026-08-10 — relative free-run win; writes dead).*
9. **E17b per-layer mid write-init 0.1** *(done_failed 2026-08-13 — mid-init not sticky; free-run ≈E17).*
10. **E17c depth-private gated working memory + causal carry pressure** *(draft 2026-08-14).*

## What we've explored so far (evidence, not verdicts)
- **E17b mid-init 0.1 (per_layer_banks, Polonez, train 2026-08-10→13, Tier-1+1.5 2026-08-13):**
  write gates opened near ~100M (max \|tanh\| 0.14) then closed to ~0.05 by 1B; RankMe 68;
  Δshuf/static≥1024 **0.0055/0.0033**; free-run `real` greedy @256 **0.20/0.60** (E17 **0.21/0.59**;
  E16b **0.04/0.94**). Mid write-init alone is not sticky under plain CE. See
  [report](../2_Experiments_Registry/run_reports/e17b_per_layer_mid_write_init_20260813.md).
- **E17 low-init 1B (per_layer_banks, Polonez, train 2026-08-07→10, Tier-1.5 2026-08-10):**
  matched init 0.01 vs E16b finished 1B with writes still dead (`|tanh|≤0.033`), Δbeyond ~0.004,
  RankMe 98. Free-run `real` greedy @256 **0.21/0.59** (E16b **0.04/0.94**; base **0.16/0.71**) —
  absolute success bar missed; relative lift vs E16b (prose, `real≈zero`, long prompts help).
  The topology is block-causal, but its write mechanism never engaged. See
  [report](../2_Experiments_Registry/run_reports/e17_lowinit_1b_generation_20260810.md).
- **E16b free-run generation vs base Gemma (Odra, Tier-1.5, 2026-08-01) — FAIL on generation; teacher-forced mechanism interpretation revised 2026-08-14:**
  matched continuation bank + context sweep on `checkpoint-7900` vs `gemma-3-1b-pt`.
  E16b `real` greedy @256: distinct-1 **0.04** / REP-3 **0.94** (digit/punctuation
  attractors); base sample @256 REP-3 **0.03**. Longer prompt prefixes help base and
  hurt E16b free-run. Chat template is not the fix.
  **Layer-0 decode probe (same day, rp=1.2 + `frozen` mode, commit `8a6bafa`):** `zero`
  is the *only* fluent mode (greedy d1@256 **0.74** / r3 **0.01**, base-like); `frozen`
  degenerates like `real` → **refutes "self-writes poison free-run"**; the driver is the
  **concept read pathway** reading a near-static `z` (write gates ≈0 → z ≈ learned
  constant). `repetition_penalty` doesn't help (turns loops into structured junk);
  sampling does. Backbone + LoRA + windowed-global-attention are sound (`zero` is
  fluent) — we did **not** break Gemma. See
  [report](../2_Experiments_Registry/run_reports/e16b_generation_quality_assessment_20260801.md).
- **E16b long-context Muon 1B (Gemma-3-1B, Odra, train 2026-07-18→20, Tier-1 2026-07-25) — recorded success; causal interpretation revised 2026-08-14:**
  shared-depth workspace at seq 4096 on `e16b_long_4k_v1` with Muon for 1B tokens
  reached offline RankMe **101** and Δshuffle/Δstatic≥1024 **2.47/2.35** (clears the
  0.01 gate by a large margin; E16a Muon was 0.0028 at 100M/2K). Δone-block≥1024
  **0.58** showed accumulated multi-block state under the registered protocol. **Revised
  reading:** the implementation rereads non-token-causal current-block writes at later
  same-block depths, so these deltas establish predictive use but not clean causal
  cross-block memory. Free-run separately failed; retain the numbers as historical
  evidence, but do not build new work on the shared-depth topology.
  See [run report](../2_Experiments_Registry/run_reports/e16b_longctx_muon_1b_20260725.md).
- **E16 shared depth-recurrent workspace (Gemma-3-1B, Odra, 2026-07-14):**
  the 50M run kept healthy geometry (within-sample RankMe 62.2; centered 125.0) and
  eval CE 1.8122, but beyond-local static/shuffle deltas were only
  +0.000499/+0.001018 nats, both below 0.01. Interleaved tied writes at four Gemma
  depths did not establish persistent concept use *under 2K CE*; the same architecture
  later cleared the registered metric under E16b’s long-context regime, whose causal
  interpretation was revised on 2026-08-14. See
  [spec](../experiments_specs/done_failed/E16_shared_depth_recurrent_concepts.md).
- **E15 supervision-calibrated delayed recall (Gemma-3-1B, Odra, 2026-07-13):** after
  resuming E14 to 12,000 total answer labels, the block-2 explicit-carry control was still
  below chance (0.98% versus 1.56%; required ≥80%). Thus more exposure alone does not make the
  one-answer-per-2K sparse task learnable, and E15 did not test the E10e memory interface. See
  [run report](../2_Experiments_Registry/run_reports/e15_supervision_calibrated_delayed_recall_20260713.md).
- **E14 forced delayed recall (Gemma-3-1B, Odra, 2026-07-13):** the registered 2M-token stop
  fired (all block-4 memory margins <0.0036 nats), while healthy geometry persisted (RankMe 91.4).
  Because the block-2 explicit-carry control also stayed at chance after only 984 supervised
  answers, the run exposed an input-token-vs-supervision budgeting flaw rather than isolating
  writer retention or read integration. See
  [run report](../2_Experiments_Registry/run_reports/e14_forced_delayed_recall_gate_20260713.md).
- **E10e calibrated concept memory at 100M (Gemma-3-1B, Odra, 2026-07-13):** versus the
  same-budget E10 pilot, CE fell 1.8150→1.7972 and within-sample RankMe rose 77.1→99.9, but
  beyond-local static/shuffle deltas reached only +0.000962/+0.001613 nats; more calibrated
  plain-CE exposure did not yield persistent memory use. See
  [run report](../2_Experiments_Registry/run_reports/e10e_calibrated_memory_100m_20260713.md).
- **E10b normalized concept read (Gemma-3-1B, Odra, 2026-07-12):** at the ~25M decision
  checkpoint, geometry remained healthy (RankMe 112.2; centered 125.1) and local CE matched E10,
  but static−real was only +0.000371 and Δshuffle +0.000179 at positions ≥1024; read normalization
  alone did not create persistent multi-block usage. See
  [run report](../2_Experiments_Registry/run_reports/e10b_normalized_concept_read_20260712.md).
- **E10 100M concept-arm pilot (Gemma-3-1B, Odra, 2026-07-11):** stable and non-collapsed
  (RankMe 77.1; centered 123.1), but every beyond-local recurrent-state ablation was <0.001 nats;
  the matched control is still required before judging the primary recovery criterion. See
  [run report](../2_Experiments_Registry/run_reports/e10_100m_concept_pilot_20260711.md).
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
- **Long-context engineering, round 2 (2026-06-27):** the real memory wall was the **output head**, not the encoder — F2's chunked CE secretly retained `[B,N,V]` in the autograd graph (fixed by `ChunkedLMHeadCE`; **256K now fits on one 3090**). **Sequence parallelism (F6) reaches 1M context on 3× 3090 at 22.6 GB/GPU** (validated ≡ single-GPU to ~1e-6 in loss + all grads). **Muon** converges ~2× faster than AdamW on wikitext-103. 10M is the hardware ceiling for 24 GB cards (needs 80 GB cards / ~30 GPUs). Full note: [long_context_memory_optimization_round2_2026_06_27.md](../4_Research_Notes/long_context_memory_optimization_round2_2026_06_27.md).
- **E05 attempt 2 — windowed decoder, diverged at step 40k (Odra, killed + fast-evaled 2026-06-28):** the LR 1e-4 / warmup 1500 retune of attempt 1 (which diverged at step ~20 under LR 3e-4) **delayed but did not prevent** divergence — same signature, just at step 40k / epoch 0.19 / 5.2B tokens instead of step 20: eval_loss 3.32 → 4.03 over 12k steps, pre-clip grad_norm escalated 9 → 56 → 219 → 903 while cosine LR was still ~8.5e-5. **Architecture is sound** — best checkpoint-40000 fast-evaled clears Stage 1 floor on every gate except the beyond-window Δshuffle target (within-sample RankMe **59.8**, early-Δshuffle **0.85**, beyond-window Δshuffle **0.35** ≥ floor 0.3 but < Stage 2 target 0.5). Compute: **81.3 GPU-h / 17.78 kWh / 5.21B tokens** (`compute/max_tokens_b`). **Takeaway:** divergence is optimization-side, not architectural — cosine-kept-hot + HF-default `max_grad_norm=1.0` let bad-direction updates dominate the sharpening late-run loss landscape. Retune for attempt 3: LR 5e-5, `max_grad_norm` 0.5 (now wired through launcher), batch 12, re-scoped to 0.5 ep (~7B tokens). See [run report](../2_Experiments_Registry/run_reports/e05_attempt2_diverged_20260628.md).
- **E05 attempt 3 — windowed decoder, completed 0.5 ep + evaluated (Odra, 2026-06-30):** the LR 5e-5 / clip 0.5 retune of attempt 2 **held — first fully completed E05 run**, and the eval is now in. Training: 0.5 ep / 10.2B tokens / 68.2 GPU-h / 18.24 kWh; eval_loss fell monotonically 5.40 → 3.83 across 17 evals; grad_norm held 0.4–0.55 through step ~48k, then rose to 40–75 in the cosine-tail region (LR ≈ 1e-6) without hurting eval_loss — the *opposite* signature from attempt 2's escalation. **Stage 1 PASS:** within-sample RankMe **37.67** (not collapsed), Δzero_beyond **6.99** (decoder reads concepts), Δshuffle_beyond **0.39** (≥ floor 0.3). **Stage 2 NOT YET MET:** Δshuffle_beyond < target 0.5; **STS-B zero-shot 0.452 is below both trivial floors** (token-embed-mean 0.486, teacher-hidden-mean 0.460) — the concept bottleneck currently *destroys* semantic-similarity signal vs averaging raw token embeddings. Free-running generations are grammatical but semantically empty repetition loops (token-F1 0.149, exact-match 0.015). SICK-R 0.183, SICK-E acc 0.634, PAWS 0.550/0.253, GLUE MRPC 0.669/0.778, GLUE STSB 0.354/0.341 (full-finetune, demoted evidence). **Takeaway:** optimization succeeded and the architecture is stable — this is a "more training / stronger objective" signal, not an architectural dead-end (cf. attempt 2's divergence). Matched A/B now justified. Two eval-script bugs fixed during eval (wandb tag truncation `730e607`, SmolLM2 pad_token `70e1fd2`). See [run report](../2_Experiments_Registry/run_reports/e05_attempt3_completed_20260630.md).
- **E05 Muon A/B — optimizer arm (Odra, 2026-07-02):** stabilized Muon (wd=0.1, adamw_lr=2e-4, LR 0.01) **converges ~5× faster + to a 1.22-nat-lower eval_loss (2.606 vs Adam's 3.83)**, stable end-to-end (no divergence). Naive Muon diverged at LR 0.02 and 0.01 — root-caused (Moonlight arXiv:2502.16982: full-rank orthogonalized updates grow weight spectral norms in the Q·K + lm_head bilinear couplings, with no weight decay to curb it + an over-hot `adamw_lr=2e-3`) and fixed (wd=0.1 + `adamw_lr=2e-4`) + a **sustained-LR (`constant_with_warmup`) calibration protocol** (a short cosine hid the delayed onset — a calibration-fidelity lesson). **Takeaway:** the optimizer is a first-class, ready lever — a compute-efficiency win for E06–E08 (re-validate Muon per objective). **Caveat:** eval_loss ≠ concept semantics (Adam had 3.83 but STS-B 0.45); the Muon semantic/geometry verdict (STS-B, RankMe, Δshuffle) awaits its eval suite. **Eval (2026-07-04, ⚠️ TENTATIVE): split** — downstream semantics up (STS-B **0.518** clearing floors, SICK-R 0.302, GLUE MRPC/STSB improved) BUT concept geometry + long-range gate **regressed** (within-sample RankMe **10.57** vs 37.67, Δshuffle_beyond **0.209** vs 0.39, Δzero_beyond 0.41 vs 6.99) → the lower loss came from the decoder's within-window bypass, not richer concepts. **Not decisive** (wd confound Muon 0.1 vs Adam 0.0; authoritative prefix→suffix `concept_ablation/*` not yet read; single seed) — open for discussion + literature on optimizer-vs-representation-collapse. Adam wd=0.0 vs Muon wd=0.1 confound (wd is part of "what Muon needs to be stable"). See [run report](../2_Experiments_Registry/run_reports/e05_muon_divergence_rootcause_20260701.md).
- **E05 Muon long (2 ep) — compute-matched to E02-long, evaluated (Odra, 2026-07-09):** the 2-ep Muon run (`concept_ar_prefix_H768L6C128D4_20260704_225659`, **300.88 GPU-h / 85.25 kWh / 40.78 B tokens** ≈ E02-long's 290.7 GPU-h) tested whether more compute de-collapses the 0.5-ep Muon bottleneck. **Result: REGRESSION — more compute collapsed it harder.** Stable end-to-end (grad_norm 0.4–0.8 mid, ~3.7 cosine tail, no divergence), eval_loss fell to a project-low **2.581** (best ckpt-272000). But **every concept gate regressed vs the 0.5-ep Muon arm**: within-sample RankMe **4.96/128** (was 10.57; centered 4.61 → genuine collapse, not offset), slot-mean 1.66, mean concept cosine 0.892; Δshuffle_beyond **0.227** (was 0.209 — flat from step 4k in W&B `concept_ablation/*`, so the decoder's long-range concept use was stationary for the whole 2-ep run); **STS-B zero-shot 0.062** (was 0.518 — now 0.42 *below* the token-embed-mean floor 0.486). Frozen-probe SICK-R mean→attn 0.048→0.160 (Δ+0.112, small distributed component). SICK-R 0.111, SICK-E 0.626, PAWS 0.562/0.305, GLUE MRPC 0.699/0.815, STSB 0.341, QQP 0.807, MNLI-m 0.613 (full-finetune, demoted). **Takeaway:** the "is it under-trained?" hypothesis for the windowed+Muon regime is **falsified** — the K=128 within-window bypass is the attractor; extra optimization makes the bypass better, not the concepts richer (opposite of E02-long's full-causal de-collapse where 5→16.7). eval_loss is orthogonal to concept quality (loss ↓ 2.606→2.581 while RankMe ↓ 10.6→5.0 and STS-B ↓ 0.518→0.062). **Closes the E05 from-scratch "more compute" branch.** Open confounds (do not block the pivot): ~~wd (Muon 0.1 vs Adam 0.0)~~ — **RESOLVED 2026-07-11 by [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md): wd is innocent**; Tier-1 protocol split (0.5-ep old / 2-ep new — seq-512 flattens collapse so the regression holds a fortiori; recompute of the 0.5-ep arm under the new protocol queued). Reinforces the E10 pretrained-backbone pivot. **Mechanism deep-dive (2026-07-09):** no weight corruption (0 NaN/Inf, 229 tensors); the collapse is rank-1 of `enc.L5.bixt.rv_lat` (834/1536 dead rows) — 128 diverse slots all fed the same single-rank document summary. The LR=0.003 "grad-norm-rises-while-loss-falls" event is a real Edge-of-Stability threshold crossing (descendable-sharpness ceiling `2/(η·s)` lifts as η falls, crossing the bypass-gorge curvature → loss breaks a 116k-step plateau); wd is the proximate collapse driver (selective shrinkage of bypass-redundant directions, `muon.py:101`) — ~~hypothesis~~ **FALSIFIED 2026-07-11 by [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md)** (Adam@wd=0.1 stays healthy: RankMe 30.88 vs Muon's 1.8–3.6 at identical wd=0.1); the collapse is Muon's full-rank whitened updates converging fast into the intrinsically low-rank bypass minimum, not wd. Turns the wd confound into a decisive cheap test (Adam@wd=0.1 control, [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md)) + a minimal anti-collapse objective extension ([E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md); lit [concept_bottleneck_collapse_mitigation](../literature_review/concept_bottleneck_collapse_mitigation.md)). **Status (2026-07-11): [E05b](../experiments_specs/done_success/E05b_wd_confound_control.md) EVALUATED — DECISIVE, wd innocent.** Adam@wd=0.1 (68.62 GPU-h / 19.13 kWh / 10.2 B tok) within-sample RankMe **30.88** (centered 32.17) — **3–6× Muon's 1.8–3.6 at identical wd=0.1**, Δshuffle_beyond **0.50** (clears Stage-2), active-slot 1.000, 0 NaN → the collapse is Muon-specific (full-rank whitened updates), not wd. [E05c](../experiments_specs/ahead/E05c_anticollapse_extension.md) (non-bypassable objective, config-only) is the next fix; [E05d](../experiments_specs/ahead/E05d_concept_vicreg.md) (VICReg) stays queued. Run report [e05b_wd_confound_control_20260711](../2_Experiments_Registry/run_reports/e05b_wd_confound_control_20260711.md). See [run report](../2_Experiments_Registry/run_reports/e05_muon_long_2ep_collapsed_20260709.md).
- Full history (with caveats): [master_experiment_log.md](../2_Experiments_Registry/master_experiment_log.md); older roadmap + TODO diary in [5_Archive/](../5_Archive/).

## Not active right now (still part of the Vision)
Recursive concept refinement and latent reasoning remain Vision goals — E08 and related
ideas stay in play; compose them only after a strictly causal platform (E17c or a later
successor) demonstrably carries content. From-scratch and other bases are not ruled out. Diffusion
decode stays parked/revivable from `parked/`. Instruction SFT, long-context, and audio
remain long-term Vision only. Multi-agent latent communication stays the Stage-2
headline (see [team_brief](../sprind_frontier_ai/team_brief.md)).

## Engineering notes (not live experiments)
Canonical eval protocol, Tier-1 data-protocol upgrade, compute audit, and training-pipeline
modularization are done — see `docs/engineering_specs/` and
[evaluation_protocol.md](../3_Evaluations_and_Baselines/evaluation_protocol.md).
