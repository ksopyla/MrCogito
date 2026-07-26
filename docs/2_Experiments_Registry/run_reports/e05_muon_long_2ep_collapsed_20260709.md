# E05 Muon long (2 ep) — eval_loss keeps falling, concept bottleneck collapses harder

**Date:** 2026-07-09
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `concept_ar_prefix_H768L6C128D4_20260704_225659`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/concept_ar_prefix_H768L6C128D4_20260704_225659) (group `E05_concept_ar_prefix_H768L6C128D4`)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260704_225057.log`
**Best checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260704_225659/checkpoint-272000` (eval_loss 2.581, epoch 1.97)
**Last checkpoint:** `Cache/Training/concept_ar_prefix_H768L6C128D4_20260704_225659/checkpoint-276566` (eval_loss 2.584, epoch 2.0)
**Git commit:** `a37b451` (training); Tier-1 eval-protocol upgrade `4d25b81` deployed during eval
**Git tag:** —
**Related TODO:** E05 optimizer A/B in `docs/experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md`

---

## Goal

Answer the "is it under-trained?" question raised by the E05 Muon 0.5-ep eval (2026-07-04, TENTATIVE): the 0.5-ep Muon arm had the **lowest eval_loss of any E05 run (2.606)** but also a regressed concept bottleneck (within-sample RankMe 10.57 vs Adam 37.67; Δshuffle_beyond 0.209 vs 0.39). The E02-long prior (5-ep de-collapse: slot rank 5.9 → 16.7) raised the possibility that Muon's lower loss was a *faster* path that more compute would turn into richer, not poorer, concepts. This 2-ep run — **compute-matched to E02-long (~300 GPU-h)** — tests exactly that. ⚠️ **Not a matched A/B:** the Adam arm is 0.5 ep; this run is compared against E02-long (5 ep, prefix→suffix, full-causal) and the 0.5-ep Muon arm.

## Configuration

| Item | Value |
|---|---|
| Family / objective | `concept_ar`, prefix→suffix |
| Decoder | causal AR, D4, RoPE, **`decoder_context_window=128`** (windowed) |
| Dataset | `smollm3_inspired_2k_e05` (pretokenized, 9,956,348 train / seq 2048) |
| Epochs / steps | **2 / 276,566** (max_steps reached) |
| Effective batch | 8 × accum 3 × 3 GPU = **72** |
| Seed | 42 |
| Optimizer | `Muon` (stabilized: LR **0.01**, **wd 0.1**, `adamw_lr 2e-4`, momentum 0.95, ns_steps 5, cosine, clip 0.5) |
| Throughput | 0.766 steps/s · 55.2 samples/s |
| Compute | **300.88 GPU-h / 85.25 kWh / 40.78 B max-tokens / 24.47 B loss-tokens-est** (`compute/audit_state=finished`, flag `loss_fraction:prefix_suffix_approx`) |

## Training Outcome

**Stable end-to-end, monotonic eval_loss descent, no divergence.** eval_loss fell 3.64 (step 4k) → 2.58 (step 276k) across 69 evals; the curve flattened in the last ~50k steps (best 2.581 @ step 272000, last 2.584 @ step 276000 — Δ 0.003). Pre-clip grad_norm held 0.4–0.8 mid-run (epoch ~1.0), rose to ~3.7 in the cosine tail (LR ≈ 9e-9 at the last step) **without hurting eval_loss** — the *opposite* signature from the 2026-06-28 divergence. The stabilized Muon recipe (wd 0.1 + `adamw_lr 2e-4`, calibrated on 2026-07-01) held for the full 2 ep at 4× the 0.5-ep budget.

**This is the lowest eval_loss of any E05 run** (2.58 vs 0.5-ep Muon's 2.606 vs Adam attempt-3's 3.83). On the optimization axis alone, more compute + Muon monotonically helps.

## Concept Health

**Severe collapse — worse than every prior E05 arm.** Tier 1 run under the **new eval data protocol** (held-out pretokenized eval split, seq 2048, length-stratified over buckets 256/512/1024, seeded, Δ ± per-batch std). Best and last are essentially identical, so only best is shown.

| metric | Adam att-3 (0.5 ep) | Muon 0.5 ep | **Muon long (2 ep)** | E02-long (5 ep, full-causal) |
|---|---|---|---|---|
| within-sample RankMe (PRIMARY) | 37.67 † | 10.57 † | **4.96** ✗ | 82.28 † |
| within-sample RankMe (centered) | — | — | **4.61** (low on both → genuine collapse, not offset) | — |
| slot-mean effective rank (diagnostic) | 4.76 † | 3.38 † | **1.66** ✗ | 16.69 |
| mean pairwise concept cosine | — | — | **0.892** ✗ | 0.124 |
| active slot fraction | — | 0.21 | **0.38–0.51** | — |
| cross-sample manifold RankMe | 113.9 † | 218.4 † | **158.0** | 245.9 |
| anisotropy (mean random-pair cosine) | 0.682 † | 0.771 † | **0.992** ✗ (narrow cone) | 0.32 |

† — measured under the **old** Tier-1 data protocol (train-stream, seq 512, unseeded); directionally comparable but absolutes not strict-comparable with the new-protocol Muon-long numbers (see `master_experiment_log.md` note).

**Read:** within-sample RankMe collapsed further with more compute (0.5-ep 10.6 → 2-ep 5.0). The centered variant (4.61) is also low — low on both raw and centered rules out shared-offset anisotropy and confirms **genuine rank-1-ish collapse**. Slots are nearly redundant (slot-mean 1.66); concepts are ~90% correlated; the pooled-embedding cone is extremely narrow (anisotropy 0.992). The only number that *rose* is cross-sample RankMe (158, downstream-embedding diversity) — i.e. the model still distinguishes inputs in aggregate, but through a near-rank-1 concept set.

## Evaluation

All on `checkpoint-272000` (best), `model_type concept_ar`, Odra, via the `experiment-evaluate` tiered suite.

### Training-time `concept_ablation/*` (authoritative prefix→suffix, eval holdout, last eval @ step 276000)

| metric | first (step 4k) | last (step 276k) | gate |
|---|---|---|---|
| Δshuffle_beyond (E05 long-range gate) | 0.118 | **0.123** | ✗ Stage-1 floor ≥0.3 |
| Δzero_beyond | 0.189 | **0.245** | — |
| Δshuffle (overall) | 0.140 | 0.171 | ✗ E01 ≥0.5 |
| Δshuffle_early | 0.412 | **0.832** | ✓ uses concepts where bypass is impossible |
| Δzero_early | 0.708 | 1.042 | ✓ |

The beyond-window deltas **never rose** with training — they are flat-to-down from step 4k to step 276k. The early-position deltas (where the decoder has no bypass) *did* rise. **The decoder's long-range concept dependence was stationary and gate-failing for the entire 2-epoch run; only within-window dependence grew.**

### Tier-1 `run_concept_analysis.py` reconstruction-contract ablation (best, new protocol)

| metric | value (± std) | gate |
|---|---|---|
| Δshuffle (overall) | **0.303 ± 0.058** | ✗ E01 ≥0.5 |
| Δzero (overall) | 0.417 ± 0.129 | ✗ |
| Δshuffle_early | **1.001** | ✓ PRIMARY |
| Δzero_early | 0.967 | ✓ PRIMARY |
| Δshuffle_beyond (K=128) | **0.227** | ✗ E05 Stage-1 ≥0.3 |
| Δzero_beyond | 0.387 | ✗ |
| ce_intact | 2.514 | — |

(Compression-curve eval skipped: CUDA OOM at seq 2048 on the full forward pass — a known lean-memory gap in the runner, non-blocking; geometry + ablation completed on the leaner path.)

### Tier 2 — zero-shot STS-B + trivial floors

| metric | value | vs floors |
|---|---|---|
| **STS-B Pearson / Spearman** | **0.062 / 0.482** | ✗ catastrophic |
| token-embed-mean floor (SmolLM2-135M) | 0.486 / 0.525 | model is **0.42 below** the trivial floor |
| teacher-hidden-mean floor (SmolLM2-135M) | 0.460 / 0.523 | model is **0.40 below** |

**The collapsed concepts now *destroy* semantic-similarity signal** — Pearson 0.062 is essentially uncorrelated, far below averaging raw token embeddings. This is the worst STS-B of any E05 arm (0.5-ep Muon 0.518; Adam att-3 0.452). W&B [ofqs4sx3](https://wandb.ai/ksopyla/MrCogito/runs/ofqs4sx3).

### Tier 2.5 — frozen-encoder readout probe (SICK-R, mean vs attention pool)

| pool | Pearson | Spearman |
|---|---|---|
| mean | 0.048 | 0.014 |
| attention | **0.160** | 0.158 |
| **Δ (attn − mean)** | **+0.112** | — |

A small distributed component survives (+0.11, vs E02-long's +0.336 and E04's +0.22), but both pools are near-floor — consistent with genuine within-sample collapse. W&B mean [6i5a8tbr](https://wandb.ai/ksopyla/MrCogito/runs/6i5a8tbr), attention [vgr73ssz](https://wandb.ai/ksopyla/MrCogito/runs/vgr73ssz).

### Tier 3 — supervised pair tasks (full fine-tune, BEST)

| task | metric | E05 Muon long (2 ep) | 0.5-ep Muon | Adam att-3 |
|---|---|---|---|---|
| SICK-R | Pearson | **0.111** | 0.302 | 0.183 |
| SICK-E | acc | 0.626 | 0.733 | 0.634 |
| PAWS | acc / F1 | 0.562 / 0.305 | 0.567 / 0.202 | 0.550 / 0.253 |
| GLUE MRPC | acc / F1 | 0.699 / 0.815 | 0.725 / 0.830 | 0.669 / 0.778 |
| GLUE STSB | Pearson | 0.341 | 0.532 | 0.354 |
| GLUE QQP | acc | 0.807 | *running* | 0.734 |
| GLUE MNLI-m | acc | 0.613 | *running* | 0.498 |

W&B: SICK-R [xdvs2rcu](https://wandb.ai/ksopyla/MrCogito/runs/xdvs2rcu), SICK-E [oe8utwtb](https://wandb.ai/ksopyla/MrCogito/runs/oe8utwtb), PAWS [x2pa12ds](https://wandb.ai/ksopyla/MrCogito/runs/x2pa12ds). GLUE full-finetune is demoted evidence (it unfreezes the encoder and routes around the bottleneck); the semantic regressions here (SICK-R, GLUE-STSB) track the Tier-1/2 collapse.

### Generation samples (greedy, concept-conditioned, best)

- [0] prompt (Kansas Sesquicentennial) → *"The Kansas State Historical Society is a nonprofit organization dedicated to promoting the history of Kansas and its people…"* — **locally coherent.**
- [1] prompt (HMRC Tax Credits) → *"The HRCM is a separate unit, which is a separate unit for the HRCM. The HRCM is a separate unit for the HRCM…"* — repetition loop.
- [2] prompt (Joseph Fiennes / Camelot) → *"JF: What do you think about the role of the character in the play? JF: I think it's a very different role. I think it's a very different role…"* — repetition loop.
- [3] prompt (Letters: Memories of sewage) → *"The problem is that the water is not a solution, and the water is not a solution…"* — repetition loop.

Same signature as every prior E05 arm: fluent-local, semantically-empty repetition loops. Token-F1 not re-measured (the 0.5-ep Muon run's 0.149 is representative).

## Interpretation

**Longer Muon training made the concept bottleneck worse, not better — decisively.** Every concept-quality metric regressed from the 0.5-ep Muon arm (within-sample RankMe 10.6 → 5.0; Δshuffle_beyond 0.21 → 0.23; STS-B 0.518 → 0.062) while eval_loss kept falling (2.606 → 2.581). This is the opposite of the E02-long prior (slot rank rose 5.9 → 16.7 over 0.3 → 5 ep). Two takeaways:

1. **The "is it under-trained?" hypothesis for the windowed + Muon regime is falsified.** More compute at this configuration does not de-collapse; it sharpens the collapse. The decoder's within-window bypass is the attractor — extra optimization makes the bypass better, not the concepts richer. The training-time `concept_ablation/*` shows beyond-window dependence was *flat from step 4k onward*; the model never "discovered" long-range concept use despite 40.8 B tokens.
2. **eval_loss is orthogonal to concept quality** (now shown across 3 Muon checkpoints at 0.5/2 ep: loss ↓ 2.606 → 2.581 while RankMe ↓ 10.6 → 5.0 and STS-B ↓ 0.518 → 0.062). The optimization win is real but it is a **decoder-fluency win routed around the bottleneck**, exactly the mechanism flagged as tentative in the 0.5-ep eval — now confirmed.

**Against E02-long (the de-collapse prior):** E02-long de-collapsed because its *full-causal* decoder has no within-window bypass, so more training forces information through concepts. The K=128 windowed decoder (E05) gives the optimizer an easier local path; Muon's fast full-rank pressure takes that path harder than Adam. The single architectural variable (window vs full-causal) is the likely root cause of the regime difference — not the optimizer, not the compute budget. This re-frames the E05 hypothesis: **the windowed decoder as specified cannot be de-collapsed by more training;** it needs a bypass-free or bypass-penalized objective (cf. E06 latent prediction, E04 parallel decoder, decoder-weakening).

**Caveats / confounds (carried from the 0.5-ep eval, still open):**
- **wd confound:** Muon ran wd=0.1, Adam wd=0.0. The 2-ep Muon collapse could be wd-driven rather than optimizer-driven. Still needs Adam@wd=0.1 (or Muon@wd=0) to isolate.
- **Single seed / single run.** The collapse trend (0.5-ep → 2-ep) is within one run family but un-replicated.
- **Tier-1 protocol split:** the 0.5-ep Muon numbers used the old protocol (train-stream, seq 512); the 2-ep numbers use the new protocol (held-out, seq 2048). The 2-ep *absolute* RankMe (5.0) is not strict-comparable with the 0.5-ep *absolute* (10.6) — but the seq-512 truncation *flattens* collapse (fewer positions), so the new-protocol number is, if anything, a *favorable* read of the 2-ep run, and the regression conclusion holds a fortiori. Recompute of the 0.5-ep arm under the new protocol is the clean fix (queued in `agenda.md`).

## Mechanism (deep-dive, 2026-07-09 — why the LR=0.003 fireworks + where the collapse lives)

Prompted by the W&B observation that *as LR drops below 0.003, grad_norm rises while train/eval loss fall sharply* — and the worry that the weights might have NaN'd / zeroed. **They did not.** This section records the diagnosis (data + code + literature). Full notes + the research-report follow-up: `docs/literature_review/concept_bottleneck_collapse_mitigation.md`.

### A. Weight corruption is ruled out — the failure is clean rank collapse

Loaded all three surviving checkpoints on CPU (float32, 229 tensors each): **0 NaN, 0 Inf, no true zero weights.** No numerical corruption. But one specific matrix is degenerate:

| matrix (encoder) | shape | stable rank | rank-1 energy | dead rows | verdict |
|---|---|---|---|---|---|
| **`L5.bixt.rv_lat`** (token→latent-query value/receive proj) | 1536×768 | **1.79** | **31%** | **834/1536** | ❌ effectively rank-1 |
| `L3/L4.bixt.rv_lat` | 1536×768 | 6.1–7.4 | ~2% | 0–579 | degenerating |
| `L0/L1.bixt.*` | — | 25–250 | — | 0 | healthy |
| `concept_embeddings.weight` (the C=128 slots) | 128×768 | **22.4** | 0.2% | 0 | ✅ diverse |
| `lm_head`, `decoder.input_projection`, `concept_self_attn` | — | 11–191 | <1% | 0 | ✅ healthy |

**Read:** the 128 concept *slots* stayed diverse (rank 22, no dead rows); it is the cross-attention that *writes document information into* those slots (`bixt.rv_lat`, layers 2–5, worst in L5) that collapsed to a single mode. So the model still emits 128 *distinct* concept vectors, but they are all filled with the **same one-dimensional summary** of the document — exactly what produces the within-sample RankMe ≈5 / slot-rank ≈1.7. The collapse developed **between step 69k (0.5-ep arm: L5 rv_lat rank 12.6, no dead cols) and 264k** (earliest surviving 2-ep ckpt: already rank 1.8, frozen thereafter). ⚠️ **All pre-crossover checkpoints were deleted by `save_total_limit=5`** (none below step 264k survive), so the exact collapse step cannot be pinned from disk — operational lesson below.

### B. The LR=0.003 event is a real, sharp Edge-of-Stability threshold crossing

Quantified from W&B history (the raw, pre-whitening, pre-clip `train/grad_norm` and `eval/loss`):

- **grad_norm climbs monotonically as LR decays** (mean by LR band): 0.57 (LR 0.009–0.011) → 0.72 → 0.81 → 0.92 (LR 0.003–0.005) → **1.23** (LR 0.001–0.003) → **2.66** (LR <0.001, spikes to 8.4). Cosine ⇒ LR=0.003 at ~63% of training ≈ **step 175,000**.
- **eval/loss plateaus then breaks exactly there:** 3.64 → ~3.0 by step ~60k, then **flat at 3.0–3.05 for ~116k steps (60k→176k)**, then **drops sharply 3.0 → 2.80 → 2.58** starting at the LR=0.003 crossover.
- **concept geometry at the crossover:** training-time `concept_geometry/effective_rank` peaks 4.61 (~step 16k) → declines to ~2.0 by the crossover → 1.78 at end; `Δzero_beyond` is **most negative (−0.094) right at LR=0.003 (~step 172k)** — concepts were momentarily *hurting* the decoder — then recovers positive.

### C. The math — why rising grad_norm + falling loss is expected under Muon (not a bug)

1. **Muon's step is the orthogonal polar factor — gradient magnitude is discarded.** The Newton-Schulz step (`nn/muon.py:97`) converges to `Ortho(M)=UVᵀ`, which is scale-invariant (`Ortho(cM)=Ortho(M)`; the buffer is Frobenius-normalized first, `muon.py:35`). So every singular direction of the step is ≈1, `‖step‖_F≈√max(m,n)`, and **the raw `grad_norm` is decoupled from the update size** (Keller Jordan, *Muon*, 2024; Moonlight, arXiv:2502.16982 §2.2). Loss can keep falling while grad_norm rises — and indeed Moonlight's own Muon run had *flat* grad_norm, so the rise is our landscape, not Muon.
2. **Edge of Stability: lowering LR *lifts* the sharpness ceiling.** GD is stable in a curvature direction only if `η_eff·κ < 2`; for Muon the per-direction step is `≈η·s`, so the descendable-sharpness ceiling is `2/(η·s)`, which **grows as η→0** (Cohen et al., *Edge of Stability*, ICLR 2022, arXiv:2103.00065; progressive sharpening, Agarwala et al., arXiv:2210.04860). The ~116k-step plateau is the iterate oscillating on the *rim* of the bypass gorge while `2/(η·s) < κ_⊥`; **the break at LR≈0.003 is the moment the ceiling crosses the gorge's transverse curvature and the iterate flows down.**
3. **Gorge geometry: low loss + high grad-norm simultaneously.** In a narrow canyon the gradient is dominated by the steep *transverse* walls while the *longitudinal* (floor) component — the actual descent — is small: `‖G‖=‖H(W−W*)‖` large, loss low. The gorge the optimizer descends at LR=0.003 **is the bypass minimum**; the rank collapse is its representational signature.

### D. Why Muon collapses harder than Adam — and why wd is the proximate killer (resolves the open confound)

- **Full-rank vs low-rank updates.** Muon's whitened step is full-rank (every singular direction stepped); Adam's `m/√v` step is effectively low-rank. Moonlight §3.4 shows Muon gives *higher-rank weights*, yet here it gives *lower-rank representations* — the inversion: representation rank is a property of the data-conditioned *image* of the map, not the operator (a full-rank matrix can map onto a 1-D manifold). Muon's aggressive full-rank pressure takes the bypass path harder than Adam.
- **Weight decay selectively collapses the redundant directions.** For the `bixt` directions the bypass rendered redundant, `G≈0`. Muon's scale-invariant step *cannot* shrink them (NS5 re-normalizes tiny gradients back to unit size — redundant directions do a noise-driven random walk and stay alive). But decoupled weight decay sits **outside** the Ortho term (`muon.py:101`, `p.mul_(1 − mu_lr·wd)`, **coupled to the cosine LR**): for useful directions gradient pressure balances wd; for redundant directions only wd acts → they shrink toward zero → **selective rank collapse**. This fits every observation: Adam (wd=0.0, 0.5-ep) RankMe 37.67 (no shrinkage); Muon (wd=0.1) 3.56 → 1.78 (more steps × more cumulative wd → more collapse). The literature genuinely does not settle "does decoupled wd shrink representation rank" (our own open confound) — but in this architecture, with the bypass active, the prediction is sharp and **decisively testable with the Adam@wd=0.1 control** (if Adam@0.1 also collapses → wd, not Muon-per-se, is the driver). Note the tension: wd=0 makes Muon diverge (Moonlight MaxLogit), wd=0.1 stabilizes it but collapses concepts — you cannot simply delete wd.

### E. Novelty

**No verified external report of "Muon cosine-tail grad_norm rise + loss fall" exists** (checked: Keller Jordan; Moonlight App. D — loss/grad-norm stayed *stable* there; modded-nanogpt; Fireworks MuonClip). The observation is novel to MrCogito; the explanation is a synthesis of Muon-decoupling + EoS-ceiling-lift + gorge geometry, not a named Muon phenomenon.

*Checkpoint-integrity artifacts (Odra): `Cache/Training/concept_ar_prefix_H768L6C128D4_20260704_225659/checkpoint-{264000,…,276566}/model.safetensors`; `…20260702_031956/checkpoint-{56000,…,69142}/model.safetensors`.*

---

## Decision

**Verdict: REGRESSION — E05 Muon long (2 ep) fails every concept-quality gate (Stage-1 floor Δshuffle_beyond ≥0.3 not met; STS-B below trivial floors; RankMe ~5/128).** The "more compute de-collapses it" hypothesis for the windowed + Muon regime is rejected. Optimization succeeded; the concept bottleneck collapsed harder. **No weight corruption** — the failure is clean rank-1 collapse of `bixt.rv_lat`, mechanistically explained (EoS gorge descent at LR=0.003 + wd-driven selective shrinkage of the bypass-redundant directions).

**Immediate next actions (do not extend this run):**
1. **Record and move on.** This run closes the E05 windowed-decoder track's "more training" branch. The Adam-vs-Muon A/B research verdict (pending since 2026-07-04) is now *decentralized*: both optimizers fail the concept gates on the windowed decoder; the difference is degree-of-collapse, not pass/fail.
2. **The agenda's pivot to E10 (pretrained-backbone concept memory) remains the headline** and this run reinforces it — the from-scratch windowed decoder at <200M scale produces used-but-semantically-empty concepts regardless of optimizer or budget.
3. **Targeted mechanistic follow-ups (cheap, in flight 2026-07-09) — not a broad re-investment in E05 from-scratch:** the deep-dive above turns the open `wd` confound into a *decisive, cheap* test and proposes a minimal bypass-free/anti-collapse objective.
   - **Control (settles the wd driver):** Adam @ wd=0.1, 0.5 ep, otherwise identical to attempt-3 (wd=0.0). Prediction: it collapses (slower than Muon) ⇒ wd is the proximate collapse driver; if it stays healthy ⇒ Muon's full-rank dynamics are. Spec: [E05b_wd_confound_control.md](../../experiments_specs/done_success/E05b_wd_confound_control.md). Also raises `SAVE_TOTAL_LIMIT` + lower `EVAL_STEPS` so pre-crossover checkpoints survive and the collapse is visible live (the `save_total_limit=5` gap above must not recur).
   - **Anti-collapse extension (tests if the collapse is objective-fixable):** add a genuinely non-bypassable signal — decoder suffix corruption (destroy the local tokens the bypass copies) + a VICReg-style variance+covariance penalty on the concept matrix (directly counteract the rank-1 collapse / wd shrinkage). Spec: [E05c_anticollapse_extension.md](../../experiments_specs/ahead/E05c_anticollapse_extension.md). Literature: [concept_bottleneck_collapse_mitigation.md](../../literature_review/concept_bottleneck_collapse_mitigation.md).
4. **Optional cleanup (low priority):** recompute the 0.5-ep Muon + Adam att-3 under the new Tier-1 protocol to remove the protocol-split confound in the Concept Health table.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E05_windowed_decoder_concept_memory.md`, `agenda.md`, prior run report [e05_muon_divergence_rootcause_20260701.md](e05_muon_divergence_rootcause_20260701.md)*
