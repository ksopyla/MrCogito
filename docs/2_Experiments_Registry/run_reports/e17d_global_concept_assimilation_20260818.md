# E17d global-attention concept layers (300M) — `backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227`

**Date:** 2026-08-18
**Machine:** Polonez (4× RTX 3090; eval on GPU 0)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227`
**WandB (training):** `backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227` (same W&B entity/project as the E17/E17c training runs)
**Raw log:** `Cache/logs/shell_perceiver_denoise_20260817_141158.log` · eval `Cache/logs/eval_E17d_20260818_104901.log` · late-bin `Cache/logs/eval_E17d_late_bin_20260818.log`
**Best checkpoint:** `Cache/Training/…20260817_141227/checkpoint-2660`
**Last checkpoint:** `…/checkpoint-2668` (= final; 8 steps after last eval)
**Git commit:** `c91ef685c46fd6b7ae8efe36cfffb993d29ec369` (`c91ef68`; Polonez training worktree)
**Git tag:** `arch/e17d-global-concept-assimilation` · `train/backbone_concept_gemma_3_1b_pt_K512_concept_20260817_141227`
**Related TODO:** 300M mechanism verdict (do not launch 1B)

**Artifacts:**
- Authoritative late-bin perm gate (B=2, ≥2048-token holdout, 24 batches): `Cache/Evaluation_reports/…141227_best_late_bin_gate.json`
- Tier-1 geometry (seq 4096, length-stratified; late-bin keys NaN — short-bucket poison): `…_best_concept_analysis.json` · `…_last_concept_analysis.json`
- Matched generation assessment vs base Gemma: `…_ckpt2660_generation_assessment.json`
- Tier 1.5 quality runner (6 prompts × continuation × 4 modes): `…_best_generation_quality.json`
- Compute audit: `Cache/Evaluation_reports/compute_audit/20260818_104903_*`

---

## Goal

Close the registered E17d 300M mechanism verdict: with concept mix in the attention residual and no token carry, does permuting all four banks raise CE on **late intra-block tokens 256–512** by **≥0.10 nats** (CI lb **>0.05**), with ≥3/4 banks showing their own positive late-bin Δ, without collapsing RankMe or free-run?

## Configuration

| Item | Value |
|---|---|
| Family | `backbone_concept` · `per_layer_banks` · dedicated reads in `attn_residual` · untied additive writers |
| Backbone | frozen Gemma-3-1B-pt + LoRA r=16/α=32 · seed 42 |
| Concepts | C=128 · K=512 · 4 private banks at global layers 5/11/17/23 |
| Carry / pressure | `MEMORY_CARRY_DROPOUT=1.0` · `INFERENCE_CARRY_POLICY=drop_after_first` · `MEMORY_PRESSURE_TOKENS=0` · uniform CE |
| Gates | `READ_GATE_INIT=0.1` · `WRITE_GATE_INIT=0.1` |
| Dataset | `e16b_long_4k_v1` · seq 4096 · causal LM · `BATCH_PACKING_MODE=length_group` |
| Optimizer | Muon · LR 0.01 · AdamW side 2e-4 · wd 0.1 |
| Budget | **300M** non-padding tokens · eff. batch **64** (bs=8 × 4 GPU × accum=2) · max_steps **2668** |
| Compute | **39.445 GPU-h** · **10.733 kWh** · max_tokens **0.699B** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |
| Throughput | train_runtime **35500 s** (~9.86 h wall) · **4.809** samples/s · real tok/s **8264** · pad_ratio **0.023** · mean seq ~2098 |

## Training Outcome

Stable finish on Polonez 2026-08-17 14:12→2026-08-18 00:04Z. `"Training completed."` No traceback, no OOM. Best = last-eval = `checkpoint-2660` (`eval_loss` **2.365** ≤ 2.50). Last = `checkpoint-2668`. Train loss **4.724**. Ignore aborted `…124945` (CheckpointError) and `…125416` (bs=3 underfilled). Health: 621 tensors, **0 NaN / 0 Inf** on best.

## Concept Health

Offline Tier-1 geometry (`checkpoint-2660`, pretokenized holdout, seq 4096, 24× B=2):

| Metric | Best | Last | Gate |
|---|---|---|---|
| within-sample RankMe (last bank) | **76.8** (centered 107.5) | 76.1 | ≥19.2 **PASS** |
| bank RankMe 0/1/2/3 | **43.2 / 58.7 / 65.9 / 76.8** | 42.7 / 58.1 / 65.0 / 76.1 | every bank ≥19.2 **PASS**; min 19.8 |
| Δshuffle ≥1024 | **0.022** ± 0.011 | 0.023 | diagnostic (not the E17d primary) |
| Δstatic / Δone-block ≥1024 | 0.032 / 0.0075 | 0.027 / 0.006 | diagnostic |
| mean pairwise cosine | 0.71 | — | correlated slots; RankMe still healthy |

Authoritative carryless / permutation protocol (B=2, 24 long holdout batches, seq ≥2048, bootstrap 95% CI). Stratified T1 JSON late-bin keys are **NaN** (short `(0,1024]` batches poison the mean); do not use them.

| Metric | Best | Gate |
|---|---|---|
| late-bin **256–512 Δpermutation** | **0.044** CI **[0.039, 0.049]** | ≥0.10 and CI lo>0.05 **FAIL** |
| bank-0 / 1 / 2 / 3 late-bin Δperm | 0.008 / **0.014** / 0.008 / 0.006 | 4/4 CI>0; magnitudes ≪ 0.10 |
| carryless first-64 Δpermutation | **0.746** CI **[0.687, 0.809]** | not the registered primary |
| bank-0 / 1 / 2 / 3 first-64 Δperm | 0.125 / **0.206** / 0.082 / 0.054 | 4/4 CI>0; **not** a bank-0 monopoly |
| Δpermutation ≥1024 | **0.020** CI [0.018, 0.023] | E17c was 0.013 |
| intra-block bins 0–64 / 64–128 / 128–256 / 256–512 | **0.75 / 0.19 / 0.097 / 0.044** | gist decays through the page |

E17c on the same long-doc protocol was late-bin **0.026**, first-64 **0.594** (bank 0 **0.38**; others ≤0.03), RankMe **6.75**. E17d keeps the gist-not-memory decay curve, recovers geometry, and spreads the *block-start* gist across banks.

## Evaluation — generation (matched E17c protocol)

Short-prompt continuation, greedy, cutoffs from `run_e16b_generation_assessment.py` (`checkpoint-2660`, 6 prompts; n=5 at 256 because one sample stopped at 125 tokens).

| Condition @256 | distinct-1 | REP-3 |
|---|---|---|
| **E17d `real`** | **0.185** | **0.595** |
| E17d `zero` | 0.239 | 0.451 |
| E17d `shuffle` | 0.185 | 0.595 |
| Base Gemma greedy | 0.163 | 0.706 |
| **E17c `real`** | **0.226** | **0.529** |
| **E17b `real`** | **0.196** | **0.601** |
| **E17 `real`** | **0.208** | **0.593** |
| **E16b `real`** | **0.04** | **0.94** |

Registered gen gate (`real`@256 d1≥0.20, REP-3≤0.60, and `real` beats `shuffle`): d1 **miss**, REP-3 **barely pass**, `real` **= `shuffle`**. `zero` is less repetitive than `real`. Chat-template greedy is worse (`real`@256 **0.10/0.81**).

Snippet (`real`, greedy, prompt *The future of renewable energy depends on*): "the ability to store energy. The ability to store energy is a key factor in the success of renewable energy. The ability to the store energy is a key factor in…"

Tier 1.5 `run_generation_quality.py` matches: continuation `real` **0.487/0.199 @64 → 0.185/0.595 @256**; `shuffle`/`static` identical to `real`; `zero` **0.239/0.451 @256**.

## Interpretation

Geometry **passed** (min RankMe 43.2 vs E17c's collapsed 6.75). Eval loss **passed** (2.365 ≤ 2.50). The registered **late-bin assimilation gate failed**: 0.044 nats, CI entirely below 0.05, only ~1.7× E17c's 0.026. Four banks all have a tiny positive late-bin CI, so this is not E17c's bank-0 monopoly — but none of them carry late-page content. The large signal is still the first 64 tokens of a new block (0.75 nats, now spread across banks, bank 1 largest). Intra-block bins 0.75→0.19→0.10→0.044 are the same gist-not-memory shape as E17c.

Putting the mix in the attention residual and deleting the token carry gave later banks a *block-start* job and kept RankMe healthy. It did not make concepts do global-attention work through the rest of the page. Free-run stays in the E17 family and misses the 0.20 distinct-1 bar; `real≈shuffle`. Do not spend 1B on this cosine.

## Decision

**Do not launch 1B.** Treat E17d as a mixed 300M close: keep healthy multi-bank geometry and the distributed first-64 Δperm as evidence that attn-residual + no-carry can give later depths a job, and treat the late-bin miss (0.044 vs 0.10) as the kill for this cell as a global-attention replacement. Next bet has to make concept content necessary *after* the local window has already refilled, not only at the page boundary.

## Notes

`run_concept_analysis.py` averages all length buckets including short sequences whose `_intra_block_bin_mean` is NaN; that is why training-eval and stratified T1 JSON show `delta_permutation_block_256_512: nan`. The long-doc gate above averages only finite batches.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E17d_global_concept_assimilation.md`, `agenda.md`*
