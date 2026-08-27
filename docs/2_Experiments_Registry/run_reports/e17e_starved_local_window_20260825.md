# E17e starve local window K=256 (300M) — `backbone_concept_gemma_3_1b_pt_K256_concept_20260822_120601`

**Date:** 2026-08-25
**Machine:** Polonez (4× RTX 3090; train 4 GPU; eval on GPU 0)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K256_concept_20260822_120601`
**WandB (training):** `backbone_concept_gemma_3_1b_pt_K256_concept_20260822_120601` (same W&B entity/project as the E17d training run)
**Raw log:** `Cache/logs/shell_perceiver_denoise_20260822_120532.log` · eval `Cache/logs/eval_E17e_20260825_175717.log`
**Best checkpoint:** `Cache/Training/…20260822_120601/checkpoint-2660`
**Last checkpoint:** `…/checkpoint-2668` (= final; 8 steps after last eval)
**Git commit:** `3e59b979aaa2f5d6ae2101fe16c548fe25db19e0` (`3e59b97`; Polonez training worktree)
**Git tag:** `arch/e17e-starved-local-window` · `train/backbone_concept_gemma_3_1b_pt_K256_concept_20260822_120601`
**Related TODO:** 300M starve verdict (do not launch 1B)

**Artifacts:**
- Authoritative late-half perm gate (B=2, seq ≥2048, 24 batches; key `block_256_512` = offsets 128–256 of each K=256 window): `Cache/Evaluation_reports/…120601_best_late_bin_gate.json` · `…_last_late_bin_gate.json`
- Tier-1 geometry (seq 4096, length-stratified): `…_best_concept_analysis.json` · `…_last_concept_analysis.json`
- Matched generation assessment vs base Gemma: `…_ckpt2660_generation_assessment.json` · `…_ckpt2668_generation_assessment.json`
- Tier 1.5 quality runner (6 prompts × continuation/chat × 4 modes): `…_best_generation_quality.json` · `…_last_generation_quality.json`
- Compute audit: `Cache/Evaluation_reports/compute_audit/20260825_175719_*`

---

## Goal

Close the registered E17e 300M starve: keep the E17d attn-residual four-bank cell and no token carry, cut local softmax + write cadence from K=512 to K=256, and test whether permuting all banks raises CE on the **late half of each 256-token window** by **≥0.10 nats** (CI lb **>0.05**).

## Configuration

| Item | Value |
|---|---|
| Family | `backbone_concept` · `per_layer_banks` · dedicated reads in `attn_residual` · untied additive writers |
| Backbone | frozen Gemma-3-1B-pt + LoRA r=16/α=32 · seed 42 · fresh (not warm-started from E17d) |
| Concepts | C=128 · **K=256** · 4 private banks at global layers 5/11/17/23 · `sliding_window` aligned to 256 |
| Carry / pressure | `MEMORY_CARRY_DROPOUT=1.0` · `INFERENCE_CARRY_POLICY=drop_after_first` · `MEMORY_PRESSURE_TOKENS=0` · uniform CE |
| Dataset | `e16b_long_4k_v1` · seq 4096 · causal LM · `BATCH_PACKING_MODE=length_group` |
| Optimizer | Muon · LR 0.01 · AdamW side 2e-4 · wd 0.1 |
| Budget | **300M** non-padding tokens · eff. batch **64** (bs=8 × 4 GPU × accum=2) · max_steps **2668** |
| Compute | **37.087 GPU-h** · **10.500 kWh** · max_tokens **0.699B** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |
| Throughput | train_runtime **33379 s** (~9.3 h wall) · **5.115** samples/s · real tok/s **9085** · pad_ratio **0.023** · mean seq ~2098 |

## Training Outcome

Stable finish on Polonez 2026-08-22 12:06→21:22Z. `"Training completed."` No traceback, no OOM. Best = last-eval = `checkpoint-2660` (`eval_loss` **2.464** ≤ 2.70). Last = `checkpoint-2668`. Train loss **4.935** (log 5.30→4.77). Health: **0 NaN / 0 Inf** on best. Config confirms `concept_block=256` and `sliding_window=256`.

## Concept Health

Offline Tier-1 geometry (`checkpoint-2660`, pretokenized holdout, seq 4096, 24× B=2). Prefer mixed-length RankMe for the de-collapse gate (same protocol as E17d).

| Metric | Best | Last | Gate |
|---|---|---|---|
| within-sample RankMe (last bank) | **57.4** (centered 95.3) | 57.5 | ≥19.2 **PASS** |
| bank RankMe 0/1/2/3 (mixed length) | **31.5 / 34.9 / 31.2 / 57.4** | 31.4 / 38.2 / 31.4 / 57.5 | every bank ≥19.2 **PASS** |
| bank RankMe 0/1/2/3 (late-bin, seq ≥2048) | 18.8 / 20.8 / **15.5** / 35.3 | 18.7 / 20.7 / 15.3 / 35.1 | bank 2 misses 19.2 on long-only; kill `<10` **not tripped** |
| Δshuffle ≥512 | **0.073** CI [0.058, 0.089] | 0.079 | diagnostic (E17d ≥1024 was 0.022) |

Authoritative carryless / permutation protocol (B=2, 24 long holdout batches, seq ≥2048, bootstrap 95% CI). Key `delta_permutation_block_256_512` is the **late half of a K=256 window** (offsets 128–256), compared to E17d's late-half **0.044**, not to E17d's 128–256 bin.

| Metric | Best | Last | Gate |
|---|---|---|---|
| late-half **Δpermutation** | **0.1043** CI **[0.0947, 0.1135]** | **0.0973** CI **[0.0876, 0.1064]** | ≥0.10 and CI lo>0.05: best **PASS** (fragile); last **MISS** 0.10 |
| bank-0 / 1 / 2 / 3 late-half Δperm | 0.0251 / 0.0251 / 0.0252 / 0.0129 | 0.0258 / 0.0254 / 0.0237 / 0.0124 | 4/4 CI>0 **PASS** ≥3/4; magnitudes ≪ 0.10 |
| first-64 Δpermutation | **1.343** CI [1.222, 1.465] | 1.264 | not the registered primary |
| bank-0 / 1 / 2 / 3 first-64 Δperm | 0.146 / **0.216** / 0.105 / 0.065 | — | 4/4 positive; **not** a bank-0 monopoly |
| intra-block 0–64 / 64–128 / 128–256 / late-half | **1.34 / 0.43 / 0.21 / 0.10** | 1.26 / — / — / 0.097 | gist decays through the page |
| eval_loss | **2.464** | — | ≤2.70 **PASS** (E17d 2.365) |

E17d on the same long-doc protocol was late-half **0.044** CI [0.039, 0.049], first-64 **0.75**, RankMe **43.2–76.8**. Halving the window lifted late-half ~2.4× and cleared the written 0.10 bar on **best only**. The decay shape is unchanged: the model still uses banks mainly at the start of a new window.

## Evaluation — generation (matched E17d protocol)

Short-prompt continuation, greedy, cutoffs from `run_e16b_generation_assessment.py` (`checkpoint-2660`, 6 prompts; n=4 at 256 because two samples stopped early).

| Condition @256 | distinct-1 | REP-3 |
|---|---|---|
| **E17e `real` (best)** | **0.162** | **0.686** |
| E17e `zero` (best) | 0.111 | 0.806 |
| E17e `shuffle` (best) | 0.162 | 0.686 |
| E17e `real` (last) | 0.195 | 0.628 |
| Base Gemma greedy | 0.163 | 0.706 |
| **E17d `real`** | **0.185** | **0.595** |
| **E17c `real`** | **0.226** | **0.529** |
| **E17b `real`** | **0.196** | **0.601** |
| **E17 `real`** | **0.208** | **0.593** |
| **E16b `real`** | **0.04** | **0.94** |

Registered gen gate (`real`@256 d1≥0.20, REP-3≤0.60, and `real` beats `shuffle`): all three **FAIL** on best. `real` **= `shuffle`**. Last is closer to E17d and still misses. Chat-template greedy is worse (`real`@256 **0.141/0.704**).

Snippet (`real`, greedy, prompt *The future of renewable energy depends on*): "the development of new technologies. The future of renewable energy depends on the development of new technologies. The future of renewable energy depends on t…"

Tier 1.5 `run_generation_quality.py` matches the assessment continuation numbers. STS-B / SICK / PAWS / GLUE did **not** produce scores: `evaluate_on_benchmark.py` / `evaluate_model_on_glue.py` reject `--model_type backbone_concept` (argparse). Not a registered E17e gate.

## Interpretation

The starve **moved the registered CE number**: late-half Δperm 0.044 → **0.104** on best, CI lb 0.095 still > 0.05, four banks all participate. Eval loss 2.464 stayed under 2.70. Mixed-length RankMe stayed healthy (31–57), below E17d's 43–77 but well above the 19.2 floor. Kill RankMe `<10` did not fire.

That is not the same as "concepts now do late-page work." Per-bank late-half is still **~0.025**. Intra-block bins still decay 1.34 → 0.43 → 0.21 → 0.10. Last checkpoint **misses** 0.10 (0.097). Generation got **slightly worse** than E17d and remains `real=shuffle`. E17/E17c are still the best free-run in this family; E16b's huge teacher-forced Δ does not transfer to free-run.

## Decision

**Do not launch 1B.** Treat E17e as a mixed 300M close: the K=256 starve is a real lift vs E17d on the written late-half gate, but the gate is fragile (best-only, CI lb 0.095, last 0.097) and generation failed. Do **not** retune another half-window in this ID. The remaining local computer is still enough for FinePDFs after ~64 tokens of a new page.

## Notes

`block_256_512` is a fraction of `concept_block`, not document offsets 256–512. Eval suite `FAILED: stsb_best:2` is the argparse rejection above; `sick_best` / `paws_best` / `glue_best` logged OK because the wrappers swallowed the same error.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E17e_starved_local_window.md`, `agenda.md`*
