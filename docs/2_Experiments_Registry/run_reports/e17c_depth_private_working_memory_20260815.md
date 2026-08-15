# E17c depth-private gated working memory (300M) — `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241`

**Date:** 2026-08-15
**Machine:** Polonez (4× RTX 3090; eval on GPU 0/1)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241`
**WandB (training):** `backbone_concept_gemma_3_1b_pt_K512_concept_20260814_133241` (same W&B entity/project as the E17/E17b training runs)
**Raw log:** `Cache/logs/shell_perceiver_denoise_20260814_133212.log` · eval `Cache/logs/eval_E17c_20260815_085904.log` · perm-gate `Cache/logs/eval_E17c_perm_20260815.log`
**Best checkpoint:** `Cache/Training/…20260814_133241/checkpoint-2370`
**Last checkpoint:** `…/checkpoint-2372` (= final; 2 steps after last eval)
**Git commit:** `9432f296038daf2ca5720bdbc68559f90c126c5a` (`9432f29`; W&B `git_dirty=True`)
**Git tag:** `arch/e17c-depth-private-working-memory-9-g9432`
**Related TODO:** 300M mechanism verdict (do not launch 1B)

**Artifacts:**
- Carryless perm gate (B=2, ≥2048-token holdout, 24 batches): `Cache/Evaluation_reports/…133241_best_perm_gate.json` · `…_last_perm_gate.json`
- Tier-1 geometry (seq 4096, B=1 so permutation skipped): `…_best_concept_analysis.json` · `…_last_concept_analysis.json`
- Matched generation assessment vs base Gemma: `…_ckpt2370_generation_assessment.json`
- Tier 1.5 quality runner (8 prompts × continuation/chat × 4 modes): `…_best_generation_quality.json`
- Compute audit: `Cache/Evaluation_reports/compute_audit/20260815_085659_*`

---

## Goal

Close the registered E17c 300M mechanism verdict: do depth-private gated cells plus 50% causal carry dropout make real banks beat batch-permuted banks by **≥0.20 nats** on carryless first-64 tokens (CI lower **>0.10**), without collapsing geometry or free-run?

## Configuration

| Item | Value |
|---|---|
| Family | `backbone_concept` · `per_layer_banks` · dedicated reads · untied `gated_replace` writers |
| Backbone | frozen Gemma-3-1B-pt + LoRA r=16/α=32 |
| Concepts | C=128 · K=512 · 4 private banks at global layers 5/11/17/23 |
| Pressure | `MEMORY_CARRY_DROPOUT=0.5` · first-64 CE ×4 · update-gate init 0.25 · read-gate init 0.1 |
| Dataset | `e16b_long_4k_v1` · seq 4096 · causal LM · `BATCH_PACKING_MODE=length_group` |
| Optimizer | Muon · LR 0.01 · AdamW side 2e-4 · wd 0.1 · warmup 500 |
| Budget | **300M** non-padding tokens · eff. batch 72 (bs=3 × 4 GPU × accum=6) · max_steps **2372** |
| Compute | **40.48 GPU-h** · **11.32 kWh** · max_tokens **0.700B** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |
| Throughput | train_runtime **36431 s** (~10.1 h wall) · **4.686** samples/s · pad_ratio **0.006** · mean seq ~2076 |

## Training Outcome

Stable finish on Polonez 2026-08-14 13:32→23:40Z. Best = last-eval = `checkpoint-2370` (`eval_loss` **2.276** ≤ E17b+0.10 bar 2.36). Weighted train loss **14.36** is the pressure objective (not comparable to eval CE). Live W&B `pressure_delta_permutation_first64` was **NaN at every eval** (length-grouped eval batches were often too short); the 100M carryless kill could not be read from the training log. Geometry RankMe was already **<19.2** at the first eval (step 237) — the any-checkpoint collapse kill was not auto-enforced.

| step (~tok) | update_0..3 | RankMe min/med/max | Δperm_beyond | pressure Δperm first64 |
|---|---|---|---|---|
| 237 (~30M) | 0.35 / 0.91 / 0.88 / 0.79 | 1.00 / 1.00 / 1.40 | 0.0006 | NaN |
| 711 (~90M) | 0.43 / 0.76 / 0.53 / 0.70 | 1.04 / 1.34 / 8.16 | 0.0049 | NaN |
| 1422 (~180M) | 0.72 / 0.45 / 0.13 / 0.83 | 1.26 / 4.37 / 23.3 | 0.0063 | NaN |
| **2370 (300M)** | **0.84 / 0.78 / 0.28 / 0.85** | **1.82 / 5.40 / 15.5** | **0.0080** | NaN |

Update gates stayed open (not E17b's close-to-zero writes). State RMS stayed tiny (~0.029). Read gates rose to 0.27 / 0.28 / 0.37 / 0.44. Health: 625 tensors, **0 NaN / 0 Inf** on best and last.

## Concept Health

Offline Tier-1 geometry (`checkpoint-2370`, pretokenized holdout, seq 4096):

| Metric | Best | Last | Gate |
|---|---|---|---|
| within-sample RankMe | **6.75** (centered 8.39) | 6.71 | ≥38.4 **FAIL**; any-bank <19.2 **KILL** |
| bank RankMe 0/1/2/3 | 6.91 / **1.84** / 14.5 / 6.75 | 6.88 / 1.83 / 14.6 / 6.71 | bank 1 collapsed |
| Δshuffle ≥1024 | **0.000** | 0.000 | (B=1 skipped permutation) |
| Δstatic ≥1024 | 0.0052 | 0.0052 | diagnostic |

Authoritative carryless / permutation protocol (B=2, 24 long holdout batches, seq ≥2048, bootstrap 95% CI):

| Metric | Best | Last | Gate |
|---|---|---|---|
| carryless first-64 **Δpermutation** | **0.594** CI **[0.543, 0.645]** | 0.591 CI [0.541, 0.642] | ≥0.20 and CI lo>0.10 **PASS** |
| bank-0 / 1 / 2 / 3 first-64 Δperm | **0.380** / 0.029 / 0.013 / 0.032 | 0.379 / 0.029 / 0.012 / 0.032 | 4/4 CI>0; **bank 0 dominates** |
| Δpermutation ≥1024 | **0.0131** CI [0.0115, 0.0147] | 0.0131 | 300M stop if <0.02 **KILL 1B** |
| Δstatic / Δone-block ≥1024 | 0.0042 / 0.0018 | 0.0044 / 0.0019 | ≪ 0.05 / 0.02 1B bars |

Live training shuffle and permutation beyond-local numbers match (~0.008–0.013): with local K-carry present, banks barely affect CE. Removing the carry (the pressure test) opens a **0.59 nat** gap, almost entirely through **layer-5 / bank 0**.

## Evaluation — generation (matched E17b protocol)

Short-prompt continuation, greedy, mean over 6 prompts (`run_e16b_generation_assessment.py`). Prefer this table over a chat+continuation mix.

| Condition @256 | distinct-1 | REP-3 |
|---|---|---|
| **E17c `real`** | **0.226** | **0.529** |
| E17c `zero` | 0.231 | 0.492 |
| E17c `shuffle` / `static` | 0.226 / 0.226 | 0.529 / 0.529 |
| Base Gemma greedy | 0.163 | 0.706 |
| **E17b mid-init `real`** | **0.196** | **0.601** |
| **E17 low-init `real`** | **0.208** | **0.593** |
| **E16b `real`** | **0.04** | **0.94** |

`real` ≈ `shuffle` ≈ `static`; `zero` is slightly less repetitive. 300M gen kill (REP-3>0.80 **and** worse than zero) does **not** fire. 1B utility bar (d1≥0.25 and REP-3≤0.50 with `real≥zero`) is close but unmet at 300M.

Snippet (`real`, greedy, prompt *The future of renewable energy depends on*): "the ability to store energy. The ability to store energy is a key factor in the success of renewable energy. The ability to the store energy is a key factor in…"

Chat-template greedy is worse (`real`@256 **0.12/0.74**), same pattern as E17/E17b.

Tier 1.5 `run_generation_quality.py` (8 prompts, same greedy decode) corroborates the matched table and shows the length-binned drop past K=512: continuation `real` **0.508/0.164 @64 → 0.203/0.563 @256 → 0.108/0.746 @512**; `shuffle`/`static` identical to `real`; `zero` slightly worse. Chat `real`@256 **0.108/0.746**. Distinct-n falls and REP-3 rises with length — the fluent-local repetition signature, not a generation win from the new cell.

## Interpretation

The registered **300M carryless number cleared** — the first time a strictly block-causal per-layer cell has shown content-bearing use when the local carry is removed. That signal lives in **one bank** (layer 5) on a **collapsed** concept set (RankMe 6.7; bank 1 at 1.84). With the K-token carry restored, beyond-local Δpermutation stays ~0.013, about 3× E17b but still below the 0.02 300M stop and far below a 0.05 1B bar. Free-run stays in the E17 family (prose, `real≈shuffle`), a small lift vs E17b, not a generation win from the new cell.

So the pressure objective does what it was designed to do on the tokens it reweights, and ordinary causal LM still does not need durable cross-block state. More tokens on this cosine will not fix that; the 1B quality run is not justified.

## Decision

**Do not launch 1B.** Treat E17c as a mixed 300M close: keep the carryless bank-0 signal as evidence that causal carry dropout can force concept use, and treat collapsed RankMe + non-transfer to normal CE as the kill for this cell as a generation memory platform. Next bet (not this track) has to explain why only the earliest bank absorbed the pressure and why the other three banks stayed near-static under open update gates.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E17c_depth_private_working_memory.md`, `agenda.md`*
