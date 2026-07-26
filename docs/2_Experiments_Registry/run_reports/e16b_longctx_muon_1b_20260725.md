# E16b — Long-context Muon unlocks shared-depth causal concepts — `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850`

**Date:** 2026-07-25
**Machine:** Odra (3× RTX 3090, 24 GB each)
**Run ID:** `backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850`
**WandB (training):** [Link](https://wandb.ai/ksopyla/MrCogito/runs/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850)
**Raw shell log:** `Cache/logs/shell_perceiver_denoise_20260718_143716.log`
**Best checkpoint:** `Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850/checkpoint-7900`
**Git commit:** `613c5fc`
**Git tag:** `odra-smoke-verified-20260712-18-g613c`
**Related TODO:** E16b long-context Muon scale-up — [spec](../../experiments_specs/done_success/E16b_longctx_muon_1b.md)

---

## Goal

Test whether the E16 shared depth-recurrent concept workspace becomes causally useful
when trained with Muon on a 4K long-document mix for 1B non-padding tokens.

This is the first Gemma-backbone run to clear the ≥0.01 beyond-local causal-use gate.
Earlier E10–E16a near-nulls are still useful evidence about **short 2K / ≤100M plain CE**;
E16b shows a **different operating regime** works. See **Why this worked** below.

## Configuration

| Item | Value |
|---|---|
| Family / backbone | `backbone_concept`; frozen `google/gemma-3-1b-pt` + LoRA r=16 |
| Memory | C=128; K=512; `shared_depth_recurrent`; read RMSNorm; gate init 0.01 |
| Optimizer | Muon 0.01 / wd 0.1 / adamw_lr 2e-4 |
| Data / objective | Gemma-tokenized `e16b_long_4k_v1`; seq **4096**; causal next-token CE |
| Budget | 1B target tokens; 7,905 steps (resumed from failed `…20260716_093047` ckpt-3950); effective batch 72; seed 42 |
| Compute | **114.92 GPU-h / 34.44 kWh / 2.331B max-token upper bound** (`compute/audit_state=finished`; flag `loss_fraction:unknown`) |

## Training Outcome

Completed stably (exit 0) on 2026-07-20 after ~38.3 h wall. Eval CE fell monotonically
2.269 → **1.621** (best at step 7900). Training-time beyond-local ablations rose nearly
monotonically across the 1B budget:

| step (approx tokens) | eval CE | Δshuffle_beyond | Δstatic_beyond | RankMe |
|---:|---:|---:|---:|---:|
| 790 (~0.1B) | 2.269 | 0.001 | 0.000 | 104 |
| 3950 (~0.5B) | 2.216 | 0.061 | 0.047 | 89 |
| 5530 (~0.7B) | 2.089 | 0.320 | 0.297 | 101 |
| **7900 (1B, best)** | **1.621** | **1.536** | **1.504** | **101** |

No NaN/Inf; no safety kill. Signal was absent early and **emerged with scale** — unlike
the flat near-null trajectories of the short-context E10–E16a runs.

## Offline Tier-1 (2026-07-25)

Protocol: `backbone_concept`, pretokenized `e16b_long_4k_v1`, seq 4096, buckets
`1024,2048`, 24 docs × 2 seeds. Best `checkpoint-7900` and last `checkpoint-7905`
are essentially identical.

| Metric | Best (ckpt-7900) |
|---|---|
| within-sample RankMe | **101.0** /128 (centered 107.9) |
| Δshuffle ≥1024 | **2.47** |
| Δstatic ≥1024 | **2.35** |
| Δone-block ≥1024 | **0.58** |
| `(2048,4096]` Δshuffle | **1.99** |
| `(2048,4096]` Δzero | **1.70** |

Artifacts:
`Cache/Evaluation_reports/e16b_backbone_concept_gemma_3_1b_pt_K512_concept_20260718_150850_ckpt7900_best_concept_analysis.json`
(and last sibling).

### Against registered criteria

| Criterion | Result |
|---|---|
| Primary: `min(Δstatic, Δshuffle)_beyond ≥ 0.01` at long positions | **PASS** — offline min≥1024 ≈ **2.35** (~235× gate); `(2048,4096]` Δshuffle **1.99** |
| Geometry: within-sample RankMe ≥38.4 | **PASS** — **101.0** |
| Depth utilization ≥3/4 read & write \|gate\|≥0.005 | **PASS** |
| Scale evidence (monotonic rise across budget) | **PASS** |
| Soft safety kills | none fired |

### Fair same-family baselines

| Run | Budget / seq | RankMe | min beyond Δ | Note |
|---|---|---:|---:|---|
| E10 / E10e | 100M / 2K | 77–100 | ~0.001 | short-ctx regime; near-null causal use |
| E16 Adam | 50M / 2K | 62 | ~0.0005 | same architecture, short-ctx |
| E16a Muon | 100M / 2K | 97 | **0.0028** | best short-ctx; still below 0.01 |
| E16b mid (ckpt-3950) | ~0.5B / 4K | 91 | ~0.09–0.11 | first clear of 0.01 mid-run |
| **E16b final** | **1B / 4K** | **101** | **~2.3** | **success — validated path** |

Generic STS-B / SICK / PAWS / GLUE were **not** part of E16b's registered gate.

## Why this worked

**The shared-depth recurrent workspace was already viable.** E10–E16a repeatedly showed
healthy concept geometry (RankMe often 60–100) with near-zero beyond-local ΔCE under
**short 2K plain CE**. E16b shows that a longer-context / higher-budget regime can
make the same interface causally load-bearing:

1. **Multi-block pressure helps.** At seq 2048 with K=512 there are only ~4 concept
   blocks, and the `smollm3_inspired_2k_e05` mix is fluency-heavy — next-token CE can
   often be satisfied from local Gemma context. At seq **4096** on **`e16b_long_4k_v1`**
   there are **8 blocks** and longer documents that reward cross-block memory.
2. **The signal grows with tokens.** Training-time beyond-Δ stayed ~0.001 early,
   crossed 0.01 around mid-budget, and reached ~1.5 at 1B. E16a’s 0.0028 at 100M/2K
   looks like an early point on a curve that short-context runs did not climb far.
3. **Compound regime change, not a brand-new architecture.** Shared depth-recurrent
   writes (E16) + calibrated gates/RMSNorm are kept. What changed is length + long-doc
   mix + Muon + 1B budget — intentional compound bet; factor isolation remains open.
4. **`Δone-block≈0.58` supports recurrence.** Shuffle/static beyond show concepts are
   content-bearing; one-block shows accumulated multi-block state beats prior-block-only.

Earlier short-ctx near-nulls remain valid evidence about that regime. They do not
cancel this success, and they do not shut down other research paths (E08, diffusion,
design-only alternatives).

## Interpretation

Against E16b’s own registered question this is a **mechanism success**: concepts in
the shared depth-recurrent workspace are causally load-bearing over multi-block
positions at this scale. That establishes **one valid path**. Open questions
(factor isolation; transferable semantics / reasoning; E08 composition; diffusion
revive) stay in play with **shifted priorities** — more weight on exploring this
regime next, without discarding other directions.

## Decision

1. Record E16b as `done_success` on the mechanism gate — a validated long-context path.
2. Keep prior E10–E16a / E14–E15 results as-is (regime evidence); lower near-term
   priority for more short-ctx micro-calibrations, not a ban.
3. **Next focus:** semantic probe (STS-B + floors) on this checkpoint; continue
   exploring this route; also keep E08 / diffusion / other design-only ideas available.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E16b_longctx_muon_1b.md`, `agenda.md`*
