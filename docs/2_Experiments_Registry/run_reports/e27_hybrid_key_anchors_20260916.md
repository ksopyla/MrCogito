# E27 hybrid key-span anchors — DNA MATCH @512 r=16 — `73wq90jq`

**Date:** 2026-09-16
**Machine:** Odra 1× RTX 3090 (GPU 0; host reports 3 GPUs)
**Run ID:** `73wq90jq`
**WandB:** [https://wandb.ai/ksopyla/MrCogito/runs/73wq90jq](https://wandb.ai/ksopyla/MrCogito/runs/73wq90jq) // pragma: allowlist secret
**Raw log:** Odra `Cache/logs/e24_s0_e27_hybrid_key_anchors_20260916_184048.log`
**Best checkpoint:** none (on-the-fly BAPO probe; models discarded)
**Git commit:** `83011b0` (W&B metadata; PR 42 launch code)
**Git tag:** —
**Related:** E25 MATCH r=16 frozen-mean 0 bits · queue note (PR 40) · diagnosis PR 38

---

## Goal

Test whether exclusive E21 at seq=512 packed `recall_single` can recover MATCH if r=16 frozen-mean slots keep a gist of values **and** DNA **key spans** are exposed as raw identity keys. Success required ≥ 0.75× live E18 bits on this replica (E25 bar ≈ 36 bits).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` dense / e18 / e21 / e18_local |
| Encoder / decoder | H=256, glob=1, stack=2, kv=1, SSMax `log`, inplace r=16, `message_identity_slots`, `message_global_anchors=key_spans` |
| Dataset | on-the-fly DNA `recall_single` `--scale bridge` seq=512, prize 48 bits, chance 0.25 |
| Objective | packed answer CE |
| Steps | 800 (`k1_mult` 4 extends **dense** to 3200); E21 banner ≤800, not extra-stepped to 8k |
| Effective batch | 32 |
| Throughput | ~0.05 s/step on e21 |
| Compute | W&B `_runtime` **108 s**. `compute/audit_state` **failed** (structural: no HF Trainer `world_size` / `global_step` / batch / `max_seq_length`). Do not hand-estimate GPU-h. |

## Training Outcome

Calibrated. Dense **100%** / 47.93 bits @400 (S0 pass). E18 **100%** / **47.87 bits** @350. E21 **0.350 acc** / **3.03 bits** / flow **0.063** @800 (best acc 0.352 @750). `e18_local` **0.253** / **0 bits** (K2 pass). `s1.txt` = FAIL. INDEX skipped.

## Concept Health

- RankMe **1.12** (n=2048, dim=32) — collapsed.
- Channel ablations (acc): `none` 0.253 · `swapped` 0.253 · `slots_only` **0.342** ≈ real **0.350**.
- Dropping key-span anchors does not change the score. Anchors-only was not a separate override; S2 not scored after S1 miss.

## Evaluation

Zero-shot STS-B / GLUE not in scope (DNA probe).

| arch | acc | bits | flow | step |
|---|---:|---:|---:|---:|
| dense | 1.000 | 47.93 | 0.999 | 400 |
| e18 | 1.000 | 47.87 | 0.997 | 350 |
| e21 | 0.350 | 3.03 | 0.063 | 800 |
| e18_local | 0.253 | 0 | 0 | 800 |

JSON: Odra `Cache/bapo_s0/e27_hybrid_key_anchors/bridge_recall_single.json`.

## Interpretation

S1 required ≥ 0.75 × 47.87 ≈ **35.9 bits**. Observed **3.03**. That is a miss, not mixed. K3's letter asked for an 8k extra-step; this replica stayed on the 800-step E21 budget and was not climbing (late 0.25→0.35 then flat). RankMe 1.12 and `slots_only`≈real already show the extra keys were not an addressable `b` channel. Fair baseline: E25 same exam identity r=1 **43 bits**, frozen-mean r=16 **0 bits**. Hybrid did not leave that 0-bit wall.

## Decision

**killed / `done_failed`.** Do not retry `key_spans` on DNA MATCH @512 r=16. Do not extra-step this hunt. Wave B (E28 bits on Odra, E29 bind starting on Polonez) is not recorded here.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E27_hybrid_key_anchors.md`, `agenda.md`*
