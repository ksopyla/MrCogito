# E26 prefix-AE exclusive slots — DNA MATCH @512 r=16 — `50t48aas`

**Date:** 2026-09-16
**Machine:** Polonez 1× RTX 3090 (GPU 0; host reports 4 GPUs)
**Run ID:** `50t48aas`
**WandB:** [https://wandb.ai/ksopyla/MrCogito/runs/50t48aas](https://wandb.ai/ksopyla/MrCogito/runs/50t48aas) // pragma: allowlist secret
**Raw log:** Polonez `Cache/logs/e24_s0_e26_prefix_ae_slots_20260916_164428.log`
**Best checkpoint:** none (on-the-fly BAPO probe; models discarded)
**Git commit:** `32ab7d1` (W&B metadata; PR 42 launch code)
**Git tag:** —
**Related:** E25 learned-pool MATCH 0 bits · queue note (PR 40) · diagnosis PR 38

---

## Goal

Test whether a weak prefix-block autoencoder (linear head, compressor grads from AE only) makes exclusive r=16 inplace slots MATCH-addressable. Success required ≥ 0.75× live E18 bits (≈ 36).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` dense / e18 / e21 / e18_local |
| Encoder / decoder | H=256, glob=1, stack=2, kv=1, SSMax `log`, inplace r=16, **no** identity_slots, `message_prefix_ae` λ=1.0 |
| Dataset | on-the-fly DNA `recall_single` `--scale bridge` seq=512, prize 48 bits, chance 0.25 |
| Objective | packed answer CE + prefix-block AE |
| Steps | 800 (`k1_mult` 4 on dense); E21 ≤800, not extra-stepped to 8k/16k |
| Effective batch | 32 |
| Throughput | ~0.05 s/step on e21 |
| Compute | W&B `_runtime` **90 s**. `compute/audit_state` **failed** (same structural miss as E27). Do not hand-estimate GPU-h. |

## Training Outcome

Calibrated. Dense **100%** / 47.97 bits @250 (S0 pass). E18 **0.993 acc** / **47.32 bits** @350. E21 **0.311 acc** / **1.03 bits** / flow **0.021** @800 (best acc 0.324). `e18_local` **0.243** / **0 bits** (K2 pass). `s1.txt` = FAIL. INDEX skipped.

First start `reyc0q2p` (16:40, wrong W&B project (cwd name; `.env` lacked entity/project), SIGTERM 143 during `e18_local`) is **not** ledger evidence.

## Concept Health

- RankMe **14.6** (n=2048, dim=32) — not collapsed.
- Prefix AE: loss 1.02 · tok_acc **0.54** · key_acc **0.66** (n=15360).
- Ablations (acc): `none` 0.247 · `swapped` 0.249 (no `slots_only`; no key-span anchors).

## Evaluation

Zero-shot STS-B / GLUE not in scope.

| arch | acc | bits | flow | step |
|---|---:|---:|---:|---:|
| dense | 1.000 | 47.97 | 0.999 | 250 |
| e18 | 0.993 | 47.32 | 0.986 | 350 |
| e21 | 0.311 | 1.03 | 0.021 | 800 |
| e18_local | 0.243 | 0 | 0 | 800 |

JSON: Polonez `Cache/bapo_s0/e26_prefix_ae_slots/bridge_recall_single.json`.

## Interpretation

S1 required ≥ 0.75 × 47.32 ≈ **35.5 bits**. Observed **1.03**. Kill letters: K3 needs AE key_acc ≥ 80% **and** MATCH flow < 0.05 — key_acc **0.66** so K3 does not fire; K4 needs key_acc < 40% — not met; K5 is MATCH at chance at 8k — this run stopped at 800 with flow 0.021 (near floor). The honest reading is still an S1 miss: geometry exists (RankMe 14.6) and the AE writes some keys (66%), but the exclusive read does not recover MATCH. Do not call this mixed.

## Decision

**killed / `done_failed`.** Do not train `u`/`delta` longer under answer CE. Do not treat 66% key reconstruction as a MATCH channel. Wave B (E28 on Odra, E29 starting on Polonez) is not recorded here.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E26_prefix_ae_exclusive_slots.md`, `agenda.md`*
