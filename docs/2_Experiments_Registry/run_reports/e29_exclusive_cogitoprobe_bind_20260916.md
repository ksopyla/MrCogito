# E29 exclusive CogitoProbe-bind @1024 — hop 0.078 bits, no dense S0

**Date:** 2026-09-16 (train + probe); recorded 2026-09-17
**Machine:** Polonez 4× RTX 3090
**Run ID:** `perceiver_ar_perceiver_H256L1g1s2N1024_20260916_170437`
**WandB (training):** [https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H256L1g1s2N1024_20260916_170437](https://wandb.ai/ksopyla/MrCogito/runs/perceiver_ar_perceiver_H256L1g1s2N1024_20260916_170437) // pragma: allowlist secret
**Raw log:** Polonez `Cache/logs/shell_perceiver_denoise_20260916_170331.log`
**Best checkpoint:** trainer `eval_loss` min at `checkpoint-200`; probe JSON scored **`checkpoint-768`** (last)
**Git commit:** `481bf32` (PR 42 launch; Polonez not pulled to `501896f`)
**Git tag:** `arch/e18-perceiver-ar-v2-229-g481b`
**Related:** queue [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md) · E28 [e28_exclusive_cogitoprobe_bits_20260916.md](e28_exclusive_cogitoprobe_bits_20260916.md)

---

## Goal

Test whether exclusive r=16 slots bind `(entity, attribute, value)` on **`ksopyla/cogito-probe-bind`**: `hop_friend_place` bits ≥ 0.75× dense whenever `attr_color` also clears that bar, with **`ksopyla/cogito-probe-props`** filler-shuffle as the gist control on the same checkpoint.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` exclusive · H=256 glob=1 stack=2 · inplace r=16 · `identity_slots=True` · boundary **48** · no prefix AE · no `key_spans` |
| Dataset | Hub `ksopyla/cogito-probe-bind` seq=1024 `fixed` · 2,048 train / 128 val · packing off · packed-answer CE |
| Steps | 12 epochs · **768** steps · batch 32 (4×4×accum 2) · lr 3e-4 · warmup 50 |
| Probe | bind test by task + props `prop_color` control vs `--shuffle_filler` |
| Compute | dry-run audit: **0.739 GPU-h** / 0.206 kWh / 0.0252B max-tokens. `compute/audit_state=flagged` (`loss_fraction:unknown`; `gpu_hours:summary_vs_ts:0.030`). World 4. Do not hand-estimate. |

Dense S0 **was not trained**. Proposition-colour shuffle **was not run**.

## Training Outcome

Finished 12 epochs. Last train CE **2.28**, eval **4.26** (best eval 3.81 @200). Throughput ~25k real tok/s. Pad ratio 0.

## Concept Health

No RankMe / STS-B. Task breakout on last ckpt (`real`):

| task | n | acc | bits | prize | none bits | swapped bits |
|---|---:|---:|---:|---:|---:|---:|
| `hop_friend_place` | 37 | **0.044** | **0.078** | 40 | 0.057 | 0.070 |
| `attr_color` | 47 | **0.024** | **0.007** | 40 | 0 | 0.010 |
| `who_place` | 44 | **0.006** | **0** | 24 | 0 | 0 |
| bind pooled | 128 | 0.023 | 0.025 | 34.5 | 0.016 | 0.026 |

Hop 8 tokens × 5 bits → 32-way chance ≈ 3.1%. **4.4% / 0.078 bits** is chance, not a binding memory. `attr_color` is also chance, so K2 (labels pass, hop fails) does **not** fire. `who_place` 0.57% is below 8-way chance (~12.5% on a 24-bit/8-token prize). Combined `real−none` **0.009 bits**; `swapped ≰ none` at this noise.

## Evaluation

Props `prop_color` (n=128, prize 30 bits), same ckpt, no extra train:

| condition | acc | bits |
|---|---:|---:|
| control (`shuffle_filler=false`) | **0.040** | **0.094** |
| filler shuffle | **0.042** | **0.102** |

Δacc **+0.13 pt** — filler shuffle does **not** drop answers (S3 filler clause holds; K3 does not fire). Proposition shuffle in `meta.propositions` is **missing**, so S3 is incomplete.

JSON: Polonez `Cache/Evaluation_reports/e29_bind_{hops,gist_attr,gist_who}_*.json` · `E29_bind_seq1024_fixed_*.json` · `e29_props_{control,filler_shuffle}_*.json`.

## Interpretation

S1 required hop bits ≥ 0.75× dense. Dense never ran, so K1 vs K4 cannot be lettered cleanly — and that does not make the hop a pass. **0.078 bits** of a 40-bit one-hop prize is a miss. Colour-list gist is not a hidden win. Do not grow length.

## Decision

**killed / `done_failed`.** Do not 4k/8k/32k. Do not close E19 (gate still unmet). Do not launch arith.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E29_exclusive_cogitoprobe_bind.md`, `agenda.md`*
