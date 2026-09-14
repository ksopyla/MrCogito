# E25 bridge 512 recall_single r=1 in-place hard identity — E21 vs E18 vs dense (rung 5l)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_512_ip_id_recall`
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge512_ip_id_recall/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge512_ip_id_recall_probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `39ebd94` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** r=16 MATCH wall [`e25_bridge512_ip_r16_mean_recall_20260914.md`](e25_bridge512_ip_r16_mean_recall_20260914.md) · r=1 INDEX identity [`e25_bridge512_ip_id_far_copy_20260913.md`](e25_bridge512_ip_id_far_copy_20260913.md)

---

## Goal

One change vs 512 `recall_single` frozen mean r=16 (E21 **0 bits** while E18 **47.94**):
keep inplace + `--message_identity_slots`, no raw_kv, and set `--message_ratio 1`.
Does exclusive **uncompressed token KV** bind a key, or can the exclusive channel
only INDEX?

H=128 (same 512 MATCH-wall width). Score vs 0.75× E18; also report vs 0.75× dense.
If E18 were ~0, would not call S1 from 0.75×0.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=128** · 0.595M (e21 0.603M) · kv=1 · **`logit_scale=none`** |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge` seq=512, gap=64, window=16, packed answer 24 / **48-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_512_ip_id_recall 0 \
  --scale bridge --recipe recall_single --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 128 --amp auto \
  --message_ratio 1 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
```

Byobu `E25_512_id_recall`. Log:
`msg_r=1  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first. Hunt JSON: `message_ratio: 1`, `message_identity_slots: true`,
`message_inplace_raw_kv: false`.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **99.9%** | 350 | 47.69 | 0.994 |
| e18 | **99.4%** | 350 | **46.67** | 0.972 |
| **e21 identity r=1** | **91.6%** | **800** | **43.08** | **0.897** |
| e21 best | **92.1%** | 750 | (final metric is @800) | — |
| e18_local | 25.3% | 800 | 0 | 0 |

E18 is **live** (46.67 bits). E21 chance through ~350, then 35.5% @400 → 92.1% @750.
r=16 frozen mean on the same task was **0 bits / floor at 800**.

S1 vs 0.75× E18: need **35.00 bits**. E21 has **43.08**.
Content vs 0.75× dense: need **35.77 bits**. E21 has **43.08**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.9% / 47.69 bits. |
| **S1 vs 0.75× E18** | **PASS.** 43.08 ≥ 35.00 bits. |
| **content vs 0.75× dense** | **PASS.** 43.08 ≥ 35.77 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered |
| **K2** | **PASS.** `e18_local` 25.3% / 0 bits. |
| **K3** | not triggered (flow 0.90). |

Do **not** extra-step to 8k: S1 already PASS at 800 (peak 92.1% @750). Do not 4k.
Do not Glyph. Do not remainder. Do not unfreeze. Do not 1024 recall.

## Interpretation

**Exclusive identity binds a key; r=16 pooling is the MATCH killer.** The exclusive
QUERY cut with uncompressed token KV copies **43.08 bits** of packed `recall_single`
(same mask that copies INDEX at r=1). Frozen 16-token means on that mask copy INDEX
and **0 MATCH bits**. Do not relabel E18's 46.67 bits as E21 — E21 recovered **43.08**.

## Decision

Keep the spec in `ahead/`. MATCH-capable recipe: inplace + `--message_identity_slots`
+ **r=1**. INDEX-capable compression stays r=16 frozen mean. Next ONE experiment:
seq=512 packed **`select_1decoy`**, inplace r=1 identity (type-cue vs MATCH). Not 4k.
Not Glyph. Not learned pool. Not 1024 recall.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
