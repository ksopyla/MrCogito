# E25 bridge_1k 1024 far_copy r=16 in-place frozen mean — E21 vs E18 vs dense (rung 5j)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_1k_ip_r16_mean` (3200) · `e25_1k_ip_r16_mean_s8k` (8000)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge1k_ip_r16_mean/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge1k_ip_r16_mean_probe.log` (8k; also `_3200.log`)
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `4c4d516` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** 512 frozen mean NEAR-PASS [`e25_bridge512_ip_r16_mean_far_copy_20260914.md`](e25_bridge512_ip_r16_mean_far_copy_20260914.md)

---

## Goal

One **scale** change vs 512 inplace r=16 frozen mean (E21 **47.36 bits @8k**):
`--scale bridge_1k` seq=1024. Compression recipe stays frozen mean
(`--message_slots_inplace --message_identity_slots`, no raw_kv, no learned `u`/`delta`).

Width is the E24 recipe that keeps E18 alive at 1024: **H=256, SSMax `--global_logit_scale log`**,
default 1 KV head. Not H=256 `logit_scale=none` (E24 dilution, E18 0 bits). Not H=512 MHA
(S0 fallback unused).

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.261M (e21 2.277M) · kv=1 · **`logit_scale=log`** |
| E21 | query boundary id 10, **r=16**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`** |
| Data | `--scale bridge_1k` seq=1024, gap=64, window=16, packed span 32 / **64-bit** prize |
| Placement | `--evidence_align right` (row gap 65/65/65) |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~3GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_1k_ip_r16_mean 0 \
  --scale bridge_1k --recipe far_copy --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --amp auto --message_ratio 16 --message_slots_inplace --message_identity_slots \
  --steps 800 --k1_mult 4
# climbing at 3200 (32 bits): extra-step --steps 8000 --k1_mult 1
```

Byobu `E25_1k_ip_mean` / `E25_1k_ip_s8k`. Log:
`seq=1024  logit_scale=log  msg_r=16  msg_inplace=True  msg_rawkv=False  msg_idslots=True`.

Dense first. First JSON dense plateaued at 94.7% through 3200 (S0 still PASS). Extra-step
dense early-stopped **100% @250**.

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense (8k JSON) | **100.0%** | 250 | 63.95 | 0.999 |
| e18 (8k JSON) | **99.7%** | 2750 | 63.12 | 0.986 |
| **e21 @3200** | 68.8% | 3200 | 32.37 | 0.506 |
| **e21 @8000** | **91.2%** | **8000** | **53.82** | **0.841** |
| e21 best | **92.7%** | 7850 | (final metric is @8000) | — |
| e18_local (8k JSON) | 25.9% | 8000 | 0 | 0 |

First JSON: dense **94.7% / 59.98 bits** @3200 (S0 PASS); E18 **100% / 63.95 bits** @950
(live control — not the dilution recipe). First-JSON E21 @800 was **41.9%** (climbing, not
floor) → extra-step. 8k curve: chance through ~800, ~45–50% through 4800, jump ~5200
(73.7%), **91.2% @8000**.

S1 vs 0.75× E18 in the 8k JSON: need **47.34 bits / flow 0.740**. E21 has **53.82 / 0.841**.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 100% / 63.95 bits (8k JSON; 94.7% / 59.98 at 3200). |
| **S1 vs 0.75× E18** | **PASS.** 53.82 ≥ 47.34 bits. |
| **S2** plots | **PASS.** |
| **K1** | not triggered (no H=512 bump). |
| **K2** | **PASS.** `e18_local` 25.9% / 0 bits. |
| **K3** | not triggered (climbing). |

Do **not** extra-step to 16k. Do not 4k. Do not Glyph. Do not remainder. Do not unfreeze.

## Interpretation

**Frozen 16-token means carry INDEX at 1024** on the SSMax-log recipe whose E18 control
copies 64 bits. E21 is slower than E18 (91% @8k vs 100% @950–2750) and recovers **53.8
bits, not 63.1** — do not relabel E18 as E21. Concat exclusive slots at 512 were 0 bits;
inplace frozen mean is a live compressed INDEX channel at both 512 (near-pass) and 1024
(pass). Learned `u`/`delta` stays off.

## Decision

Keep the spec in `ahead/`. Working recipe: inplace + `--message_identity_slots`, H=256
SSMax log at 1024. Next ONE experiment: seq=512 packed **`recall_single`**, same frozen
mean (content vs INDEX). Not 4k. Not Glyph. Not learned pool.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
