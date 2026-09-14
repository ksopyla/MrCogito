# E25 bridge seq=288 chain_ordered --key_len 13 `--global_layers 2` — dense S0 PASS, E18/E21 hops FAIL (rung 5bd)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_288_chain_k13_glob2` (800 advertised; dense-matched 3200; e18 / e21 / e18_local scored; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge288_chain_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `835d15d` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=256 glob2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md) · seq=320 glob2 K1 [`e25_bridge320_chain_k13_glob2_20260914.md`](e25_bridge320_chain_k13_glob2_20260914.md)

---

## Goal

Measured wall: seq=256 packed `chain_ordered --key_len 13` hops=2, H=256 SSMax
log, **two** global layers — dense **99.2% / 25.46 bits**; E21 **S1 PASS 25.85**;
live E18 **5.56 bits**. Same recipe at seq=320/384/512 was dense **K1**.
Hypothesis: packed 288 (midpoint of **(256 S0 PASS, 320 K1]**) is dense-solvable
at two global Blocks. Same DNA + compressor as the 256 glob2 PASS, **only**
`--seq_len 288` (bridge default is 512). Recalibrate dense S0 at two global
blocks in this JSON. If dense S0 PASSes, score e18/e21.

If dense misses 75% (K1), stop; do not score E21. Chance at 800 → no 8k.
Climbing short of S1 → 8k only then. `--hops 1` is illegal. Do not relabel
E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · **2.802M** (e21 **2.835M**) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 288`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 169/188/206 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.2GB, 0.04–0.05 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_288_chain_k13_glob2 0 \
  --scale bridge --seq_len 288 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# --seq_len 288 required (bridge default is 512)
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused)
# chance at 800: do not extra-step; do not 16k
```

Byobu `E25_288_chain_glob2`. Log:
`seq=288  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=169/188/206`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 288`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`. Hunt exit **0**.
Params <100M. Dense S0 recalibrated at two full layers (2.802M vs glob=1 2.261M).

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **93.1%** | 3200 | **22.20** | **0.854** |
| e18 | **26.8%** | 3200 | **0.00** | **0.000** |
| **e21 @800** | **26.8%** | 800 | **0.00** | **0** |
| **e21 final** | **24.5%** | 3200 | **0.00** | **0.000** |
| e18_local | **24.5%** | 3200 | **0.00** | **0** |

Dense crossed 75% at 2200 (76.1%), best **98.2% / CE 0.054** @3000–3100, final
**93.1% / 22.20 bits** @3200 (late dip, still S0). JSON `calibrated: true`.
Hunt exit 0.

E18 sat at chance every eval (CE at ln(4) ≈ 1.386; best 26.8% is noise around
25%). Final **26.8% / 0.00 bits** @3200. Live **~0**, not the 256 glob2 5.56-bit
climb. Do **not** pass S1 via 0.75×0.

E21 sat at chance through the advertised 800 floor (26.8% / CE 1.386) **and**
through the dense-matched 3200 budget (final 24.5% / **0.00 bits**; CE at
ln(4)). Not climbing.

**8k not run** (chance floor at 800; same floor at 3200). Do not 16k. Do not
relabel the skipped-looking E18 arm as E21 — E18 was scored and is 0 bits.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 93.1% / 22.20 bits @3200 ≥ 75%. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not a pass.** E18 is ~0 bits; do not use 0.75×0. |
| **content vs 0.75× dense** | **FAIL.** 0.00 vs 16.65 bits. |
| **S2** plots | **PASS** (learning-curves / heatmap / recovered-bits / flow / bytes-per-token). |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 24.5% / 0 bits. |
| **K3** | **triggered.** E21 chance floor @800 and @3200. |

## Interpretation

**Packed 288 hops is dense-solvable at two global Blocks, but neither E18 nor
E21 composes the DFA.** The same compressor and glob=2 recipe that composed
hops at seq=256 (E21 25.85 / E18 5.56) is chance at seq=288 for both global-read
arms, while dense recovers **22.20 of 26 prize bits**. This is not an E21-only
compression failure: uncompressed E18 is also 0 bits (unlike 256 glob2, where
E18 was live but weak).

Dense hops wall at glob=2 is now **(288 S0 PASS, 320 K1]**. Exclusive / one-read
hops wall at glob=2 is **(256 S1 PASS, 288 FAIL]**.

## Decision

Keep the spec in `ahead/`. Do **not** extra-step to 8k (chance floor). Do not
16k. Do not hops seq shrink below 256. Do not Glyph. Do not unfreeze `u`/`delta`.
Do not restore full raw prefix KV (that is E18). Do not reopen SELECT
**(692, 696]**. Do not INDEX extra-steps. Code default `--global_layers` stays 1.
Next ONE (do not run): seq=**272** packed `chain_ordered --key_len 13`
`--global_layers 2` (localize the E21/E18 hops wall **(256 S1 PASS, 288 FAIL]**;
dense is already S0 PASS at 288). Recalibrate dense S0. `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
