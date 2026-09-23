# E25 bridge seq=320 chain_ordered --key_len 13 `--global_layers 2` — dense K1 (rung 5bc)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_320_chain_k13_glob2` (800 advertised; dense K1 budget 3200; e18/e21/`e18_local` skipped; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge320_chain_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge320_chain_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `6d063e8` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** seq=256 glob2 S1 PASS [`e25_bridge256_chain_k13_glob2_20260914.md`](e25_bridge256_chain_k13_glob2_20260914.md) · seq=384 glob2 K1 [`e25_bridge384_chain_k13_glob2_20260914.md`](e25_bridge384_chain_k13_glob2_20260914.md) · seq=512 glob2 K1 [`e25_bridge512_chain_k13_glob2_20260914.md`](e25_bridge512_chain_k13_glob2_20260914.md)

---

## Goal

Measured wall: seq=256 packed `chain_ordered --key_len 13` hops=2, H=256 SSMax
log, **two** global layers — dense **99.2% / 25.46 bits**; E21 **S1 PASS 25.85**;
live E18 **5.56 bits**. Same recipe at seq=384 and seq=512 was dense **K1**.
Hypothesis: packed 320 (midpoint of **(256 S0 PASS, 384 K1]**) is dense-solvable
at two global Blocks. Same DNA + compressor as the 256 glob2 PASS, **only**
`--seq_len 320` (bridge default is 512). Recalibrate dense S0 at two global
blocks in this JSON.

If dense misses 75% (K1), stop; do not score E21. Chance at 800 → no 8k.
`--hops 1` is illegal. Do not relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | dense first; e18 / e21 / e18_local skip on K1 |
| Width | **H=256** · **2.802M** · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 (not scored) | query boundary id 10, **r=1**, remainder **off**, **`msg_inplace=True`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 320`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16**. Right-align gap 186/203/238 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · 0.04 s/step, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_320_chain_k13_glob2 0 \
  --scale bridge --seq_len 320 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# --seq_len 320 required (bridge default is 512)
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused)
# dense K1: do not extra-step; do not score E21
```

Byobu `E25_320_chain_glob2`. Log:
`seq=320  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=186/203/238`
dense `patterns=[('full', 0)×5]  glob_layers=2` · 2.802M.

Hunt JSON: `seq_len: 320`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: false`. Hunt exit **2**.
Params <100M. Dense S0 recalibrated at two full layers (2.802M vs glob=1 2.261M).

## Training Outcome

| arch | acc | step | recovered bits | flow |
|---|---|---|---|---|
| dense | **26.2%** | 3200 | **0.00** | **0.000** |
| e18 | skipped (K1) | — | — | — |
| e21 | skipped (K1) | — | — | — |
| e18_local | skipped (K1) | — | — | — |

Dense sat at chance every eval through the full K1 budget (CE at ln(4) ≈ 1.386).
Best acc **27.9%** @2400 (noise around 25%). Final **26.2% / 0 bits** @3200.
JSON `calibrated: false`. Hunt exit 2.

Do **not** score S1 vs E18 or vs dense. There is no E21 number. Do not relabel
the skipped E18 arm as E21.

**8k not run** (dense chance floor / K1). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **FAIL / K1.** Dense 26.2% / 0 bits @3200 < 75%. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **not scored.** |
| **content vs 0.75× dense** | **not scored.** |
| **S2** plots | **PASS** (dense-only uncalibrated curves). |
| **K1** | **triggered.** Stop. Do not score E21. |
| **K2** | **not scored.** |
| **K3** | **not scored.** |

## Interpretation

**Two sequential global attend+FFN Blocks do not make packed 320 hops
dense-solvable.** The same compressor and glob=2 recipe that composed hops at
seq=256 (dense 25.46 / E21 25.85) is chance at seq=320 for dense — matching
seq=384 glob=2 K1 (24.5% / 0 bits) and seq=512 glob=2 K1 (22.5% / 0 bits).
The dense hops wall at glob=2 is **context length**, not a missing midpoint
between 256 and 384. Wall is now **(256 S0 PASS, 320 K1]**.

## Decision

Keep the spec in `ahead/`. Do **not** extra-step to 8k (dense chance floor).
Do not 16k. Do not score E21. Do not hops seq shrink below 256. Do not Glyph.
Do not unfreeze `u`/`delta`. Do not restore full raw prefix KV (that is E18).
Do not reopen SELECT **(692, 696]**. Do not INDEX extra-steps. Code default
`--global_layers` stays 1. Next ONE (do not run): seq=**288** packed
`chain_ordered --key_len 13` `--global_layers 2` (localize the dense hops wall
**(256 S0 PASS, 320 K1]**). Recalibrate dense S0. `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
