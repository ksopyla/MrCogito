# E25 bridge seq=256 chain_ordered --key_len 13 `--global_layers 2` — E21 hops S1 PASS (rung 5az)

**Date:** 2026-09-14
**Machine:** Odra 1× RTX 3090 (GPU 0). Polonez left empty. E22 Byobu sessions untouched.
**Run ID:** `e25_256_chain_k13_glob2` (800 advertised; dense-matched 1800; E21 early-stop 1650 — S1 PASS; 8k not run)
**WandB:** n/a (probe; no `compute/*` audit — there is no W&B run)
**Raw JSON/plots:** `/opt/cursor/artifacts/e25_bridge256_chain_k13_glob2/`
**Raw log:** `/opt/cursor/artifacts/e25_bridge256_chain_k13_glob2/probe.log`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git commit:** `2870430` (Odra E24 worktree; E22 main checkout untouched)
**Git tag:** —
**Related:** glob=1 hops FAIL [`e25_bridge256_chain_k13_20260914.md`](e25_bridge256_chain_k13_20260914.md) · 1024 SELECT glob2 chance [`e25_bridge1k_ip_id_glob2_select_20260914.md`](e25_bridge1k_ip_id_glob2_select_20260914.md)

---

## Goal

Measured wall: seq=256 packed `chain_ordered --key_len 13` hops=2, H=256 SSMax
log, **one** global layer — dense **96.4% / 24.12 bits**; **E18 and E21 both 0**.
Not E21-only. Hypothesis: hops need **two full sequential global attend+FFN
Blocks**, not a second attend inside one Attention (already failed on 1024
SELECT). Same DNA + compressor recipe as the FAIL, **only** add
`--global_layers 2`. E21 still exclusive identity slots; E18 still raw prefix.
Dense S0 at two global layers is a recalibration (same JSON).

Score vs 0.75× live E18; if E18 were ~0, score vs 0.75× dense. Chance at 800 →
no 8k. Climbing short of S1 → 8k only then. Do not 16k. Do not Glyph. Do not
`--hops 1`. Do not relabel E18 as E21.

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar`: dense / e18 / e21 / e18_local |
| Width | **H=256** · 2.802M (e21 2.835M) · kv=1 · **`logit_scale=log`** · stack=2 |
| E21 | query boundary id 10, **r=1**, remainder **off**, **`msg_override=real`**, **`msg_inplace=True`**, **`msg_rawkv=False`**, **`msg_idslots=True`**, **`msg_keepswa=False`**, **`msg_extrahops=0`**, **`msg_updatekv=False`**, **`msg_anchors=none`**, **`glob_layers=2`** |
| Data | `--scale bridge --seq_len 256`, **`--key_len 13`**, hops=2, **26-bit** prize |
| Geometry | min_gap **64** > local_window **16** (not tiny_wide). Right-align gap 153/161/178 |
| Init | `--warm_residuals` |
| Device | CUDA bf16 (`--amp auto`), `backend=sdpa` · ~1.1GB, no OOM |

```
bash scripts/e24_bapo_hunt.sh e25_256_chain_k13_glob2 0 \
  --scale bridge --seq_len 256 --recipe chain_ordered --arch dense e18 e21 e18_local \
  --evidence_align right --warm_residuals --hidden 256 --global_logit_scale log \
  --key_len 13 --amp auto --message_ratio 1 --message_slots_inplace \
  --message_identity_slots --global_layers 2 --steps 800 --k1_mult 4
# no --hops (recipe hops=2; --hops 1 is illegal)
# no --message_pool_remainder
# no --message_keep_local_swa
# no --message_extra_slot_attends (default 0)
# no --message_update_slot_kv
# no --message_global_anchors (default none)
# dense first (underscore --no-dense_first unused)
# S1 already PASS: do not extra-step
```

Byobu `E25_256_chain_glob2`. Log:
`seq=256  gap=64  window=16  prize=26.00 bits  answer_len=13
row_gap[min/med/max]=153/161/178`
dense `patterns=[('full', 0)×5]  glob_layers=2`
e18/e21 `patterns=[('swa', 16), ('full', 0), ('full', 0), ('swa', 16), ('swa', 16)]`
e21 `msg_r=1  msg_remainder=False  msg_inplace=True  msg_rawkv=False
msg_idslots=True  msg_keepswa=False  glob_layers=2  msg_extrahops=0
msg_updatekv=False  msg_anchors=none`.

Hunt JSON: `seq_len: 256`, `min_gap: 64`, `local_window: 16`, `hidden: 256`,
`global_logit_scale: log`, **`global_layers: 2`**, `message_ratio: 1`,
`message_pool_remainder: false`, `message_slots_inplace: true`,
`message_identity_slots: true`, `message_inplace_raw_kv: false`,
`message_keep_local_swa: false`, `message_extra_slot_attends: 0`,
`message_update_slot_kv: false`, `message_global_anchors: none`,
`hops: 2`, `prize_bits: 26.0`, `calibrated: true`. Hunt exit **0**.
Params <100M. Dense S0 recalibrated at two full layers (2.802M vs glob=1 2.261M).

## Training Outcome

| arch | acc | step | recovered bits | flow | B/tok |
|---|---|---|---|---|---|
| dense | **99.2%** | 1800 | **25.46** | 0.979 | 0.01243 |
| e18 | **42.3%** | 1800 | **5.56** | 0.214 | 0.00272 |
| **e21 @800** | **24.0%** | 800 | **0.02** | ~0 | ~0 |
| **e21 final** | **99.6%** | 1650 | **25.85** | **0.994** | **0.01262** |
| e18_local | **23.3%** | 1800 | **0.00** | 0 | 0 |

Dense crossed 75% at 1350, early-stop **99.2% / 25.46 bits** @1800 (best = final).
JSON `calibrated: true`. Hunt exit 0.

E18 is **live, not ~0**: 29.1% / 0.28 bits @800, best 44.8% / 5.77 bits @1550,
final **42.3% / 5.56 bits** @1800 (climbing, short of solving hops). Do **not**
pass S1 via 0.75×0.

E21 sat at chance through the advertised 800 floor (24.0% / 0.02 bits; CE at
ln(4)). Dense-matched budget continued to 1800 (existing hunt protocol). Click
started ~1150 (33.9% / 1.11 bits), **50.5% / 12.47 bits @1450**, **97.8% /
24.88 bits @1550**, early-stop **99.6% / 25.85 bits @1650**.

S1 vs 0.75× live E18 **4.17 bits**: E21 **25.85 PASS**. Content vs 0.75× dense
**19.10 bits**: E21 **25.85 PASS**. Exclusive identity with two global blocks
even **beats** live E18 (5.56 bits) on this rung. Do not relabel E18 as E21.

**8k not run** (S1 already PASS inside the dense-matched budget). Do not 16k.

## Gates vs this rung

| gate | result |
|---|---|
| **S0** | **PASS.** Dense 99.2% / 25.46 bits @1800 ≥ 75%. Recalibrated at `global_layers=2`. |
| **S1 vs 0.75× live E18** | **PASS.** 25.85 ≥ 4.17 bits. E18 is live (5.56), not ~0. |
| **content vs 0.75× dense** | **PASS.** 25.85 ≥ 19.10 bits. |
| **S2** plots | **PASS.** learning-curves / heatmap / recovered-bits / flow / bytes-per-token. |
| **K1** | not triggered. |
| **K2** | **PASS.** `e18_local` 23.3% / 0 bits. |
| **K3** | not triggered (E21 not chance at finish). |

## Interpretation

**Two sequential global attend+FFN Blocks compose hops at seq=256.** The same
recipe at `global_layers=1` recovered **0 bits** for both E18 and E21 while
dense solved it. Adding a second full global Block (not extra hops inside one
Attention, not extra SWA `stack_layers`) opens the DFA: E21 exclusive identity
copies **25.85 of 26 prize bits**. Uncompressed E18 with the same two raw
global layers is live but weak (5.56 bits) — dilution over the raw prefix, not
an E21 compression failure. 1024 SELECT still failed at `global_layers=2`; this
is hops-specific, not a SELECT rescue.

## Decision

Keep the spec in `ahead/`. Do **not** extra-step to 8k (S1 PASS). Do not 16k.
Do not hops seq shrink. Do not Glyph. Do not unfreeze `u`/`delta`. Do not
restore full raw prefix KV (that is E18). Code default `--global_layers` stays
1 (E18-loadable); hops hunts that need two reads pass `--global_layers 2`.
Next ONE (do not run): seq=**512** packed `chain_ordered --key_len 13`
`--global_layers 2` (same compressor; previously dense **K1** at one global
layer). Recalibrate dense S0 at two global blocks. `--hops 1` is illegal.

*Related: `master_experiment_log.md`, `docs/experiments_specs/ahead/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
