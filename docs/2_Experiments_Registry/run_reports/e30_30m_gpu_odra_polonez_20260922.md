# E30 ~31M GPU capacity hunt — Odra + Polonez

**Date:** 2026-09-22
**Machine:** Odra 3×RTX 3090 (seq=128) · Polonez 4×RTX 3090 (seq=256/512) · isolated E30 worktree @ `08e1978`
**Run IDs:** `e30_30m_gpu` (Byobu `E30_30m_g0..g3`, plus `E30_30m_twfix` / `twmatchfix` / `twe30m`)
**WandB:** n/a (BAPO probe; no `compute/*` audit)
**Raw JSON:** Odra `Cache/bapo_e30_30m/tiny_*` · Polonez `Cache/bapo_e30_30m/{tw_*,bridge_*}` · local `/opt/cursor/artifacts/e30_30m_gpu_{odra,polonez}/`
**Best checkpoint:** none (on-the-fly rows; models discarded)
**Git tag:** —
**Related:** [E30 spec](../../experiments_specs/ahead/E30_sliding_window_perceiver.md) · [small-model protocol](../../engineering_specs/small_model_capability_protocol.md) · CPU width hunt [e30_tiny_5m_9m_capability_20260921.md](e30_tiny_5m_9m_capability_20260921.md)

---

## Goal

Move the 31M (H=960, 4-layer) hunt off CPU onto idle Odra/Polonez GPUs in parallel.
Fair write is still concat frozen-mean e21 (`--message_identity_slots`) vs concat CA e30.
Seq=128 / 256 / 512, INDEX + MATCH; SELECT at seq=128; `--warm_residuals` from seq=256
when 1e-3 zero-init missed S0.

## Configuration

| knob | value |
|---|---|
| hidden / head_dim / stack | 960 / 64 / 2 (~31.00M dense · 31.12M e21 · 31.13M e30) |
| SWP | K auto-fit, 8 heads, qdim 128, coverage 8× |
| E21 | identity concat r=16, no inplace |
| amp | `auto` (bf16 autocast; unfused RMSNorm warning only) |
| batch | 32 · ~0.04–0.14 s/step vs ~1.3 s/step CPU |

LR used for the **scored** rows: 1e-3 at seq=128 for dense/e18/e21; **3e-4 for e30**
(1e-3 RankMe-collapsed e30 on CPU at this width). Seq=256/512 scored at **3e-4 +
`--warm_residuals`** after 1e-3 zero-init K1'd e18/e21 INDEX and dense MATCH at 256.

## Results (recovered bits)

Prize: INDEX 64b @128 / 48b @256 / 64b @512. MATCH 32b @128 / 48b @256 / 48b @512.
SELECT 32b @128.

| scale | recipe | dense | e18 | e21 mean | e30 SWP |
|---|---|---|---|---|---|
| tiny 128 | INDEX | 62.9 | 63.0 | 42.5 | **59.8** |
| tiny 128 | MATCH | 31.2 | 11.4 | 14.7 | **28.0** |
| tiny 128 | SELECT | 31.4 | 31.3 | 8.8 | **27.8** |
| tiny_wide 256 | INDEX (3e-4 warm for e18/e21; e30 1e-3) | 47.3 | 47.2 | 42.3 | **46.1** |
| tiny_wide 256 | MATCH (3e-4 warm) | 47.1 | 47.2 | 40.9 | **46.9** |
| bridge 512 | INDEX (3e-4 warm, right-align) | 64.0 | 64.0 | 61.4 | **63.1** |
| bridge 512 | MATCH (3e-4 warm) | 48.0 | 48.0 | 44.0 | **47.6** |

1e-3 zero-init at seq=256 is a **false kill**: e18/e21 INDEX 0 bits and dense MATCH 0.2
bits. Retry at 3e-4 + warm restored them. Do not score those 1e-3 rows.

Write diagnostics (e30): `none` always chance. Seq=128 MATCH RankMe 3.84,
entropy/log W 0.68. Seq=512 MATCH RankMe 2.63, entropy/log W **0.87** (S2 smear
threshold is 0.85 — borderline pick). Seq=512 INDEX n_windows=3, K=32, C=96.

## Gates vs E30 spec

| gate | seq=128 31M | seq=512 31M (claim length) |
|---|---|---|
| S0 dense ≥75% | PASS (INDEX 99%; MATCH 99%; SELECT 99%) | PASS (INDEX 100%; MATCH 100% @100 steps) |
| S1 MATCH ≥0.75× live E18, E21-mean wall | 28.0 vs 0.75×11.4=8.6 **pass**, but E21 is also live (14.7) | 47.6 vs 0.75×48.0=36 **pass**; **E21 is not dead** (44.0 bits) so the "means-die" clause does not fire |
| S2 entropy/log W <0.85 and `none` chance | MATCH 0.68 yes | MATCH 0.87 borderline; `none` chance |
| S3 INDEX ≥0.75× live E18 | 59.8 vs 47.3 pass | 63.1 vs 48.0 pass |
| K5 INDEX collapse vs e21 | e30 beats e21 | e30 63.1 > e21 61.4 |

Do not kill. At 31M the frozen-mean MATCH wall from E25 (<10M, seq=512, 0 bits) is
**gone**. E30 tracks uncompressed E18 on INDEX/MATCH at 256 and 512 and **beats e21
on SELECT at 128** (27.8 vs 8.8 vs e18 31.3). The distinctive "CA where means die"
claim needs a harder exam (SELECT/hops at 512, or the <10M seq=512 wall), not more
width.

## Protocol lessons

1. 31M E30 needs **3e-4** (1e-3 collapsed on CPU). Same LR + `--warm_residuals` from
   seq=256 at this width, or e18/e21 INDEX look dead.
2. GPU ~30× CPU (0.04 s/step seq=128). Split one probe per GPU; one LR per process.
3. Amp bf16 on 3090 works; RMSNorm falls back to unfused.

Odra GPUs idle after tiny. Polonez GPUs idle after bridge + tiny_wide retries.
