# E25 DNA S0-capable USER_CORE rungs mapped at stop resolution (rung 5cg)

**Date:** 2026-09-15
**Machine:** none (audit; no GPU hunt)
**Run ID:** `e25_dna_s0_ladder_mapped` (docs-only; no probe JSON)
**WandB:** n/a (no run; no `compute/*`)
**Raw JSON/plots:** none
**Git commit:** `67d0d0e` (audit base; docs SHA is this report's commit)
**Git tag:** —
**Related:** last hunt 1536 MATCH identity extra hop [`e25_bridge1k_1536_ip_id_h256_recall_extrahop_20260915.md`](e25_bridge1k_1536_ip_id_h256_recall_extrahop_20260915.md) · spec [`E25_e21_bapo_capability_ladder.md`](../../experiments_specs/done_success/E25_e21_bapo_capability_ladder.md)

---

## Goal

List `USER_CORE_TASKS` in `data/bapo_ladder.py` (`far_copy`, `recall`, `select`,
`chain_ordered`, `chain`) whose dense control can plausibly pass **S0** (dense ≥ 75%)
and that are still unmapped at stop resolution. If none remain, do **not** invent a
hunt. Extra hop is hops-composition only (shuffled `n_dist=1/2` PASS; MATCH
pooling/identity/MATCH2 length FAIL; ordered hops 288 FAIL).

Gates in this file: **S0** = dense ≥ 75% in that replica; **S1** = E21 recovered bits
≥ 0.75 × live E18 in the same JSON, or vs 0.75 × dense when E18 ≈ 0 (never via
`0.75 × 0`); **K1** = dense missed S0 (E21 not scored); **K2** = `e18_local` leak.

## Configuration

No probe. Audit of named recipes vs closed E25 reports on
`cursor/e21-capability-ladder-df1a` at `67d0d0e`. Architecture lock unchanged:
E18 = SWA + global read over raw prefix KV; E21 = `message_boundary_token_id`
(DNA query id 10 at A=4) severs SWA; global sees `KVCompressor` slots only.

## Capability map (USER_CORE)

| recipe | S0-capable cell | E21 wall | extra-hop transfer |
|---|---|---|---|
| `far_copy` INDEX r=16 mean | 512 near-pass / 1024 **S1 PASS** 53.82 vs 47.34; 1536+ dense S0, E18 ~0 | **(1024 PASS, 1536 FAIL]** shared with E18 | not run (E18 dead; INDEX 1280 hard-stopped) |
| `recall_single` MATCH identity | 512 / 1280 **S1 PASS** 63.94; 1536 dense+E18 live, E21 1.50 then extra-hop 0 | **(1280 PASS, 1536 FAIL]** even glob=2 | **FAIL** 0 vs live E18 63.94 |
| `recall_single` MATCH r=8 pool | 1024 **S1 PASS** 60.35; 1280 dense+E18 live, E21 0 | **(1024 PASS, 1280 FAIL]** | **FAIL** 0 vs live E18 63.99 |
| `recall_single` MATCH r at 1280 | r=1 identity PASS; r=4 rem-off 24.91@800 / 0@8k; rem-on leftover-2 45.31@1550 / 0@8k | **(r=1 PASS, r=4 FAIL]** | pooling extra hop at r=8 FAIL (composition does not transfer) |
| `recall` MATCH2 `n_dist=1` | 512 **S1 PASS** 20.78 vs 19.99; 1024 **S1 PASS** 61.81 vs 44.11 (E18 ~0); 1280 dense S0, E21 0 | **(1024 S1 PASS, 1280 FAIL]**; 1152 dense **K1** hole | **FAIL** 0 vs 0.75× dense 47.40 |
| `recall` MATCH2 `n_dist=2` | 512 dense 55.8% / 18.44 @3200 | **K1** (3-item not S0) | n/a (not S0) |
| `select_1decoy` length | 512 / 640 / 672 / 688 / 692 **S1 PASS**; 696 dense+E18 live, E21 0 even keepswa | **(692 PASS, 696 FAIL]** | 1024 extra hop FAIL; 696 extra hop hard-stopped |
| `select_1decoy` pooling @512 | r=8 rem-off **S1 PASS** 45.53; r=12 rem-on **S1 PASS** 43.29; r=16 rem-on live 84.5% / 35.94 vs 35.95 | r=16 is the SELECT S1 edge (0.016-bit miss) | not transferred (extra hop is hops-composition only) |
| `chain_ordered` hops glob=2 | 256 / 264 **S1 PASS**; 272 FAIL severed, PASS keepswa + extra hop @8k; 288 dense S0, E21 0; 320+ **K1** | dense **(288 S0 PASS, 320 K1]**; E21 severed **(264 S1 PASS, 272 FAIL]**; keepswa **(272 S1 PASS, 288 K1]**; extra-hop **(272 S1 PASS, 288 FAIL]** | **PASS** 272 / **FAIL** 288 |
| `chain` shuffled glob=2 | `n_dist=0` **S1 PASS** 16.60 vs 12.28; `n_dist=1` FAIL 6.28 (keepswa 14.77 miss); `n_dist=2` FAIL 0 | exclusive **(n_dist=0 S1 PASS, n_dist=1 S1 FAIL]** | **PASS** packed `n_dist=1` 24.08 and `n_dist=2` 22.60; **STOP** `n_dist=3` |

Unmapped USER_CORE recipes that can plausibly pass dense S0: **none**.

Not hunts: default `select` (tiny dense ~39% **K1**); Glyph; HARD_TASKS (`unique` /
`match3` / `count` / `majority`); hops 320; glob=3 / `update_slot_kv` / window 32 on
hops; SELECT 694; SELECT 696 extra hop; INDEX 1280; MATCH 1408/2048; MATCH r=2/r=6;
MATCH glob=2 stacks; MATCH r=8 extra hop (ran FAIL); MATCH 1536 identity extra hop
(ran FAIL); MATCH2 1280 extra hop (ran FAIL); MATCH2 length extra-steps; shuffled
`n_dist=3`; stack keepswa+extra hop; unfreeze `u`/`delta`; 16k; change code defaults.

## Extra-hop transfer (closed)

| wall | extra hop | verdict |
|---|---|---|
| ordered hops 272 | 1 | **S1 PASS** 25.48 vs 0.75× live E18 19.12 |
| ordered hops 288 | 1 | **S1 FAIL** 0 vs 0.75× dense 19.32 (E18 0) |
| shuffled `n_dist=1` | 1 | **S1 PASS** 24.08 vs 7.31 |
| shuffled `n_dist=2` | 1 | **S1 PASS** 22.60 vs 17.21 (E18 0; vs 0.75× dense) |
| MATCH2 1280 identity | 1 | **S1 FAIL** 0 vs 47.40 |
| MATCH 1280 r=8 pool | 1 | **S1 FAIL** 0 vs live E18 47.99 |
| MATCH 1536 identity | 1 | **S1 FAIL** 0 vs live E18 47.96 |
| SELECT 1024 identity | 1 | **S1 FAIL** 0 (prior; not this turn) |

Extra hop composes extra shuffled/ordered hop-edges. It does not rescue MATCH
pooling, MATCH identity capacity, MATCH2 length, or SELECT type-then-value.

## Decision

Parent later moved the spec/plan to `done_success/` after the original-objective
audit. Do **not** invent a hunt. Code defaults unchanged
(`--message_extra_slot_attends` stays 0; `--message_keep_local_swa` stays false;
`--global_layers` stays 1 except hops hunts already closed). Next: **STOP:
ladder mapped**.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_success/E25_e21_bapo_capability_ladder.md`, `agenda.md`*
