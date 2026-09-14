# E25 — E21 capability limits on a calibrated BAPO DNA ladder (one rung at a time)

- **Status:** active (tiny DNA limits; 512 concat 0 bits; raw/identity/raw-KV PASS;
  r=16 frozen mean NEAR-PASS @512; learned pool 0 bits; **1024 frozen mean S1 PASS**
  53.82 vs 47.34 bits; **512 recall frozen mean 0 bits**; **512 recall r=1 identity
  S1 PASS 43.08 vs 35.00**; **512 recall r=4 frozen mean S1 PASS 47.04 vs 35.97**;
  **512 recall r=8 frozen mean S1 PASS 47.16 vs 35.97**;
  **512 recall r=12 frozen mean S1 FAIL 34.33 vs 35.97** (climbing, remainder off);
  **512 recall r=10 frozen mean 0 bits** (chance floor @800 remainder off);
  **512 recall r=10 remainder-on S1 PASS 44.21 vs 35.85**;
  **512 recall r=12 remainder-on S1 PASS 46.06 vs 35.97**;
  **512 recall r=16 remainder-on S1 PASS 40.92 vs 35.89** (leftover/alignment was
  the MATCH wall through r=16; remainder-off r=16 was chance);
  **512 select r=16 remainder-on S1 FAIL 35.94 vs 35.95** (live 84.5% @8k, 0.016-bit
  miss; type-cue slightly harder than MATCH on the same pooler);
  **512 select r=12 remainder-on S1 PASS 43.29 vs 35.92** (r=16 is the SELECT S1 edge);
  **512 select r=8 remainder-off S1 PASS 45.53 vs 35.93** (default MATCH grid enough for SELECT);
  **1024 recall r=8 rem-off H=128 S1 FAIL 0.01 vs 46.98** (8k floor; 800 climb 13.33 did not replicate);
  **1024 recall r=8 rem-off H=256 SSMax log S1 PASS 60.35 vs 46.96** (MATCH at 1024 needs INDEX width, not a new pooler);
  **1024 select r=8 rem-off H=256 SSMax log S1 FAIL 0 bits** (chance @800; SELECT does not scale with MATCH);
  **1024 select r=1 identity H=256 SSMax log S1 FAIL 0 bits** (chance @800; exclusive SELECT dead at 1024 even with identity slots);
  **1024 select r=1 identity SWA-unsever H=256 log S1 FAIL 0.01 bits** (chance @800;
  `--message_keep_local_swa`; exclusive SELECT still dead);
  **1024 select r=1 identity extra exclusive hop H=256 log S1 FAIL 0 bits** (chance @800;
  `--message_extra_slot_attends 1`; two exclusive reads over frozen slots still dead);
  **1024 select r=1 identity type_marks anchors H=256 log S1 FAIL 31.90 vs 47.99**
  (0 bits @800 floor; late click 31.90 @1050; this-JSON E18 0; scored vs 0.75× dense;
  `--message_global_anchors type_marks`; extra non-slot count 0 at r=1 identity);
  **1024 select type_marks 8k extra-step S1 FAIL 6.48 vs 47.97** (live E18 **63.96**
  @3250; E21 0 @800/1050, climb @4850, plateau ~14.43 bits, final 6.48 @8000;
  did not pass 31.90; `--no-dense_first`);
  **1024 select extra hop + unfrozen slot K/V H=256 log S1 FAIL 0 bits**
  (chance @800; `--message_extra_slot_attends 1 --message_update_slot_kv`;
  extra hop is real updated-Q / frozen-KV, not a no-op; rewriting slot K/V
  between hops still dead);
  **1024 select r=1 identity second exclusive global layer `--global_layers 2`
  H=256 log S1 FAIL 0 bits** (chance @800; two full exclusive attend+FFN
  Blocks over slots, extra hops 0; not `stack_layers=2`; E18 live 63.97);
  **1024 select r=1 identity QUERY-side anchors `--message_global_anchors query_side`
  H=256 log S1 FAIL 0 bits** (chance @800; leak QUERY+2 asked-key symbols+ANSWER
  = 4 tokens vs seq=1024, extra non-slot count 4, not already r=1 prefix slots;
  this-JSON E18 live 46.77 late-click @800);
  **768 select r=1 identity H=256 log S1 FAIL 0.01 bits** (chance @800;
  `--scale bridge_1k --seq_len 768`; exclusive SELECT dead between 512 PASS
  and 768; E18 live 63.97);
  **640 select r=1 identity H=256 log S1 PASS 63.95 vs 47.93** (100% @450;
  `--scale bridge_1k --seq_len 640`; exclusive SELECT live at 640 on
  1024-passing width; E18 live 63.91);
  **704 select r=1 identity H=256 log S1 FAIL 0 bits** (chance @800;
  `--scale bridge_1k --seq_len 704`; exclusive SELECT dead between 640
  PASS and 704; wall was (640, 704]; E18 live 63.89);
  **672 select r=1 identity H=256 log S1 PASS 63.96 vs 47.95** (100%
  @500; `--scale bridge_1k --seq_len 672`; exclusive SELECT live at 672
  on 1024-passing width; E18 live 63.93; wall was (672, 704]);
  **688 select r=1 identity H=256 log S1 PASS 62.68 vs 47.97** (99.5%
  @400 extra-step; 800 climb 72.4% / 31.57 bits, best 88.9% @750;
  `--scale bridge_1k --seq_len 688`; exclusive SELECT live at 688 on
  1024-passing width; E18 live 63.96; wall was (688, 704]);
  **696 select r=1 identity H=256 log S1 FAIL 0 bits** (chance @800;
  `--scale bridge_1k --seq_len 696`; exclusive SELECT dead between 688
  PASS and 696; wall was (688, 696]; E18 live 63.93);
  **692 select r=1 identity H=256 log S1 PASS 62.76 vs 47.91** (99.8%
  @500; `--scale bridge_1k --seq_len 692`; exclusive SELECT live at 692
  on 1024-passing width; E18 live 63.89; wall now (692, 696]);
  **696 select r=1 identity `--message_pack_stride 32` H=256 log S1 FAIL
  31.67 vs 47.50 @800 / 0 bits @8k** (800 climb 49.7%; 8k chance floor;
  r=1 remainder is a no-op; leftover drop equalizes 692/696 to 640
  exclusive slots and does not rescue 696; E18 live 63.33/63.91);
  **696 select `--evidence_align spread` H=256 log K1** (dense 24.7% /
  0 bits @3200; e18/e21 skipped; QUERY 658 leftover 18 unchanged;
  seed-0 fact0 202 vs right 528; 8k not run);
  **696 select r=1 identity `--local_window 32` H=256 log S1 FAIL 0 bits**
  (chance @800; window 32 < gap 64; SWA-16 QUERY-align hypothesis; E18
  live 63.62);
  **4k INDEX r=16 rem-off H=256 log S1 FAIL 0 bits** (dense 63.94 S0 PASS; E18 ~0;
  scored vs 0.75× dense 47.96; chance @800);
  **2048 INDEX r=16 rem-off H=256 log S1 FAIL 0 bits** (dense 63.79 S0 PASS; E18 ~0;
  scored vs 0.75× dense 47.84; chance @800; `--scale bridge_1k --seq_len 2048`);
  **1536 INDEX r=16 rem-off H=256 log S1 FAIL 0 bits** (dense 62.41 S0 PASS; E18 ~0;
  scored vs 0.75× dense 46.81; chance @800; `--scale bridge_1k --seq_len 1536`;
  shared wall now **(1024 PASS, 1536 FAIL]**);
  **512 select r=1 identity S1 PASS 47.98 vs 35.94**; **512 chain H=128 K1** dense
  24.2%; **512 chain S0 hunt K1** H=256 log 32.4% / H=512 MHA 24.1%; **512 chain
  key_len=13 K1** dense 22.5%; **256 chain hops FAIL** dense 96.4% / E18 0 / E21
  0 (glob=1); **256 chain hops `--global_layers 2` S1 PASS** 25.85 vs 4.17
  (dense 25.46; E18 live 5.56 climbing; glob=1 was 0/0; 8k not run);
  **512 chain hops `--global_layers 2` K1** dense 22.5% / 0 bits @3200
  (2.802M; e18/e21 skipped; glob=1 was also K1; 8k not run). Keep in `ahead/`. Do not score Glyph yet.
- **Serves:** the Vision's "does the compressed channel carry *addressable* content, and how
  many bits?" question, now on **E21** (exclusive compressed read) rather than E18's raw
  one-global-read. Observation base for later gating / compression. Reuses E24's DNA instrument.
- **Implementation plan:** [E25_e21_bapo_capability_ladder_plan.md](E25_e21_bapo_capability_ladder_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-13
- **Engineering foundation:** [bapo_capability_ladder.md](../../engineering_specs/bapo_capability_ladder.md)
  · E21 spec on `cursor/strategy-sota-review-2026-09-e212`

> One experiment = one coherent claim about *what E21 can and cannot do* on a calibrated
> ladder, scored **one rung at a time**. Architecture changes only after a measured wall.
> Not a new training fork. Not a relabel of E24's E18 tables.

## Hypothesis
If the E18 platform is run as E21 — a **message boundary** at the DNA `query` token that severs
every sliding-window / n-gram path, and the global read sees the prefix **only as compressed
slots** (one learned slot per `r = 16` tokens, `KVCompressor`) — then on tiny packed
`far_copy` (seq=128, 64-bit prize), whose dense control already hit **99.4% / 63.3 bits** and
whose uncompressed E18 control already hit **99.2% / 63.0 bits**, E21 recovers
**≥ 0.75 × E18's recovered bits** (≥ 47 bits, information_flow ≥ 0.74)
**because** INDEX is a positional copy; mean-pool slots at init plus a trained pooling query
are enough to carry a marked span that already sits behind `min_gap`, and the E21 cut only
removes a local path `e18_local` already proved cannot use. **Kill the rung** (and change the
architecture before scoring recall or 512) if E21 stays at chance after 4× the dense step
budget while `e18_local` also stays at chance — then the slots are not a copy channel at this
scale, not a leak.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (E18 + E21 config: `message_boundary_token_id`,
  `message_compress_ratio`, `KVCompressor`), `evaluation/bapo_models.py` (`e21` arch),
  `data/symbolic_tasks.py` + `data/bapo_ladder.py`, `verification/bapo_capability_probe.py`.
  Shared causal-LM entrypoint unchanged.
- **Init / checkpoint:** random init, <100M, same tiny recipe as E24 (H=128, packed span 32).
  No pretrained warm-start. Compressor at init is mean-pool (`u = 0`, `delta = 0`).
- **Baseline to beat:** E24 tiny `far_copy` E18 **99.2% / 63.0 bits** (uncompressed one-read
  ceiling) and dense **99.4% / 63.3 bits** (solvability). `e18_local` **~25% / 0 bits** (K2).
- **Materially new:** the E21 *exclusive compressed read* on the same rows E18 already copies.
  E24 measured raw keys. This measures slots. Do not reuse E18 accuracy as an E21 score.

## The architectural bet
```
DNA row:  [BOS] filler [spanmark span] filler [QUERY] [ANSWER] y [END] [EOS]
                                                     ▲
                                          message_boundary_token_id = query

sender (t < QUERY):  E18 forward; slots = Pool_1(block of r=16 tokens), complete homogeneous blocks only
receiver (t ≥ QUERY):
   SWA / n-grams:  QUERY is a document start  →  local path cannot see the span
   global read:    raw keys of the receiver side  ∪  slots of earlier sides
                   (nothing from the prefix except slots)
```
`e18` on the same probe is arm U (`ratio` unused, no boundary). `e21` is the compressed exclusive
channel. Dense stays the solvability gate. `e18_local` stays the leak check.

**Out of scope for this first rung:** recall, select, chains, 512/1024/4k, Glyph, ratio
sweeps, SSMax, remainder-block pooling. Those are the *next* small experiments, each frozen
only after this result.

## Why this is not a safe retread
E18 already copies 64 bits at seq=128 with **raw** keys. E21 asks whether **~5–8 slots**
carry the same prize when the local bypass is structurally cut. That is the BAPO `(a,b)`
question E18 never posed: `b=0` on the prefix, `a` = slot KV. The E21 LM spec was a 32k
pretraining run; this is the cheap controlled exam that spec's K1/K2 needed and did not run.

## Success criteria (set BEFORE running)
- **S0 (instrument, already measured):** tiny packed `far_copy` dense ≥ 75%. Cite E24:
  99.4%. Re-run dense in the same probe so the JSON is self-contained; skip later arches if
  this replica misses 75% (K1).
- **S1 (this rung):** E21 `far_copy` information_flow ≥ 0.75 × E18 in the same run, and
  `e18_local` stays near chance (K2).
- **S2 (plots):** learning-curve / heatmap / recovered-bits plots exist for this rung.

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% in this replica — instrument broken; do not score E21.
- **K2:** `e18_local` > chance + 0.15 — QUERY/gap leak; stop and fix the generator.
- **K3:** E21 at chance (flow < 0.05) after `k1_mult ×` dense's finish step. Then slots are
  not an INDEX channel at r=16 / seq=128. **Do not** step to recall. Next experiment is one
  architecture change (remainder-block slot, or r that divides the prefix), not a knob sweep.

## Plan
- **Data:** on-the-fly DNA `far_copy`, `--scale tiny`, packed `span_len=32`. Boundary = existing
  `query` control id. No new generator.
- **Compute:** Cloud CPU for tiny (same machine as E24 tiny). GPU later rungs on Odra/Polonez.
- **Steps:** probe defaults (800, `k1_mult=4`, dense first, early-stop 99%).
- **Launch:** `uv run python verification/bapo_capability_probe.py --scale tiny --recipe far_copy --arch dense e18 e21 e18_local --out …`
- **New foundation code:** E21 `KVCompressor` + message mask on `nn/perceiver_ar_lm.py`
  (off by default; byte-identical to E18). Probe arch `e21`. No new `train_*.py`.

## Result
- Run id: `e25_tiny_far_copy` · `e25_tiny_far_copy_remainder` · `e25_tiny_far_copy_query_align` · `e25_tiny_far_copy_e21_steps` · `e25_tiny_recall_single` · `e25_tiny_recall_single_e21_steps` · `e25_tiny_select_1decoy` · `e25_tiny_chain_ordered` · `e25_tiny_chain_ordered_e21_steps` · `e25_512_fc` · `e25_512_r64` · `e25_512_r1` · `e25_512_raw` · `e25_512_ip_r1` · `e25_512_ip_rawkv` · `e25_512_ip_id` · `e25_512_ip_r16_mean` · `e25_512_ip_r16_mean_s3k` · `e25_512_ip_r16_mean_s8k` · `e25_512_ip_r16_learned` · `e25_1k_ip_r16_mean` · `e25_1k_ip_r16_mean_s8k` · `e25_512_ip_r16_mean_recall` · `e25_512_ip_id_recall` · `e25_512_ip_id_select` · `e25_512_ip_id_chain` · `e25_512_chain_s0` · `e25_512_chain_s0_h512` · `e25_512_chain_k13` · `e25_256_chain_k13` · `e25_512_ip_r4_mean_recall` · `e25_512_ip_r4_mean_recall_s8k` · `e25_512_ip_r8_mean_recall` · `e25_512_ip_r8_mean_recall_s8k` · `e25_512_ip_r12_mean_recall` · `e25_512_ip_r12_mean_recall_s8k` · `e25_512_ip_r10_mean_recall` · `e25_512_ip_r10_rem_recall` · `e25_512_ip_r10_rem_recall_s8k` · `e25_512_ip_r12_rem_recall` · `e25_512_ip_r12_rem_recall_s8k` · `e25_512_ip_r16_rem_recall` · `e25_512_ip_r16_rem_recall_s8k` · `e25_512_ip_r16_rem_select` · `e25_512_ip_r16_rem_select_s8k` · `e25_512_ip_r12_rem_select` · `e25_512_ip_r12_rem_select_s8k` · `e25_512_ip_r8_select` · `e25_512_ip_r8_select_s8k` · `e25_1k_ip_r8_recall` · `e25_1k_ip_r8_recall_s8k` · `e25_1k_ip_r8_h256_recall` · `e25_1k_ip_r8_h256_recall_s8k` · `e25_1k_ip_r8_h256_select` · `e25_1k_ip_id_h256_select` · `e25_1k_ip_id_keepswa_select` · `e25_1k_ip_id_extrahop_select` · `e25_1k_ip_id_anchors_select` · `e25_1k_ip_id_anchors_e18` · `e25_1k_ip_id_anchors_select_s8k` · `e25_1k_ip_id_updatekv_select` · `e25_1k_ip_id_glob2_select` · `e25_1k_ip_id_qside_select` · `e25_4k_ip_r16_mean` · `e25_2k_ip_r16_mean` · `e25_768_ip_id_h256_select` · `e25_640_ip_id_h256_select` · `e25_704_ip_id_h256_select` · `e25_672_ip_id_h256_select` · `e25_688_ip_id_h256_select` · `e25_688_ip_id_h256_select_s8k` · `e25_696_ip_id_h256_select` · `e25_692_ip_id_h256_select` · `e25_696_pack32_h256_select` · `e25_696_pack32_h256_select_s8k` · `e25_696_spread_ip_id_h256_select` · `e25_696_w32_ip_id_h256_select` · `e25_1536_ip_r16_mean` · `e25_256_chain_k13_glob2` · `e25_512_chain_k13_glob2`
- WandB: n/a (probe; no `compute/*`)
- Run reports: [`tiny`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md) · [`remainder`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md) · [`QUERY-align`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md) · [`INDEX extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_e21_steps_20260913.md) · [`recall`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_20260913.md) · [`recall extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_e21_steps_20260913.md) · [`select`](../../2_Experiments_Registry/run_reports/e25_tiny_select_1decoy_20260913.md) · [`chain`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_20260913.md) · [`chain extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_e21_steps_20260913.md) · [`512 r=16`](../../2_Experiments_Registry/run_reports/e25_bridge512_far_copy_20260913.md) · [`512 r=64`](../../2_Experiments_Registry/run_reports/e25_bridge512_r64_far_copy_20260913.md) · [`512 r=1`](../../2_Experiments_Registry/run_reports/e25_bridge512_r1_far_copy_20260913.md) · [`512 raw`](../../2_Experiments_Registry/run_reports/e25_bridge512_raw_far_copy_20260913.md) · [`512 in-place r=1`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r1_far_copy_20260913.md) · [`512 in-place raw KV`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_rawkv_far_copy_20260913.md) · [`512 in-place identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_far_copy_20260913.md) · [`512 in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_far_copy_20260914.md) · [`512 in-place r=16 learned`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_learned_far_copy_20260914.md) · [`1024 in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r16_mean_far_copy_20260914.md) · [`512 recall in-place r=16 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_mean_recall_20260914.md) · [`512 recall in-place r=1 identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_recall_20260914.md) · [`512 select in-place r=1 identity`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_select_20260914.md) · [`512 chain H=128 K1`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_id_chain_20260914.md) · [`512 chain S0 hunt`](../../2_Experiments_Registry/run_reports/e25_bridge512_chain_s0_20260914.md) · [`512 chain key_len=13`](../../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_20260914.md) · [`256 chain hops`](../../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_20260914.md) · [`512 recall r=4 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r4_mean_recall_20260914.md) · [`512 recall r=8 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_mean_recall_20260914.md) · [`512 recall r=12 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_mean_recall_20260914.md) · [`512 recall r=10 mean`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_mean_recall_20260914.md) · [`512 recall r=10 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r10_rem_recall_20260914.md) · [`512 recall r=12 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_recall_20260914.md) · [`512 recall r=16 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_recall_20260914.md) · [`512 select r=16 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r16_rem_select_20260914.md) · [`512 select r=12 remainder`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r12_rem_select_20260914.md) · [`512 select r=8 rem-off`](../../2_Experiments_Registry/run_reports/e25_bridge512_ip_r8_select_20260914.md) · [`1024 recall r=8 rem-off`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_recall_20260914.md) · [`1024 recall r=8 H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_recall_20260914.md) · [`1024 select r=8 H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_r8_h256_select_20260914.md) · [`1024 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_h256_select_20260914.md) · [`1024 select SWA-unsever`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_keepswa_select_20260914.md) · [`1024 select extra hop`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_extrahop_select_20260914.md) · [`1024 select type_marks anchors`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_20260914.md) · [`1024 select type_marks 8k`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_anchors_select_s8k_20260914.md) · [`1024 select extra hop + unfrozen KV`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_updatekv_select_20260914.md) · [`1024 select second global layer`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_glob2_select_20260914.md) · [`1024 select QUERY-side anchors`](../../2_Experiments_Registry/run_reports/e25_bridge1k_ip_id_qside_select_20260914.md) · [`4k INDEX r=16 H=256 log`](../../2_Experiments_Registry/run_reports/e25_medium_4k_ip_r16_mean_far_copy_20260914.md) · [`2048 INDEX r=16 H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_2k_ip_r16_mean_far_copy_20260914.md) · [`768 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_768_ip_id_h256_select_20260914.md) · [`640 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_640_ip_id_h256_select_20260914.md) · [`704 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_704_ip_id_h256_select_20260914.md) · [`672 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_672_ip_id_h256_select_20260914.md) · [`688 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_688_ip_id_h256_select_20260914.md) · [`696 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_696_ip_id_h256_select_20260914.md) · [`692 select r=1 identity H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_692_ip_id_h256_select_20260914.md) · [`696 select pack_stride 32`](../../2_Experiments_Registry/run_reports/e25_bridge1k_696_pack32_ip_id_h256_select_20260914.md) · [`696 select spread`](../../2_Experiments_Registry/run_reports/e25_bridge1k_696_spread_ip_id_h256_select_20260914.md) · [`696 select window 32`](../../2_Experiments_Registry/run_reports/e25_bridge1k_696_w32_ip_id_h256_select_20260914.md) · [`1536 INDEX r=16 H=256 log`](../../2_Experiments_Registry/run_reports/e25_bridge1k_1536_ip_r16_mean_far_copy_20260914.md) · [`256 chain hops glob2`](../../2_Experiments_Registry/run_reports/e25_bridge256_chain_k13_glob2_20260914.md) · [`512 chain hops glob2`](../../2_Experiments_Registry/run_reports/e25_bridge512_chain_k13_glob2_20260914.md)
- Verdict: **mixed.** Tiny INDEX near-pass at 8k. Tiny MATCH/select/chain walls. GPU seq=512
  exclusive concat slots **0 bits** (r=16/64/1). `--message_override raw` **PASSES** (100% /
  63.96 bits). r=1 in-place learned compressor **0 bits**. In-place raw KV **PASSES** (63.29
  bits). In-place **hard identity PASSES** (99.8% / 62.64 bits @750). In-place **r=16
  frozen mean-pool NEAR-PASS** (84.3% / 47.36 bits @8k vs 0.75× E18 47.95). Scatter is
  fine; 16-token means carry INDEX (slow, same shape as tiny). In-place **r=16 learned
  pool 0 bits** @800 (chance floor). Learning wrecks; keep frozen mean. Seq=1024 inplace
  r=16 frozen mean **S1 PASS** (91.2% / 53.82 bits @8k vs 0.75× E18 47.34; H=256 SSMax
  log). Seq=512 packed `recall_single` inplace r=16 frozen mean **0 bits** @800 (E18
  live 47.94; vs 0.75× E18 35.95 and vs 0.75× dense 35.23). Frozen means carry INDEX,
  not MATCH. Seq=512 packed `recall_single` inplace r=1 identity **S1 PASS** (91.6% /
  43.08 bits @800 vs 0.75× E18 35.00 and vs 0.75× dense 35.77). Exclusive identity
  binds a key; r=16 pooling is the MATCH killer. Seq=512 packed `select_1decoy`
  inplace r=1 identity **S1 PASS** (100% / 47.98 bits @700 vs 0.75× E18 35.94 and vs
  0.75× dense 35.85). Exclusive identity does type-cue select. Seq=512 packed `chain_ordered` H=128
  **K1** (dense 24.2% / 0 bits @3200; e18/e21 skipped). Seq=512 packed `chain_ordered`
  dense S0 hunt **K1** at H=256 SSMax log (32.4% / 1.10 bits) and H=512 full MHA
  (24.1% / 0 bits). Seq=512 `--key_len 13` (tiny 26-bit keys, hops=2) **K1**
  (dense 22.5% / 0 bits @3200). Seq=256 `--key_len 13` hops=2 **S0 PASS** (dense
  96.4% / 24.12 bits); E18 **0 bits**; E21 **0 bits** (floor). Do not pass S1 via
  0.75×0; vs 0.75× dense 18.09 **FAIL**. Dense hops wall is between 256 and 512.
  Seq=256 same recipe **`--global_layers 2` S1 PASS** (E21 **99.6% / 25.85 bits**
  @1650 vs 0.75× live E18 4.17 and vs 0.75× dense 19.10; E18 live **5.56 bits**
  climbing 42.3%; glob=1 was 0/0). 8k not run.
  Seq=512 packed `chain_ordered --key_len 13` hops=2 **`--global_layers 2`
  K1** (dense **22.5% / 0 bits** @3200; e18 / e21 / e18_local skipped; glob=1
  was also K1). Do not score E21. 8k not run.
  Seq=512 packed `recall_single` inplace r=4 frozen mean **S1 PASS** (99.3% /
  47.04 bits @1900 vs 0.75× E18 35.97 and vs 0.75× dense 35.97; climbing 29.76
  bits @800). MATCH survives 4-token means. Seq=512 packed `recall_single`
  inplace r=8 frozen mean **S1 PASS** (99.1% / 47.16 bits @7100 vs 0.75× E18
  35.97 and vs 0.75× dense 35.96; climbing 15.90 bits @800). MATCH survives
  8-token means; wall is between r=8 and r=16. Seq=512 packed `recall_single`
  inplace r=12 frozen mean remainder-off **S1 FAIL** (81.7% / 34.33 bits @8000 vs
  0.75× E18 35.97; climbing 5.79 bits @800, not chance). Seq=512 packed
  `recall_single` inplace r=10 frozen mean remainder-off **0 bits** @800 (chance
  every eval; CE at ln(4); vs 0.75× E18 35.96 and vs 0.75× dense 35.92).
  Unexpected vs r=12 live at 800. Pooling width is not monotone. Seq=512 packed
  `recall_single` inplace r=10 frozen mean remainder-on **S1 PASS** (95.1% /
  44.21 bits @8000 vs 0.75× E18 35.85 and vs 0.75× dense 35.91; climbing 8.65
  bits @800). Leftover 2 tokens were the r=10 floor, not the 10-token grid.
  Seq=512 packed `recall_single` inplace r=12 frozen mean remainder-on **S1
  PASS** (97.9% / 46.06 bits @8000 vs 0.75× E18 35.97 and vs 0.75× dense 35.90;
  climbing 6.14 bits @800). Remainder-off r=12 was 34.33 S1 FAIL. Leftover 8
  was the 1.64-bit miss, not the r=12 grid. Seq=512 packed `recall_single`
  inplace r=16 frozen mean remainder-on **S1 PASS** (92.6% / 40.92 bits @8000 vs
  0.75× E18 35.89 and vs 0.75× dense 35.96; climbing 4.72 bits @800; best 96.4%
  @7750). Remainder-off r=16 MATCH was 0 bits chance. Leftover/alignment was
  the MATCH wall through r=16. Seq=512 packed `select_1decoy` inplace r=16
  frozen mean remainder-on **S1 FAIL** (84.5% / 35.94 bits @8000 vs 0.75× E18
  35.95, miss 0.016; vs 0.75× dense 35.78 PASS; climbing 4.56 bits @800; best
  91.4% @7950). Live, not chance. Type-cue is slightly harder than MATCH on
  the same pooler. Seq=512 packed `select_1decoy` inplace r=12 frozen mean
  remainder-on **S1 PASS** (93.8% / 43.29 bits @8000 vs 0.75× E18 35.92 and vs
  0.75× dense 35.95; climbing 8.92 bits @800; best 98.0% @7900). SELECT clears
  at r=12 rem-on; r=16 is the SELECT S1 edge. Seq=512 packed `select_1decoy`
  inplace r=8 frozen mean remainder-off **S1 PASS** (96.4% / 45.53 bits @8000 vs
  0.75× E18 35.93 and vs 0.75× dense 35.94; climbing 11.73 bits @800; best
  97.9% @7450). SELECT works on the default MATCH r=8 grid; remainder-on is
  not required. Seq=1024 packed `recall_single` inplace r=8 frozen mean
  remainder-off H=128 **S1 FAIL** (26.8% / 0.01 bits @8000 vs 0.75× E18 46.98
  and vs 0.75× dense 47.79; 800 climb 13.33 bits did not replicate; E18 live
  62.64 in the 8k JSON). MATCH does not scale to 1024 on the 512 default
  H=128 recipe. Seq=1024 packed `recall_single` inplace r=8 frozen mean
  remainder-off **H=256 SSMax log S1 PASS** (96.5% / 60.35 bits @8000 vs 0.75×
  E18 46.96 and vs 0.75× dense 47.18; climbing 24.35 bits @3200; best 97.6%
  @7500). MATCH at 1024 needs the INDEX-passing width, not a new pooler.
  Seq=1024 packed `select_1decoy` inplace r=8 frozen mean remainder-off
  **H=256 SSMax log S1 FAIL** (25.7% / 0 bits @800 vs 0.75× E18 47.89 and vs
  0.75× dense 47.97; chance every eval). SELECT does not scale with MATCH at
  1024. Seq=1024 packed `select_1decoy` inplace **r=1 hard identity** remainder-off
  **H=256 SSMax log S1 FAIL** (25.7% / 0 bits @800 vs 0.75× E18 47.85 and vs
  0.75× dense 47.97; chance every eval). Exclusive SELECT is dead at 1024 even
  with identity slots. The 512 MATCH r=16 mean → r=1 identity split does not
  hold here. Next: **stop 1024 SELECT**. Seq=4096 packed `far_copy` inplace
  r=16 frozen mean remainder-off **H=256 SSMax log S1 FAIL** (25.7% / 0 bits
  @800 vs 0.75× dense 47.96; E18 ~0 live; chance every eval). Exclusive INDEX
  that passed at 1024 is chance at 4k. No 4k scale code change (`medium`).
  Seq=2048 packed `far_copy` `--scale bridge_1k --seq_len 2048` inplace r=16
  frozen mean remainder-off **H=256 SSMax log S1 FAIL** (25.0% / 0 bits @800
  vs 0.75× dense 47.84; E18 ~0 live; chance every eval). Exclusive INDEX that
  passed at 1024 is chance at 2048 with the same gap=64. No new scale enum.
  Seq=1536 packed `far_copy` `--scale bridge_1k --seq_len 1536` inplace r=16
  frozen mean remainder-off **H=256 SSMax log S1 FAIL** (24.8% / 0 bits @800
  vs 0.75× dense 46.81; E18 ~0 live; chance every eval). Shared INDEX wall
  is **(1024 PASS, 1536 FAIL]**. 8k not run. No new scale enum.
  Seq=1024 packed `select_1decoy` inplace r=1 identity **SWA-unsever**
  (`--message_keep_local_swa`) remainder-off **H=256 SSMax log S1 FAIL**
  (25.8% / **0.01 bits** @800 vs 0.75× E18 47.95 and vs 0.75× dense 47.27;
  chance every eval). Unsevering SWA does not bind type-cue at 1024
  (window 16 cannot reach gap 100). Exclusive identity global remains the
  E21-specific wall. Seq=1024 packed `select_1decoy` inplace r=1 identity
  **extra exclusive hop** (`--message_extra_slot_attends 1`) remainder-off
  **H=256 SSMax log S1 FAIL** (25.7% / **0 bits** @800 vs 0.75× E18 47.94
  and vs 0.75× dense 47.97; chance every eval). A second exclusive read
  over the same frozen slots does not rescue SELECT. Default extra hops
  stay 0. Seq=1024 packed `select_1decoy` inplace r=1 identity
  **type_marks anchors** (`--message_global_anchors type_marks`) remainder-off
  **H=256 SSMax log S1 FAIL** (25.7% / **0 bits** @800 floor; 48.1% /
  **31.90 bits** @1050 vs 0.75× dense 47.99; this-JSON E18 0 — do not pass
  via 0.75×0; vs 0.75× prior live E18 47.94). Type-mark leak is 2 tokens vs
  seq=1024; at r=1 identity those marks are already replace slots (extra
  non-slot count 0).   Late click after chance-through-950 is not an S1 pass.
  Default `--message_global_anchors` stays `none`.
  Seq=1024 packed `select_1decoy` type_marks **8k extra-step**
  (`--no-dense_first --arch e18 e21 e18_local --steps 8000 --k1_mult 1`)
  **H=256 SSMax log S1 FAIL** (39.3% / **6.48 bits** @8000 vs 0.75× live
  E18 **47.97**; live E18 **100% / 63.96 bits** @3250; E21 0 @800 and @1050;
  first climb @4850; best 14.43 bits @7100 / 42.9% @7850; did **not** climb
  past 31.90). Recalibrated E18 live in this JSON. Do not pass via 0.75×0.
  Seq=1024 packed `select_1decoy` inplace r=1 identity **extra hop + unfrozen
  slot K/V** (`--message_extra_slot_attends 1 --message_update_slot_kv`)
  remainder-off **H=256 SSMax log S1 FAIL** (25.7% / **0 bits** @800 vs 0.75×
  live E18 47.85 and vs 0.75× dense 47.96; chance every eval). Inspection:
  extra hop is two sequential attends with updated Q (QUERY Q can contain
  type) and frozen K/V — not a no-op. Rewriting exclusive slot K/V from the
  post-attend residual still does not rescue 1024 SELECT. Default
  `--message_update_slot_kv` stays off. Default extra hops stay 0.
  Seq=1024 packed `select_1decoy` inplace r=1 identity **second exclusive
  global layer** (`--global_layers 2`, extra hops 0, update_slot_kv off)
  remainder-off **H=256 SSMax log S1 FAIL** (25.7% / **0 bits** @800 vs 0.75×
  live E18 47.97 and vs 0.75× dense 47.96; chance every eval). Two full
  exclusive attend+FFN Blocks over slots — not extra hops inside one
  Attention and not extra SWA (`stack_layers` already 2). Does not rescue
  1024 SELECT. Default `--global_layers` stays 1.
  Seq=1024 packed `select_1decoy` inplace r=1 identity **QUERY-side anchors**
  (`--message_global_anchors query_side`) remainder-off **H=256 SSMax log
  S1 FAIL** (25.7% / **0 bits** @800 vs 0.75× this-JSON E18 35.07 and vs
  0.75× dense 47.97; chance every eval). Leak is QUERY + 2 asked-key
  symbols + ANSWER (**4 vs seq=1024**; extra non-slot count 4; not r=1
  prefix replace slots). Existing `query_nbhd` extra count is 0 at r=1.
  This-JSON E18 late-clicked 75.1% / 46.77 bits @800 (live). Default
`--message_global_anchors` stays `none`.
  Seq=768 packed `select_1decoy` `--scale bridge_1k --seq_len 768` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 FAIL** (25.6% / **0.01
  bits** @800 vs 0.75× this-JSON E18 **47.97** and vs 0.75× dense 47.96;
  chance every eval; E18 live **63.97** @350). Exclusive SELECT that
  passed at 512 is chance at 768 on the 1024-passing width. Do not extra-step
  (floor). No new scale enum.
  Seq=640 packed `select_1decoy` `--scale bridge_1k --seq_len 640` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 PASS** (100% / **63.95
  bits** @450 vs 0.75× this-JSON E18 **47.93** and vs 0.75× dense 47.82;
  chance through 350, click 51.3% @400; E18 live **63.91** @300). Exclusive
  SELECT that passed at 512 is live at 640 and chance at 768 on the
  1024-passing width. Do not extra-step (S1 PASS). No new scale enum.
  Seq=704 packed `select_1decoy` `--scale bridge_1k --seq_len 704` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 FAIL** (25.7% / **0 bits**
  @800 vs 0.75× this-JSON E18 **47.92** and vs 0.75× dense 47.96; chance
  every eval; E18 live **63.89** @350). Exclusive SELECT that passed at 640
  is chance at 704 on the 1024-passing width. Wall was **(640, 704]**. Do not
  extra-step (floor). No new scale enum.
  Seq=672 packed `select_1decoy` `--scale bridge_1k --seq_len 672` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 PASS** (100% / **63.96
  bits** @500 vs 0.75× this-JSON E18 **47.95** and vs 0.75× dense 47.82;
  chance through 400, click 47.6% @450; E18 live **63.93** @350). Exclusive
  SELECT that passed at 640 is live at 672 and chance at 704 on the
  1024-passing width. Wall was **(672, 704]**. Do not extra-step (S1 PASS).
  No new scale enum.
  Seq=688 packed `select_1decoy` `--scale bridge_1k --seq_len 688` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 PASS** (99.5% / **62.68
  bits** @400 extra-step vs 0.75× this-JSON E18 **47.97** and vs 0.75×
  dense 47.97; 800 climb 72.4% / 31.57 bits, best 88.9% @750; E18 live
  **63.96** @350). Exclusive SELECT that passed at 672 is live at 688 and
  chance at 704 on the 1024-passing width. Wall was **(688, 704]**. Do not
  16k (S1 PASS). No new scale enum.
  Seq=696 packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 FAIL** (25.0% / **0
  bits** @800 vs 0.75× this-JSON E18 **47.95** and vs 0.75× dense 47.96;
  chance every eval; E18 live **63.93** @350). Exclusive SELECT that
  passed at 688 is chance at 696 on the 1024-passing width. Wall was
  **(688, 696]**. Do not extra-step (floor). No new scale enum.
  Seq=692 packed `select_1decoy` `--scale bridge_1k --seq_len 692` inplace
  r=1 identity remainder-off **H=256 SSMax log S1 PASS** (99.8% / **62.76
  bits** @500 vs 0.75× this-JSON E18 **47.91** and vs 0.75× dense 47.98;
  chance through 450, click 99.8% @500; E18 live **63.89** @350). Exclusive
  SELECT that passed at 688 is live at 692 and chance at 696 on the
  1024-passing width. Wall now **(692, 696]**. Do not extra-step (S1 PASS).
  No new scale enum.
  Seq=696 packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
  r=1 identity **`--message_pack_stride 32`** remainder-off **H=256 SSMax
  log S1 FAIL** (49.7% / **31.67 bits** @800 vs 0.75× this-JSON E18
  **47.50**, climbing; **24.1% / 0 bits** @8000 chance floor vs 0.75×
  E18 **47.94**; E18 live **63.33** @550 / **63.91** @350). r=1
  remainder-on is a no-op. Leftover drop equalizes 692/696 exclusive
  slots to 640 and does not rescue 696. Do not 16k. No new scale enum.
  Seq=696 packed `select_1decoy` `--scale bridge_1k --seq_len 696`
  `--evidence_align spread` inplace r=1 identity remainder-off pack_stride
  0 **H=256 SSMax log K1** (dense **24.7% / 0 bits** @3200; e18/e21/
  e18_local skipped; JSON `calibrated: false`; hunt exit 2). QUERY 658
  leftover 18 unchanged vs right-align; seed-0 fact0 **202** vs right
  **528**; left-filler 201 vs 527. Spread turns near-copy S0 into
  find-the-mark (row gap 336/506/623 vs right 100). Do not extra-step
  (dense chance floor). Do not 16k. No new scale enum.
  Seq=696 packed `select_1decoy` `--scale bridge_1k --seq_len 696`
  `--evidence_align right` `--local_window 32` inplace r=1 identity
  remainder-off pack_stride 0 **H=256 SSMax log S1 FAIL** (25.0% /
  **0 bits** @800 vs 0.75× this-JSON E18 **47.71** and vs 0.75× dense
  47.90; chance every eval; E18 live **63.62** @600). Banner
  `gap=64 window=32` (32 < 64; row gap 100). Patterns `('swa', 32)`.
  QUERY 658 leftover 18 unchanged; 658%16=2 vs 654%16=14. Widening SWA
  does not rescue 696. Wall still **(692, 696]**. Do not extra-step
  (floor). Do not 16k. Default window stays 16. No new scale enum.

## Follow-up (rung 1b — remainder pooling)
Ran. S1 **FAIL** at 2400 (12.7 bits). Geometry stacking did not help.

## Follow-up (rung 1c — align QUERY to r)
Ran seq=132. S1 **FAIL** at 2950 (8.7 bits). Complete coverage ≠ the missing bits at short budget.

## Follow-up (rung 1d — extra steps)
Ran 8000 steps on seq=128 complete-block E21. **NEAR-PASS** (47.01 vs 47.29 bits). Channel is slow INDEX, not dead.

## Follow-up (rung 2 — tiny recall_single)
Ran at 2850 steps. Dense S0 **PASS**. E18 0 bits. E21 **3.19 bits** (not chance; not dense).

## Follow-up (rung 2b — extra steps on tiny recall)
Ran 8000 steps. E21 **9.81 bits / flow 0.307** (best acc 48.5% @7700) vs 0.75× dense 23.53 /
0.735. MATCH wall vs dense; still beats E18's 0 bits. Do not 16k.

## Follow-up (rung 3 — tiny select_1decoy)
Ran at 1900 steps. Dense S0 **PASS**. E18 **23.2 bits**. E21 **1.47 bits / flow 0.046**
(type-cue wall; K3 borderline). Do not extra-step this rung.

## Follow-up (rung 4 — tiny chain_ordered)
Ran at 3200 steps. Dense S0 **PASS** (93.1%). E18 **24.6 bits**. E21 **8.22 bits / flow 0.316**.
S1 fail; still climbing (INDEX-like).

## Follow-up (rung 4b — extra steps on tiny chain)
Ran 8000 steps. E21 **49.4% / 10.02 bits / flow 0.386** (best 51.7% @5450). Acc did not
rise vs 3200. Composition wall. Do not 16k.

## Follow-up (rung 5 — GPU seq=512 far_copy)
Ran on Odra GPU 0, E24 right-align recipe. Dense S0 **PASS** (99.5% / 61.2 bits). E18 **99.4% /
63.2 bits**. E21 **25.1% / 0 bits / flow 0** @800 (chance; not climbing). S1 **FAIL**. K2 **PASS**.
Do not extra-step (curve is a floor). Do not 1024.

## Follow-up (rung 5b — match tiny slot count)
Ran `--message_ratio 64` on Odra. Dense **100% / 63.3 bits**. E18 **100% / 63.9 bits**. E21
**25.1% / 0.01 bits** @800 (chance, not climbing). Eight slots do not rescue 512 INDEX.

## Follow-up (rung 5c — identity slots)
Ran `--message_ratio 1` on Odra. Dense **100% / 63.95 bits**. E18 **100% / 63.94 bits**. E21
**25.1% / 0.00 bits** @800 (chance, not climbing). Identity slots do not rescue 512 INDEX.
Compression is not the killer; the exclusive QUERY cut is.

## Follow-up (rung 5d — raw override)
Ran `--message_override raw` on Odra. Dense **100% / 63.94 bits**. E18 **99.6% / 62.23 bits**.
E21 raw **100% / 63.96 bits / flow 0.999** @450 (chance through 350, 77.8% @400). S1 **PASS**.
K2 **PASS**. Slot routing (concat extra KV) is the 512 killer, not document-start, not pooling.

## Follow-up (rung 5e — in-place slots)
Ran `--message_slots_inplace` + `--message_ratio 1` on Odra (`msg_override=real`). Dense
**100% / 63.95 bits**. E18 **100% / 63.93 bits**. E21 r=1 in-place **25.1% / 0.00 bits /
flow 0** @800 (chance, not climbing). S1 **FAIL**. K2 **PASS**. K3 **triggered**. KV_LEN=S
is not enough; do not stack r=16. Next: exclusive mask vs in-place values.

## Follow-up (rung 5f — mask vs values)
Ran `--message_inplace_raw_kv` + inplace r=1 on Odra (`msg_override=real`). Dense **100% /
63.95 bits**. E18 **100% / 63.90 bits**. E21 inplace raw KV **99.5% / 63.29 bits / flow
0.989** @450 (chance through 350, 81.9% @400). S1 **PASS**. K2 **PASS**. Compressor values
(even r=1) are the remaining 512 killer; exclusive `~replace` is not. Do not stack r=16.

## Follow-up (rung 5g — hard identity at r=1)
Ran `--message_identity_slots` + inplace r=1 on Odra (`msg_override=real`, no raw_kv).
Dense **100% / 63.92 bits**. E18 **100% / 63.94 bits**. E21 inplace identity **99.8% /
62.64 bits / flow 0.979** @750 (chance through 650, 56.6% @700). S1 **PASS**. K2 **PASS**.
Scatter is fine; learned `delta` wrecked claimed r=1 identity (one-token softmax ignores
`u`). Do not run r=16 in this turn.

## Follow-up (rung 5h — inplace r=16 frozen mean-pool)
Ran `--message_identity_slots` + inplace r=16 on Odra (`msg_override=real`, no raw_kv).
Dense **100% / 63.91 bits**. E18 **100% / 63.94 bits**. E21 frozen mean **84.3% /
47.36 bits / flow 0.740** @8000 (42.4% / 8.18 bits @800, climbing; 48.0% / 22.98 @3200;
best 88.7% @7900). S1 **NEAR-PASS** (47.36 vs 47.95). K2 **PASS**. 16-token means carry
INDEX at 512; not chance, not a lost span. Do not 16k. Next: inplace r=16 learned pool
(unfreeze `u`/`delta`), no identity flag, no raw_kv. Not 1024. Not Glyph.

## Follow-up (rung 5i — inplace r=16 learned pool)
Ran inplace r=16 **without** `--message_identity_slots` on Odra (`msg_override=real`,
no raw_kv). Dense **100% / 63.94 bits**. E18 **100% / 63.94 bits**. E21 learned pool
**25.1% / 0.00 bits / flow 0** @800 (chance every eval; CE at floor). S1 **FAIL**. K2
**PASS**. Do not extra-step (floor; frozen mean was already climbing at 800). Learning
wrecks r=16 the way `delta` wrecked r=1. Keep frozen mean. Next: seq=1024 inplace r=16
frozen mean. Not Glyph. Not remainder.

## Follow-up (rung 5j — seq=1024 inplace r=16 frozen mean)
Ran `--scale bridge_1k` H=256 `--global_logit_scale log` + inplace identity r=16 on Odra.
Dense **100% / 63.95 bits**. E18 **99.7% / 63.12 bits** @2750 (live; not dilution). E21
frozen mean **91.2% / 53.82 bits / flow 0.841** @8000 (68.8% / 32.37 @3200, climbing;
best 92.7% @7850). S1 **PASS** (53.82 vs 47.34). K2 **PASS**. Frozen means carry INDEX
at 1024. Do not 16k. Next: 512 `recall_single` frozen mean. Not 4k. Not Glyph.

## Follow-up (rung 5k — seq=512 recall_single inplace r=16 frozen mean)
Ran `--scale bridge` packed `recall_single` H=128 + inplace identity r=16 on Odra.
Dense **99.0% / 46.97 bits**. E18 **100% / 47.94 bits** @400 (live in this JSON; not
E24's 0-bit 512-recall control). E21 frozen mean **25.5% / 0.00 bits / flow 0** @800
(chance every eval; CE at floor). S1 **FAIL** (0 vs 35.95). Content vs 0.75× dense
**FAIL** (0 vs 35.23). K2 **PASS**. Do not extra-step (floor). Frozen means carry
INDEX, not MATCH, at 512. Next: 512 recall inplace r=1 identity. Not 4k. Not Glyph.

## Follow-up (rung 5l — seq=512 recall_single inplace r=1 hard identity)
Ran `--scale bridge` packed `recall_single` H=128 + inplace identity **r=1** on Odra
(no raw_kv). Dense **99.9% / 47.69 bits**. E18 **99.4% / 46.67 bits** @350 (live).
E21 identity **91.6% / 43.08 bits / flow 0.897** @800 (chance through ~350; best
92.1% @750). S1 **PASS** (43.08 vs 35.00). Content vs 0.75× dense **PASS** (43.08 vs
35.77). K2 **PASS**. Do not 8k (S1 already PASS). Exclusive identity binds a key;
r=16 pooling is the MATCH killer. Next: 512 `select_1decoy` r=1 identity. Not 4k.
Not Glyph.

## Follow-up (rung 5m — seq=512 select_1decoy inplace r=1 hard identity)
Ran `--scale bridge` packed `select_1decoy` H=128 + inplace identity **r=1** on Odra
(no raw_kv). Dense **99.9% / 47.80 bits**. E18 **100% / 47.92 bits** @400 (live).
E21 identity **100% / 47.98 bits / flow 1.000** @700 (chance through ~500; early
stop). S1 **PASS** (47.98 vs 35.94). Content vs 0.75× dense **PASS** (47.98 vs
35.85). K2 **PASS**. Do not 8k. Exclusive identity does type-cue select, not only
MATCH/INDEX. Next: 512 `chain_ordered` r=1 identity. Not 4k. Not Glyph.

## Follow-up (rung 5n — seq=512 chain_ordered inplace r=1 identity, H=128)
Ran `--scale bridge` packed `chain_ordered` H=128 + inplace identity **r=1** on Odra.
Dense **24.2% / 0 bits** @3200 (chance every eval; CE at ln(4)). **K1.** e18 / e21 /
e18_local **skipped**. Do not score S1. Packed 512 chain is not dense-solvable at
this width. Next: 512 chain **dense S0 hunt** H=256 `--global_logit_scale log`.
Not 4k. Not Glyph. Do not score E21 until S0.

## Follow-up (rung 5o — seq=512 chain_ordered dense S0 hunt)
Ran packed `chain_ordered` dense-first on Odra. H=256 `--global_logit_scale log`:
dense **32.4% / 1.10 bits** @3200 (**K1**). One bump `--hidden 512 --kv_heads 0`:
dense **24.1% / 0 bits** @3200 (**K1**). e18 / e21 skipped both times. Width is
not the lever. Next: 512 chain **`--key_len 13`** (tiny packed 26-bit keys) H=256
log, dense-first. Not 4k. Not Glyph. Do not invent a third width. Do not score
E21 until S0.

## Follow-up (rung 5p — seq=512 chain_ordered --key_len 13)
Ran packed `chain_ordered --key_len 13` (tiny 26-bit keys, hops=2) H=256
`--global_logit_scale log` dense-first on Odra. Dense **22.5% / 0 bits** @3200
(**K1**; chance floor). e18 / e21 skipped. Tiny keys still fail at seq=512 —
context, not 24-token packing. Next: seq=**256** chain `--key_len 13` hops=2
(H=256 log). `--hops 1` is illegal (`chain needs hops >= 2`). Not 4k. Not Glyph.
Do not bump width. Do not score E21 until S0.

## Follow-up (rung 5q — seq=256 chain_ordered --key_len 13)
Ran `--scale bridge --seq_len 256` (min_gap 64 > window 16; not tiny_wide)
`--key_len 13` hops=2 H=256 `--global_logit_scale log` on Odra. Dense **96.4% /
24.12 bits** @3200 (**S0 PASS**). E18 **26.2% / 0 bits**. E21 identity **24.9% /
0 bits** (chance floor). K2 **PASS**. Do not extra-step (floor). Do not pass S1
via 0.75×0; vs 0.75× dense 18.09 **FAIL**. Dense hops wall is between 256 and
512. Next: stop hops extra-steps. Not seq=192. Not 4k. Not Glyph.

## Follow-up (rung 5r — seq=512 recall_single r=4 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 4`
(no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra. Dense **100% / 47.96
bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21 **78.3% / 29.76 bits** @800
(climbing) then **99.3% / 47.04 bits** @1900 (**S1 PASS** vs 0.75× E18 35.97 and
vs 0.75× dense 35.97). K2 **PASS**. MATCH survives 4-token means (r=16 was 0 bits;
r=1 was 43.08). Next: 512 recall **r=8 frozen mean**. Not hops. Not 4k. Not Glyph.
Do not unfreeze `u`/`delta`.

## Follow-up (rung 5s — seq=512 recall_single r=8 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 8`
(no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra. Dense **100% / 47.95
bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21 **56.5% / 15.90 bits** @800
(climbing) then **99.1% / 47.16 bits** @7100 (**S1 PASS** vs 0.75× E18 35.97 and
vs 0.75× dense 35.96). K2 **PASS**. MATCH survives 8-token means; wall is
between r=8 and r=16. Next: 512 recall **r=12 frozen mean**. Not hops. Not 4k.
Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5t — seq=512 recall_single r=12 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 12`
(remainder **off**, no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra.
Dense **100% / 47.91 bits** (**S0 PASS**). E18 **100% / 47.96 bits**. E21
**39.8% / 5.79 bits** @800 (climbing) then **81.7% / 34.33 bits** @8000 (**S1
FAIL** vs 0.75× E18 35.97 and vs 0.75× dense 35.93). K2 **PASS**. Not chance.
S1-passing MATCH dies between r=8 and r=12 at the 8k budget. Next: 512 recall
**r=10 frozen mean**. Do not 16k. Not hops. Not 4k. Not Glyph. Do not remainder.
Do not unfreeze `u`/`delta`.

## Follow-up (rung 5u — seq=512 recall_single r=10 frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 10`
(remainder **off**, no raw_kv, `u`/`delta` frozen) H=128 dense-first on Odra.
Dense **99.9% / 47.89 bits** (**S0 PASS**). E18 **100% / 47.94 bits**. E21
**25.5% / 0.00 bits** @800 (**S1 FAIL** vs 0.75× E18 35.96 and vs 0.75× dense
35.92). K2 **PASS**. Chance every eval; CE at ln(4). Do not extra-step (floor).
Unexpected vs r=12 live at 800. Pooling width is not monotone. Stop stacking
MATCH r-sweeps. Do not r=9. Do not r=11. Do not remainder-on. Do not 16k.
Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5v — seq=512 recall_single r=10 remainder-on frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 10`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **99.9% / 47.88 bits** (**S0 PASS**). E18 **100% / 47.79 bits**.
E21 **44.2% / 8.65 bits** @800 (climbing) then **95.1% / 44.21 bits** @8000
(best 96.4% @7800; **S1 PASS** vs 0.75× E18 35.85 and vs 0.75× dense 35.91).
K2 **PASS**. Remainder-off r=10 was 0 bits at 800. Leftover 2 tokens were the
floor, not the r=10 grid. Do not 16k. Next: 512 recall **r=12 remainder-on**.
Not r=9/11. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5w — seq=512 recall_single r=12 remainder-on frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 12`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **100% / 47.87 bits** (**S0 PASS**). E18 **100% / 47.96 bits**.
E21 **41.0% / 6.14 bits** @800 (climbing) then **97.9% / 46.06 bits** @8000
(best 98.2% @7850; **S1 PASS** vs 0.75× E18 35.97 and vs 0.75× dense 35.90).
K2 **PASS**. Remainder-off r=12 was 34.33 S1 FAIL. Leftover 8 was the 1.64-bit
miss, not the r=12 grid. Do not 16k. Next: 512 recall **r=16 remainder-on**.
Not r=9/11. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5x — seq=512 recall_single r=16 remainder-on frozen mean)
Ran packed `recall_single` inplace `--message_identity_slots --message_ratio 16`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **100% / 47.95 bits** (**S0 PASS**). E18 **100% / 47.86 bits**.
E21 **36.4% / 4.72 bits** @800 (climbing) then **92.6% / 40.92 bits** @8000
(best 96.4% @7750; **S1 PASS** vs 0.75× E18 35.89 and vs 0.75× dense 35.96).
K2 **PASS**. Remainder-off r=16 was 0 bits chance. Leftover/alignment was the
MATCH wall through r=16. Do not 16k. Next: 512 **select_1decoy r=16 remainder-on**.
Not r=9/11. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5y — seq=512 select_1decoy r=16 remainder-on frozen mean)
Ran packed `select_1decoy` inplace `--message_identity_slots --message_ratio 16`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **99.8% / 47.71 bits** (**S0 PASS**). E18 **100% / 47.94 bits**.
E21 **35.6% / 4.56 bits** @800 (climbing) then **84.5% / 35.94 bits** @8000
(best 91.4% @7950; **S1 FAIL** vs 0.75× E18 35.95 by 0.016 bits; **PASS** vs
0.75× dense 35.78). K2 **PASS**. Live, not chance. Type-cue slightly harder
than MATCH (40.92 PASS) on the same pooler. Do not 16k. Next: 512
**select_1decoy r=12 remainder-on**. Not r=9/11. Not hops. Not 4k. Not Glyph.
Do not unfreeze `u`/`delta`.

## Follow-up (rung 5z — seq=512 select_1decoy r=12 remainder-on frozen mean)
Ran packed `select_1decoy` inplace `--message_identity_slots --message_ratio 12`
`--message_pool_remainder` (no raw_kv, `u`/`delta` frozen) H=128 dense-first on
Odra. Dense **100% / 47.93 bits** (**S0 PASS**). E18 **100% / 47.90 bits**.
E21 **45.1% / 8.92 bits** @800 (climbing) then **93.8% / 43.29 bits** @8000
(best 98.0% @7900; **S1 PASS** vs 0.75× E18 35.92 and vs 0.75× dense 35.95).
K2 **PASS**. SELECT clears at r=12 rem-on; r=16 is the SELECT S1 edge. Do not
16k. Next: 512 **select_1decoy r=8 remainder-off**. Not r=9/11. Not hops. Not
4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5aa — seq=512 select_1decoy r=8 remainder-off frozen mean)
Ran packed `select_1decoy` inplace `--message_identity_slots --message_ratio 8`
(no `--message_pool_remainder`, no raw_kv, `u`/`delta` frozen) H=128 dense-first
on Odra. Dense **100% / 47.91 bits** (**S0 PASS**). E18 **100% / 47.90 bits**.
E21 **43.3% / 11.73 bits** @800 (climbing) then **96.4% / 45.53 bits** @8000
(best 97.9% @7450; **S1 PASS** vs 0.75× E18 35.93 and vs 0.75× dense 35.94).
K2 **PASS**. SELECT works on the default MATCH r=8 grid; remainder-on is not
required. Do not 16k. Next: 1024 **recall_single r=8 rem-off** (scale MATCH).
Not another SELECT r. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ab — seq=1024 recall_single r=8 remainder-off frozen mean H=128)
Ran packed `recall_single` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 8` (no `--message_pool_remainder`, no raw_kv, `u`/`delta` frozen)
**H=128 `logit_scale=none`** dense-first on Odra. Dense **100% / 63.73 bits**
(**S0 PASS**, not K1). E18 **99.5% / 62.64 bits** @500 (live in 8k JSON). E21
**43.1% / 13.33 bits** @800 (climbing) then **26.8% / 0.01 bits** @8000
(**S1 FAIL** vs 0.75× E18 46.98 and vs 0.75× dense 47.79). K2 **PASS**. 8k is
a floor; 800 climb did not replicate. MATCH does not survive 1024 on the 512
default H=128 recipe. Do not 16k. Do not bump H=256 this turn (dense not K1).
Next: 1024 **recall_single r=8 rem-off H=256 `--global_logit_scale log`**. Not
another SELECT r. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ac — seq=1024 recall_single r=8 remainder-off H=256 SSMax log)
Ran packed `recall_single` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 8` (no `--message_pool_remainder`, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Dense **99.4% / 62.90
bits** (**S0 PASS**). E18 **100% / 62.61 bits** @300 (live). E21 **38.1%** @800
then **49.0% / 24.35 bits** @3200 (climbing) then **96.5% / 60.35 bits** @8000
(best 97.6% @7500; **S1 PASS** vs 0.75× E18 46.96 and vs 0.75× dense 47.18).
K2 **PASS**. MATCH at 1024 needs INDEX-passing width, not a new pooler. Do not
16k. Next: 1024 **select_1decoy r=8 rem-off H=256 SSMax log**. Not hops. Not
remainder-on. Not r=1 identity. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ad — seq=1024 select_1decoy r=8 remainder-off H=256 SSMax log)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 8` (no `--message_pool_remainder`, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Dense **100% / 63.96
bits** (**S0 PASS**). E18 **100% / 63.86 bits** @350 (live). E21 **25.7% / 0
bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18 47.89
and vs 0.75× dense 47.97). K2 **PASS**. SELECT does not scale with MATCH at
1024. Do not extra-step (floor). Do not 16k. Next: 1024 **select_1decoy r=1
identity H=256 SSMax log** (pooling vs exclusive channel). Not remainder-on.
Not extra width. Not hops. Not 4k. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ae — seq=1024 select_1decoy r=1 identity H=256 SSMax log)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace` (no `--message_pool_remainder`, no
raw_kv, `u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Dense **100% / 63.96 bits** (**S0 PASS**). E18 **100% / 63.80 bits** @350
(live). E21 **25.7% / 0 bits** @800 (chance every eval; CE at ln(4); **S1 FAIL**
vs 0.75× E18 47.85 and vs 0.75× dense 47.97). K2 **PASS**. Exclusive SELECT is
dead at 1024 even with identity slots. r=8 means are not uniquely the killer.
Do not extra-step (floor). Do not 16k. Next: **stop 1024 SELECT** (no
remainder-on, no extra width, no 8k a chance floor). Remaining walls: 512 chain
K1, 256 hops FAIL. Not hops. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5af — seq=4096 far_copy r=16 rem-off H=256 SSMax log)
Ran packed `far_copy` `--scale medium` inplace `--message_identity_slots
--message_ratio 16 --message_slots_inplace` (no `--message_pool_remainder`, no
raw_kv, `u`/`delta` frozen) **H=256 `--global_logit_scale log` batch 8**
dense-first on Odra. Dense **100% / 63.94 bits** @400 (**S0 PASS**; H=512
unused). E18 **25.7% / 0.00 bits** @800 (live ~0). E21 **25.7% / 0 bits**
@800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× dense 47.96).
Do **not** pass S1 via 0.75×0. K2 **PASS**. Exclusive INDEX that passed at
1024 is chance at 4k. No 4k scale code change (`medium` already exists).
Do not extra-step (floor). Do not remainder-on. Do not 4k H=512 (dense S0
already passed). Next: seq=**2048** packed `far_copy` same compressor (localize
the INDEX wall). Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ag — seq=2048 far_copy r=16 rem-off H=256 SSMax log)
Ran packed `far_copy` `--scale bridge_1k --seq_len 2048` inplace
`--message_identity_slots --message_ratio 16 --message_slots_inplace` (no
`--message_pool_remainder`, no raw_kv, `u`/`delta` frozen) **H=256
`--global_logit_scale log` batch 32** dense-first on Odra. Dense **100% /
63.79 bits** @250 (**S0 PASS**). E18 **23.9% / 0.00 bits** @800 (live ~0).
E21 **25.0% / 0 bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs
0.75× dense 47.84). Do **not** pass S1 via 0.75×0. K2 **PASS**. Exclusive
INDEX that passed at 1024 is chance at 2048 with the same gap=64 / window=16.
No new scale enum (`--seq_len` override). Do not extra-step (floor). Do not
remainder-on. Do not 2048 H=512 (dense S0 already passed). Next: **stop INDEX
length extra-steps** (do not seq=1536). Then **one-knob SWA-unsever** on the
measured 1024 SELECT identity floor. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ah — seq=1024 select SWA-unsever, identity slots)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_keep_local_swa` (no
`--message_pool_remainder`, no raw_kv, `u`/`delta` frozen) **H=256
`--global_logit_scale log`** dense-first on Odra. Dense **100% / 63.03 bits**
@200 (**S0 PASS**). E18 **100% / 63.94 bits** @400 (live). E21 **25.8% / 0.01
bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18 47.95
and vs 0.75× dense 47.27). K2 **PASS**. Inspection: r=1 identity already
exposes every sender token KV to the exclusive global read (not a coverage
hole). Unsevering SWA does not rescue 1024 SELECT. Do not extra-step (floor).
Do not restore raw global KV (that is E18). Do not remainder-on. Do not
H=512. Next: **one extra exclusive hop over frozen slots** (not DNA `--hops`).
Not Glyph. Do not unfreeze `u`/`delta`. Default `--message_keep_local_swa`
stays off.

## Follow-up (rung 5ai — seq=1024 select extra exclusive hop, identity slots)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_extra_slot_attends 1` (no
`--message_pool_remainder`, no `--message_keep_local_swa`, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on Odra.
Dense **100% / 63.95 bits** @200 (**S0 PASS**). E18 **100% / 63.92 bits**
@350 (live). E21 **25.7% / 0 bits** @800 (chance every eval; CE at ln(4);
**S1 FAIL** vs 0.75× E18 47.94 and vs 0.75× dense 47.97). K2 **PASS**.
Inspection: slots already post-pre-SWA; type-cue tokens already in r=1
identity slots; extra hop re-reads the same exclusive frozen K/V (not raw
prefix). Two exclusive reads do not rescue 1024 SELECT. Do not extra-step
(floor). Do not restore raw global KV (that is E18). Do not remainder-on.
Do not H=512. Next: **sparse exclusive-plus-anchors** leak of a few non-slot
tokens (QUERY neighborhood or type markers only), still not full raw prefix;
min config flag default OFF. Not hops seq shrink. Not Glyph. Do not unfreeze
`u`/`delta`. Default `--message_extra_slot_attends` stays 0. Default
`--message_keep_local_swa` stays off.

## Follow-up (rung 5aj — seq=1024 select type_marks anchors, identity slots)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_global_anchors type_marks`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0, no
raw_kv, `u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Dense **100% / 63.98 bits** @1050 (**S0 PASS**; plateau ~88% then late
click). E18 **24.9% / 0 bits** @1050 (this replica missed; e18-only 800 replica
also 0). E21 **25.7% / 0 bits** @800 (chance floor) then **48.1% / 31.90 bits**
@1050 (click @1000 after chance-through-950; **S1 FAIL** vs 0.75× dense 47.99
and vs 0.75× prior live E18 47.94). Do **not** pass S1 via 0.75×0. K2 **PASS**.
Type-mark leak is **2 tokens vs seq=1024**; at r=1 identity those marks are
already replace slots (extra non-slot count 0). Sparse exclusive-plus-anchors
does not clear 1024 SELECT S1. Do not extra-step this turn (800-floor chance).
Do not restore raw global KV (that is E18). Do not remainder-on. Do not H=512.
Next: **extra-step this late click to 8k** (include e18). Not hops seq shrink.
Not Glyph. Do not unfreeze `u`/`delta`. Default `--message_global_anchors`
stays `none`. Default extra hops stay 0. Default `--message_keep_local_swa`
stays off.

## Follow-up (rung 5ak — seq=1024 select type_marks 8k extra-step, live E18)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_global_anchors type_marks`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0, no
raw_kv, `u`/`delta` frozen) **H=256 `--global_logit_scale log`**
`--arch e18 e21 e18_local --steps 8000 --no-dense_first --k1_mult 1` on Odra.
Dense skipped (prior S0 **100% / 63.98 bits**). E18 **100% / 63.96 bits**
@3250 (**live**; this-JSON E18 at 1050 was 0). E21 **25.7% / 0 bits** @800
and @1050 (prior 31.90 @1050 did not replicate) then climb @4850, plateau
~42% / **14.43 bits** @7100 (best acc 42.9% @7850), final **39.3% / 6.48
bits** @8000 (**S1 FAIL** vs 0.75× live E18 47.97 and vs 0.75× dense 47.99).
Did **not** climb past 31.90. K2 **PASS**. Type-mark leak remains a no-op at
r=1 identity. Do **not** 16k (plateau 6050–8000). Do not restore raw global
KV (that is E18). Do not remainder-on. Do not H=512. Next: **stop 1024
SELECT extra-steps**. Remaining DNA walls: 512 chain **K1**, 256 hops
**FAIL**, 2048+ INDEX shared with E18. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Default `--message_global_anchors` stays `none`.
Default extra hops stay 0. Default `--message_keep_local_swa` stays off.

## Follow-up (rung 5al — seq=1024 select extra hop + unfrozen slot K/V)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_extra_slot_attends 1
--message_update_slot_kv` (no `--message_pool_remainder`, no
`--message_keep_local_swa`, anchors none, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Inspection: extra
hop is two sequential attends with updated Q and frozen K/V (QUERY Q can
contain type); not a no-op. This knob rewrites exclusive slot K/V from the
post-attend residual before hop 2. Dense **100% / 63.95 bits** @250 (**S0
PASS**). E18 **100% / 63.80 bits** @450 (live). E21 **25.7% / 0 bits** @800
(chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18 47.85 and vs
0.75× dense 47.96). K2 **PASS**. Unfreezing slot K/V between hops does not
rescue 1024 SELECT. Do not extra-step (floor). Do not restore raw global KV
(that is E18). Do not remainder-on. Do not H=512. Next: **stop 1024 SELECT
architecture hunts**. Remaining DNA walls: 512 chain **K1**, 256 hops
**FAIL**, 2048+ INDEX shared with E18. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Default `--message_update_slot_kv` stays off.
Default `--message_extra_slot_attends` stays 0. Default
`--message_keep_local_swa` stays off. Default `--message_global_anchors`
stays `none`.

## Follow-up (rung 5am — seq=1024 select second exclusive global layer)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --global_layers 2` (no
`--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0, no
`--message_update_slot_kv`, anchors none, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Existing probe flag
(default 1, E18-loadable) — two sequential exclusive attend+FFN Blocks, not
`--stack_layers` (already 2 / SWA) and not `--message_extra_slot_attends`.
Dense **100% / 63.95 bits** @200 (**S0 PASS**). E18 **100% / 63.97 bits**
@750 (live). E21 **25.7% / 0 bits** @800 (chance every eval; CE at ln(4);
**S1 FAIL** vs 0.75× E18 47.97 and vs 0.75× dense 47.96). K2 **PASS**. A
second exclusive global Block does not rescue 1024 SELECT. Do not extra-step
(floor). Do not restore raw global KV (that is E18). Do not remainder-on.
Do not H=512. Next: **stop 1024 SELECT architecture hunts**. Remaining DNA
walls: 512 chain **K1**, 256 hops **FAIL**, 2048+ INDEX shared with E18.
Not hops seq shrink. Not Glyph. Do not unfreeze `u`/`delta`. Default
`--global_layers` stays 1. Default extra hops stay 0. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5an — seq=1024 select QUERY-side anchors, identity slots)
Ran packed `select_1decoy` `--scale bridge_1k` inplace `--message_identity_slots
--message_ratio 1 --message_slots_inplace --message_global_anchors query_side`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0, no
`--message_update_slot_kv`, `global_layers` 1, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Existing
`query_nbhd` is prefix-before-QUERY (r=1 slots, extra 0). `query_side` leaks
QUERY + 2 asked-key symbols + ANSWER (**4 tokens vs seq=1024**, extra
non-slot count 4, not prefix replace slots). Dense **100% / 63.96 bits**
@250 (**S0 PASS**). E18 **75.1% / 46.77 bits** @800 (live late-click; chance
through 750). E21 **25.7% / 0 bits** @800 (chance every eval; CE at ln(4);
**S1 FAIL** vs 0.75× E18 35.07 and vs 0.75× dense 47.97). K2 **PASS**.
QUERY-side neighborhood leak does not rescue 1024 SELECT. Do not extra-step
(floor). Do not restore raw global KV (that is E18). Do not remainder-on.
Do not H=512. Next: **stop 1024 SELECT architecture hunts**. Remaining DNA
walls: 512 chain **K1**, 256 hops **FAIL**, 2048+ INDEX shared with E18.
Not hops seq shrink. Not Glyph. Do not unfreeze `u`/`delta`. Default
`--message_global_anchors` stays `none`. Default extra hops stay 0. Default
`--global_layers` stays 1. Default `--message_update_slot_kv` stays off.
Default `--message_keep_local_swa` stays off.

## Follow-up (rung 5ao — seq=768 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 768` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 2048 INDEX); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Dense **100% / 63.95 bits** @250
(**S0 PASS**). E18 **100% / 63.97 bits** @350 (live). E21 **25.6% / 0.01
bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18
47.97 and vs 0.75× dense 47.96). K2 **PASS**. Exclusive identity SELECT
that passed at 512 is chance at 768 on the 1024-passing width. Do not
extra-step (floor). Do not restore raw global KV (that is E18). Do not
remainder-on. Do not H=512. Next: seq=**640** same recipe (tighten
(512, 768]). Not architecture knobs. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Default remainder stays off. Default extra hops
stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5ap — seq=640 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 640` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 768); `bridge_1k` keeps the 32-token /
64-bit pack (`bridge` would pack 24). Window 16 < gap 64; right-align row
gap 100/100/100. Dense **99.8% / 63.76 bits** @200 (**S0 PASS**). E18
**100% / 63.91 bits** @300 (live). E21 **100% / 63.95 bits** @450 (chance
through 350, 51.3% @400, early-stop; **S1 PASS** vs 0.75× E18 47.93 and vs
0.75× dense 47.82). K2 **PASS**. Exclusive identity SELECT that passed at
512 is live at 640 and chance at 768 on the 1024-passing width. Wall
**(640, 768]**. Do not extra-step (S1 already PASS). Do not restore raw
global KV (that is E18). Do not remainder-on. Do not H=512. Next: seq=**704**
same recipe (tighten (640, 768]). Not architecture knobs. Not hops seq
shrink. Not Glyph. Do not unfreeze `u`/`delta`. Default remainder stays
off. Default extra hops stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5aq — seq=704 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 704` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 640/768); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Dense **100% / 63.95 bits** @300
(**S0 PASS**). E18 **100% / 63.89 bits** @350 (live). E21 **25.7% / 0.00
bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18
47.92 and vs 0.75× dense 47.96). K2 **PASS**. Exclusive identity SELECT
that passed at 640 is chance at 704 on the 1024-passing width. Wall
**(640, 704]**. Do not extra-step (floor). Do not restore raw global KV
(that is E18). Do not remainder-on. Do not H=512. Next: seq=**672** same
recipe (tighten (640, 704]). Not architecture knobs. Not hops seq shrink.
Not Glyph. Do not unfreeze `u`/`delta`. Default remainder stays off.
Default extra hops stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5ar — seq=672 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 672` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 640/704); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Dense **100% / 63.76 bits** @200
(**S0 PASS**). E18 **100% / 63.93 bits** @350 (live). E21 **100% / 63.96
bits** @500 (chance through 400, 47.6% @450, early-stop; **S1 PASS** vs
0.75× E18 47.95 and vs 0.75× dense 47.82). K2 **PASS**. Exclusive identity
SELECT that passed at 640 is live at 672 and chance at 704 on the
1024-passing width. Wall **(672, 704]**. Do not extra-step (S1 already
PASS). Do not restore raw global KV (that is E18). Do not remainder-on.
Do not H=512. Next: seq=**688** same recipe (tighten (672, 704]). Not
architecture knobs. Not hops seq shrink. Not Glyph. Do not unfreeze
`u`/`delta`. Default remainder stays off. Default extra hops stay 0.
Default `--global_layers` stays 1. Default `--message_update_slot_kv`
stays off. Default `--message_keep_local_swa` stays off. Default
`--message_global_anchors` stays `none`.

## Follow-up (rung 5as — seq=688 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 688` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 672/704); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Dense **100% / 63.96 bits** @250
(**S0 PASS**, 800 JSON) and **100% / 63.96 bits** @200 (8k JSON). E18
**100% / 63.73 bits** @350 (800 JSON) and **100% / 63.96 bits** @350
(8k live). E21 **72.4% / 31.57 bits** @800 (chance through 450, click
@500, best **88.9% @750**, dip at 800 — climb, not floor; short of S1
47.80) then extra-step **99.5% / 62.68 bits** @400 (**S1 PASS** vs 0.75×
E18 47.97 and vs 0.75× dense 47.97). K2 **PASS**. Exclusive identity
SELECT that passed at 672 is live at 688 and chance at 704 on the
1024-passing width. Wall **(688, 704]**. Do not 16k (S1 already PASS).
Do not restore raw global KV (that is E18). Do not remainder-on. Do not
H=512. Next: seq=**696** same recipe (tighten (688, 704]). Not
architecture knobs. Not hops seq shrink. Not Glyph. Do not unfreeze
`u`/`delta`. Default remainder stays off. Default extra hops stay 0.
Default `--global_layers` stays 1. Default `--message_update_slot_kv`
stays off. Default `--message_keep_local_swa` stays off. Default
`--message_global_anchors` stays `none`.

## Follow-up (rung 5at — seq=696 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 688/704); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Dense **100% / 63.95 bits** @200
(**S0 PASS**). E18 **100% / 63.93 bits** @350 (live). E21 **25.0% / 0.00
bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75× E18
47.95 and vs 0.75× dense 47.96). K2 **PASS**. Exclusive identity SELECT
that passed at 688 is chance at 696 on the 1024-passing width. Wall
**(688, 696]**. Do not extra-step (floor — unlike 688's 800-step climb).
Do not 16k. Do not restore raw global KV (that is E18). Do not
remainder-on. Do not H=512. Next: seq=**692** same recipe (tighten
(688, 696]). Not architecture knobs. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Default remainder stays off. Default extra hops
stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5au — seq=692 select_1decoy r=1 identity, H=256 log)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 692` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace`
(no `--message_pool_remainder`, no `--message_keep_local_swa`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no raw_kv,
`u`/`delta` frozen) **H=256 `--global_logit_scale log`** dense-first on
Odra. Existing `--seq_len` (same as 688/696); `bridge_1k` keeps the
32-token / 64-bit pack (`bridge` would pack 24). Window 16 < gap 64;
right-align row gap 100/100/100. Seq=692 ≡ 20 (mod 32) / 4 (mod 16) — not
a packed-answer or SWA-window boundary. Dense **100% / 63.97 bits** @200
(**S0 PASS**). E18 **100% / 63.89 bits** @350 (live). E21 **99.8% / 62.76
bits** @500 (chance through 450, click 99.8% @500, early-stop; **S1 PASS**
vs 0.75× E18 47.91 and vs 0.75× dense 47.98). K2 **PASS**. Exclusive
identity SELECT that passed at 688 is live at 692 and chance at 696 on
the 1024-passing width. Wall **(692, 696]**. Do not extra-step (S1 already
PASS). Do not 16k. Do not restore raw global KV (that is E18). Do not
remainder-on. Do not H=512. Next: seq=**694** same recipe (tighten
(692, 696]). Not architecture knobs. Not hops seq shrink. Not Glyph. Do
not unfreeze `u`/`delta`. Default remainder stays off. Default extra hops
stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5av — seq=696 select leftover drop `--message_pack_stride 32`)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace
--message_pack_stride 32` (no `--message_pool_remainder`, extra hops 0,
no `--message_update_slot_kv`, `global_layers` 1, anchors none, no
raw_kv, `u`/`delta` frozen) **H=256 `--global_logit_scale log`**
dense-first on Odra. Geometry: 688/692/696 have 2 complete 35-token KV
packs; 693–696 add 4 left-filler tokens after BOS, not an incomplete
DNA block; QUERY 650/654/658; leftover 10/14/18; tail 38; gap filler
60. r=1 remainder is a no-op (658 exclusive slots on or off). Pack
stride 32 remainder-off drops leftover → **640** exclusive slots at
both 692 and 696. Dense **100% / 63.95 bits** @200 (**S0 PASS**, 800
JSON). E18 **99.8% / 63.33 bits** @550 (live). E21 **49.7% / 31.67
bits** @800 (click 50.2% @750; **S1 FAIL** vs 0.75× E18 47.50,
climbing). 8k extra-step: dense **100% / 63.96**; E18 **100% / 63.91**
@350; E21 **24.1% / 0 bits** @8000 (chance every eval; **S1 FAIL** vs
0.75× E18 47.94). K2 **PASS**. Leftover *count* is not the 4-token
cliff. Wall still **(692, 696]**. Do not 16k. Do not seq=694. Do not
restore raw global KV. Default `--message_pack_stride` stays **0**.
Default remainder stays off. Next: seq=**696** `--evidence_align
spread` same identity recipe, pack_stride 0 (do not run here). QUERY
stays 658; slack is no longer locked as left-filler after BOS. Not
architecture knobs. Not hops seq shrink. Not Glyph. Do not unfreeze
`u`/`delta`. Default extra hops stay 0. Default `--global_layers`
stays 1. Default `--message_update_slot_kv` stays off. Default
`--message_keep_local_swa` stays off. Default `--message_global_anchors`
stays `none`.

## Follow-up (rung 5aw — seq=696 select `--evidence_align spread`)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace
--evidence_align spread` (no `--message_pack_stride`, no
`--message_pool_remainder`, extra hops 0, no `--message_update_slot_kv`,
`global_layers` 1, anchors none, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Geometry seed
0: QUERY **658** leftover **18** on both aligns; right fact0=**528**
decoy0=563 left-filler **527**; spread fact0=**202** decoy0=534
left-filler **201** mid-filler 297. Probe row gap 336/506/623 vs right
100/100/100. Dense **24.7% / 0 bits** @3200 (**S0 FAIL / K1**; chance
every eval; CE at ln(4)). e18 / e21 / e18_local **skipped**. Hunt exit
2. Do not extra-step (floor). Do not 16k. Do not score E21. Spread at
696 is uncalibrated (same class as 4k spread INDEX). Wall still
**(692, 696]** on **right-align**. Default `--evidence_align` for
SELECT length rungs stays **right**. Default `--message_pack_stride`
stays **0**. Default remainder stays off. Next: seq=**694** right-align
same identity recipe, pack_stride 0 (do not run here). Not architecture
knobs. Not hops seq shrink. Not Glyph. Do not unfreeze `u`/`delta`.
Default extra hops stay 0. Default `--global_layers` stays 1. Default
`--message_update_slot_kv` stays off. Default `--message_keep_local_swa`
stays off. Default `--message_global_anchors` stays `none`.

## Follow-up (rung 5ax — seq=696 select `--local_window 32`)
Ran packed `select_1decoy` `--scale bridge_1k --seq_len 696` inplace
`--message_identity_slots --message_ratio 1 --message_slots_inplace
--evidence_align right --local_window 32` (no `--message_pack_stride`,
no `--message_pool_remainder`, extra hops 0, no `--message_update_slot_kv`,
`global_layers` 1, anchors none, no raw_kv, `u`/`delta` frozen)
**H=256 `--global_logit_scale log`** dense-first on Odra. Banner
`seq=696 gap=64 window=32` (**32 < 64**; row gap 100/100/100). E18/E21
patterns `('swa', 32)`. QUERY **658** leftover **18** (658%16=2 vs
654%16=14; 658%32=18 vs 654%32=14). Dense **99.9% / 63.87 bits** @200
(**S0 PASS**). E18 **99.9% / 63.62 bits** @600 (live). E21 **25.0% /
0 bits** @800 (**S1 FAIL** vs 0.75× E18 47.71; chance every eval; CE
at ln(4)). `e18_local` **24.7% / 0 bits** (**K2 PASS**). Hunt exit 0.
Do not extra-step (floor). Do not 16k. Widening SWA does not rescue
696. Wall still **(692, 696]** on **right-align**. Default
`--local_window` stays **16**. Default `--message_pack_stride` stays
**0**. Default remainder stays off. Default `--evidence_align` for
SELECT length rungs stays **right**. Next: seq=**694** right-align
same identity recipe, pack_stride 0, default window 16 (do not run
here). Not another window. Not hops seq shrink. Not Glyph. Do not
unfreeze `u`/`delta`. Default extra hops stay 0. Default
`--global_layers` stays 1. Default `--message_update_slot_kv` stays
off. Default `--message_keep_local_swa` stays off. Default
`--message_global_anchors` stays `none`.

## Follow-up (rung 5ay — seq=1536 far_copy r=16 rem-off H=256 SSMax log)
Ran packed `far_copy` `--scale bridge_1k --seq_len 1536` inplace
`--message_identity_slots --message_ratio 16 --message_slots_inplace` (no
`--message_pool_remainder`, no raw_kv, `u`/`delta` frozen) **H=256
`--global_logit_scale log` batch 32** dense-first on Odra. Banner
`seq=1536 gap=64 window=16` (16 < 64; row gap 65/65/65). QUERY leftover
**12** (same residue as 1024/2048). Dense **99.1% / 62.41 bits** @300
(**S0 PASS**). E18 **25.8% / 0.00 bits** @800 (live ~0). E21 **24.8% /
0 bits** @800 (chance every eval; CE at ln(4); **S1 FAIL** vs 0.75×
dense 46.81). Do **not** pass S1 via 0.75×0. K2 **PASS**. Exclusive
INDEX that passed at 1024 is chance at 1536 with the same gap=64 /
window=16. Shared wall **(1024 PASS, 1536 FAIL]**. No new scale enum
(`--seq_len` override). Do not extra-step (floor). Do not remainder-on.
Do not 1536 H=512 (dense S0 already passed). Next: **stop INDEX length
extra-steps** (do not seq=1280). Do not reopen SELECT **(692, 696]**.
Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5az — seq=256 chain_ordered --key_len 13 `--global_layers 2`)
Ran the measured glob=1 hops FAIL recipe (`--scale bridge --seq_len 256`
`--key_len 13` hops=2 H=256 `--global_logit_scale log` inplace r=1 identity,
remainder off, extra hops 0, keep_local_swa off, update_slot_kv off, anchors
none) **plus `--global_layers 2`** dense-first on Odra. Window 16 < gap 64.
Dense **99.2% / 25.46 bits** @1800 (**S0 PASS**; recalibrated at two full
layers, 2.802M). E18 **42.3% / 5.56 bits** @1800 (live, climbing; not ~0).
E21 **24.0% / 0.02 bits** @800 (chance floor) then **99.6% / 25.85 bits**
@1650 (**S1 PASS** vs 0.75× E18 4.17 and vs 0.75× dense 19.10). K2 **PASS**.
Two sequential global attend+FFN Blocks compose hops at 256; one global
layer did not (E18 and E21 both 0). Exclusive identity even beats live E18
on this rung. Do not relabel E18 as E21. 8k not run (S1 already PASS). Do
not 16k. Code default `--global_layers` stays 1. Next: seq=**512** packed
`chain_ordered --key_len 13` `--global_layers 2` (previously dense K1 at one
global layer). `--hops 1` is illegal. Not Glyph. Do not unfreeze `u`/`delta`.

## Follow-up (rung 5ba — seq=512 chain_ordered --key_len 13 `--global_layers 2`)
Ran the measured 256 glob2 S1 PASS compressor (`--scale bridge` seq=512
default `--key_len 13` hops=2 H=256 `--global_logit_scale log` inplace r=1
identity, remainder off, extra hops 0, keep_local_swa off, update_slot_kv
off, anchors none) **plus `--global_layers 2`** dense-first on Odra. Window
16 < gap 64. Dense **22.5% / 0 bits** @3200 (**S0 FAIL / K1**; chance every
eval; CE at ln(4); best 27.9%; 2.802M). e18 / e21 / e18_local **skipped**.
Hunt exit 2. JSON `calibrated: false`. glob=1 at seq=512 was also K1
(22.5% / 0 bits, 2.261M). Two global Blocks that compose hops at 256 do
not make packed 512 dense-solvable. Do not score E21. Do not extra-step
(floor). Do not 16k. Code default `--global_layers` stays 1. Next: seq=**384**
packed `chain_ordered --key_len 13` `--global_layers 2` (localize dense hops
wall **(256 S0 PASS, 512 K1]**). `--hops 1` is illegal. Not Glyph. Do not
unfreeze `u`/`delta`.
