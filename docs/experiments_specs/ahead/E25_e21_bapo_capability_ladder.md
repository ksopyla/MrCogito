# E25 — E21 capability limits on a calibrated BAPO DNA ladder (one rung at a time)

- **Status:** active (tiny DNA limits + 512 INDEX wall at r=16 / 0 bits). Keep in `ahead/`.
  Do not score Glyph or 1024 yet.
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
- Run id: `e25_tiny_far_copy` · `e25_tiny_far_copy_remainder` · `e25_tiny_far_copy_query_align` · `e25_tiny_far_copy_e21_steps` · `e25_tiny_recall_single` · `e25_tiny_recall_single_e21_steps` · `e25_tiny_select_1decoy` · `e25_tiny_chain_ordered` · `e25_tiny_chain_ordered_e21_steps` · `e25_512_fc`
- WandB: n/a (probe; no `compute/*`)
- Run reports: [`tiny`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md) · [`remainder`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md) · [`QUERY-align`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md) · [`INDEX extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_e21_steps_20260913.md) · [`recall`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_20260913.md) · [`recall extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_recall_single_e21_steps_20260913.md) · [`select`](../../2_Experiments_Registry/run_reports/e25_tiny_select_1decoy_20260913.md) · [`chain`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_20260913.md) · [`chain extra steps`](../../2_Experiments_Registry/run_reports/e25_tiny_chain_ordered_e21_steps_20260913.md) · [`512`](../../2_Experiments_Registry/run_reports/e25_bridge512_far_copy_20260913.md)
- Verdict: **mixed.** Tiny INDEX near-pass at 8k. Tiny MATCH/select/chain walls. GPU seq=512
  right-align `far_copy`: dense **61.2 bits**, E18 **63.2 bits**, E21 **0 bits / chance**
  (S1 fail, K3). Next: r=64 at seq=512 (~8 slots), not 1024, not remainder.

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
Next ONE change: `--message_ratio 64` at seq=512 (~8 slots, the width that near-passed tiny
INDEX). Remainder stays off. Recalibrate dense S0 in the same JSON. Not Glyph.
