# E25 — E21 capability limits on a calibrated BAPO DNA ladder (one rung at a time)

- **Status:** active (tiny INDEX wall under three placements; extra steps next). Keep in
  `ahead/` — K3 not hit. Do not score recall or 512.
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
- Run id: `e25_tiny_far_copy` (`73f2a31`) · `e25_tiny_far_copy_remainder` (`8b76ba5`) · `e25_tiny_far_copy_query_align` (seq=132)
- WandB: n/a (probe; no `compute/*`)
- Run reports: [`e25_tiny_far_copy_20260913.md`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_20260913.md) · [`e25_tiny_far_copy_remainder_20260913.md`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_remainder_20260913.md) · [`e25_tiny_far_copy_query_align_20260913.md`](../../2_Experiments_Registry/run_reports/e25_tiny_far_copy_query_align_20260913.md)
- Verdict: **mixed** — three tiny `far_copy` placements all S0/K2 pass, S1 fail, K3 not hit. Complete-block **17.7 bits**, remainder **12.7**, QUERY-align seq=132 **8.7**. Slot channel is live and lossy; missing remainder was not the INDEX wall. Do not score recall or 512.

## Follow-up (rung 1b — remainder pooling)
Ran. S1 **FAIL** (12.7 bits vs ≥47). K3 not triggered. Stop architecture stacking on the compressor.

## Follow-up (rung 1c — align QUERY to r)
Ran `--seq_len 132` (QUERY at 96). Dense S0 **99.1% / 62 bits**. E21 **40.0% / 8.7 bits** (flow 0.136). S1 **FAIL**. Complete coverage does not recover the prize.

## Follow-up (rung 1d — extra steps)
Next ONE change: extra step budget on seq=128 complete-block E21 (best so far, still climbing at 2400). Not a new architecture. If it plateaus below 47 bits, tiny INDEX is the measured E21 limit.
