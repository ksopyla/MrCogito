# E17a — Untied per-bank concept writers (4 writers)

- **Status:** draft — **conditional on E17; do not run unless E17's result activates it** (see Activation gate). Not yet implemented.
- **Serves:** the capacity/specialization counterfactual to E17. E17 deliberately tied the writer (one shared `ConceptWriteHead`) so the *only* variable vs E16b was state topology. E17a asks the follow-up: **was that tied writer itself the bottleneck?**
- **Implementation plan:** *(not yet written — author via `implementation-plan` only if E17 activates E17a)*
- **Owner / dates:** Krzysztof Sopyła · opened 2026-08-01 · closed —

> One experiment = one changed variable vs E17: the **write operator** goes from one tied
> `ConceptWriteHead` (shared across the 4 banks) to **four untied** `ConceptWriteHead`s
> (one per global layer/bank). The 4 private banks, per-bank gates, per-layer reads,
> backbone, LoRA, data, optimizer, and 1B-token budget are all identical to E17.

## Hypothesis
If each of the four per-layer concept banks is written by its **own untied** `ConceptWriteHead`
(4 writers, +3× writer-transform params) instead of E17's one shared tied writer, then at matched
1B tokens beyond-window CE drops by **≥0.05 nats/bank** and free-run `real` distinct-1@256 rises
by **≥0.10** over E17 — because each writer can specialize to its layer's feature distribution
(low-level at layer 5, high-level at layer 23) instead of one shared writer having to serve all
four depths.

## Builds-on
- **Foundation:** E17's `concept_io_mode="per_layer_banks"` path in `nn/backbone_concept_lm.py`
  (the tied `ConceptWriteHead`, `_forward_per_layer_banks_block`); a new reusable config flag
  `tie_concept_writer` (default `True` = E17). No new training script — a config value on the
  shared `BackboneConceptLM`.
- **Init / checkpoint:** fresh frozen `google/gemma-3-1b-pt` + LoRA r=16, seed 42 — **the same
  init recipe as E17** (not a resume), so writer tying is the only variable.
- **Baseline to beat:** **E17** at matched 1B tokens (the E17 checkpoint) on beyond-window CE,
  per-bank `Δshuffle_beyond`, and free-run diversity. E17 is the direct A/B; E16b is the
  grandparent (shared state + tied writer).
- **Materially new vs E17:** writer tying `tied → untied` — `self.write_head` (1 shared
  `ConceptWriteHead`) becomes `self.write_heads` (`ModuleList` of 4, one per global layer). This
  adds **+3× the writer-transform params** (BiXT + norms; the gates stay 4, one per writer). That
  capacity increase is exactly the variable under test, not a confound to avoid.

## The architectural bet
Replace E17's single tied writer:

```text
E17 (per_layer_banks, tied writer):
  one ConceptWriteHead (shared BiXT/norms/sandwich); depth_alphas[g] gates bank g's write
  layer g:  bank_g = write_head(bank_g, h_block_g, depth_index=g)   # SAME weights for all g

E17a (per_layer_banks, untied writers):
  four ConceptWriteHeads write_heads[0..3], one per global layer; each has its own BiXT/norms/sandwich + its own gate
  layer g:  bank_g = write_heads[g](bank_g, h_block_g)              # DIFFERENT weights per g
```

Everything else is held fixed vs E17: 4 private banks, per-layer reads (each layer's own Q/K/V/O),
per-bank read/write gates, frozen Gemma-3-1b + LoRA r16, `e16b_long_4k_v1`, K=512, C=128, Muon,
gate init 0.01, raw causal-LM CE, eff batch 72, 1B tokens, report @100M.

**Out of scope:** changing the number of banks, the read interface, the gate parameterization,
or the data — those are E17/E12/E13 territory. E17a isolates writer tying only.

## Why this is not a safe retread
This is the direct **specialization counterfactual** to E17's deliberate single-writer tying.
E17 tied the writer for a clean test (only state topology changes vs E16b); E17a tests whether
that conservative choice left quality on the table. **Analogy:** a standard transformer gives
every layer its OWN weights (layers are not weight-tied); E16b violated that for the *state*
(one shared bank) and E17 fixed the state but kept the *write op* tied. E17a restores per-layer
weights at the write op — the natural completion of "each layer owns its memory machinery."
BAPO lens unchanged from E17: still a write-dynamics / effective-`a` fix, not a reasoning-capacity
claim.

## Activation gate (run ONLY if E17 shows all of)
1. **The topology fix worked:** E17's per-bank write gates opened (`|tanh(α)| ≥ 0.1` by 100M–1B,
   unlike E16b's dead ±0.05). If E17's gates stay dead, the topology itself failed and E17a is moot.
2. **But concept quality is capped** by the tied writer — at least one of:
   - per-bank `Δshuffle_beyond` plateaus below ~E16b-relative expectations, **or**
   - free-run `real` distinct-1@256 recovers only partway (e.g. **< 0.6**, with base Gemma ≈0.78)
     despite open gates — i.e. the selfish writes opened but one shared writer can't summarize
     all four depths well enough to fully close the generation gap.

**Cancel E17a (move to `canceled/`) if E17's tied writer already fully recovers generation**
(free-run `real` distinct-1@256 ≥ ~0.7, `real ≥ zero`, fluent) — the tied writer was sufficient
and untied capacity is unnecessary.

## Success criteria (set BEFORE running — all vs E17 at matched 1B tokens)
- **Beyond-window CE:** ≥ **0.05 nats/bank lower** than E17 (mean of the 4 banks, positions ≥1024).
- **Per-bank effective `a`:** `Δshuffle_beyond ≥ E17 + 0.02` nats (per bank, reported as min + median).
- **Free-run diversity (the symptom):** `real` greedy distinct-1@256 **≥ E17 + 0.10** (toward base
  0.78), with `real ≥ zero` maintained.
- **Specialization evidence (attribution):** the 4 untied writers must **diverge** (per-bank
  `Δshuffle_beyond` spread > E17's, and writer-weight cosine-similarity across banks < ~0.9) — a
  uniform +params win would not establish the tying was the bottleneck.
- **Non-regression:** per-bank write gates still open (`|tanh(α)| ≥ 0.1`); within-bank RankMe ≥ 38.4.

## Kill criteria (set BEFORE running)
- At the **100M report**, stop if E17a is **not beating E17** on beyond-window CE **and** not on
  `Δshuffle_beyond` **and** not on free-run diversity (and the writers haven't diverged) — the tied
  writer was sufficient; E17a is unnecessary.
- Stop on non-finite loss/gradients, eval CE rising for three consecutive evaluations, or any
  bank's within-sample RankMe < 19.2/128.
- **Do not run at all** if E17 is canceled or its gates never open (E17a is conditional on E17).

## Plan
- **Data:** `e16b_long_4k_v1` (Gemma-tokenized, 4K) — the same manifest as E17 (must already be
  pretokenized on the run host from E17). Raw causal-LM CE, seq 4096.
- **Compute:** Polonez 4× RTX 3090 (or Odra 3×), eff batch 72 (per-device 3 × accum 6 × 4),
  1B tokens, warmup 500, report @100M, divergence-only kills. ≈ E17's ~115 GPU-h.
- **Launch (once E17 activates it and the flag exists):**
  ```bash
  EXPERIMENT_ID=E17a CONCEPT_IO_MODE=per_layer_banks TIE_CONCEPT_WRITER=false \
  READ_GATE_INIT=0.01 WRITE_GATE_INIT=0.01 OPTIMIZER=muon LEARNING_RATE=0.01 \
  MUON_ADAMW_LR=2e-4 MUON_MOMENTUM=0.95 WEIGHT_DECAY=0.1 CONCEPT_MEMORY_LR= \
  MAX_SEQ_LENGTH=4096 PRETOKENIZE_MIX=e16b_long_4k_v1 TARGET_TOKENS=1000000000 \
  WARMUP_STEPS=500 SAVE_TOTAL_LIMIT=12 SKIP_PRETOKENIZE=1 \
  PER_DEVICE_BATCH_SIZE=3 GRADIENT_ACCUMULATION_STEPS=6 \
  bash scripts/launch_e17.sh
  ```
- **New foundation code (reusable, when implemented — NOT now):** a `tie_concept_writer` field on
  `BackboneConceptConfig` (default `True`); when `False`, build `self.write_heads =
  ModuleList([ConceptWriteHead(...) for _ in range(G)])` and route `_forward_per_layer_banks_block`
  to `write_heads[g]`. Thread `TIE_CONCEPT_WRITER` through `launch_e17.sh` → `launch_e10.sh`.
  Config-selectable; no fork. Add a param-count test (E17a trainable = E17 + 3× writer-transform).

## Result
<Filled in AFTER, by experiment-track — and only if E17 activates E17a.>
- Run id: —
- WandB: —
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: —

## References
- E17 (baseline, done_failed init-0.01 / 1B): [spec](../done_success/E17_four_bank_concept_memory.md) · [plan](../done_success/E17_four_bank_concept_memory_plan.md)
- E16b write-path/topology diagnosis: [report](../../2_Experiments_Registry/run_reports/e16b_write_path_and_topology_diagnosis_20260801.md)
- E13 (full 26-layer per-layer memory): [spec](E13_layerwise_recurrent_kv_memory.md) — same "tie the writer by default, untie as a follow-up" discipline
