# E18c — Concept-compressed read: does trained retrieval survive pooling the prefix K/V into one concept slot per r tokens?

- **Status:** draft — **gated on E18b passing S1–S3**; no code written
- **Serves:** the project Vision directly (compress long context into concept vectors) in the one place
  where compression is cheap to test against an honest uncompressed baseline; the 10M-context path
  ([blockers note](../../4_Research_Notes/e18_10m_context_blockers.md)); E21's message size (a 32k
  document → 2k slots ≈ 2 MB).
- **Implementation plan:** *(to be authored by `implementation-plan` after E18b's result)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-10 · closed —

> E10–E17 compressed context into concepts as the **only** channel and, under plain CE, the model routed
> around it. E18 showed why: next-token loss gives long-range channels no gradient (arm C). E18b (if it
> passes) supplies a channel that is *trained to retrieve*. E18c compresses **that** channel and measures
> **retrieval retention** — a metric that cannot be satisfied by ignoring the channel.

## Hypothesis
If the global read's prefix K/V for every *completed* block of r tokens is replaced by **c learned concept
slots** produced by attention-pooling the block's pre-encoded states (the current, incomplete block stays
raw), and the model is trained from E18b's checkpoint with the same 5% retrieval mix, then at **r = 16,
c = 1** passkey retrieval at 32k retains **≥ 90% of E18b's accuracy** and LM loss stays within **0.5%** —
**because** arm A's per-token tail showed the read's information is sparse and localized (27% of tokens
depend on it, the worst 1% carry the mass), and a retrieval target is a *local* pattern that a pooled
block key can still address if the pooling query is trained by the retrieval loss; the prefix cache then
shrinks 16× (1M tokens → 64 MB, 10M → 640 MB).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` — new reusable `KVCompressor` (config-selectable
  `global_kv_compress_ratio=r`, `global_kv_slots_per_block=c`; 0 = off, byte-identical), one extra
  `mask_mod` for the read (query p attends to slots of blocks ending ≤ ⌊p/r⌋·r and to raw keys inside its
  own block), `prefix_kv()` returns slots (E21's message object), `reach_override` unchanged;
  E18b's data builder and probes.
- **Init / checkpoint:** warm start from E18b arm R `final`; compressor zero-/identity-initialised so step
  0 equals the uncompressed model on the raw block and starts from mean-pooling on past blocks.
- **Baseline to beat:** E18b arm R passkey at 32k / 64k / 128k and its LM eval loss (uncompressed read,
  same data, same protocol).
- **Materially new:** compression of a *functional* retrieval channel, judged by retrieval retention rather
  than by CE (the metric under which E10–E17's concepts were ignored); causal block pooling with a raw
  current block (Compressive-Transformer-style memory, but inside one read with RoPE at block-end
  positions, so the training sweep identity still holds and kernels stay standard); the slot count is
  length-proportional (S/r), not fixed — the Perceiver bottleneck is *soft*.

## The architectural bet
```
pre-encoded states h[0:S]  ─► per block b of r tokens: slots_b = Pool_c(h[b·r : (b+1)·r])   # learned queries
global read at query p:  keys = { slots_b : (b+1)·r ≤ p }  ∪  { raw k_j : ⌊j/r⌋ = ⌊p/r⌋, j ≤ p }
cache per token: 1 KB / r  (+ r raw)      — r = 16 → 64 B/token
```
Pooling = one cross-attention with c learned queries per block (Perceiver-style), K/V from the block,
positions = block end (RoPE), value embeddings of the block max-pooled into the slot's V. Train with LM
CE + the retrieval rows; no auxiliary loss.

## Why this is not a safe retread
The ledger's compression experiments failed on *usage*; this one cannot fail that way — the compressed
channel is the only route to the retrieval targets, so the loss must use it. The question becomes purely
informational: how many bits per token does exact retrieval need? r=16 asks whether one 256-dim slot per
16 tokens (16 B/token) suffices for content addressing — the same ratio at which gist/AutoCompressor
methods report retrieval degradation, so a pass is informative either way.

## Success criteria (set BEFORE running)
- **S1 retention:** passkey at 32k ≥ 0.9 × E18b-R's, and at 128k ≥ 0.8 × E18b-R's, at r = 16, c = 1.
- **S2 LM cost:** trainer eval loss within 0.5% of E18b-R; PG-19 buckets ≥ 8k within 0.5%.
- **S3 cache:** measured prefix-cache bytes at 32k ≤ 1/12 of E18b-R's (accounting for the raw block).
- **S4 message hook:** `prefix_kv()` slots round-trip: a receiver model given only the slots of a 32k
  document answers passkey ≥ 0.9 × S1 (E21 pre-check).

## Kill criteria (set BEFORE running)
- **K1:** r = 16 fails S1 → run r = 4 (one iteration). r = 4 retention < 80% → pooled slots cannot be
  addressed; stop. The 10M path then needs *sparse* (top-k block) retrieval instead of compression, and
  the concept-vector idea is recorded as falsified *for exact retrieval* at this scale.
- **K2:** r = 16 passes S1 but fails S2 by > 1.5% → compression taxes LM; try c = 2 (one iteration).

## Plan
- **Data:** E18b's merged manifest (LM 95% + retrieval 5%), 32k.
- **Compute:** Polonez, 1 arm ≈ 8 GPU-h (+ 8 for the r = 4 fallback) + probes.
- **Steps / epochs:** 0.3B tokens at 32k from E18b-R (the addressing circuit already exists; the run
  teaches the pooling), corrected extension protocol.
- **Launch:** `E18_STAGE=32k EXPERIMENT_ID=E18c MODEL_NAME_OR_PATH=<E18b-R final>
  PAR_GLOBAL_KV_COMPRESS_RATIO=16 PAR_GLOBAL_KV_SLOTS=1 LEARNING_RATE=0.002 WARMUP_STEPS=100
  LR_SCHEDULER_TYPE=cosine PRETOKENIZED_MANIFEST=<E18b manifest> bash scripts/launch_e18.sh`
- **New foundation code:** `KVCompressor` + read mask in `nn/perceiver_ar_lm.py` (tests: off = byte-identical;
  causality — no slot of an incomplete block is visible; sweep identity; cache-size accounting;
  `prefix_kv` slot round-trip), launcher knobs, probe for S3/S4.

## Result
<Filled in AFTER, by experiment-track.>
- Run id: `<run_id>` · Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
