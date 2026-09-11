# E21 — Latent-message pretraining: one model, split into sender and receiver, that may only talk through its compressed global read

- **Status:** approved 2026-09-11 (user go: plan + implement + Polonez smoke); **implemented 2026-09-11** as config over the E18 foundation (see [plan](E21_latent_message_pretraining_plan.md) and CHANGELOG 2026-09-11) — `KVCompressor` landed with E21 itself and compresses only the keys that cross the boundary, so E18c is no longer a code dependency; Polonez data prep / smoke pending (server busy). Warm-starts from E18b arm R if available, else stage A.
- **Serves:** the Vision's three cells at once — compress long context into concept slots (priority 2), reason from them (priority 1/3), and **latent agent-to-agent communication** (priority 4) — by making the latent message the *training objective* rather than the last demo. Rationale: [strategy synthesis](../../4_Research_Notes/strategy_synthesis_latent_channel_20260911.md) · positioning: [positioning_and_funding](../../1_Strategy_and_Plans/positioning_and_funding.md) · literature: [latent_agent_communication](../../literature_review/latent_agent_communication.md).
- **Implementation plan:** [E21_latent_message_pretraining_plan.md](E21_latent_message_pretraining_plan.md) (2026-09-11; compression applies only to keys crossing the boundary — see plan §1 for the deliberate deviation from the E18c-style sketch below)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-11 · closed —

> The E18 family was originally sequenced platform (E18) → retrieval read (E18b) → compressed read
> (E18c) → reasoning (E19) → diffusion (E20) → agent messages (E21). This spec pulls E21 forward
> and changes its role: **the message is the objective that trains the compression**, because the
> ledger says a compressed channel trained under an objective that a raw-token path can satisfy goes
> dead (E05, E10–E17, E18 arm C), while the one objective that de-collapsed concepts with scale
> (E02 prefix→suffix) is the one where the decoder had no raw access to the prefix. E18c keeps a
> 20-layer SWA stack whose chained reach (~80k) covers a 32k row, so its slots are again bypassable
> for LM loss and it must lean on retrieval labels alone. E21 removes the bypass by construction.

## Hypothesis
If, on a fraction *q* of pretraining rows, the E18 platform is run as **sender ∥ receiver with shared
weights** — a *message boundary* at position P cuts every sliding-window layer and the n-gram tables'
context, so tokens ≥ P can see tokens < P **only** through the global read, and that read sees the
prefix **only as compressed concept slots** (one learned slot per r = 16 tokens, E18c's `KVCompressor`)
— and the loss is ordinary CE on every token (sender side normal LM; receiver side "suffix from
message") plus E18b's dense retrieval rows with sources before P and targets after P, then the slots
become a **load-bearing, content-specific, compressed message**: receiver early-suffix CE improves
by **≥ 0.30 nats** with the message vs. without (E02 measured 1.43 at a 512-token scale with an
uncompressed message), retains **≥ 80% of the uncompressed (r = 1) message's gain** and **≥ 90% of
E18b-R's keyed-recall accuracy** through the boundary, costs **≤ 0.5%** LM loss on ordinary rows, and
**collapses under a swapped message** — **because** with the raw channel `b` cut at P (BAPO: a
two-party prefix/suffix protocol with bounded message `a`), every bit of prefix→suffix dependency
must pass through the slots, so CE itself supervises them densely at every receiver position, and
because sender and receiver share weights, the read's K/V space is the message space by construction
(no projector, no alignment, unlike C2C/Interlat/StateBridge).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`perceiver_ar` family: global read, `prefix_kv()`,
  `reach_override`, block-mask memo, E18c's `KVCompressor` + slot mask), `data/packed_dataset.py`
  (`doc_ids` — the message boundary is a *partial* document boundary: SWA layers treat P as a doc
  start, the global read does not), `scripts/build_retrieval_mix_dataset.py` (E18b rows; add a
  `--boundary_aware` flag so sources land before P and targets after), `evaluation/long_context_probes.py`
  (`--probe passkey/buckets/reach`; add `--probe message`), `scripts/launch_e18.sh`, Muon split.
- **Init / checkpoint:** warm start from **E18b arm R `final`** (retrieval-trained read; if E18b has
  not closed, stage A `checkpoint-9030`). Compressor init as in E18c (identity on the raw current
  block, mean-pool on past blocks) so step 0 = the uncompressed model.
- **Baseline to beat:** (i) **the same checkpoint probed with `message_override=none`** (receiver
  gets no prefix at all — the floor); (ii) **arm U** trained identically with r = 1 (uncompressed
  raw K/V across the boundary — the ceiling); (iii) E18b-R keyed-recall / passkey at 32k; (iv) E02-long
  early-position Δzero **1.43** / Δshuffle **1.04** as the historical order of magnitude.
- **Materially new:** (a) the message boundary — a mask that severs the local channel while leaving
  the global read open — has not been used in any ledger experiment; E05's window cap severed
  nothing (windows chain), E02 severed everything except a fixed 128-slot summary at 512 tokens;
  (b) weight-shared sender/receiver over the read's own K/V space (the literature bolts channels
  onto frozen text models with projectors); (c) compression *trained by* the receiver's CE, not
  judged by retrieval retention only (E18c); (d) pre-registered causal controls (none / swapped /
  raw) that the latent-MAS audit literature asks for and no paper provides.

## The architectural bet
```
row x[0:S], S = 32k;  with prob. q = 0.5 draw P ~ U[4k, S − 4k]  (else P = S: ordinary E18c row)
sender side  (t <  P): normal E18/E18c forward; slots_b = Pool_1(h[b·16:(b+1)·16]) for completed blocks
receiver side (t ≥ P):
   SWA layers, n-gram context:   doc_ids' := doc_ids ⊕ [t ≥ P]     # P is a document start for local layers
   global read at query t:       keys = {slots_b : (b+1)·16 ≤ P}  ∪  {raw k_j : P ≤ j ≤ t, same block}
                                 # nothing from [0,P) except slots; nothing of [P,t] is compressed away
loss = mean CE over all S tokens (+ E18b retrieval rows built boundary-aware: key/source < P, target ≥ P)
```
The same weights play both roles; nothing new is learned *about* roles. `prefix_kv(x[0:P])`
**is** the message object: `S/16` slots × 2 kv-heads × 128 × bf16 ≈ **64 B/token → 2 MB for a 32k
document, 64 MB for 1M**. Because the boundary is a mask, training-time identity with inference
holds (a receiver at inference literally loads another copy's `prefix_kv()`).

**Probe `--probe message`** (paired per row, like the reach ablation): receiver CE on positions
[P, P+512) and [P+512, P+4096) under `message_override ∈ {real, none, swapped, raw}` where
*swapped* = slots of a different row of equal length and *raw* = r = 1 K/V of the true prefix
(only meaningful on arm U or via the compressor's identity path). Keyed-recall / passkey with the
key before P and the query after P.

## Why this is not a safe retread
It is the first ledger experiment where **CE alone cannot ignore the concept channel**: E05/E10–E17/E18
all left a raw path open and watched the channel die; E02 closed it but at 512 tokens with a fixed
summary and no exact-recall pressure. It is also the first where the compressed object is *the same
object* a second model reads (E21's original purpose), so a positive result is simultaneously a
long-context cache, a reasoning state (E19 refines these slots) and an agent message. Cross-domain
hook: Yao's two-party communication complexity / BAPO — Alice holds the prefix, Bob the suffix, the
protocol is bounded by the message size; the corpus callosum — two hemispheres with a
bandwidth-limited bridge learn complementary, transmissible codes precisely because neither can
read the other's raw input. The rate–distortion curve (Δ-nats vs bytes/token across r) is the
scientific deliverable, not a single point.

## Success criteria (set BEFORE running)
- **S1 load-bearing:** arm R, `real − none` on [P, P+512) ≥ **0.30 nats** (paired, ≥ 10σ) and > 0 on
  [P+512, P+4096).
- **S2 retention:** arm R's `real − none` ≥ **0.8 ×** arm U's (compression keeps ≥ 80% of the raw
  message's value at 16× fewer slots).
- **S3 recall through the message:** boundary-aware keyed-recall and passkey (key < P, query ≥ P) at
  32k ≥ **0.9 ×** E18b-R's in-row accuracy; ≥ 0.8 × at 64k.
- **S4 content, not steering:** `swapped − none` ≤ **0** on [P, P+512) (a wrong message is no better
  than none; the StateBridge random-prefix failure mode is absent).
- **S5 LM cost:** trainer eval loss on ordinary rows (P = S) within **0.5%** of E18c-r16 / E18b-R.
- **S6 message hook:** a second process loading only `prefix_kv(x[0:P])` reproduces the receiver's
  logits bit-exactly (the E21 round-trip; E18c's S4 subsumed).

## Kill criteria (set BEFORE running)
- **K1 dead channel:** S1 < 0.10 nats at the end of the budget while arm U's `real − none` ≥ 0.30 →
  compression is the failure; **one iteration** at r = 4 (E18c's fallback ladder). If r = 4 also fails
  S1 → pooled slots cannot carry suffix-relevant content; record the concept-slot idea as falsified for
  this platform at this scale and move the long-context claim to sparse retrieval.
- **K2 no signal even raw:** arm U's `real − none` < 0.10 nats → the receiver formulation itself is
  wrong (or 32k prefixes carry too little suffix information at 125M); stop, run the 512-token E02
  geometry as a sanity anchor before any retry.
- **K3 steering artifact:** S4 fails (`swapped > none` by > 0.05 nats) → the read learned a generic
  prior, not content; add a swapped-message *negative* term to the objective (one iteration) — if it
  persists, stop.
- **K4 cost:** S5 fails by > 1.5% → q = 0.5 is too large; halve to 0.25 (one iteration).
- Any: eval loss rising over 3 consecutive evals → stop, retune per the Muon protocol.

## Plan
- **Data:** E18b's merged manifest (LM 95% + retrieval 5%, 32k) with the retrieval rows rebuilt
  `--boundary_aware` (targets after a row-specific P that is written into the row metadata and honoured
  by the collator). q = 0.5 of LM rows get a boundary; P drawn per row, fixed by seed.
- **Compute:** Polonez 4×3090 (Odra fallback). Arms: **R** (r = 16, the claim) and **U** (r = 1, the
  ceiling); ≈ 8–10 GPU-h each at 32k + probes ≈ **25 GPU-h**. The "none" and "swapped" readings are
  probes on the trained arms, not trained models (same protocol as the reach ablation).
- **Steps / epochs:** 0.5B tokens per arm at seq 32k, `PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4`,
  corrected extension protocol (≤ 20% peak lr warm start, 100-step warmup, cosine).
- **Launch (arm R):**
  ```bash
  E18_STAGE=32k EXPERIMENT_ID=E21 MODEL_NAME_OR_PATH=<E18b-R final> \
  PAR_GLOBAL_KV_COMPRESS_RATIO=16 PAR_GLOBAL_KV_SLOTS=1 \
  PAR_MESSAGE_BOUNDARY_FRAC=0.5 PAR_MESSAGE_BOUNDARY_MIN=4096 \
  LEARNING_RATE=0.002 MUON_ADAMW_LR=4e-5 WARMUP_STEPS=100 LR_SCHEDULER_TYPE=cosine \
  PRETOKENIZED_MANIFEST=$DATASETS_TOK_DIR/e21_lm95_ret05_boundary_manifest.json \
  PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/launch_e18.sh
  ```
  Arm U: `PAR_GLOBAL_KV_COMPRESS_RATIO=1`. Probes: `evaluation/long_context_probes.py --probe message
  --message_override none,swapped,raw --context_lengths 32768,65536` and `--probe passkey --boundary_aware`.
- **New foundation code (reusable, config-selectable):** in `nn/perceiver_ar_lm.py` a `message_boundary`
  input (per-row P, or `None`) that (i) augments `doc_ids` for SWA layers and the n-gram context,
  (ii) restricts the global read's keys for queries ≥ P to slots of blocks ending ≤ P plus raw keys in
  [P, t]; `message_override` for the probe; collator/packing support for a per-row boundary; the
  `--boundary_aware` builder flag; the `message` probe. Tests: boundary off ⇒ byte-identical to E18c;
  no token < P reachable except via slots (masks + a gradient-flow test); `prefix_kv(x[0:P])` round-trip
  (S6); `swapped` uses a different row.
- **Not in scope (follow-ups after a positive signal):** multi-boundary rows (multi-hop agents), a text-
  summary-at-equal-bytes baseline (the demo's comparison; needs a summariser — pilot uses `none`/`raw`),
  cross-checkpoint receivers, **E19** write-back refinement of the slots (accuracy vs steps) — E19's
  metric becomes "S1 grows with refinement steps K".

## Decision this spec feeds
- **S1–S5 pass →** the 600M main run adopts the boundary objective in stage 1 (q ≈ 0.3), M2 is
  re-scoped to *message* metrics (RULER/HELMET @128k answered from slots; bytes/token), the HF demo is
  built on `prefix_kv()`, and E19 starts from arm R.
- **K1 (both r) or K2 →** the concept-slot message is recorded as falsified at this scale; the platform
  keeps its retrieval claim (E18b) and the lab pivots to the reasoning axis (E19 on raw K/V).

## Result
<Filled in AFTER, by experiment-track.>
- Run ids: `<R>`, `<U>` · WandB: <link>
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: promising | mixed | regression | killed — <one line>
