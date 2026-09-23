# E22 — Perceiver Concept LM: encoder → positional concept array → latent transformer → block-local decoder, from scratch at 32k

- **Status:** approved 2026-09-12 (KS: "design the experiment and implement it … make it right and modern, with proper long data set and long evaluation … iterate on this architecture") · ran 2026-09-12 (Odra A/C, 560 steps ≈ 0.44B tokens each, 88% of budget) · **killed 2026-09-12 — K1 met, S1/S2/S4 missed, S5 passed** → `done_failed`; successor [E23](../ahead/E23_exclusive_concept_channel.md)
- **Serves:** the Vision's priorities 1–2 directly — compress long text into a length-proportional concept array, *reason over the array* (concept↔concept attention, depth as a knob), decode from it — as the platform the latent-message (E21/E19) and 1M/10M work will run on. Rationale and evidence: [Perceiver revisit synthesis 2026-09-12](../../4_Research_Notes/perceiver_revisit_synthesis_20260912.md).
- **Implementation plan:** [E22_perceiver_concept_lm_plan.md](E22_perceiver_concept_lm_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-12 · closed 2026-09-12

> One coherent architectural bet: the encoder→concepts→decoder Perceiver idea rebuilt on the
> three laws the ledger and the frontier agree on (no raw long-range bypass at any layer;
> positional slot allocation; contextualise before compressing), plus the one thing no shipped
> model has — a latent transformer *over the concept array*. Everything below is one claim;
> the arms (no-concept control, dense control) exist to falsify it, not to hedge it. r, c, K,
> encoder depth and the decoder window are registered post-signal iterations, not part of the bet.

## Hypothesis
If a from-scratch ~130M-compute LM (SmolLM3 vocabulary, 32k rows) is built as **(i)** a 6-layer
causal sliding-window encoder over tokens, **(ii)** a concept array with **one learned-query
pooled slot per 16 tokens** (C = S/16, positions = block ends), **(iii)** a 4-layer causal
transformer over the concept array, and **(iv)** an 8-layer decoder whose self-attention is
**confined to its own 1024-token segment** and whose only access to anything earlier is
cross-attention to the concepts — then, under plain next-token CE (+ 5% dense keyed-recall rows),
the concept array becomes **load-bearing and useful**: ablating it costs ≥ 0.30 nats on tokens with
≥ 4k of history, a matched decoder *without* the concept path is ≥ 3% worse on those tokens, the
model stays within 3% of a matched dense transformer that sees everything raw, and passkey through
the concepts reaches ≥ 0.5 at 32k — **because** every dependency longer than one segment has exactly
one route (BAPO: `b` is capped at the segment, `a` is the array, CE must use `a`), each slot is
guaranteed by position rather than won in competition (no E05-style free-latent collapse), the slots
are formed from 6-layer-contextualised states rather than raw or 1-layer embeddings (the E18b defect),
and the latent transformer lets slots combine before the decoder reads them (the cell HCA/CSA leave empty).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` primitives — `TinyHashedEmbedding` (tiny factorised token
  table + hashed 2/3-grams), `Attention` (GQA, QK-norm, RoPE, value embeddings), `Block` (RMSNorm +
  SwiGLU, zero-init residual writers), `attend` (FlexAttention / SDPA with document masks),
  `chunked_softcap_ce` / Liger fused CE, `_positions`; `training/train_concept_pretraining.py` +
  `PerceiverDenoiseTrainer` (Muon/AdamW split, packing, `doc_ids`); `data/packed_dataset.py`;
  `scripts/train_concept_pretraining_multigpu.sh`; the `perceiver_ar` eval layer
  (`evaluation/long_context_probes.py`, `analysis/check_model_health.py`, `scripts/eval_perceiver_ar_suite.sh`).
  **New reusable family** `perceiver_concept` (`nn/perceiver_concept_lm.py`) selected by
  `MODEL_FAMILY=perceiver_concept`; objective `causal_lm`.
- **Init / checkpoint:** random init, from scratch. No warm start from E18/E18b (their read sits
  at layer 1 and their stack is a chained SWA bypass — the geometry this spec removes).
- **Baseline to beat:** (1) **arm C**, the same model with the concept path removed
  (`concept_mode=none`: decoder segments only) — the *usefulness* control, the lesson of E18 arm C;
  (2) **dense control**, `perceiver_ar` `PAR_MODE=dense` with 18 full-causal layers at the same
  width, data, tokens and batch — the *compression price* reference; (3) anchors, not matched:
  E18 stage A 3.790 @1B/8k, E18b arm 0 buckets 2.876 / 2.319 / 2.279 @32k, E18b DT passkey 0.725.
- **Materially new vs the ledger:** E01–E05 pooled a whole 512–2048-token input into 128 *free*
  latents with a decoder that kept a raw window (bypass + competition); E10–E17 bolted concept banks
  onto a backbone that kept full attention; E18/E18b/E21 have **no latent array at all** (one K/V
  read, pooled or not, never processed). E22 is the first ledger design with (a) length-proportional,
  positionally allocated slots, (b) a transformer *over* the slots, (c) a decoder with no raw path
  past its segment, (d) trained from scratch under CE at 32k on a long-document mix.

## The architectural bet
```
tokens x[0:S]  (S = 32768, packed documents with doc_ids)
  ─► TinyHashedEmbedding (e=256 + hashed 2/3-grams → d=768)
  ─► ENCODER: 6 × Block(swa, window 512, causal, doc-masked)            h_enc [B,S,d]
  ─► POOL (per block j of r=16 tokens, per document):                    z0   [B,C,d], C = ⌈S/r⌉
        z0_j = mean(h_enc[block j]) + XAttn_c(q_learned → h_enc[block j])   # c=1; out-proj zero-init
        (learnable within-block positional bias on the pooling logits; QK-norm)
        pos(z_j) = position of the block's last token;  doc(z_j) = doc of that token
  ─► LATENT TRANSFORMER: 4 × Block(full, causal over concepts, doc-masked, RoPE at pos(z))  z [B,C,d]
        (weight-tied repeats K = 1 in the bet; K ∈ {2,4} is the post-signal reasoning knob)
  ─► DECODER: 8 × [ self-attn confined to the token's 1024-token segment (segment id ⊕ doc id)
                     → cross-attn to {z_j : pos(z_j) ≤ p, doc(z_j) = doc(p)}   (q: RoPE(p), k: RoPE(pos z))
                     → SwiGLU ]                                            y [B,S,d]
  ─► RMSNorm → untied lm_head (softcap 30, z-loss 1e-4, Liger fused CE); loss on every token
```
**Training-time identity.** Every arrow is causal, so one forward computes all positions; at
inference the concept array is the only unbounded state (d bf16 per 16 tokens = **96 B/token**,
1M → 96 MB) plus one 1024-token segment per decoder layer. Encoder and pooling run once per
token; the latent transformer is O(C²) with C = S/16; the decoder is O(S·1024 + S·C).
**Controls share every line above** except the one they remove: arm C deletes the cross-attention
(decoder = segments only); the dense control is a plain 18-layer causal transformer.

**Sizes (pilot):** d 768, 6 q-heads × 128, 2 kv-heads, SwiGLU 2048, vocab 128,256. Compute
parameters ≈ enc 38M + latent 25M + dec 63M + pooler ≈ 2M ≈ **128M**; plus token table 32.8M,
untied head 98.5M, n-gram tables 33.6M, value-embedding tables (enc 0,3; dec 0) 24.6M → ≈ **318M
total** (reported as compute / dense / total in the run log — the E18 "125M" mislabel is not repeated).

## Why this is not a safe retread
It is not "cross-attention to 128 concepts again": the slot count scales with length, allocation
is positional, the decoder cannot bypass the array, and the array is processed by its own
transformer before being read. It is not E18 with a compressor either: E18/E21 have one K/V read
and no latent stack. Cross-domain hook: the design is Yao's two-party protocol at every segment
boundary (Alice = the array, Bob = the segment; the message is the only channel) and a
thalamocortical loop (fast local processing in the segment, slow integration over the array).
Frontier anchors for each law: DeepSeek V4 HCA (positional 128:1 pooling, 128-token raw window at
every layer), V4.1-Flash CED (deep encoder → one projected read), LCLM (mean pooling > CLS pooling,
encoder window ≥ 256, loss on uncompressed tokens interleaved with compressed spans).

## Success criteria (set BEFORE running; pilot, Odra 3×3090, 32k, 0.5B tokens per arm)
- **S1 load-bearing:** paired concept ablation (`--probe concept`, `concept_override ∈ {real, none,
  shuffled}`) on ≥ 64 held-out PG-19 rows ≥ 32k: **CE(none) − CE(real) ≥ 0.30 nats** on tokens with
  ≥ 4096 tokens of history (mean over rows, ≥ 10σ); **CE(shuffled) − CE(real) ≥ 0.20** (content, not
  a generic prior); tokens in the first concept block (positions < r) are identical under every mode (built-in noise-floor check: no slot is visible to them).
- **S2 useful, not just used:** arm A eval loss on positions ≥ 4096 ≤ **0.97 ×** arm C's at equal
  tokens (E18 arm A = arm C was the kill; this is the same test).
- **S3 compression price:** arm A eval loss (all positions) within **3%** of the dense control at
  equal tokens; per-position buckets [0,1k) within 2% (segment-local tokens see identical raw context).
- **S4 recall through concepts:** passkey at 32k with the needle ≥ 4096 tokens before the query
  ≥ **0.5**; held-out keyed-recall rows (E18b protocol, `--probe tasks`) first-token accuracy ≥ **25%**
  (E18b-R 4.2%, dense DT 99.3%).
- **S5 geometry (diagnostic, not a gate):** per-sample RankMe of z ≥ 64 (of 768); no dead slots.

## Kill criteria (set BEFORE running)
- **K1:** at 40% of the budget, arm A's loss on positions ≥ 4096 is not below arm C's → the array adds
  nothing a segment-only decoder cannot recover → stop, run the mask/gradient-flow tests before any
  architecture change (a structural-closure design that fails S2 is far more likely a bug than a result).
- **K2:** S1 Δ_none < 0.05 nats at the end → dead channel despite closure → same diagnosis path; do
  not iterate r/c/K blind.
- **K3:** eval loss rising over 3 consecutive evals → stop, retune per the Muon protocol (E05).
- **K4:** throughput < 40% of the dense control at 32k → fix the kernel path before spending.
- Any pilot arm > 12 GPU-h without a checkpoint that evaluates → stop and debug.

## Plan
- **Data:** the E18b merged manifest on Polonez (`$DATASETS_TOK_DIR/../datasets_tok_smollm3_32k/e18b_lm_ret05_manifest.json`:
  PG-19, FinePDFs-100BT, FineWeb-Edu, stack-edu-py, 5% keyed-recall rows; SmolLM3 tokenizer, rows ≤ 32k),
  copied to Odra via the NAS. **Long-document reweighting** by `DATASET_MIX_WEIGHT_OVERRIDE` (row
  weights chosen for token share ≈ pg19 35% / finepdfs 35% / fineweb 15% / stack 10% / retrieval 5%)
  and `BATCH_PACKING_MODE=pack` so every training row is a full 32k sequence with `doc_ids`.
  Eval: manifest eval splits (PG-19 rows ≥ 32k feed the position-bucket and concept probes), the
  held-out keyed-recall rows, synthetic passkey / multikey / vt / fwe.
- **Compute:** Odra 3×3090 for arm A and arm C (Polonez after E21 finishes for the dense control).
  ≈ 8–12 GPU-h per arm at 32k (calibrate batch first).
- **Steps / epochs:** 0.5B tokens per arm; effective batch ≈ 0.5M tokens (per-device 1 × 3 GPUs ×
  accumulation 5 packed 32k rows); Muon lr 0.01 / AdamW 2e-4 / wd 0.1 / clip 0.5 (E05/E18-calibrated),
  100-step warmup, cosine to 10%. Same schedule for all arms.
- **Launch:** `bash scripts/launch_e22.sh` (thin wrapper; pins the protocol and delegates to the
  generic launcher) — arm A default; `PCL_CONCEPT_MODE=none bash scripts/launch_e22.sh` = arm C;
  `E22_ARM=dense bash scripts/launch_e22.sh` = the `perceiver_ar` dense control at 18 layers.
- **New foundation code (reusable, config-selectable):** `nn/perceiver_concept_lm.py`
  (`PerceiverConceptConfig`, `ConceptPooler`, `ConceptCrossAttention`, `PerceiverConceptLM` with
  `hidden_states`, `concept_override`, `concepts()`; per-token loss path shared with the probes);
  family registration in `training/concept_pretraining_factories.py` + `concept_pretraining_args.py`
  (`PCL_*` knobs); `DATASET_MIX_WEIGHT_OVERRIDE` plumbing in the generic launcher; `--probe concept`
  in `evaluation/long_context_probes.py` and family-aware model loading in the eval layer; tests.
- **Registered post-signal iterations (only after S1–S2 pass):** r ∈ {8, 32}; c = 2; latent
  repeats K ∈ {2, 4} (weight-tied) — the reasoning-bandwidth curve; encoder depth 6 → 10; decoder
  segment 512 / 2048 and `dec_local=swa` (window 256, DeepSeek-style) as the bypass ablation; 64k–128k
  stage on a PG-19/ProLong-heavy mix.

## Result
- Run ids: A `perceiver_concept_H768e6r16c1l4d8s1024_20260912_131735` (+ `_resumed`), C `…_152511`
  (Odra, 560 steps × 24 packed 32k rows ≈ 0.44B tokens each, 88% of the 0.5B budget); A′ `…_143726` (Polonez, duplicate); dense
  `perceiver_ar_dense_H768L0g0s18N2048_20260912_161135` (crashed ≈ step 390, checkpoint lost).
- Run report: [e22_pilot_verdict_20260912](../../2_Experiments_Registry/run_reports/e22_pilot_verdict_20260912.md) ·
  root cause: [e22_root_cause_20260912](../../4_Research_Notes/e22_root_cause_20260912.md)
- Verdict: **killed** — arm A = arm C on far tokens (S2 ratio ≈ 1.00, K1 met); Δ_none 0.25 but 0.22 of it
  already inside segment 0; the far slots' marginal is **0.05 nats, flat 1k→32k** (`near`/`far` ablation);
  passkey 0.0, keyed recall 4.8% (S4); RankMe 265 (S5 ✅ — the array is diverse, the decoder just never
  reads it for content). Root causes: CE on natural text pays ≈ 0.05 nats for far context at this scale,
  and the `cpos ≤ pos` mask let the array serve as local depth. Successor: E23 (exclusive scope + an
  objective that pays for far content).
