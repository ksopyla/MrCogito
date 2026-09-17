# E27 — Hybrid (a, b): identity keys + pooled gist on exclusive E21 slots

- **Status:** draft 2026-09-16 (queue rank 1; awaiting KS approval) · not launched
- **Serves:** Vision “does the compressed channel carry *addressable* content?” — BAPO
  hybrid rather than uniform r=1 or uniform r=16. Queue:
  [e21_improvement_queue.md](../../4_Research_Notes/e21_improvement_queue.md).
- **Implementation plan:** [E27_hybrid_key_anchors_plan.md](E27_hybrid_key_anchors_plan.md)
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-16 · closed —

> One coherent bet: MATCH dies under mean-pool because **keys are not independently
> addressable**, not because values need full identity KV. E25 `type_marks` at r=1 added
> **zero** extra keys (marks already were replace slots). This is the untested hybrid.

## Hypothesis
If exclusive E21 at seq=512 packed `recall_single` keeps **r=16 frozen-mean slots** for
the bulk prefix and additionally exposes the DNA **key spans** (tokens after each
`keymark`, not the values) as raw identity keys, then recovered bits reach **≥ 0.75 ×
live E18** on that recipe (E25 E18 **47.94 bits**, gate ≈ **36 bits**) **because** MATCH
needs a `b` channel of addressable keys and an `a` channel of pooled values (BAPO MATCH
`a·b = Ω(n)` is a *count of items*, not slot width) — **while** anchors-only (slots
masked) stay near chance and `e18_local` stays at chance.

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`KVCompressor`, `build_message_anchors`,
  `message_global_anchors`), `evaluation/bapo_models.py` (`e21`),
  `verification/bapo_capability_probe.py`, `data/symbolic_tasks.py` recall blocks
  `[keymark, *key, *val]`. Shared probe; no new `train_*.py`.
- **Init / checkpoint:** random, H=256, E25 GPU recipe (inplace, identity_slots = frozen
  mean at r=16, SSMax log). No pretrained warm-start.
- **Baseline to beat:** E25 seq=512 `recall_single` r=16 frozen mean **0 bits**; same
  recipe identity r=1 **43.08 bits**; live E18 **47.94 bits**. Dense S0 already ≥ 75%.
- **Materially new:** `message_global_anchors=key_spans` — sparse raw keys on the *key
  field*, with r=16 mean on everything else. Not E25 type_marks-at-r=1. Not a width sweep.

## The architectural bet
```
prefix  →  r=16 frozen-mean slots (a = gist of values + filler)
        ∪  identity raw K/V on tokens in each key span (b = addressable keys)
QUERY severs SWA; answer attends exclusive slots ∪ key-span anchors only
```
**Out of scope:** prefix AE (E26); extra hop; `global_layers=2`; SELECT leftover;
CogitoProbe (Wave B); unfreezing `u`/`delta` under answer CE.

## Why this is not a safe retread
Uniform r=1 restores MATCH and **throws away compression**. Uniform r=16 copies INDEX
and **smears MATCH**. Literature lever 2 says spend the `b` budget on the keys that
must be matched, not on every token and not on wider heads. Surprising if it works: a
handful of identity key tokens plus a 16-token mean of the value is enough.

## Success criteria (set BEFORE running)
- **S0:** dense ≥ 75% on this replica (E25: ~99% / ~64 bits). Skip E21 if it misses.
- **S1 (the claim):** E21 hybrid recovered bits ≥ **0.75 × live E18** (cite this JSON;
  E25 bar was 35.95). `e18_local` at chance (K2).
- **S2 (hybrid, not leak):** anchors-only (`message_override` that keeps anchors and
  drops slots, or slots masked) recovered bits **< 0.20 ×** the hybrid. Slots-only
  (anchors `none`) stays near E25’s 0 bits.
- **S3 (don’t regress INDEX):** same compressor, `far_copy` seq=1024 r=16, flow ≥ 0.75
  × E18 (E25: 53.82 vs 0.75×47.34). Run only if S1 passes.

## Kill criteria (set BEFORE running)
- **K1:** dense < 75% — instrument; do not score E21.
- **K2:** `e18_local` > chance + 0.15 — QUERY leak.
- **K3:** S1 miss after 8k steps (4× E25’s 800, same extra-step as MATCH hunts) while
  identity r=1 in this replica still ≥ 0.75 × E18 — hybrid does not repair smear.
- **K4:** S2 miss (anchors-only ≈ hybrid) — the values leaked into the key spans; not
  an `(a,b)` result. Stop; do not call it compression.

## Plan
- **Data:** on-the-fly DNA `recall_single`, `--scale bridge --seq_len 512`.
- **Compute:** Odra 1×3090 (Polonez fallback). **~2 GPU-h** (dense + E18 + E21 hybrid
  + anchors-only + slots-only; 800 then 8k extra-step if climbing).
- **Launch:**
  ```bash
  uv run python verification/bapo_capability_probe.py \
    --scale bridge --seq_len 512 --recipe recall_single \
    --arch dense e18 e21 e18_local \
    --hidden 256 --global_logit_scale log \
    --message_ratio 16 --message_slots_inplace --message_identity_slots \
    --message_global_anchors key_spans \
    --steps 800 --k1_mult 4 \
    --out Cache/Evaluation_reports/e27_hybrid_key_anchors
  ```
- **New foundation code:** one reusable anchor mode `key_spans` in
  `build_message_anchors` (default `none`; E18-loadable). Specify the flag; do not
  ship a training fork. Probe CLI already takes `--message_global_anchors`.

## Result
- Run id: `<run_id>`
- WandB: n/a (probe) unless identity is wired
- Run report: `docs/2_Experiments_Registry/run_reports/<...>.md`
- Verdict: — 
