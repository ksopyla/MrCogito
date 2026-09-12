# Long-context + reasoning evaluation layer for `perceiver_ar` — engineering spec (decision record)

- **Type:** engineering (eval-foundation). Not an `E0NN` experiment: it changes *how we measure*
  the E18 platform and its successors (E21 latent-message pretraining), not what we train.
- **Status:** implemented 2026-09-11 (code + tests); baseline runs on the E18 checkpoints follow
  as soon as Polonez is free.
- **Owner:** Krzysztof Sopyla
- **Serves:** the agenda's "proper evaluation before E21" gate. Every E18/E21 decision so far
  rested on training CE plus the ad-hoc passkey probe; neither tells us whether the model
  *reasons* (short-context) or *uses* its context beyond needle retrieval (long-context).

## Problem
1. The existing tiers (concept geometry, STS-B/SICK/PAWS/GLUE, generation vibe check) assume a
   concept bottleneck and a sentence-pair route. `perceiver_ar` has neither — the E18 family had
   **no reasoning evaluation at all**.
2. Long-context evidence was one probe (`passkey`) with exact match only, hand-launched per
   checkpoint from `Cache/jobs/*.sh`. Passkey is a retrieval test; it says nothing about
   aggregation (count / track / compare across the whole window), which is what E21's latent
   message must carry.
3. Numbers were not comparable across checkpoints or to public models: no shared schema, no
   reference rows, no aggregator.

## Decision
### Reasoning: lm-evaluation-harness (EleutherAI, v0.4.x), loglikelihood tasks only
- Adapter `evaluation/lm_eval_perceiver_ar.py` registers `perceiver_ar` as an `HFLM` subclass.
  The harness provides batching, request de-duplication, `acc`/`acc_norm`, stderr, and the same
  task definitions used by the SmolLM2 / Pythia / OLMo model cards, so our rows sit next to theirs.
- Loads with `attn_backend=sdpa`, `attn_pad_multiple=1`, `use_liger=False`: short prompts must
  not be padded to the 2048 training block (flex would also recompile per length). `_model_call`
  returns the **tanh-softcapped** logits the model was trained under; the test suite checks that
  `loglikelihood` equals a manual log-softmax of those logits.
- `_model_generate` raises: the family has no HF-compatible KV-cache `generate`, so
  generation-based tasks are out of scope for now (see "deferred").
- Tiers (`evaluation/run_lm_eval_suite.py`): `core` = hellaswag, arc_easy, arc_challenge,
  piqa, winogrande, openbookqa, boolq, social_iqa, commonsense_qa, lambada_openai, wikitext;
  `full` adds mmlu, sciq, copa. Main metric per task follows the SmolLM2 card (`acc_norm`
  for hellaswag / arc / piqa / openbookqa / sciq, `acc` otherwise, `word_perplexity` for wikitext).
  `--hf_model` runs any HF causal LM (SmolLM2-135M/360M reference rows) through the same runner.
- Output: `Cache/Evaluation_reports/lm_eval/<tag>.json` (full harness results + our summary)
  and `summary.csv` (one row per tag, upserted).

### Long context: teacher-forced RULER-lite in `evaluation/long_context_probes.py`
RULER's synthetic families, rewritten so a **base** model can be scored without instruction
following: the question is phrased as a natural continuation and the answer is scored by
teacher-forced greedy match (exact, token accuracy, first-token accuracy), per context length.
- `passkey` — single needle (existing probe, kept for continuity with E18/E18b logs).
- `multikey` — N=4 needles with distinct NATO-word keys, one queried (distractor robustness).
- `vt` — variable tracking: chains `VAR ABC = 12345`, `VAR DEF = ABC`, …; answer lists every
  variable bound to the value (multi-hop across the whole context).
- `fwe` — frequent-words extraction: Zipf-weighted coded words fill the context; answer is the
  three most frequent (global aggregation, no needle).
- `buckets` — held-out CE per length bucket from the manifest's eval split (fluency control).
- `reach` — sweep restricting every full layer to swa(W) for W in {512, 2048, 8192, full};
  positive control for "is the global read used".
- `--probe suite` runs several probes from one model load; per-probe failures are recorded in
  `errors` instead of aborting. Filler comes from real eval rows (concatenated cyclically) so
  the distractor text has natural statistics.

### Orchestration
`scripts/eval_perceiver_ar_suite.sh <ckpt> <tag>` runs health → lm-eval on one GPU while the
long-context suite + reach run on a second; every step is failure tolerant and the exit codes
are summarised. `evaluation/summarize_eval_suite.py` merges any set of tags into one markdown
table (reasoning block + long-context block, `-` for missing pieces).
`analysis/check_model_health.py --model_type perceiver_ar` is the Tier-0 health check.

## Why not …
- **lm-eval `ruler` / `niah_*` groups, HELMET, LongBench v2, InfiniteBench** — all
  generation-based and written for instruct models (≥1B). A 125M base model produces
  near-zero scores for reasons unrelated to context use, and the family has no `generate`
  with KV cache. **Deferred** until an instruct-tuned E21 checkpoint exists; the adapter's
  `_model_generate` is the single place to add it.
- **Custom reasoning benchmarks** — would not be comparable to any published number.
- **Concept-geometry tiers** — no concepts in `perceiver_ar`; E21's r=16 message slots get
  their own probe (`--probe message`, E21 plan) rather than a forced fit into the old tiers.

## Cost
Core tier at 0-shot on a 125M model: ~10–15 min on one GPU (hellaswag 10k + wikitext dominate).
Long-context suite at 8k + 32k with 8 trials × 4 probes + 64 bucket rows: ~10–20 min with
flex; the 64k/128k sweep adds roughly the same again. Whole suite < 1 GPU-hour per checkpoint.

## Success criteria for the layer itself
- Reference rows (SmolLM2-135M) reproduce the model-card numbers within stderr.
- The same checkpoint evaluated twice gives identical `summary.csv` rows (seeded).
- Every E18 baseline checkpoint yields a complete table row without manual intervention.

## Files
`evaluation/lm_eval_perceiver_ar.py`, `evaluation/run_lm_eval_suite.py`,
`evaluation/long_context_probes.py`, `evaluation/summarize_eval_suite.py`,
`scripts/eval_perceiver_ar_suite.sh`, `analysis/check_model_health.py`,
`tests/test_lm_eval_perceiver_ar.py`, `tests/test_long_context_probes.py`,
`tests/test_summarize_eval_suite.py`, `.cursor/skills/experiment-evaluate/SKILL.md`.
