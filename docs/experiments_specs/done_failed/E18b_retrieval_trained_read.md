# E18b — Retrieval-trained single read: does dense retrieval supervision turn the global read into a general, length-extrapolating retriever?

- **Status:** killed (arms trained 2026-09-10 → 2026-09-11; evaluated 2026-09-12) — see Result
- **Serves:** the re-scoped E18 long-context platform — *retrieval-class* long context at a 1 KB/token cache, dense-parity short-context quality — and the E19/E21 hooks (a read that retrieves is the message space they need). Decides the AWS main run.
- **Implementation plan:** [E18b_retrieval_trained_read_plan.md](E18b_retrieval_trained_read_plan.md) *(authored by `implementation-plan`; the HOW)*
- **Owner / dates:** Krzysztof Sopyla · opened 2026-09-10 · closed 2026-09-12

> Follow-up to [E18](../ahead/E18_perceiver_ar_v2_baseline.md). What E18's pilot established
> ([reach ablation + geometry arms](../../2_Experiments_Registry/run_reports/e18_reach_ablation_20260909.md)):
> the single global read is **free** at short context (P1), is a **working exact-retrieval organ** when
> labels demand it (P2: 99.9998% copy at a 16k offset; cut its reach two tokens short → 0.4%), is
> **used** when the local stack cannot reach (arm A: 0.03 nats, 27% of tokens, and it works past the
> training length) — but adds **nothing** to next-token loss that a read-free stack cannot recover
> (arm C: 4.091 vs 4.090). Natural text does not supervise long-range addressing. This spec asks whether
> a small amount of dense synthetic supervision does — and whether what it teaches is *general*.

## Hypothesis
If the single global read is trained with **5% dense-label synthetic retrieval rows** (variable-offset
span copy and key→value lookup, many targets per row, embedded in real text at 32k) alongside the
ordinary LM mix, then it learns a **content-addressing circuit that is general and length-agnostic**:
passkey retrieval on natural text it never saw rises from **0% to ≥ 90% at 32k**, still scores
**≥ 80% at 128k (4× the training length)**, and LM eval loss moves by **≤ 0.5%** — **because** softmax
attention over K/V is an associative memory whose *addressing* weights receive almost no gradient from
next-token prediction (arm C), while dense retrieval labels supply exactly that gradient (P2 converged
in < 500 steps), and content addressing, once formed, does not depend on absolute distance (arm A's
read extrapolated beyond 8k while the dense control's far reach hurt).

## Builds-on
- **Foundation:** `nn/perceiver_ar_lm.py` (`perceiver_ar` family, unchanged), `scripts/launch_e18.sh`
  (`E18_STAGE=32k` with the **corrected extension protocol**: warm start at ≤ 20% peak lr, 100-step
  warmup, decay — [AWS plan step 6](../ahead/E18_main_run_aws_plan.md)), `scripts/build_copy_task_dataset.py`
  (pattern for the new builder), pretokenized-manifest flow (multi-source manifests with weights),
  `evaluation/long_context_probes.py` (`--probe passkey`, `--probe reach`, `--probe buckets`).
- **Init / checkpoint:** warm start from stage A `perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943/checkpoint-9030`
  (125M, N=2048, 1.0B tokens at 8k). Dense arm warm-starts from
  `perceiver_ar_dense_H768L1g1s12N2048_20260907_193351/final`.
- **Baseline to beat:** passkey on every 125M/1B-token pilot model so far: **0.00 at 32k**, 0.075 at 16k
  (stage A, stage B, dense; `Cache/eval/e18/*_passkey.json`). LM reference: stage A position buckets at 32k
  (3.192 / 2.940 / 2.645 / 2.611) — the control arm must not regress against them.
- **Materially new:** (i) retrieval supervision on a **raw K/V read that is proven to retrieve** (P2), not
  on a compressed concept channel — E14/E15 ("forced / supervision-calibrated delayed recall") trained
  recall into a *bottleneck* on a frozen Gemma backbone and found it unused; here the channel is
  demonstrably functional and the question is *generality*, not existence; (ii) **dense-label** task
  design: a passkey row carries 5 supervised tokens, a copy row 16k — the pilot's copy converged on ~8k
  rows, so needle-style rows in the mix would be ~3,000× too sparse; (iii) transfer is the metric — no
  passkey-format rows are ever trained on; (iv) a dense control on the *same* mix, so the deck's claim
  ("the read extrapolates, dense does not") is measured, not assumed.

## The architectural bet
Nothing changes in the network. The bet is that **the read's missing ingredient is supervision density,
not capacity or depth**, and that the circuit it forms under dense synthetic supervision is the general
one (content addressing), not a task-specific one (fixed-offset copy). Three arms, one hypothesis:

| arm | model | mix | role |
|---|---|---|---|
| **R** | perceiver (stage A warm start) | LM 95% + retrieval 5% | the claim |
| **0** | perceiver (stage A warm start) | LM 100% | LM-cost of the mix; **re-validates the extension protocol** (stage B redo) |
| **D** | dense control (warm start) | LM 95% + retrieval 5% | architecture control: can 14 unbounded layers learn the same, and do they extrapolate? |

**Retrieval rows** (new reusable builder, `scripts/build_retrieval_mix_dataset.py`): 32k-token rows whose
filler is real text from the LM mix's *train* split (never eval), with embedded tasks:
- *span copy*: 8–16 spans of 64–512 tokens, each repeated once at a random distance 1k–30k later; labels
  on the repeat (≈ 2–4k supervised tokens per row);
- *key → value lookup*: 16–32 `(key, value)` pairs (random 4-token keys, 8-token values) scattered in the
  filler; later `key →` queries in random order; labels on the values;
- span/pair boundaries marked by two reserved tokens so the model can learn the task frame.
No passkey/NIAH format appears in training. The retrieval source contributes **no rows to the trainer's
eval split** (LM eval loss stays comparable across arms).

**Probes** (all three arms, `final`): passkey at 8k / 32k / 64k / 128k (filler = concatenated eval rows;
small probe extension), position buckets on the 32k and 16k PG-19 row sets, reach ablation (arm R must
show its read *load-bearing on passkey*: reach → 8k at 32k passkey must collapse it).

## Why this is not a safe retread
It is the first experiment in the ledger where the memory channel is *known to work* before supervision is
added (P2), so a negative result means "does not generalise", not "channel unused" (the E10–E17 outcome).
The analogy is a Hopfield/associative memory: energy landscape (K/V) is free, the *address decoder*
(query projection) is what training must shape, and LM loss gives it no gradient. If the circuit is
content-based it must be distance-agnostic — hence the 4× extrapolation criterion is part of the claim,
not an add-on.

## Success criteria (set BEFORE running)
- **S1 transfer:** arm R passkey (5-digit key, all digits argmax-correct, 5 depths × 8 trials) **≥ 90% at 32k**.
- **S2 extrapolation:** arm R passkey **≥ 80% at 128k** (4× training length) and ≥ 85% at 64k.
- **S3 LM cost:** arm R trainer eval loss (LM-only rows) within **0.5%** of arm 0; PG-19 buckets ≥ 8k within 0.5%.
- **S4 protocol:** arm 0 buckets at 32k **≤ stage A's** at every bucket (the corrected extension protocol does not regress a converged model; stage B was +2%).
- **S5 architecture:** arm R ≥ arm D at 128k by ≥ 15 points *or* arm D fails S2 — the deck claim.

## Kill criteria (set BEFORE running)
- **K1 no transfer:** arm R passkey at 32k < 50% at the end of the budget while its training-task accuracy
  (span copy / lookup on held-out rows) is ≥ 95% → the circuit is task-specific; **one fix iteration**:
  add two task families (reverse lookup value → key; sorted-list retrieval) at the same 5%; if still < 50%,
  stop — the single read cannot be made a general retriever this cheaply.
- **K2 no extrapolation:** S1 passes but 128k < 50% → **one fix iteration**: `PAR_GLOBAL_LOGIT_SCALE=log`
  (SSMax, tiny-study-safe) and rerun arm R; if still < 50%, the 1M claim needs a positional change (YaRN on
  the read) and moves to the main-run spec as a risk, not a pilot.
- **K3 protocol:** arm 0 regresses vs stage A by > 1% at any bucket → stop everything; the extension
  protocol is still wrong and no long-context stage may be launched until it is fixed.
- **K4 cost:** arm R LM loss > arm 0 by > 1.5% → the mix is too large; halve to 2.5% (one iteration).

## Plan
- **Data:** LM = `e18_pilot_longdoc_v1` (32k tree, existing). Retrieval = 6,000 train rows × 32k
  (≈ 200M tokens; 5% of the 0.5B budget ≈ 25M tokens seen ≈ 760 rows, > 2M supervised tokens — vs P2's
  convergence at ~8k rows × 16k labels this is 50× fewer labels, deliberately: the claim is that a
  *fraction* of the supervision suffices once mixed with a warm-started LM; K1 covers the alternative)
  + 200 held-out rows for the training-task accuracy probe. Merged manifest written by the builder
  (`--base_manifest … --fraction 0.05 --out_manifest …`).
- **Compute:** Polonez 4×3090 (Odra if thermals misbehave). ~8 GPU-h per arm at 32k + ~1 h probes each
  → **≈ 27 GPU-h**, run as a chain R → 0 → D (`Cache/jobs/e18b_chain.sh`).
- **Steps / epochs:** 0.5B tokens per arm at seq 32k, `PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4`
  (32 rows/step, as stage B), `length_group` batching.
- **Launch (arm R):**
  ```bash
  E18_STAGE=32k EXPERIMENT_ID=E18b \
  MODEL_NAME_OR_PATH=Cache/Training/perceiver_ar_perceiver_H768L1g1s12N2048_20260907_080943/checkpoint-9030 \
  LEARNING_RATE=0.002 MUON_ADAMW_LR=4e-5 WARMUP_STEPS=100 LR_SCHEDULER_TYPE=cosine \
  PRETOKENIZED_MANIFEST=$DATASETS_TOK_DIR/e18b_lm95_retrieval05_manifest.json \
  PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/launch_e18.sh
  ```
  Arm 0: same with the LM manifest. Arm D: `PAR_MODE=dense MODEL_NAME_OR_PATH=<dense final>`.
  Probes: `evaluation/long_context_probes.py --probe passkey --context_lengths 8192,32768,65536,131072`,
  `--probe buckets`, `--probe reach --reach_windows 8192,full` on the passkey set.
- **Data verified 2026-09-10 (Polonez, `Cache/jobs/e18b_prep.sh`):** 6,000 train / 200 eval rows at exactly 32,768
  tokens; merged manifest `e18b_lm_ret05_manifest.json` — row weight **0.00457** (base mean row 2,858 tokens →
  achieved token share **5.0%**), retrieval source `in_eval: false`; token stats 8.18B tokens/epoch (the
  all-exhausted interleave cycles the 6k rows ≈ 2.1×), length cache 2.73M rows (mean 2,994). Decoded row check:
  label density 8.1% mean (4.8–13.2%) ≈ 2.6k supervised tokens/row; 24 items in row 0; first target's source
  2,967 tokens earlier; keys are three ordinary subword tokens, values and filler are real PG-19/FinePDFs text.
- **New foundation code:** `scripts/build_retrieval_mix_dataset.py` (reusable, task-family flags,
  writes arrow + merged manifest; tests: label density, no eval leakage, reserved-token framing);
  `evaluation/long_context_probes.py`: passkey filler from concatenated rows for lengths beyond the
  tokenized row length, plus a `--probe tasks` accuracy on the held-out retrieval rows. No model changes.
- **Not in scope (follow-ups):** read placement (arm B of E18 decides), NoPE read (tiny study: stalls
  positional copy), compression of the read (E18c, gated on this spec passing).

## Decision this spec feeds
- **S1–S4 pass →** the AWS main run is re-scoped and unblocked: M2 = RULER ≥ 80 @128k, NIAH ≥ 95% @256k
  / ≥ 80% @1M (retrieval claims), the "≥ 3% lower loss" clause is dropped; 5% retrieval mix in stage 1;
  extension protocol per the AWS plan; read position per arm B; `PAR_GLOBAL_LOGIT_SCALE=log` iff K2 fired.
- **K1 or K3 →** do not launch AWS; E18 family closes as *platform validated, retrieval claim not
  reachable at this cost*; the platform stays as the E19/E21 substrate.

## Result
- Run ids: R `perceiver_ar_perceiver_H768L1g1s12N2048_20260910_184948` · 0 `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_021311` ·
  R2 (R + value embedding on the read, added 2026-09-11) `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_130626` ·
  T (retrieval rows only, 350M tokens, learnability control, added 2026-09-11) `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_210251` ·
  D `perceiver_ar_dense_H768L1g1s12N2048_20260911_100947` (terminated after 78 steps; not evaluated)
- WandB: job type `train_perceiver_ar_causal_lm`, run names = run ids
- Run report: [e18_baseline_eval_suite_20260912.md](../../2_Experiments_Registry/run_reports/e18_baseline_eval_suite_20260912.md)
- Verdict (2026-09-12): **killed** — S1 fails (passkey @32k **0** on every arm, also at 64k/128k); S3 passes (arm R LM loss
  +0.04% vs arm 0); S4 passes (arm 0 ≤ stage A at every bucket). K1's premise is unmet: the training task itself was not
  learned (held-out first-token accuracy R 4.2% ≈ arm 0's 2.4%, task-only arm T 4.5%; token accuracy ≈ 0.31 on all arms is
  the LM predicting natural-text values). The fix iteration (more task families) is not run: the bottleneck is learnability
  of the dense-label lookup by a plain-CE read at this label budget and framing, not the mix fraction. AWS main run stays
  **no** on the current E18 spec; the platform is the E21 substrate.
