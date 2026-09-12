# E18 / E18b — first pass of the long-context + reasoning eval layer, and the E18b retrieval-mix arms (2026-09-12)

**Date:** 2026-09-12 (evaluation; E18b arms trained 2026-09-10 → 2026-09-11)
**Machine:** Polonez (4× RTX 3090); lm-eval on GPU 0/1, RULER-lite on GPU 2/3; no training during the suite
**Run IDs (E18b arms, all warm-started from stage A `checkpoint-9030`):**
- arm **R** (95% LM + 5% retrieval rows): `perceiver_ar_perceiver_H768L1g1s12N2048_20260910_184948`
- arm **0** (LM rows only — protocol control): `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_021311`
- arm **R2** (arm R + value embedding on the global read): `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_130626`
- arm **T** (retrieval rows only, 350M tokens — learnability control, added 2026-09-11): `perceiver_ar_perceiver_H768L1g1s12N2048_20260911_210251`
- arm **D** (dense 32k control): `perceiver_ar_dense_H768L1g1s12N2048_20260911_100947` — terminated (SIGTERM) after 78 steps when arm R2 was queued in its place; not evaluated
**WandB:** job type `train_perceiver_ar_causal_lm`, run names = run ids (project `ksopyla`)
**Raw logs:** `Cache/logs/e18b_arm_{R,0,R2,T,D}_*.log` · suite `Cache/logs/e18_baseline_eval.log`, `Cache/logs/e18_lmeval_redo_pair1.log` · probes `Cache/logs/eval_e18b_arm*.log`
**Artifacts:** `Cache/eval/{e18_stageA_ck9030,e18_dense_1b,e18b_arm0,e18b_armR,e18b_armR2,smollm2_135m}/` (lm-eval JSON, `longctx_suite.json`, `reach.json`, `health.log`), `Cache/eval/e18b/{0,R,R2,T}_{tasks,passkey,buckets16k,buckets32k}.json`, aggregate `Cache/eval/summary_e18_baselines.md`, `Cache/Evaluation_reports/lm_eval/summary.csv`
**Git commit:** eval layer `4e2352b` … `d137adf` (branch `cursor/strategy-sota-review-2026-09-e212`); arms trained at `1a89736`–`515820e`
**Git tag:** —
**Spec:** [E18b](../../experiments_specs/done_failed/E18b_retrieval_trained_read.md) · [plan](../../experiments_specs/done_failed/E18b_retrieval_trained_read_plan.md) · platform [E18](../../experiments_specs/ahead/E18_perceiver_ar_v2_baseline.md) · eval layer [engineering spec](../../engineering_specs/long_context_reasoning_eval_layer.md)

---

## Goal

Two things at once. (1) Put the new evaluation layer (lm-evaluation-harness `perceiver_ar` adapter on the
SmolLM2-card 0-shot tiers + teacher-forced RULER-lite probes behind one runner) on every checkpoint the
E18 family has produced, so E21 and later runs are judged against fixed numbers rather than ad-hoc
probes. (2) Close E18b: does 5% dense-label synthetic retrieval in the 32k mix turn the single global
read into a general, length-extrapolating retriever (S1 passkey ≥ 90% @32k, S2 ≥ 80% @128k) at ≤ 0.5%
LM cost (S3), without the extension protocol regressing a converged model (S4)?

## Configuration

| Item | Value |
|---|---|
| Family | `perceiver_ar` · `par_mode=perceiver` (1 pre-layer @512 → 1 full-causal global read → 12 stack layers @2048), 125M non-embedding (278.7M total with hashed n-gram tables); dense control = same depth, all layers full-causal |
| E18b arms | seq 32k, `length_group` batching, 32 rows/step (2 × 4 GPU × accum 4), Muon lr 0.002 (AdamW side 4e-5), cosine, warmup 100; 0.5B tokens per arm (5,219 steps ≈ 7.3–7.8 h; arm T 350M) |
| Data | LM = `e18_pilot_longdoc_v1` 32k tree; retrieval = `e18b_lm_ret05_manifest.json` (6,000 × 32k rows, row weight 0.00457 → 5.0% of tokens; 3-subword keys, natural-text values, ≈ 2.6k supervised tokens/row) |
| Reasoning eval | lm-eval 0.4.11, 0-shot: hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa, boolq, social_iqa (Hub parquet), commonsense_qa, lambada_openai, wikitext (word ppl); `batch_size=auto`; reference `HuggingFaceTB/SmolLM2-135M` through the same harness |
| Long-context eval | `evaluation/long_context_probes.py` teacher-forced, 64 rows: `buckets` (position-bucket CE at 32k), `passkey` / `multikey` / `vt` / `fwe` (exact = all target tokens argmax-correct; `_tok` = per-token accuracy) at 8k and 32k, `reach` (paired Δ CE when the global read is windowed) |
| Compute | not audited (`compute/audit_state` absent on the E18b runs); wall clock from logs: R 7.29 h, 0 7.84 h, R2 7.82 h, T 1.56 h on 4× 3090 |

## Training outcome (E18b)

All perceiver arms trained cleanly at 32k from the stage-A warm start. Final trainer eval loss (256 PG-19-heavy
rows): arm 0 **3.5208**, arm R **3.5221** (+0.04%), arm R2 **3.5221**, arm T 3.904 (task rows only — LM
forgetting expected). Train loss R vs R2 differs in the 6th digit: the value embedding on the global read
changed nothing. Arm D was stopped after 78 steps (see Run IDs).

## Evaluation

### Reasoning tier (lm-eval, 0-shot, acc / acc_norm as on the SmolLM2 card)

| model | avg 10 tasks | hellaswag | arc_e | piqa | winogrande | boolq | lambada | wikitext ppl |
|---|---|---|---|---|---|---|---|---|
| stage A `checkpoint-9030` (8k, 1.0B tok) | 0.348 | 0.266 | 0.348 | 0.567 | 0.519 | 0.621 | 0.121 | 109.0 |
| dense control (1.0B tok, 8k) | 0.348 | 0.265 | 0.354 | 0.564 | 0.511 | 0.613 | 0.117 | 110.2 |
| E18b arm 0 (+0.5B @32k) | **0.362** | 0.275 | 0.380 | 0.583 | 0.525 | 0.623 | **0.181** | **73.9** |
| E18b arm R | 0.361 | 0.274 | 0.382 | 0.572 | 0.528 | 0.622 | 0.175 | 74.3 |
| E18b arm R2 | 0.361 | 0.274 | 0.383 | 0.573 | 0.530 | 0.622 | 0.175 | 74.3 |
| SmolLM2-135M (reference, 2T tok) | 0.447 | 0.432 | 0.587 | 0.685 | 0.534 | 0.604 | 0.427 | 23.1 |

Chance level for the 10-task average is ≈ 0.30; the 1.0B-token checkpoints sit ≈ 5 points above it, the
1.5B-token E18b arms ≈ 6 points. Perceiver and dense are indistinguishable at equal tokens (P1 again, now on
public tasks). Full per-task table: `Cache/eval/summary_e18_baselines.md`.

### Long-context tier (teacher-forced, 64 rows)

| model | CE [0,8k) | CE [8k,32k) | passkey @8k | passkey @32k | multikey @8k / @32k | vt / fwe exact | Δ CE, read → 512 |
|---|---|---|---|---|---|---|---|
| stage A | 3.958 | 3.912 | 0.025 | 0 | 0 / 0 | 0 / 0 | +0.0006 |
| dense control | 3.958 | 3.965 | 0.025 | 0 | 0.025 / 0 | 0 / 0 | +0.12 (all layers; @2048: +0.03) |
| E18b arm 0 | 3.616 | 3.567 | 0.075 | 0 | 0.100 / 0 | 0 / 0 | ≈ 0 |
| E18b arm R | 3.612 | 3.555 | 0.075 | 0 | 0.075 / 0 | 0 / 0 | ≈ 0 |
| E18b arm R2 | 3.612 | 3.555 | 0.075 | 0 | 0.075 / 0 | 0 / 0 | ≈ 0 |

Per-token accuracies on the synthetic probes (`multikey_tok` 0.27–0.28, `vt_tok` 0.30, `fwe_tok` 0.48–0.53 at
8k) are LM guessing on the natural-text-shaped targets, identical across arms. The dense control's full
attention *hurts* beyond its 8k training length (CE [8k,32k) 3.965 vs 3.898 with an 8192 window).

### E18b task probe (held-out retrieval rows, 50 rows, 132,841 labelled tokens) and extrapolation

| arm | saw the task | first-token acc | token acc | passkey 8k / 32k / 64k / 128k |
|---|---|---|---|---|
| 0 | never | 2.4% | 0.307 | 0.075 / 0 / 0 / 0 |
| R | 5% of 0.5B | 4.2% | 0.316 | 0.075 / 0 / 0 / 0 |
| R2 | 5% of 0.5B (+ value embed on read) | 4.4% | 0.317 | 0.075 / 0 / 0 / 0 |
| T | 100% of 0.35B | 4.5% | 0.319 | 0.125 / 0 / — / 0 |

## Interpretation

Against E18b's own criteria: **S1 fails** (passkey @32k = 0 on every arm, target ≥ 90%); S2/S5 moot;
**S3 passes** (arm R LM loss +0.04% vs arm 0, ≤ 0.5%); **S4 passes** (arm 0 ≤ stage A at every bucket —
3.567 vs 3.912 at [8k,32k) — the corrected extension protocol is fine; K3 did not fire).
**K1's premise is not met either:** the training task itself was not learned. Arm R's first-token accuracy
on held-out retrieval rows (4.2%) is within noise of arm 0, which never saw a retrieval row (2.4%), and the
task-only arm T — 14× more supervised tokens than arm R saw, ~28M labelled tokens — reaches 4.5%. Token
accuracy ≈ 0.31 on all arms is the LM predicting natural-text values. So the question the spec asked
(does a learned retrieval circuit *transfer* to passkey and *extrapolate*?) was never reached: at this
label budget (P2 converged with ≈ 130M labels of a trivially framed copy task) and this framing (3-subword
keys among 24 items, natural-text values, 32k rows) the single read does not acquire the lookup at all.
The reach ablation says the same from the LM side: the global read carries ≈ 0 nats in every perceiver
arm, while the dense control shows a 0.03–0.12 nat prize exists.

Fair-comparison notes: perceiver vs dense is matched in tokens, depth and data (P1 holds on public tasks);
SmolLM2-135M is a 2T-token reference for the eval layer, not a competitor at 1.5B tokens. The suite is
teacher-forced and 64-row for the long-context probes (standard error on a 40-trial passkey ≈ 0.04), which
is enough to read 0 vs ≥ 0.5 but not to rank the arms.

## Decision

E18b → `done_failed` (kill zone: no transfer, and the mechanism could not be established because the
task was not learned; spec verdict **killed**). Do not run K1's fix iteration (more task families) on this
platform: arm T shows the bottleneck is learnability of the dense-label task by a plain-CE read, not the
mix fraction. The AWS main run stays **no** on the current E18 spec (unchanged from 2026-09-10).

Consequences for E21 (next on Polonez): E21 keeps the 5% E18b rows only as boundary-aware *probe* material;
its claim rests on the LM CE through the compressed message (`none`/`swapped`/`raw` ablations), which
this result does not touch. The `passkey`/`multikey`/`vt`/`fwe` exact scores are all 0 on every E18
checkpoint — the eval layer's headline long-context metric for E21 is therefore the message probe's Δ CE
(real vs none/swapped) and the position-bucket CE, with the RULER-lite exact scores as a stretch signal.
Eval-layer fixes made during this pass (Hub parquet `social_iqa`, runner `wait` deadlock, chunked
log-probs + `batch_size=auto`) are in CHANGELOG 2026-09-12.

*Related: `master_experiment_log.md`, `docs/experiments_specs/done_failed/E18b_retrieval_trained_read.md`, `e18_reach_ablation_20260909.md`, `e18_pilot_stageA_20260907.md`, `agenda.md`*
