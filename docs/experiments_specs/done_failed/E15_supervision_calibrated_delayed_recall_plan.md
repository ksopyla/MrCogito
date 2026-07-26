# E15 — Supervision-Calibrated Delayed Recall: Implementation Plan

- **Spec:** [E15_supervision_calibrated_delayed_recall.md](E15_supervision_calibrated_delayed_recall.md) · **Status:** implemented and run; objective protocol killed 2026-07-13
- **Authored by:** `implementation-plan` · for → `research-implement`

> Implement the approved spec's one change: extend E14's supervised-answer exposure.
> The checkpoint, data, objective, architecture, and causal evaluator remain unchanged.

## 1. Source & fit
- **Origin:** E14's preregistered gate in
  [the run report](../../2_Experiments_Registry/run_reports/e14_forced_delayed_recall_gate_20260713.md)
  fired after only 984 supervised answers; even its block-2 explicit-carry positive control
  remained at chance.
- **Synthesis verdict:** **Adapt** — calibrate the existing forced-use diagnostic by answer-label
  count before introducing denser multi-fact rows or changing the memory interface.
- **Architecture mapping (ONE):** this touches only the **training exposure / scheduler horizon**.
  It adds no model, data, loss, or evaluation behavior.

## 2. Reuse map

| Component | Action | Where |
|---|---|---|
| `BackboneConceptLM` / `BackboneConceptConfig` | reuse E14 checkpoint and real/static/zero/permutation paths as-is | `nn/backbone_concept_lm.py` |
| `DataCollatorForCausalLM(preserve_precomputed_labels=True)` | reuse as-is | `data/data_collators.py` |
| pretokenized E14 dataset + manifest | reuse immutable artifacts as-is | `$DATASETS_TOK_DIR/e14_delayed_recall*` |
| `run_delayed_recall_eval.py` | reuse block-2/3/4 evaluator as-is | `analysis/run_delayed_recall_eval.py` |
| generic E10 launcher | configure directly with environment overrides | `scripts/launch_e10.sh` |

No source-code component, config field, launcher wrapper, model registry entry, or checkpoint
contract changes. The cosine schedule keeps the same family, base LRs, and warmup; its total-step
horizon is mechanically extended from E14's 768-step budget to E15's 2,000-step budget.

## 3. Forward pass and metrics

Symbols: `B`=2 rows/GPU, `G`=3 GPUs, effective batch `E`=6 rows, `N`=2048,
`K`=512, `C`=128, `H`=1152, `V`=262144.

```text
input_ids, labels       (B,N), (B,N); labels=-100 except one answer token
z0                      (B,C,H)

for block b=0..3:
  explicit token input  previous block + current block, at most (B,2K)
  token hidden          frozen Gemma + LoRA + E10e concept reads -> (B,<=2K,H)
  sparse answer CE      lm_head at one valid position per row -> (B,V)
  recurrent write       z <- E10e BiXT write(z, current hidden) -> (B,C,H)

optimizer step          6 supervised answer labels and 6*2048 input tokens
global step 1000        6,000 labels / 12,288,000 input tokens
global step 2000        12,000 labels / 24,576,000 input tokens
```

The E14 checkpoint resumes at global step 164 with optimizer, scheduler, scaler, and RNG state.
Training targets global step 2,000 from the start so the resumed cosine scheduler has the final
frozen horizon. When checkpoint 1,000 is fully written, stop the process and discard any
post-checkpoint steps. Evaluate checkpoint 1,000; if the midpoint gate passes, resume that
checkpoint with the same 24.576M-token target.

## 4. Inputs & data
- **Dataset:** reuse E14's 4,608 train rows, 4,096 held-out block-4 rows, and 512-row block-2/3
  diagnostic views from
  `$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json`.
- **Collator:** reuse
  `data/data_collators.py:DataCollatorForCausalLM(preserve_precomputed_labels=True)`.
- **Masking:** exactly one answer label per 2,048-token row; therefore total supervised labels
  equal `global_step * effective_batch`.
- **Splits:** unchanged and immutable. Repeated train epochs are allowed; all held-out pair IDs
  and key/value assignments remain train-disjoint.
- **Identity checks:** record the manifest SHA256 and verify it matches E14's report; verify the
  checkpoint config has C=128/K=512/global_kv and checkpoint trainer state has global step 164.

## 5. Loss & training objective
- **Loss:** existing `BackboneConceptLM` answer-only next-token CE; no `LossManager` component.
- **Objective:** unchanged `objective_variant="causal_lm"` with precomputed sparse labels.
- **Weighting:** one answer per row, uniform.
- **Optimization:** restore E14 optimizer state; task LR 1e-4, concept-memory LR 3e-4,
  read/write gate init 0.01, read RMSNorm on, warmup 50, effective batch 6. The only
  optimization change is the budget-derived cosine horizon of 2,000 total steps.

## 6. Config, launch, and evaluation
- **New config fields:** none.
- **Registry/eval routing:** unchanged (`backbone_concept`, evaluation contract v1).
- **Source code:** no implementation work required.

Launch on Odra inside a fresh Byobu session:

```bash
source scripts/remote_paths.sh
EXPERIMENT_ID=E15 \
PRETOKENIZE_MIX=e14_delayed_recall \
MANIFEST="$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json" \
SKIP_PRETOKENIZE=1 \
TARGET_TOKENS=24576000 \
PRESERVE_PRECOMPUTED_LABELS=true \
PER_DEVICE_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=1 \
READ_CONCEPT_NORM=true \
READ_GATE_INIT=0.01 \
WRITE_GATE_INIT=0.01 \
CONCEPT_MEMORY_LR=3e-4 \
WARMUP_STEPS=50 \
AUTO_INTERVALS=0 \
EVAL_STEPS=1000 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=3 \
MAX_EVAL_SAMPLES=256 \
EVAL_BATCH_SIZE=2 \
RESUME_FROM_CHECKPOINT="$PWD/Cache/Training/backbone_concept_gemma_3_1b_pt_K512_concept_20260713_172219/checkpoint-164" \
bash scripts/launch_e10.sh
```

Midpoint evaluation after stopping at the completed checkpoint 1,000:

```bash
uv run python analysis/run_delayed_recall_eval.py \
  --checkpoint "Cache/Training/<e15-run>/checkpoint-1000" \
  --manifest "$DATASETS_TOK_DIR/e14_delayed_recall_gemma_manifest.json" \
  --eval_view block2 --num_pairs 256 --batch_size 16 \
  --output "Cache/Evaluation_reports/E15_checkpoint-1000_block2.json"
```

If block-2 real accuracy is at least 20%, resume checkpoint 1,000 with the exact launch
configuration above, changing only `RESUME_FROM_CHECKPOINT` to E15's checkpoint 1,000.
Evaluate the final checkpoint at block 2 (256 pairs), block 3 (256 pairs), and block 4
(2,048 pairs). The block-2 80% gate is evaluated before interpreting block-4 causality.

## 7. Tests & smoke
- No source changed, so no new unit test is required.
- Before launch, run `scripts/manifest_token_stats.py` with target 24,576,000 and effective batch
  6; require `estimated_optimizer_steps=2000`.
- Verify the E14 checkpoint loads and `trainer_state.json` reports `global_step=164`.
- Verify the direct launcher preflight prints E15, the immutable E14 manifest, effective batch
  6, target 24,576,000, sparse labels enabled, and resume checkpoint 164.
- Do not run a local Gemma-1B training smoke on macOS; E14 already verified the identical
  forward/data path.

## 8. Risks & tradeoffs
- **Local control still fails.** Cheapest signal: block-2 accuracy at step 1,000. Kill below
  20%; do not blame the memory interface.
- **Resume changes the cosine horizon.** This is the unavoidable mechanical consequence of the
  one changed exposure budget; base LRs, warmup, optimizer state, and schedule family stay fixed.
  Record the LR immediately before/after resume.
- **Repeated rows encourage memorization.** Held-out key/value assignments are train-disjoint,
  so the positive-control accuracy gate still requires learning the copy/query operation rather
  than recalling train targets.
- **Training advances after checkpoint 1,000 while monitoring notices the save.** Stop only after
  checkpoint integrity is confirmed, then discard later in-memory progress and resume the
  registered checkpoint.
- **Task passes but natural text does not.** Expected; E15 only validates or falsifies the
  forced-use diagnostic and E10e mechanism.
